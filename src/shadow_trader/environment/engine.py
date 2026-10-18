"""The pure, Gymnasium-free engine: the Execution model as ``step``.

``step(state, config, market, target) -> (new_state, events)`` is a pure function:
immutable state, frozen config, no RNG, no clock. Numerics are plain Python
``float`` (float64) for money and prices and ``int`` for units, with a fixed
operation order and no reassociation, so results are bitwise reproducible on one
platform.

Accounting is cash plus marked position value. A buy fills at the open bid plus the
open spread plus slippage, a sell at the open bid minus slippage. A long position
is marked at the close bid and a short at the close ask (bid plus close spread), so
the spread is paid once per round trip. A reversal is a single net fill: the spread
is paid on the full traded amount.

Instruments are indexed by position in ``config.instruments``; positions and
targets are tuples aligned with it while cash and equity are pooled (v1 is one
Instrument, and ``step`` also accepts a bare ``Market`` and a bare target for that
case).

Open positions are held as FIFO lots (``Lot``, signed units at the fill price).
A fill closes lots oldest first, each (partly) closed lot is reported as a
``ClosedLot`` with its realised P&L, and any remainder opens a new lot (a flip
through zero closes everything, then opens the opposite lot). Lot prices are
fill prices, so spread and slippage are already inside realised P&L and the
mark-to-market of open lots; no cost is subtracted again.

Rollover is accrued on the position held after the Bar's fills, on its notional
at the mark, ``rollover_days`` times the per-day rate (positive charges, negative
credits), and comes out of cash so it is part of equity.

Margin: ``clip_order`` is the one clipping path. An order may add exposure only
up to ``floor(equity x leverage_cap / open_bid)`` units; the rest is the shortfall.
The unit target stays frozen, so the shortfall is retried every Bar, until it fills
or a different level replaces the target. A clip never forces a reduction.

Closeout: after rollover and the close mark, used margin above zero with equity
below ``closeout_fraction`` x used margin closes the whole position at once, at
this Bar's close with normal spread and slippage (the env episode ends, so there is
no next open), flags the fill and resets the target to flat. ``StepEvents.used_margin``
keeps the margin that triggered it.

Session-end force-flat: on a Bar with ``is_session_final`` the engine overrides the
agent's target with flat, so any position is closed at that Bar's open with normal
spread and slippage. The fill is flagged ``forced_flat`` and its cost lands in this
(last) step; the target is left flat and a flat account is simply not traded.

Unfillable Bars (a Gap or a distrusted Bar): nothing trades and the Bar's own
prices are ignored. ``EngineState.trusted_marks`` holds each Instrument's last
trusted close bid and close spread (refreshed on every fillable Bar); an unfillable
Bar marks equity there (``StepEvents.carried_mark``), still accrues rollover on that
notional, never runs the closeout check, and leaves the target pending (it is
accepted and sized at the next fillable Bar). If it is the Session-final Bar, any
position is liquidated at the trusted close with normal spread and slippage
(``Fill.unfilled_forced_flat``). ``step`` raises ``ValueError`` if there is no
trusted close to carry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .config import EngineConfig

__all__ = [
    "ClosedLot",
    "EngineState",
    "Fill",
    "Lot",
    "Market",
    "OrderPlan",
    "StepEvents",
    "clip_order",
    "initial_state",
    "level_to_units",
    "step",
]


@dataclass(frozen=True, slots=True)
class Market:
    """One Instrument's prices and calendar facts for the Bar being stepped.

    ``fillable`` is priced and not distrusted, ``is_session_final`` marks the
    Session's final Bar and ``rollover_days`` is 0, 1 or 3. The calendar helper
    derives the last three from timestamps; the engine never reads a clock.
    """

    open_bid: float
    open_spread: float
    close_bid: float
    close_spread: float
    fillable: bool = True
    is_session_final: bool = False
    rollover_days: int = 0
    bars_to_session_final: int = 0


@dataclass(frozen=True, slots=True)
class Fill:
    """One executed order. ``units`` is signed (positive buys); costs are in USD."""

    instrument: int
    units: int
    price: float
    spread_cost: float
    slippage_cost: float
    clipped: bool = False  # the order was cut back by the margin limit
    closeout: bool = False  # the fill closes the position on margin closeout
    forced_flat: bool = False  # the fill is the Session-end force-flat
    unfilled_forced_flat: bool = False  # synthetic liquidation on an unfillable Bar


@dataclass(frozen=True, slots=True)
class Lot:
    """An open FIFO lot: signed ``units`` (positive long) filled at ``price``."""

    units: int
    price: float


@dataclass(frozen=True, slots=True)
class ClosedLot:
    """A lot (or the part of one) closed by a fill.

    ``units`` is signed like the lot it closed (positive for a long) and
    ``realised_pnl`` is ``units * (close_price - open_price)`` in USD.
    """

    instrument: int
    units: int
    open_price: float
    close_price: float
    realised_pnl: float


@dataclass(frozen=True, slots=True)
class EngineState:
    """Account state between steps. ``target_levels`` and ``target_units`` are the
    last accepted level and the unit target frozen when it changed."""

    cash: float
    equity: float
    positions: tuple[int, ...]
    target_levels: tuple[float, ...]
    target_units: tuple[int, ...]
    lots: tuple[tuple[Lot, ...], ...] = ()  # per Instrument, oldest first
    # per Instrument (close bid, close spread) of the last trusted (fillable) Bar
    trusted_marks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class StepEvents:
    """Everything a step did: the input to the ledger and to any summary."""

    fills: tuple[Fill, ...]
    equity: float
    used_margin: float
    rollover_charge: float
    positions: tuple[int, ...]
    mark_prices: tuple[float, ...]
    closed_lots: tuple[ClosedLot, ...] = ()
    clipped: bool = False  # some order was cut back by the margin limit
    shortfall: tuple[int, ...] = ()  # signed units still wanted, per Instrument
    closeout: bool = False  # the position was closed on margin closeout
    forced_flat: bool = False  # a Session-end force-flat order was filled
    carried_mark: bool = False  # some Bar was unfillable: marked at the trusted close
    unfilled_forced_flat: bool = False  # synthetic liquidation on an unfillable Bar


@dataclass(frozen=True, slots=True)
class OrderPlan:
    """What an order wants and what the margin limit allows (signed units)."""

    desired_delta: int
    allowed_delta: int
    clipped: bool
    shortfall: int  # target minus the post-fill position; 0 when nothing is clipped


def initial_state(
    config: EngineConfig,
    equity: float | None = None,
    trusted_marks: tuple[tuple[float, float], ...] = (),
) -> EngineState:
    """A flat account with every target flat, at ``equity`` (default: the config's
    starting equity; an evaluation sweep passes the equity carried in).

    ``trusted_marks`` seeds the (close bid, close spread) an unfillable first Bar
    carries: the close of the Bar the episode starts on."""
    n = len(config.instruments)
    start = config.starting_equity if equity is None else equity
    return EngineState(
        cash=start,
        equity=start,
        positions=(0,) * n,
        target_levels=(0.0,) * n,
        target_units=(0,) * n,
        lots=((),) * n,
        trusted_marks=trusted_marks,
    )


def _apply_fill(
    lots: list[Lot], instrument: int, delta: int, price: float
) -> list[ClosedLot]:
    """Apply a signed fill to one Instrument's FIFO lots, in place.

    Returns the lots closed, oldest first. Whatever of ``delta`` is left after
    closing opposite-signed lots opens a new lot at ``price``.
    """
    closed: list[ClosedLot] = []
    remaining = delta
    while remaining != 0 and lots and (lots[0].units > 0) != (remaining > 0):
        head = lots[0]
        if abs(remaining) >= abs(head.units):
            taken = head.units
            lots.pop(0)
        else:
            taken = -remaining  # same sign as the lot, smaller than it
            lots[0] = Lot(head.units - taken, head.price)
        closed.append(
            ClosedLot(
                instrument, taken, head.price, price, taken * (price - head.price)
            )
        )
        remaining += taken
    if remaining != 0:
        lots.append(Lot(remaining, price))
    return closed


def level_to_units(level: float, equity: float, price: float, cap: float) -> int:
    """Signed integer units for a level: floor(|level| x equity x cap / price),
    at least 1 for any non-zero level."""
    if level == 0.0:
        return 0
    units = max(1, math.floor(abs(level) * equity * cap / price))
    return units if level > 0 else -units


def clip_order(
    position: int, target_units: int, equity: float, price: float, cap: float
) -> OrderPlan:
    """Clip the order from ``position`` to ``target_units`` to the margin limit.

    The limit is ``floor(equity x cap / price)`` units on either side (at least the
    1-unit minimum while equity is positive, none otherwise). Reducing
    exposure is never clipped, and a position already beyond the limit is left as
    it is (never force-sold) while any addition becomes shortfall.
    """
    limit = max(1, math.floor(equity * cap / price)) if equity > 0 else 0
    desired = target_units - position
    sign = 1 if target_units > 0 else -1
    if abs(target_units) <= limit:
        allowed = target_units
    elif position != 0 and (position > 0) == (sign > 0):
        allowed = sign * min(abs(target_units), max(limit, abs(position)))
    else:
        allowed = sign * limit
    clipped = allowed != target_units
    return OrderPlan(desired, allowed - position, clipped, target_units - allowed)


def step(
    state: EngineState,
    config: EngineConfig,
    market: Market | Sequence[Market],
    target: float | Sequence[float],
) -> tuple[EngineState, StepEvents]:
    """Advance one Bar: accept the target, fill at the open, mark to the close."""
    markets = (market,) if isinstance(market, Market) else tuple(market)
    targets = (target,) if isinstance(target, int | float) else tuple(target)
    n = len(config.instruments)
    if len(markets) != n or len(targets) != n or len(state.positions) != n:
        raise ValueError(f"expected {n} market(s) and target(s) for {config.symbols}")
    for level in targets:
        if not -1.0 <= level <= 1.0:  # False for NaN as well
            raise ValueError(f"target {level} is outside [-1, 1]")
    for bar in markets:
        if bar.rollover_days not in (0, 1, 3):
            raise ValueError(f"rollover_days {bar.rollover_days} is not 0, 1 or 3")
    trusted = state.trusted_marks
    if not all(bar.fillable for bar in markets) and len(trusted) != n:
        raise ValueError("an unfillable Bar needs a trusted close to carry")
    closes = [
        (bar.close_bid, bar.close_spread) if bar.fillable else trusted[i]
        for i, bar in enumerate(markets)
    ]

    cap = config.effective_leverage_cap
    cash = state.cash
    positions = list(state.positions)
    target_units = list(state.target_units)
    lots = [list(held) for held in state.lots] or [[] for _ in range(n)]
    fills: list[Fill] = []
    closed_lots: list[ClosedLot] = []
    shortfall = [0] * n
    clipped = False
    forced_flat = False
    unfilled_forced_flat = False
    levels = [float(level) for level in targets]
    for i, spec in enumerate(config.instruments):
        bar = markets[i]
        if not bar.fillable:
            if not bar.is_session_final:  # the target stays pending, unaccepted
                levels[i] = state.target_levels[i]
                continue
            levels[i], target_units[i] = 0.0, 0
            held = positions[i]
            if held == 0:
                continue
            mark, mark_spread = closes[i]  # liquidate at the last trusted close
            if held > 0:
                price, spread_cost = mark - spec.slippage, 0.0
            else:
                price = mark + mark_spread + spec.slippage
                spread_cost = -held * mark_spread
            cash -= -held * price
            positions[i] = 0
            unfilled_forced_flat = True
            fills.append(
                Fill(
                    i,
                    -held,
                    price,
                    spread_cost,
                    abs(held) * spec.slippage,
                    unfilled_forced_flat=True,
                )
            )
            closed_lots.extend(_apply_fill(lots[i], i, -held, price))
            continue
        if bar.is_session_final:  # the engine overrides the target: be flat
            levels[i] = 0.0
            target_units[i] = 0
        elif targets[i] != state.target_levels[i]:  # a level change freezes new units
            target_units[i] = level_to_units(
                targets[i], state.equity, bar.open_bid, cap
            )
        plan = clip_order(
            positions[i], target_units[i], state.equity, bar.open_bid, cap
        )
        shortfall[i] = plan.shortfall
        clipped = clipped or plan.clipped
        delta = plan.allowed_delta
        if delta == 0:
            continue
        slippage = spec.slippage
        if delta > 0:
            price = bar.open_bid + bar.open_spread + slippage
            spread_cost = delta * bar.open_spread
        else:
            price = bar.open_bid - slippage
            spread_cost = 0.0
        cash -= delta * price
        positions[i] += delta
        forced = bar.is_session_final
        forced_flat = forced_flat or forced
        fills.append(
            Fill(
                i,
                delta,
                price,
                spread_cost,
                abs(delta) * slippage,
                plan.clipped,
                forced_flat=forced,
            )
        )
        closed_lots.extend(_apply_fill(lots[i], i, delta, price))

    # Rollover: a position held through the 17:00 New York marker pays the long or
    # short rate times the rollover days (3 on Wednesday) on its notional at the
    # mark. A positive rate is a charge, a negative one a credit. Fills at this
    # Bar's open come first, so a position closed at the open pays nothing.
    rollover_charge = 0.0
    for i, spec in enumerate(config.instruments):
        bar = markets[i]
        if bar.rollover_days == 0 or positions[i] == 0:
            continue
        close_bid, close_spread = closes[i]
        if positions[i] > 0:
            rate, mark = spec.rollover_long, close_bid
        else:
            rate, mark = spec.rollover_short, close_bid + close_spread
        rollover_charge += abs(positions[i]) * mark * rate * bar.rollover_days
    cash -= rollover_charge

    equity = cash
    used_margin = 0.0
    marks: list[float] = []
    for i, spec in enumerate(config.instruments):
        close_bid, close_spread = closes[i]
        mark = close_bid + close_spread if positions[i] < 0 else close_bid
        marks.append(mark)
        equity += positions[i] * mark
        used_margin += abs(positions[i]) * mark * spec.margin_rate

    # Closeout: equity under the closeout fraction of used margin closes everything.
    all_fillable = all(bar.fillable for bar in markets)  # only trusted prices count
    closeout = (
        all_fillable
        and used_margin > 0.0
        and equity < config.closeout_fraction * used_margin
    )
    if closeout:
        for i, spec in enumerate(config.instruments):
            held = positions[i]
            if held == 0:
                continue
            bar = markets[i]
            if held > 0:
                price = bar.close_bid - spec.slippage
                spread_cost = 0.0
            else:
                price = bar.close_bid + bar.close_spread + spec.slippage
                spread_cost = -held * bar.close_spread
            cash -= -held * price
            fills.append(
                Fill(
                    i, -held, price, spread_cost, abs(held) * spec.slippage, False, True
                )
            )
            closed_lots.extend(_apply_fill(lots[i], i, -held, price))
            positions[i] = 0
            target_units[i] = 0
            levels[i] = 0.0
        equity = cash

    new_state = EngineState(
        cash=cash,
        equity=equity,
        positions=tuple(positions),
        target_levels=tuple(levels),
        target_units=tuple(target_units),
        lots=tuple(tuple(held) for held in lots),
        trusted_marks=tuple(closes),
    )
    events = StepEvents(
        fills=tuple(fills),
        equity=equity,
        used_margin=used_margin,
        rollover_charge=rollover_charge,
        positions=new_state.positions,
        mark_prices=tuple(marks),
        closed_lots=tuple(closed_lots),
        clipped=clipped,
        shortfall=tuple(shortfall),
        closeout=closeout,
        forced_flat=forced_flat,
        carried_mark=not all_fillable,
        unfilled_forced_flat=unfilled_forced_flat,
    )
    return new_state, events
