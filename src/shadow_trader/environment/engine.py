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

Not yet handled (later tickets): margin clipping and shortfall retry,
closeout, rollover accrual, Session-end force-flat and unfillable Bars. ``step``
raises ``NotImplementedError`` for a ``Market`` that needs them rather than
silently ignoring it.
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
    "StepEvents",
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


@dataclass(frozen=True, slots=True)
class Fill:
    """One executed order. ``units`` is signed (positive buys); costs are in USD."""

    instrument: int
    units: int
    price: float
    spread_cost: float
    slippage_cost: float


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


def initial_state(config: EngineConfig) -> EngineState:
    """A flat account at the starting equity with every target flat."""
    n = len(config.instruments)
    return EngineState(
        cash=config.starting_equity,
        equity=config.starting_equity,
        positions=(0,) * n,
        target_levels=(0.0,) * n,
        target_units=(0,) * n,
        lots=((),) * n,
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
        if not bar.fillable or bar.is_session_final or bar.rollover_days != 0:
            raise NotImplementedError(
                "unfillable, Session-final and rollover Bars are not handled yet"
            )

    cap = config.effective_leverage_cap
    cash = state.cash
    positions = list(state.positions)
    target_units = list(state.target_units)
    lots = [list(held) for held in state.lots] or [[] for _ in range(n)]
    fills: list[Fill] = []
    closed_lots: list[ClosedLot] = []
    for i, spec in enumerate(config.instruments):
        bar = markets[i]
        if targets[i] != state.target_levels[i]:  # a level change freezes new units
            target_units[i] = level_to_units(
                targets[i], state.equity, bar.open_bid, cap
            )
        delta = target_units[i] - positions[i]
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
        fills.append(Fill(i, delta, price, spread_cost, abs(delta) * slippage))
        closed_lots.extend(_apply_fill(lots[i], i, delta, price))

    equity = cash
    used_margin = 0.0
    marks: list[float] = []
    for i, spec in enumerate(config.instruments):
        bar = markets[i]
        mark = bar.close_bid + bar.close_spread if positions[i] < 0 else bar.close_bid
        marks.append(mark)
        equity += positions[i] * mark
        used_margin += abs(positions[i]) * mark * spec.margin_rate

    new_state = EngineState(
        cash=cash,
        equity=equity,
        positions=tuple(positions),
        target_levels=tuple(float(level) for level in targets),
        target_units=tuple(target_units),
        lots=tuple(tuple(held) for held in lots),
    )
    events = StepEvents(
        fills=tuple(fills),
        equity=equity,
        used_margin=used_margin,
        rollover_charge=0.0,
        positions=new_state.positions,
        mark_prices=tuple(marks),
        closed_lots=tuple(closed_lots),
    )
    return new_state, events
