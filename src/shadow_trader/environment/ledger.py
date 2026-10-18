"""Ledger tables and ``verify_ledger``.

The Ledger is the reward-independent record of a run: ``steps`` (one row per
stepped Bar), ``fills`` (one row per executed order) and ``closed_lots`` (one row
per FIFO lot closed, with its realised P&L). Later tickets add ``episodes`` plus
the persisted form and manifest; this module is the in-memory part. No reward
appears anywhere.

``verify_ledger`` checks, from the tables alone, that the record is internally
consistent: values are finite, step indexes are contiguous, fills reference real
steps, the position of each step equals the previous position plus that step's
fills (continuity), and equity reconciles with the cash implied by the fills and
rollover plus the position marked at the recorded mark price. Lot conservation
replays the fills through FIFO and requires ``closed_lots`` to match exactly; the
open lots left must sum to the final position, and equity must equal starting
equity plus cumulative realised P&L plus the mark-to-market of the open lots minus
cumulative rollover. Lot prices are fill prices, so spread and slippage are
already inside those terms and are not subtracted again.

Margin and closeout: when ``Ledger.leverage_cap`` is known, a step that grows the
position must stay within ``equity before the step x cap`` of notional at the
fill's open bid (recovered from the fill price less its recorded costs). A closeout
row is exempt from nothing else: it must be the last row, flat, and carry a
closeout-flagged fill, and closeout-flagged fills appear nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from .config import EngineConfig
from .engine import StepEvents

__all__ = [
    "CLOSED_LOTS_SCHEMA",
    "FILLS_SCHEMA",
    "STEPS_SCHEMA",
    "Ledger",
    "LedgerBuilder",
    "LedgerError",
    "verify_ledger",
]

UTC_US = pl.Datetime("us", "UTC")

STEPS_SCHEMA: dict[str, pl.DataType] = {
    "step": pl.Int64(),
    "open_time": UTC_US,
    "episode_id": pl.Int64(),
    "session_id": pl.Int64(),
    "equity": pl.Float64(),
    "used_margin": pl.Float64(),
    "position_units": pl.Int64(),
    "rollover_charge": pl.Float64(),
    "mark_price": pl.Float64(),
    "clipped": pl.Boolean(),
    "shortfall_units": pl.Int64(),
    "closeout": pl.Boolean(),
}
FILLS_SCHEMA: dict[str, pl.DataType] = {
    "step": pl.Int64(),
    "time": UTC_US,
    "instrument": pl.String(),
    "units": pl.Int64(),
    "price": pl.Float64(),
    "spread_cost": pl.Float64(),
    "slippage_cost": pl.Float64(),
    "clipped": pl.Boolean(),
    "closeout": pl.Boolean(),
}

CLOSED_LOTS_SCHEMA: dict[str, pl.DataType] = {
    "step": pl.Int64(),
    "time": UTC_US,
    "instrument": pl.String(),
    "units": pl.Int64(),
    "open_price": pl.Float64(),
    "close_price": pl.Float64(),
    "realised_pnl": pl.Float64(),
}

RECONCILIATION_RTOL = 1e-9


class LedgerError(Exception):
    """A ledger violates an invariant; the message lists every violation found."""


@dataclass(frozen=True)
class Ledger:
    """In-memory ledger tables plus the account facts needed to verify them."""

    steps: pl.DataFrame
    fills: pl.DataFrame
    starting_equity: float
    closed_lots: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=CLOSED_LOTS_SCHEMA)
    )
    leverage_cap: float | None = None  # effective cap; None skips the margin check


class LedgerBuilder:
    """Accumulates ``StepEvents`` into ``steps`` and ``fills`` frames.

    One builder records one run of a single-Instrument engine; ``step`` indexes
    count rows added. Multi-Instrument ledgers need a per-Instrument ``steps``
    layout and are not supported yet.
    """

    def __init__(self, config: EngineConfig) -> None:
        if len(config.instruments) != 1:
            raise NotImplementedError("ledger supports a single Instrument for now")
        self._config = config
        self._symbol = config.instruments[0].symbol
        self._steps: list[tuple[object, ...]] = []
        self._fills: list[tuple[object, ...]] = []
        self._closed: list[tuple[object, ...]] = []

    def add_step(
        self,
        open_time: datetime,
        events: StepEvents,
        *,
        episode_id: int = 0,
        session_id: int = 0,
    ) -> None:
        index = len(self._steps)
        self._steps.append(
            (
                index,
                open_time,
                episode_id,
                session_id,
                events.equity,
                events.used_margin,
                events.positions[0],
                events.rollover_charge,
                events.mark_prices[0],
                events.clipped,
                events.shortfall[0] if events.shortfall else 0,
                events.closeout,
            )
        )
        for fill in events.fills:
            self._fills.append(
                (
                    index,
                    open_time,
                    self._symbol,
                    fill.units,
                    fill.price,
                    fill.spread_cost,
                    fill.slippage_cost,
                    fill.clipped,
                    fill.closeout,
                )
            )

        for lot in events.closed_lots:
            self._closed.append(
                (
                    index,
                    open_time,
                    self._symbol,
                    lot.units,
                    lot.open_price,
                    lot.close_price,
                    lot.realised_pnl,
                )
            )

    def build(self) -> Ledger:
        return Ledger(
            steps=pl.DataFrame(self._steps, schema=STEPS_SCHEMA, orient="row"),
            fills=pl.DataFrame(self._fills, schema=FILLS_SCHEMA, orient="row"),
            starting_equity=self._config.starting_equity,
            closed_lots=pl.DataFrame(
                self._closed, schema=CLOSED_LOTS_SCHEMA, orient="row"
            ),
            leverage_cap=self._config.effective_leverage_cap,
        )


def verify_ledger(ledger: Ledger) -> None:
    """Raise ``LedgerError`` listing every violated invariant; return None if clean."""
    problems: list[str] = []
    steps, fills = ledger.steps, ledger.fills

    closed = ledger.closed_lots
    for name, frame in (("steps", steps), ("fills", fills), ("closed_lots", closed)):
        for column in frame.columns:
            if frame.schema[column].is_float():
                values = frame[column].to_list()
                if any(v is None or not math.isfinite(v) for v in values):
                    problems.append(f"{name}.{column} has non-finite values")

    index = steps["step"].to_list()
    if index != list(range(len(index))):
        problems.append("steps.step is not contiguous from 0")
    elif steps.height:
        fill_steps = fills["step"].to_list()
        if any(not 0 <= s < steps.height for s in fill_steps):
            problems.append("fills reference a step that is not in steps")
        elif not problems:
            problems.extend(_replay(ledger))
            problems.extend(_check_closeout(ledger))
    elif fills.height:
        problems.append("fills reference a step that is not in steps")

    if problems:
        raise LedgerError("; ".join(problems))


def _replay(ledger: Ledger) -> list[str]:
    """Position continuity, FIFO lot conservation and equity reconciliation."""
    steps, fills = ledger.steps, ledger.fills
    by_step: dict[int, list[tuple[int, float]]] = {}
    open_bid: dict[int, float] = {}  # per step: open bid behind its last opening fill
    for s, units, price, spread, slip, closeout in zip(
        fills["step"].to_list(),
        fills["units"].to_list(),
        fills["price"].to_list(),
        fills["spread_cost"].to_list(),
        fills["slippage_cost"].to_list(),
        fills["closeout"].to_list(),
        strict=True,
    ):
        by_step.setdefault(s, []).append((units, price))
        if not closeout:  # a closeout fill is priced at the close, not the open
            per_unit = (spread + slip) / abs(units)
            open_bid[s] = price - per_unit if units > 0 else price + per_unit

    problems: list[str] = []
    lots: list[tuple[int, float]] = []  # open FIFO lots: (signed units, price)
    expected_closed: list[tuple[int, int, float, float, float]] = []
    cash = ledger.starting_equity
    position = 0
    realised = 0.0
    rollover = 0.0
    previous_equity = ledger.starting_equity
    for row in steps.iter_rows(named=True):
        s = row["step"]
        before = position
        spent = 0.0  # sum of units x price in fill order
        for units, price in by_step.get(s, ()):
            spent += units * price
            position += units
            remaining = units
            while remaining != 0 and lots and (lots[0][0] > 0) != (remaining > 0):
                held, open_price = lots[0]
                if abs(remaining) >= abs(held):
                    taken = held
                    lots.pop(0)
                else:
                    taken = -remaining
                    lots[0] = (held - taken, open_price)
                pnl = taken * (price - open_price)
                expected_closed.append((s, taken, open_price, price, pnl))
                realised += pnl
                remaining += taken
            if remaining != 0:
                lots.append((remaining, price))
        if row["position_units"] != position:
            problems.append(
                f"position at step {s} is {row['position_units']} but the fills "
                f"imply {position}"
            )
            break  # later rows would only repeat the discrepancy
        if ledger.leverage_cap is not None and s in open_bid:
            peak = max(abs(before), abs(position))  # position right after the open
            if peak > max(abs(before), 1):  # one unit is the minimum size
                limit = previous_equity * ledger.leverage_cap
                if peak * open_bid[s] > limit * (1 + RECONCILIATION_RTOL) + 1e-9:
                    problems.append(
                        f"step {s} holds {peak} units, beyond the leverage limit "
                        f"of {limit!r} notional on equity {previous_equity!r}"
                    )
        previous_equity = row["equity"]
        cash = cash - spent - row["rollover_charge"]
        rollover += row["rollover_charge"]
        value = position * row["mark_price"]
        expected = cash + value
        scale = max(abs(expected), abs(cash), abs(value), ledger.starting_equity)
        if abs(row["equity"] - expected) > RECONCILIATION_RTOL * scale:
            problems.append(
                f"equity at step {s} is {row['equity']!r} but cash and position "
                f"imply {expected!r}"
            )
            break
        # the same equity from lots: realised P&L plus open lots marked, less rollover
        mtm = sum(units * (row["mark_price"] - price) for units, price in lots)
        by_lots = ledger.starting_equity + realised + mtm - rollover
        scale = max(abs(by_lots), abs(realised), abs(mtm), ledger.starting_equity)
        if abs(row["equity"] - by_lots) > RECONCILIATION_RTOL * scale:
            problems.append(
                f"equity at step {s} is {row['equity']!r} but realised P&L and "
                f"open lots imply {by_lots!r}"
            )
            break

    if not problems:
        if sum(units for units, _ in lots) != position:
            problems.append("open lots do not sum to the final position")
        problems.extend(_check_closed_lots(ledger.closed_lots, expected_closed))
    return problems


def _check_closeout(ledger: Ledger) -> list[str]:
    """A closeout is the last row, flat, with a closeout-flagged fill."""
    steps, fills = ledger.steps, ledger.fills
    flagged = steps.filter(pl.col("closeout"))["step"].to_list()
    fill_steps = set(fills.filter(pl.col("closeout"))["step"].to_list())
    problems: list[str] = []
    if len(flagged) > 1 or (flagged and flagged[0] != steps.height - 1):
        problems.append("a closeout row is not the last row of the episode")
    if fill_steps - set(flagged):
        problems.append("a closeout-flagged fill is on a row that is not a closeout")
    for s in flagged:
        if steps["position_units"][s] != 0:
            problems.append(f"closeout row {s} is not flat")
        if s not in fill_steps:
            problems.append(f"closeout row {s} has no closeout-flagged fill")
    return problems


def _check_closed_lots(
    table: pl.DataFrame, expected: list[tuple[int, int, float, float, float]]
) -> list[str]:
    """The closed_lots table must equal the FIFO replay of the fills, row for row."""
    actual = list(
        zip(
            table["step"].to_list(),
            table["units"].to_list(),
            table["open_price"].to_list(),
            table["close_price"].to_list(),
            table["realised_pnl"].to_list(),
            strict=True,
        )
    )
    if len(actual) != len(expected):
        return [
            f"closed_lots has {len(actual)} rows but FIFO on the fills closes "
            f"{len(expected)} lots"
        ]
    for n, (got, want) in enumerate(zip(actual, expected, strict=True)):
        same = got[:2] == want[:2] and all(
            abs(g - w) <= RECONCILIATION_RTOL * max(1.0, abs(w))
            for g, w in zip(got[2:], want[2:], strict=True)
        )
        if not same:
            return [f"closed lot {n} is {got!r} but FIFO on the fills gives {want!r}"]
    return []
