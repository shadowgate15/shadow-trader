"""Ledger tables and ``verify_ledger``.

The Ledger is the reward-independent record of a run: ``steps`` (one row per
stepped Bar) and ``fills`` (one row per executed order). Later tickets add
``closed_lots`` and ``episodes`` plus the persisted form and manifest; this module
is the in-memory part. No reward appears anywhere.

``verify_ledger`` checks, from the tables alone, that the record is internally
consistent: values are finite, step indexes are contiguous, fills reference real
steps, the position of each step equals the previous position plus that step's
fills (continuity), and equity reconciles with the cash implied by the fills and
rollover plus the position marked at the recorded mark price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from .config import EngineConfig
from .engine import StepEvents

__all__ = [
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
}
FILLS_SCHEMA: dict[str, pl.DataType] = {
    "step": pl.Int64(),
    "time": UTC_US,
    "instrument": pl.String(),
    "units": pl.Int64(),
    "price": pl.Float64(),
    "spread_cost": pl.Float64(),
    "slippage_cost": pl.Float64(),
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
                )
            )

    def build(self) -> Ledger:
        return Ledger(
            steps=pl.DataFrame(self._steps, schema=STEPS_SCHEMA, orient="row"),
            fills=pl.DataFrame(self._fills, schema=FILLS_SCHEMA, orient="row"),
            starting_equity=self._config.starting_equity,
        )


def verify_ledger(ledger: Ledger) -> None:
    """Raise ``LedgerError`` listing every violated invariant; return None if clean."""
    problems: list[str] = []
    steps, fills = ledger.steps, ledger.fills

    for name, frame in (("steps", steps), ("fills", fills)):
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
    elif fills.height:
        problems.append("fills reference a step that is not in steps")

    if problems:
        raise LedgerError("; ".join(problems))


def _replay(ledger: Ledger) -> list[str]:
    """Position continuity and equity reconciliation, replayed row by row."""
    steps, fills = ledger.steps, ledger.fills
    traded = [0] * steps.height  # signed units filled per step
    cash_out = [0.0] * steps.height  # sum of units x price per step, in fill order
    for s, units, price in zip(
        fills["step"].to_list(),
        fills["units"].to_list(),
        fills["price"].to_list(),
        strict=True,
    ):
        traded[s] += units
        cash_out[s] += units * price

    problems: list[str] = []
    cash = ledger.starting_equity
    position = 0
    for row, spent, units in zip(
        steps.iter_rows(named=True), cash_out, traded, strict=True
    ):
        s = row["step"]
        position += units
        if row["position_units"] != position:
            problems.append(
                f"position at step {s} is {row['position_units']} but the fills "
                f"imply {position}"
            )
            break  # later rows would only repeat the discrepancy
        cash = cash - spent - row["rollover_charge"]
        value = position * row["mark_price"]
        expected = cash + value
        scale = max(abs(expected), abs(cash), abs(value), ledger.starting_equity)
        if abs(row["equity"] - expected) > RECONCILIATION_RTOL * scale:
            problems.append(
                f"equity at step {s} is {row['equity']!r} but cash and position "
                f"imply {expected!r}"
            )
            break
    return problems
