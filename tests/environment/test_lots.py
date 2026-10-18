"""FIFO lots, StepEvents.closed_lots and the ledger closed_lots table.

Reference config: 1,000 USD, leverage cap 10, zero spread and zero slippage at a
price of 1.00, so level 1.0 is 10,000 units. Lot prices are fill prices, so costs
are already inside realised P&L.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow_trader.environment import (
    EngineConfig,
    Ledger,
    LedgerBuilder,
    LedgerError,
    Market,
    initial_state,
    step,
    verify_ledger,
)

SPEC = replace(EngineConfig().instruments[0], slippage_pips=0.0)
CONFIG = EngineConfig(instruments=(SPEC,), starting_equity=1_000.0)
T0 = datetime(2024, 1, 2, tzinfo=UTC)


def bar(open_bid: float, close_bid: float | None = None) -> Market:
    return Market(
        open_bid=open_bid,
        open_spread=0.0,
        close_bid=open_bid if close_bid is None else close_bid,
        close_spread=0.0,
    )


def run(script: list[tuple[Market, float]], config: EngineConfig = CONFIG):
    state = initial_state(config)
    builder = LedgerBuilder(config)
    all_events = []
    for i, (market, target) in enumerate(script):
        state, events = step(state, config, market, target)
        builder.add_step(T0 + timedelta(minutes=5 * i), events)
        all_events.append(events)
    return state, all_events, builder.build()


def test_opening_a_position_closes_nothing_and_keeps_one_lot() -> None:
    state, (events,), _ = run([(bar(1.0), 1.0)])
    assert events.closed_lots == ()
    assert [(lot.units, lot.price) for lot in state.lots[0]] == [(10_000, 1.0)]


def test_partial_close_realises_pnl_on_the_closed_units() -> None:
    state, events, ledger = run([(bar(1.0), 1.0), (bar(1.10, 1.10), 0.5)])
    # level change at equity 1,000: 0.5 -> floor(0.5 * 1000 * 10 / 1.1) = 4,545
    (closed,) = events[1].closed_lots
    assert closed.units == 5_455
    assert closed.open_price == 1.0
    assert closed.close_price == 1.10
    assert closed.realised_pnl == pytest.approx(545.5, abs=1e-9)
    assert [(lot.units, lot.price) for lot in state.lots[0]] == [(4_545, 1.0)]
    assert ledger.closed_lots.height == 1
    verify_ledger(ledger)


def test_flip_through_zero_closes_the_long_and_opens_a_short_lot() -> None:
    script = [(bar(1.0), 1.0), (bar(1.10), -1.0), (bar(1.05), 0.0)]
    state, events, ledger = run(script)
    # flip: short floor(1000 * 10 / 1.1) = 9,090 opened at 1.10
    (flip_close,) = events[1].closed_lots
    assert flip_close.units == 10_000
    assert flip_close.realised_pnl == pytest.approx(1_000.0, abs=1e-9)
    # cover the short at 1.05: -9,090 * (1.05 - 1.10)
    (cover,) = events[2].closed_lots
    assert cover.units == -9_090
    assert cover.open_price == 1.10
    assert cover.close_price == 1.05
    assert cover.realised_pnl == pytest.approx(454.5, abs=1e-9)
    assert state.lots[0] == ()
    assert ledger.closed_lots["realised_pnl"].sum() == pytest.approx(1_454.5)
    assert ledger.steps["equity"][-1] == pytest.approx(1_000 + 1_454.5, abs=1e-8)
    verify_ledger(ledger)


def test_lots_close_first_in_first_out() -> None:
    script = [(bar(1.0), 0.5), (bar(1.2, 1.2), 1.0), (bar(1.3), 0.0)]
    state, events, ledger = run(script)
    # equity is still 1,000 at the 1.2 open: 8,333 units, so 3,333 more at 1.2
    first, second = events[2].closed_lots
    assert (first.units, first.open_price) == (5_000, 1.0)
    assert first.realised_pnl == pytest.approx(1_500.0, abs=1e-9)
    assert (second.units, second.open_price) == (3_333, 1.2)
    assert second.realised_pnl == pytest.approx(333.3, abs=1e-9)
    assert state.lots[0] == ()
    verify_ledger(ledger)


def test_closed_lots_table_columns() -> None:
    _, _, ledger = run([(bar(1.0), 1.0), (bar(1.1), 0.0)])
    assert ledger.closed_lots.columns == [
        "step",
        "time",
        "instrument",
        "units",
        "open_price",
        "close_price",
        "realised_pnl",
    ]
    assert ledger.closed_lots["step"].to_list() == [1]
    assert ledger.closed_lots["instrument"].to_list() == ["EURUSD"]


def _tampered(ledger: Ledger, **changes: pl.DataFrame) -> Ledger:
    return replace(ledger, **changes)


def _flip_ledger() -> Ledger:
    return run([(bar(1.0), 1.0), (bar(1.10), -1.0), (bar(1.05), 0.0)])[2]


def test_verify_rejects_altered_realised_pnl() -> None:
    ledger = _flip_ledger()
    bad = ledger.closed_lots.with_columns(pl.col("realised_pnl") + 1.0)
    with pytest.raises(LedgerError, match="lot"):
        verify_ledger(_tampered(ledger, closed_lots=bad))


def test_verify_rejects_a_missing_closed_lot() -> None:
    ledger = _flip_ledger()
    with pytest.raises(LedgerError, match="lot"):
        verify_ledger(_tampered(ledger, closed_lots=ledger.closed_lots.head(1)))


def test_verify_rejects_a_closed_lot_with_wrong_units() -> None:
    ledger = _flip_ledger()
    bad = ledger.closed_lots.with_columns(pl.col("units") - 1)
    with pytest.raises(LedgerError, match="lot"):
        verify_ledger(_tampered(ledger, closed_lots=bad))


price = st.floats(min_value=0.5, max_value=2.0, allow_nan=False)
spread = st.floats(min_value=0.0, max_value=0.001, allow_nan=False)
markets_st = st.builds(
    lambda o, c, s1, s2: Market(
        open_bid=o, open_spread=s1, close_bid=c, close_spread=s2
    ),
    price,
    price,
    spread,
    spread,
)
level_st = st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0])
series_st = st.lists(st.tuples(markets_st, level_st), min_size=1, max_size=60)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(series=series_st)
def test_random_series_keep_lot_conservation(
    series: list[tuple[Market, float]],
) -> None:
    default = EngineConfig()
    state = initial_state(default)
    realised = 0.0
    for market, target in series:
        state, events = step(state, default, market, target)
        realised += sum(c.realised_pnl for c in events.closed_lots)
        lots = state.lots[0]
        assert sum(lot.units for lot in lots) == state.positions[0]
        assert all((lot.units > 0) == (state.positions[0] > 0) for lot in lots)
        assert all(lot.units != 0 for lot in lots)
        mtm = sum(lot.units * (events.mark_prices[0] - lot.price) for lot in lots)
        total = default.starting_equity + realised + mtm
        assert events.equity == pytest.approx(total, rel=1e-9, abs=1e-6)
    _, _, ledger = run(series, default)
    verify_ledger(ledger)
