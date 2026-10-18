"""Ledger tables and verify_ledger: reference runs, tamper detection, properties."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow_trader.environment import (
    EngineConfig,
    InstrumentSpec,
    Ledger,
    LedgerBuilder,
    LedgerError,
    Market,
    initial_state,
    step,
    verify_ledger,
)

CONFIG = EngineConfig()
T0 = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
BAR = timedelta(minutes=5)


def run(markets: list[Market], targets: list[float]) -> Ledger:
    state = initial_state(CONFIG)
    builder = LedgerBuilder(CONFIG)
    for i, (market, target) in enumerate(zip(markets, targets, strict=True)):
        state, events = step(state, CONFIG, market, target)
        builder.add_step(T0 + i * BAR, events)
        if events.closeout:  # a closeout ends the episode
            break
    return builder.build()


def market(open_bid: float, close_bid: float, spread: float = 0.0001) -> Market:
    return Market(
        open_bid=open_bid, open_spread=spread, close_bid=close_bid, close_spread=spread
    )


SCRIPT_MARKETS = [
    market(1.1000, 1.1005),
    market(1.1005, 1.1010),
    market(1.1010, 1.1000),
    market(1.1000, 1.0990),
    market(1.0990, 1.0995),
]
SCRIPT_TARGETS = [0.0, 1.0, 1.0, -1.0, 0.0]


def test_steps_and_fills_tables_have_the_documented_columns() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)

    assert ledger.steps.columns == [
        "step",
        "open_time",
        "episode_id",
        "session_id",
        "equity",
        "used_margin",
        "position_units",
        "rollover_charge",
        "mark_price",
        "clipped",
        "shortfall_units",
        "closeout",
        "is_decision",
        "carried_mark",
        "blind_run",
    ]
    assert ledger.fills.columns == [
        "step",
        "time",
        "instrument",
        "units",
        "price",
        "spread_cost",
        "slippage_cost",
        "clipped",
        "closeout",
        "forced_flat",
        "unfilled_forced_flat",
    ]
    assert ledger.steps.height == 5
    assert ledger.steps["position_units"].to_list()[:3] == [
        0,
        90_867,
        90_867,
    ]  # 1e5 / 1.1005
    assert ledger.fills["step"].to_list() == [1, 3, 4]
    assert ledger.fills["instrument"].unique().to_list() == ["EURUSD"]
    assert ledger.steps["open_time"][1] == T0 + BAR
    assert "reward" not in ledger.steps.columns + ledger.fills.columns
    verify_ledger(ledger)


def test_empty_ledger_verifies() -> None:
    verify_ledger(LedgerBuilder(CONFIG).build())


def test_builder_rejects_multi_instrument_configs_for_now() -> None:
    config = EngineConfig(
        instruments=(
            *CONFIG.instruments,
            replace(CONFIG.instruments[0], symbol="GBPUSD"),
        )
    )
    with pytest.raises(NotImplementedError):
        LedgerBuilder(config)


def tampered(ledger: Ledger, **changes: pl.DataFrame) -> Ledger:
    return replace(ledger, **changes)


def test_verify_rejects_an_equity_that_does_not_reconcile() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)
    steps = ledger.steps.with_columns(
        pl.when(pl.col("step") == 2)
        .then(pl.col("equity") + 0.01)
        .otherwise(pl.col("equity"))
        .alias("equity")
    )
    with pytest.raises(LedgerError, match="equity"):
        verify_ledger(tampered(ledger, steps=steps))


def test_verify_rejects_a_position_that_breaks_continuity() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)
    steps = ledger.steps.with_columns(
        pl.when(pl.col("step") == 2)
        .then(pl.col("position_units") + 1)
        .otherwise(pl.col("position_units"))
        .alias("position_units")
    )
    with pytest.raises(LedgerError, match="position"):
        verify_ledger(tampered(ledger, steps=steps))


def test_verify_rejects_a_dropped_fill() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)
    with pytest.raises(LedgerError, match="position"):
        verify_ledger(tampered(ledger, fills=ledger.fills.slice(1)))


def test_verify_rejects_non_finite_values() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)
    steps = ledger.steps.with_columns(
        pl.when(pl.col("step") == 1)
        .then(float("nan"))
        .otherwise(pl.col("equity"))
        .alias("equity")
    )
    with pytest.raises(LedgerError, match="finite"):
        verify_ledger(tampered(ledger, steps=steps))


def test_verify_rejects_a_fill_on_a_missing_step() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)
    fills = ledger.fills.with_columns(pl.col("step") + 100)
    with pytest.raises(LedgerError, match="step"):
        verify_ledger(tampered(ledger, fills=fills))


def test_verify_rejects_a_gap_in_the_step_index() -> None:
    ledger = run(SCRIPT_MARKETS, SCRIPT_TARGETS)
    with pytest.raises(LedgerError, match="step"):
        verify_ledger(tampered(ledger, steps=ledger.steps.filter(pl.col("step") != 2)))


# --- properties -----------------------------------------------------------------

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
PROPERTY = settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@PROPERTY
@given(series=series_st)
def test_random_markets_and_targets_always_verify(
    series: list[tuple[Market, float]],
) -> None:
    ledger = run([m for m, _ in series], [t for _, t in series])
    verify_ledger(ledger)


@PROPERTY
@given(series=series_st)
def test_step_is_pure_and_does_not_mutate_inputs(
    series: list[tuple[Market, float]],
) -> None:
    state = initial_state(CONFIG)
    for bar, target in series:
        snapshot = (state, bar, CONFIG, CONFIG.config_hash())
        first = step(state, CONFIG, bar, target)
        second = step(state, CONFIG, bar, target)
        assert first == second
        assert snapshot == (state, bar, CONFIG, CONFIG.config_hash())
        state = first[0]


@PROPERTY
@given(markets=st.lists(markets_st, min_size=1, max_size=40))
def test_all_flat_targets_give_no_fills_and_constant_equity(
    markets: list[Market],
) -> None:
    ledger = run(markets, [0.0] * len(markets))
    assert ledger.fills.height == 0
    assert set(ledger.steps["equity"].to_list()) == {CONFIG.starting_equity}
    verify_ledger(ledger)


def test_equity_reconciliation_includes_rollover_and_catches_its_omission() -> None:
    config = EngineConfig(instruments=(InstrumentSpec(rollover_long=0.0001),))
    builder = LedgerBuilder(config)
    state = initial_state(config)
    for i, days in enumerate((0, 3, 0)):
        mkt = Market(1.1, 0.0001, 1.1005, 0.0001, rollover_days=days)
        state, events = step(state, config, mkt, 1.0)
        builder.add_step(T0 + i * BAR, events)
    ledger = builder.build()
    assert ledger.steps["rollover_charge"].to_list()[1] > 0.0
    verify_ledger(ledger)

    zeroed = ledger.steps.with_columns(pl.lit(0.0).alias("rollover_charge"))
    with pytest.raises(LedgerError, match="equity"):
        verify_ledger(tampered(ledger, steps=zeroed))
