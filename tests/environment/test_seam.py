"""The versioned seam tier and the live-parity tests (A, B, C)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from shadow_trader.environment import seam
from shadow_trader.environment.engine import clip_order
from shadow_trader.environment.ledger import verify_ledger
from shadow_trader.environment.seam import (
    AccountSnapshot,
    EngineConfig,
    Market,
    ObservationConfig,
    build_observation,
    engine_state_from_snapshot,
    halt_flatten,
    initial_state,
    memo_from_state,
    plan_order,
    snapshot_from_state,
    step,
    window_inputs,
)

CONFIG = EngineConfig()
OPEN = Market(
    open_bid=1.1000, open_spread=0.0001, close_bid=1.1005, close_spread=0.0001
)
NEXT = Market(
    open_bid=1.1010, open_spread=0.0001, close_bid=1.1012, close_spread=0.0001
)
FINAL = Market(
    open_bid=1.1020,
    open_spread=0.0001,
    close_bid=1.1025,
    close_spread=0.0001,
    is_session_final=True,
)


def test_seam_exports_an_integer_version_and_the_tier() -> None:
    assert isinstance(seam.SEAM_VERSION, int) and not isinstance(
        seam.SEAM_VERSION, bool
    )
    for name in (
        "EngineConfig",
        "InstrumentSpec",
        "EngineState",
        "step",
        "plan_order",
        "engine_state_from_snapshot",
        "StepEvents",
        "AccountSnapshot",
        "TargetMemo",
        "Market",
        "market_from_row",
        "window_inputs",
        "build_observation",
        "ObservationConfig",
        "Observation",
        "LedgerBuilder",
        "Ledger",
        "STEPS_SCHEMA",
        "FILLS_SCHEMA",
        "CLOSED_LOTS_SCHEMA",
        "EPISODES_SCHEMA",
        "halt_flatten",
    ):
        assert hasattr(seam, name), name
        assert name in seam.__all__, name


# Parity C: step's clip is plan_order's clip is clip_order.


def test_plan_order_unclipped_reports_the_wanted_delta() -> None:
    plan = plan_order(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    assert plan.desired_delta == plan.allowed_delta == 90_909
    assert plan.clipped is False and plan.shortfall == 0
    assert plan.override is None


def test_step_clip_equals_a_direct_clip_call_via_plan_order() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    # equity falls, so re-trading the frozen target is clipped by the margin limit
    state = replace(state, cash=state.cash - 4_000.0, equity=state.equity - 4_000.0)
    plan = plan_order(state, CONFIG, NEXT, 1.0)
    direct = clip_order(
        state.positions[0],
        state.target_units[0],
        state.equity,
        NEXT.open_bid,
        CONFIG.leverage_cap,
    )
    assert (plan.desired_delta, plan.allowed_delta) == (
        direct.desired_delta,
        direct.allowed_delta,
    )
    assert (plan.clipped, plan.shortfall) == (direct.clipped, direct.shortfall)

    # a level change that is clipped on entry, through step
    flat = initial_state(CONFIG, equity=10_000.0)
    big = replace(flat, cash=3_000.0, equity=3_000.0)
    plan = plan_order(big, CONFIG, OPEN, -1.0)
    _, events = step(big, CONFIG, OPEN, -1.0)
    assert events.fills[0].units == plan.allowed_delta
    assert events.clipped == plan.clipped
    assert events.shortfall == (plan.shortfall,)


def test_plan_order_applies_the_session_end_override_itself() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    plan = plan_order(state, CONFIG, FINAL, 1.0)  # the Policy still asks for long
    assert plan.override == "session_end"
    assert plan.allowed_delta == -90_909 and plan.clipped is False
    _, events = step(state, CONFIG, FINAL, 1.0)
    assert events.fills[0].units == plan.allowed_delta
    assert events.fills[0].forced_flat is True


def test_plan_order_applies_the_closeout_override_itself() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    # a gap leaves the long deep under water at the next open
    crash = Market(
        open_bid=1.0000, open_spread=0.0001, close_bid=1.0, close_spread=0.0001
    )
    plan = plan_order(state, CONFIG, crash, 1.0)
    assert plan.override == "closeout"
    assert plan.allowed_delta == -state.positions[0]
    _, events = step(state, CONFIG, crash, 1.0)
    assert events.closeout is True
    assert events.fills[0].closeout is True
    assert events.positions == (0,)


# Parity A: snapshot round trip.


def test_snapshot_round_trip_gives_the_same_observation_and_next_step() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    state, _ = step(state, CONFIG, NEXT, 1.0)
    snapshot = snapshot_from_state(state)
    assert isinstance(snapshot, AccountSnapshot)
    rebuilt = engine_state_from_snapshot(snapshot, memo_from_state(state), CONFIG)

    assert rebuilt.positions == state.positions
    assert rebuilt.equity == pytest.approx(state.equity, rel=1e-12)
    rows = np.array([[[0.5, -0.5]]])
    valid = np.array([[True]])
    obs_config = ObservationConfig()
    a = build_observation(rows, valid, state, NEXT, CONFIG, obs_config)
    b = build_observation(rows, valid, rebuilt, NEXT, CONFIG, obs_config)
    np.testing.assert_array_equal(a.position, b.position)
    np.testing.assert_array_equal(a.account, b.account)

    later = Market(
        open_bid=1.1020, open_spread=0.0001, close_bid=1.1018, close_spread=0.0001
    )
    for target in (1.0, 0.0, -1.0):
        _, ea = step(state, CONFIG, later, target)
        _, eb = step(rebuilt, CONFIG, later, target)
        assert [f.units for f in ea.fills] == [f.units for f in eb.fills]
        assert eb.equity == pytest.approx(ea.equity, rel=1e-12)
        assert eb.used_margin == pytest.approx(ea.used_margin, rel=1e-12)
        assert sum(c.realised_pnl for c in eb.closed_lots) == pytest.approx(
            sum(c.realised_pnl for c in ea.closed_lots), rel=1e-9
        )


def test_the_snapshot_wins_over_the_memo_for_position_and_equity() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    memo = memo_from_state(state)
    snapshot = AccountSnapshot(
        cash=5_000.0, equity=9_000.0, positions=(1_000,), avg_prices=(1.1,)
    )
    rebuilt = engine_state_from_snapshot(snapshot, memo, CONFIG)
    assert rebuilt.positions == (1_000,)
    assert rebuilt.equity == 9_000.0 and rebuilt.cash == 5_000.0
    assert rebuilt.target_levels == memo.target_levels
    assert rebuilt.target_units == memo.target_units
    assert rebuilt.lots[0][0].units == 1_000 and rebuilt.lots[0][0].price == 1.1


def test_a_missing_memo_means_flat_with_nothing_pending() -> None:
    snapshot = AccountSnapshot(cash=10_000.0, equity=10_000.0, positions=(0,))
    rebuilt = engine_state_from_snapshot(snapshot, None, CONFIG)
    assert rebuilt.target_levels == (0.0,)
    assert rebuilt.target_units == (0,)
    _, events = step(rebuilt, CONFIG, OPEN, 0.0)
    assert events.fills == ()
    assert events.shortfall == (0,)


def test_a_position_without_a_known_price_is_rejected() -> None:
    snapshot = AccountSnapshot(cash=0.0, equity=10_000.0, positions=(5,))
    with pytest.raises(ValueError, match="average price"):
        engine_state_from_snapshot(snapshot, None, CONFIG)


# Halt flatten.


def test_halt_flatten_bypasses_plan_order_and_carries_the_halt_flag() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    new_state, events = halt_flatten(state, CONFIG, NEXT)
    (fill,) = events.fills
    assert fill.halt is True and fill.units == -90_909
    assert fill.forced_flat is False and fill.closeout is False
    assert new_state.positions == (0,)
    assert new_state.target_levels == (0.0,) and new_state.target_units == (0,)


def test_halt_flag_reaches_the_ledger_and_the_ledger_verifies() -> None:
    from shadow_trader.environment.seam import LedgerBuilder

    builder = LedgerBuilder(CONFIG)
    t0 = datetime(2024, 1, 8, 12, 0, tzinfo=OPEN_TZ)
    builder.begin_episode(
        episode_id=0, session_id=0, start_time=t0, start_equity=10_000.0
    )
    state = initial_state(CONFIG)
    state, ev = step(state, CONFIG, OPEN, 1.0)
    builder.add_step(t0, ev)
    state, ev = halt_flatten(state, CONFIG, NEXT)
    builder.add_step(t0 + timedelta(minutes=5), ev)
    builder.end_episode("halt")
    ledger = builder.build()
    assert ledger.fills["halt"].to_list() == [False, True]
    verify_ledger(ledger)
    assert "halt" in seam.FILLS_SCHEMA


OPEN_TZ = __import__("datetime").UTC


# Parity B: live-path assembly equals the offline path.


def test_live_path_assembly_matches_the_offline_observation() -> None:
    from shadow_trader.environment.market_data import from_prepared
    from shadow_trader.environment.seam import market_from_row
    from shadow_trader.preparation import Scaler, live_spec
    from shadow_trader.preparation.live import evaluate_tail
    from tests.environment.fixture import INSTRUMENT, SPEC, fixture_bars, load_fixture

    dataset, scaler = load_fixture()
    assert isinstance(scaler, Scaler)
    bars = fixture_bars()
    window_bars = 3
    row = 150  # inside Session 0
    spec_hash = dataset.manifest["feature_spec"]["hash"]

    # offline path
    data = from_prepared(dataset, scaler, INSTRUMENT, 5)
    offline_rows, offline_valid = data.window(row, window_bars)
    offline_market = data.market(row)

    # live path: the tail as seen at each decision time, features per tail,
    # scaled by the same artifact
    live = live_spec(SPEC)
    scaled_rows = []
    for k in range(row - window_bars + 1, row + 1):
        tail = bars[: k + 1]
        feats = evaluate_tail(tail, live)
        frame = feats.select(SPEC.names).with_columns(
            pl.lit(INSTRUMENT).alias("instrument")
        )
        scaled = scaler.apply(frame, spec_hash=spec_hash)
        scaled_rows.append(scaled.select(SPEC.names).row(0))
    live_array = np.array([scaled_rows], dtype=np.float64)  # [1, W, F]
    live_features, live_valid = window_inputs(
        live_array, ObservationConfig(window_bars=window_bars)
    )
    bar = bars[row]
    live_market = market_from_row(
        {
            "open_time": bar.open_time,
            "open": bar.open,
            "close": bar.close,
            "spread_close": bar.spread_close,
            "spread_mean": bar.spread_mean,
            "is_gap": bar.is_gap,
            "quality_flags": bar.quality_flags,
        },
        5,
    )
    assert live_market == offline_market

    state, _ = step(initial_state(CONFIG), CONFIG, live_market, 1.0)
    obs_config = ObservationConfig(window_bars=window_bars)
    live_obs = build_observation(
        live_features[0:1].astype(np.float64),
        live_valid,
        state,
        live_market,
        CONFIG,
        obs_config,
    )
    offline_obs = build_observation(
        offline_rows.astype(np.float64),
        offline_valid,
        state,
        offline_market,
        CONFIG,
        obs_config,
    )
    np.testing.assert_array_equal(live_obs.features, offline_obs.features)
    np.testing.assert_array_equal(live_obs.valid, offline_obs.valid)
    np.testing.assert_array_equal(live_obs.position, offline_obs.position)
    np.testing.assert_array_equal(live_obs.account, offline_obs.account)
