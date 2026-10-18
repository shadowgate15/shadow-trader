"""Unfillable Bars in the engine: carried marks, pending fills, closeout and the
synthetic liquidation. Hand-computed, default config (10,000 USD, slippage 1e-5)."""

from __future__ import annotations

import math

import pytest

from shadow_trader.environment import (
    EngineConfig,
    InstrumentSpec,
    Market,
    initial_state,
    step,
)

CONFIG = EngineConfig()
NAN = math.nan
OPEN = Market(
    open_bid=1.1000, open_spread=0.0001, close_bid=1.1005, close_spread=0.0001
)
GAP = Market(NAN, NAN, NAN, NAN, fillable=False)


def seeded(config: EngineConfig = CONFIG, target: float = 1.0):
    """A state long (or short) 90,909 units after one fillable Bar."""
    return step(initial_state(config), config, OPEN, target)


def test_an_unfillable_bar_carries_the_target_and_marks_at_the_last_trusted_close():
    state, before = seeded()
    assert before.carried_mark is False

    state, events = step(state, CONFIG, GAP, 1.0)

    assert events.fills == ()
    assert events.carried_mark is True
    assert events.positions == (90_909,)
    assert events.mark_prices == (1.1005,)
    assert events.equity == before.equity  # nothing moved: still 10_035.45451
    assert events.used_margin == before.used_margin
    assert not events.closeout
    assert state.target_units == (90_909,)


def test_the_pending_target_fills_at_the_first_fillable_open_after_a_run():
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 0.0)  # seeds the trusted close
    for _ in range(3):
        state, events = step(state, CONFIG, GAP, 1.0)
        assert events.fills == () and events.positions == (0,)
    later = Market(1.2000, 0.0002, 1.2010, 0.0002)

    state, events = step(state, CONFIG, later, 1.0)

    (fill,) = events.fills
    assert fill.units == 83_333  # floor(10_000 * 10 / 1.2)
    assert fill.price == pytest.approx(1.2000 + 0.0002 + 1e-5, abs=1e-12)
    assert events.carried_mark is False


def test_rollover_accrues_on_the_carried_notional_of_an_unfillable_bar():
    config = EngineConfig(instruments=(InstrumentSpec(rollover_long=0.0001),))
    state, _ = seeded(config)
    gap = Market(NAN, NAN, NAN, NAN, fillable=False, rollover_days=3)

    _, events = step(state, config, gap, 1.0)

    # 90_909 * 1.1005 * 0.0001 * 3
    assert events.rollover_charge == pytest.approx(30.01360635, abs=1e-8)
    assert events.equity == pytest.approx(10_035.45451 - 30.01360635, abs=1e-6)


def test_the_closeout_check_runs_only_on_fillable_bars():
    config = EngineConfig(instruments=(InstrumentSpec(rollover_long=0.1),))
    state, _ = seeded(config)
    gap = Market(NAN, NAN, NAN, NAN, fillable=False, rollover_days=1)

    state, events = step(state, config, gap, 1.0)

    # equity 10_035.45451 - 10_004.53545 = 30.91906, far under 50% of 2_000.9 margin
    assert events.equity == pytest.approx(30.91906, abs=1e-5)
    assert events.equity < config.closeout_fraction * events.used_margin
    assert not events.closeout
    assert events.positions == (90_909,)

    state, events = step(state, config, OPEN, 1.0)  # the next fillable Bar checks

    assert events.closeout
    assert events.positions == (0,)
    assert events.fills[0].closeout


def test_an_unfillable_session_final_bar_liquidates_a_long_at_the_last_trusted_close():
    state, _ = seeded()
    final = Market(NAN, NAN, NAN, NAN, fillable=False, is_session_final=True)

    state, events = step(state, CONFIG, final, 1.0)

    (fill,) = events.fills
    assert fill.units == -90_909
    assert fill.price == pytest.approx(1.10049, abs=1e-12)  # 1.1005 - 1e-5
    assert fill.spread_cost == 0.0
    assert fill.slippage_cost == pytest.approx(0.90909, abs=1e-9)
    assert fill.unfilled_forced_flat is True
    assert fill.forced_flat is False
    assert events.unfilled_forced_flat is True
    assert events.forced_flat is False
    assert events.carried_mark is True
    assert events.positions == (0,)
    assert events.equity == pytest.approx(10_000 + 90_909 * 0.00038, abs=1e-5)
    assert state.target_levels == (0.0,) and state.target_units == (0,)
    (lot,) = events.closed_lots
    assert lot.close_price == fill.price


def test_an_unfillable_session_final_bar_buys_a_short_back_with_the_spread():
    state, _ = seeded(CONFIG, -1.0)
    final = Market(NAN, NAN, NAN, NAN, fillable=False, is_session_final=True)

    _, events = step(state, CONFIG, final, -1.0)

    (fill,) = events.fills
    assert fill.units == 90_909
    assert fill.price == pytest.approx(1.10061, abs=1e-12)  # 1.1005 + 1e-4 + 1e-5
    assert fill.spread_cost == pytest.approx(9.0909, abs=1e-9)
    assert events.equity == pytest.approx(
        10_000 + 90_909 * (1.09999 - 1.10061), abs=1e-5
    )


def test_a_flat_account_on_an_unfillable_final_bar_trades_nothing():
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 0.0)
    final = Market(NAN, NAN, NAN, NAN, fillable=False, is_session_final=True)

    _, events = step(state, CONFIG, final, 1.0)

    assert events.fills == ()
    assert events.unfilled_forced_flat is False
