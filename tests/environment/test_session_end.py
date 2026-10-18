"""Session-end force-flat: reference cases (hand-computed, default config)."""

from __future__ import annotations

import pytest

from shadow_trader.environment import (
    EngineConfig,
    Market,
    initial_state,
    step,
)

CONFIG = EngineConfig()
OPEN = Market(
    open_bid=1.1000, open_spread=0.0001, close_bid=1.1005, close_spread=0.0001
)
FINAL = Market(
    open_bid=1.1020,
    open_spread=0.0001,
    close_bid=1.1025,
    close_spread=0.0001,
    is_session_final=True,
)


def test_a_long_is_sold_at_the_final_bar_open_whatever_the_target() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    assert state.positions == (90_909,)

    state, events = step(state, CONFIG, FINAL, 1.0)  # the agent still asks for long

    (fill,) = events.fills
    assert fill.units == -90_909
    assert fill.price == pytest.approx(1.10199, abs=1e-12)  # 1.1020 - 1e-5
    assert fill.forced_flat is True
    assert fill.slippage_cost == pytest.approx(0.90909, abs=1e-9)
    assert events.forced_flat is True
    assert events.positions == (0,)
    # 10_000 + 90_909 * (1.10199 - 1.10011): the cost is in this last step
    assert events.equity == pytest.approx(10_170.90892, abs=1e-5)
    assert events.used_margin == 0.0
    assert state.target_levels == (0.0,)
    assert state.target_units == (0,)


def test_a_short_is_bought_back_at_the_final_bar_open_with_the_spread() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, OPEN, -1.0)

    _, events = step(state, CONFIG, FINAL, -1.0)

    (fill,) = events.fills
    assert fill.units == 90_909
    assert fill.price == pytest.approx(1.10211, abs=1e-12)  # 1.1020 + 0.0001 + 1e-5
    assert fill.spread_cost == pytest.approx(90_909 * 0.0001, rel=1e-12)
    assert fill.forced_flat is True
    assert events.positions == (0,)
    # 10_000 + 90_909 * (1.09999 - 1.10211)
    assert events.equity == pytest.approx(9_807.27, abs=1e-2)


def test_a_flat_account_is_not_traded_on_the_final_bar() -> None:
    state, events = step(initial_state(CONFIG), CONFIG, FINAL, 1.0)  # long is refused

    assert events.fills == ()
    assert events.forced_flat is False
    assert events.positions == (0,)
    assert state.target_units == (0,)
    assert events.equity == 10_000.0


def test_a_bar_that_is_not_final_never_flags_forced_flat() -> None:
    _, events = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    assert events.forced_flat is False
    assert not events.fills[0].forced_flat
