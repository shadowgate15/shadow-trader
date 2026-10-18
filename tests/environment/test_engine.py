"""Engine reference cases: hand-computed fills, costs and marks.

All cases use the default config: EURUSD, 10,000 USD, effective leverage cap 10,
slippage 0.1 pip (1e-5), so a +1 level on 10,000 USD at 1.1000 is
floor(10_000 * 10 / 1.1) = 90_909 units.
"""

from __future__ import annotations

import pytest

from shadow_trader.environment import (
    EngineConfig,
    InstrumentSpec,
    Market,
    initial_state,
    step,
)

CONFIG = EngineConfig()
FLAT_MARKET = Market(
    open_bid=1.1000, open_spread=0.0001, close_bid=1.1005, close_spread=0.0001
)


def test_initial_state_is_flat_at_starting_equity() -> None:
    state = initial_state(CONFIG)
    assert state.equity == 10_000.0
    assert state.cash == 10_000.0
    assert state.positions == (0,)


def test_long_fills_at_bid_plus_spread_plus_slippage_and_marks_at_close_bid() -> None:
    state, events = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, 1.0)

    (fill,) = events.fills
    assert fill.units == 90_909
    assert fill.price == pytest.approx(1.10011, abs=1e-12)  # 1.1000 + 0.0001 + 1e-5
    assert fill.spread_cost == pytest.approx(90_909 * 0.0001, rel=1e-12)  # 9.0909
    assert fill.slippage_cost == pytest.approx(90_909 * 1e-5, rel=1e-12)  # 0.90909
    assert events.positions == (90_909,)
    assert events.mark_prices == (1.1005,)
    # equity = 10_000 + 90_909 * (1.1005 - 1.10011)
    assert events.equity == pytest.approx(10_035.45451, abs=1e-6)
    assert state.equity == events.equity
    assert events.used_margin == pytest.approx(90_909 * 1.1005 * 0.02, rel=1e-12)
    assert events.rollover_charge == 0.0


def test_short_fills_at_bid_minus_slippage_and_marks_at_close_ask() -> None:
    _, events = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, -1.0)

    (fill,) = events.fills
    assert fill.units == -90_909
    assert fill.price == pytest.approx(1.09999, abs=1e-12)  # 1.1000 - 1e-5
    assert fill.spread_cost == 0.0  # a sell crosses no spread at the fill
    assert fill.slippage_cost == pytest.approx(0.90909, abs=1e-9)
    assert events.mark_prices == (pytest.approx(1.1006, abs=1e-12),)  # close ask
    # equity = 10_000 + 90_909 * (1.09999 - 1.1006)
    assert events.equity == pytest.approx(9_944.54551, abs=1e-6)


def test_closing_a_short_buys_at_the_ask() -> None:
    short, _ = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, -1.0)
    _, events = step(short, CONFIG, FLAT_MARKET, 0.0)

    (fill,) = events.fills
    assert fill.units == 90_909
    assert fill.price == pytest.approx(1.10011, abs=1e-12)
    assert events.positions == (0,)
    assert events.used_margin == 0.0
    # round trip: sold 1.09999, bought 1.10011 on 90_909 units
    assert events.equity == pytest.approx(10_000 - 90_909 * 0.00012, abs=1e-6)


def test_reversal_is_one_fill_with_spread_paid_once_on_the_full_amount() -> None:
    short, _ = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, -1.0)
    # short equity 9_944.54551 -> new long target floor(99_445.4551 / 1.1) = 90_404
    long, events = step(short, CONFIG, FLAT_MARKET, 1.0)

    (fill,) = events.fills
    assert fill.units == 90_404 + 90_909  # close the short and open the long
    assert fill.spread_cost == pytest.approx(181_313 * 0.0001, rel=1e-12)
    assert events.positions == (90_404,)
    assert long.positions == (90_404,)


def test_long_to_short_reversal_is_one_sell_fill() -> None:
    long, _ = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, 1.0)
    # long equity 10_035.45451 -> floor(100_354.5451 / 1.1) = 91_231
    _, events = step(long, CONFIG, FLAT_MARKET, -1.0)

    (fill,) = events.fills
    assert fill.units == -91_231 - 90_909
    assert events.positions == (-91_231,)


def test_non_zero_level_trades_at_least_one_unit() -> None:
    config = EngineConfig(starting_equity=0.05)  # floor(0.05 * 10 / 1.1) == 0
    _, events = step(initial_state(config), config, FLAT_MARKET, 1.0)
    assert events.positions == (1,)
    _, events = step(initial_state(config), config, FLAT_MARKET, -1.0)
    assert events.positions == (-1,)


def test_fractional_level_scales_the_unit_target() -> None:
    _, events = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, 0.5)
    assert events.positions == (45_454,)  # floor(0.5 * 100_000 / 1.1)


def test_flat_target_from_flat_does_nothing() -> None:
    state, events = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, 0.0)
    assert events.fills == ()
    assert events.equity == 10_000.0
    assert state == initial_state(CONFIG)


def test_holding_the_current_target_trades_nothing_as_equity_drifts() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, 1.0)
    moved = Market(
        open_bid=1.2000, open_spread=0.0001, close_bid=1.2500, close_spread=0.0001
    )
    for _ in range(3):  # equity is far from the sizing equity; the target is frozen
        state, events = step(state, CONFIG, moved, 1.0)
        assert events.fills == ()
        assert events.positions == (90_909,)


def test_a_level_change_resizes_to_current_equity() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, FLAT_MARKET, 0.5)
    state, events = step(state, CONFIG, FLAT_MARKET, 1.0)
    (fill,) = events.fills
    assert fill.units == events.positions[0] - 45_454
    assert events.positions[0] > 90_909  # equity grew, so the new unit target did


@pytest.mark.parametrize("target", [1.01, -1.01, float("nan"), float("inf")])
def test_out_of_range_target_raises(target: float) -> None:
    with pytest.raises(ValueError, match="target"):
        step(initial_state(CONFIG), CONFIG, FLAT_MARKET, target)


def test_mismatched_instrument_count_raises() -> None:
    with pytest.raises(ValueError):
        step(initial_state(CONFIG), CONFIG, (FLAT_MARKET, FLAT_MARKET), (1.0, 1.0))


def test_unhandled_market_conditions_fail_loudly() -> None:
    # unfillable handling belongs to a later ticket;
    # until then the engine must not silently ignore it.
    for market in (Market(1.1, 0.0001, 1.1, 0.0001, fillable=False),):
        with pytest.raises(NotImplementedError):
            step(initial_state(CONFIG), CONFIG, market, 1.0)


def test_two_instruments_share_one_pooled_equity() -> None:
    config = EngineConfig(
        instruments=(InstrumentSpec(symbol="EURUSD"), InstrumentSpec(symbol="GBPUSD"))
    )
    gbp = Market(
        open_bid=1.3000, open_spread=0.0002, close_bid=1.3010, close_spread=0.0002
    )
    state, events = step(initial_state(config), config, (FLAT_MARKET, gbp), (1.0, -1.0))
    assert events.positions == (90_909, -76_923)  # floor(1e5 / 1.1), floor(1e5 / 1.3)
    assert [fill.instrument for fill in events.fills] == [0, 1]
    eur = 90_909 * (1.1005 - 1.10011)
    gbp_pnl = -76_923 * ((1.3010 + 0.0002) - (1.3000 - 1e-5))
    assert events.equity == pytest.approx(10_000 + eur + gbp_pnl, abs=1e-6)
    assert state.positions == events.positions


ROLLOVER_CONFIG = EngineConfig(
    instruments=(InstrumentSpec(rollover_long=0.0001, rollover_short=-0.00005),)
)


def rollover_market(days: int) -> Market:
    return Market(
        open_bid=1.1000,
        open_spread=0.0001,
        close_bid=1.1005,
        close_spread=0.0001,
        rollover_days=days,
    )


def test_rollover_day_0_charges_nothing() -> None:
    _, events = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(0), 1.0
    )
    assert events.rollover_charge == 0.0
    assert events.equity == pytest.approx(10_035.45451, abs=1e-6)


def test_long_rollover_day_1_charges_rate_on_notional_at_the_close() -> None:
    state, events = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(1), 1.0
    )
    # notional 90_909 * 1.1005 = 100_045.3545; charge = 0.0001 of it
    assert events.rollover_charge == pytest.approx(10.00453545, abs=1e-8)
    assert events.equity == pytest.approx(10_035.45451 - 10.00453545, abs=1e-6)
    assert state.equity == events.equity
    assert state.cash == pytest.approx(
        10_000 - 90_909 * 1.10011 - 10.00453545, abs=1e-6
    )


def test_long_rollover_day_3_is_triple() -> None:
    _, events = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(3), 1.0
    )
    assert events.rollover_charge == pytest.approx(30.01360635, abs=1e-8)
    assert events.equity == pytest.approx(10_035.45451 - 30.01360635, abs=1e-6)


def test_short_rollover_uses_the_short_rate_on_the_close_ask_notional() -> None:
    # notional 90_909 * 1.1006 = 100_054.4454; rate -0.00005 is a credit
    _, day1 = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(1), -1.0
    )
    assert day1.rollover_charge == pytest.approx(-5.00272227, abs=1e-8)
    assert day1.equity == pytest.approx(9_944.54551 + 5.00272227, abs=1e-6)
    _, day3 = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(3), -1.0
    )
    assert day3.rollover_charge == pytest.approx(-15.00816681, abs=1e-8)


def test_a_positive_short_rate_is_a_charge() -> None:
    config = EngineConfig(instruments=(InstrumentSpec(rollover_short=0.00005),))
    _, events = step(initial_state(config), config, rollover_market(1), -1.0)
    assert events.rollover_charge == pytest.approx(5.00272227, abs=1e-8)


def test_flat_account_pays_no_rollover() -> None:
    _, events = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(3), 0.0
    )
    assert events.rollover_charge == 0.0
    assert events.equity == 10_000.0


def test_a_position_closed_at_the_rollover_open_pays_nothing() -> None:
    state, _ = step(
        initial_state(ROLLOVER_CONFIG), ROLLOVER_CONFIG, rollover_market(0), 1.0
    )
    _, events = step(state, ROLLOVER_CONFIG, rollover_market(1), 0.0)
    assert events.positions == (0,)
    assert events.rollover_charge == 0.0
