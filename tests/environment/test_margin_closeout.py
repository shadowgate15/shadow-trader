"""Margin clipping, shortfall retry and closeout: hand-computed reference cases.

Default config: EURUSD, 10,000 USD, effective leverage cap 10, slippage 1e-5,
closeout fraction 0.5, margin rate 2%. A unit limit at equity E and open bid p is
floor(E x 10 / p).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from shadow_trader.environment import (
    EngineConfig,
    EngineState,
    LedgerBuilder,
    LedgerError,
    Market,
    initial_state,
    step,
    verify_ledger,
)
from shadow_trader.environment.engine import clip_order

CONFIG = EngineConfig()
T0 = datetime(2024, 1, 2, tzinfo=UTC)


def mkt(open_bid: float = 1.1, close_bid: float = 1.1, spread: float = 0.0001):
    return Market(open_bid, spread, close_bid, spread)


def pending_long(equity: float, units: int = 90_909) -> EngineState:
    """Flat account whose full-size long target was frozen when equity was higher."""
    return EngineState(
        cash=equity,
        equity=equity,
        positions=(0,),
        target_levels=(1.0,),
        target_units=(units,),
        lots=((),),
    )


# --- the shared clipping path ---------------------------------------------------


def test_clip_order_passes_a_fitting_order_through() -> None:
    plan = clip_order(0, 50_000, 10_000.0, 1.1, 10.0)
    assert (plan.allowed_delta, plan.clipped, plan.shortfall) == (50_000, False, 0)


def test_clip_order_limits_to_the_largest_size_that_fits() -> None:
    plan = clip_order(0, 90_909, 5_000.0, 1.1, 10.0)  # floor(50_000 / 1.1) = 45_454
    assert plan.desired_delta == 90_909
    assert plan.allowed_delta == 45_454
    assert plan.clipped
    assert plan.shortfall == 45_455


def test_clip_order_never_forces_a_reduction_of_an_oversized_position() -> None:
    # already 60_000 long with a limit of 45_454: no sell, and the add is a shortfall
    plan = clip_order(60_000, 90_909, 5_000.0, 1.1, 10.0)
    assert (plan.allowed_delta, plan.clipped, plan.shortfall) == (0, True, 30_909)


def test_clip_order_reduces_freely_and_clips_the_far_side_of_a_flip() -> None:
    assert clip_order(60_000, 0, 5_000.0, 1.1, 10.0).clipped is False
    flip = clip_order(60_000, -90_909, 5_000.0, 1.1, 10.0)
    assert flip.allowed_delta == -60_000 - 45_454
    assert flip.shortfall == -(90_909 - 45_454)


# --- margin clipping in step, with a retried shortfall ---------------------------


def test_step_clips_flags_and_keeps_the_shortfall_pending() -> None:
    state, events = step(pending_long(5_000.0), CONFIG, mkt(), 1.0)

    (fill,) = events.fills
    assert fill.units == 45_454
    assert fill.clipped and not fill.closeout
    assert events.clipped and events.shortfall == (45_455,)
    assert state.target_units == (90_909,)  # the frozen target survives the clip


def test_shortfall_is_retried_each_bar_until_it_fills() -> None:
    state, first = step(pending_long(5_000.0), CONFIG, mkt(), 1.0)
    assert first.positions == (45_454,)

    # equity is still about 5_000 (limit 45_454 already used): nothing more fits
    state, second = step(state, CONFIG, mkt(), 1.0)
    assert second.fills == ()
    assert second.clipped and second.shortfall[0] > 0

    # equity recovers to 10_000 +: the rest fills in one go and the clip clears
    rich = EngineState(
        cash=state.cash + 6_000.0,
        equity=state.equity + 6_000.0,
        positions=state.positions,
        target_levels=state.target_levels,
        target_units=state.target_units,
        lots=state.lots,
    )
    state, third = step(rich, CONFIG, mkt(), 1.0)
    (fill,) = third.fills
    assert fill.units == 90_909 - 45_454
    assert not fill.clipped and not third.clipped and third.shortfall == (0,)
    assert state.positions == (90_909,)


def test_a_newer_action_replaces_the_pending_target_without_queueing() -> None:
    state, _ = step(pending_long(5_000.0), CONFIG, mkt(), 1.0)
    # the agent now wants half size: frozen afresh from current equity, shortfall gone
    state, events = step(state, CONFIG, mkt(), 0.5)
    assert 0 < state.target_units[0] < 22_728  # half of floor(~5_000 x 10 / 1.1)
    assert events.shortfall == (0,)
    assert not events.clipped
    (fill,) = events.fills
    assert fill.units < 0 and not fill.clipped  # sells down to the new, smaller target


# --- closeout -------------------------------------------------------------------

U = 90_909
FILL = 1.1 + 0.0001 + 0.00001
# long at 10_000 equity; closeout when 10_000 + U (c - FILL) < 0.5 x 0.02 x U x c
THRESHOLD = (U * FILL - 10_000.0) / (0.99 * U)  # about 1.0000


def long_then(close_bid: float):
    state, _ = step(initial_state(CONFIG), CONFIG, mkt(), 1.0)
    return step(state, CONFIG, mkt(1.1, close_bid), 1.0)


def test_no_closeout_just_above_the_threshold() -> None:
    state, events = long_then(THRESHOLD + 1e-4)
    assert not events.closeout
    assert events.positions == (U,)
    assert state.target_levels == (1.0,)


def test_closeout_just_below_the_threshold_closes_flags_and_resets_target() -> None:
    state, events = long_then(THRESHOLD - 1e-4)

    assert events.closeout
    assert events.positions == (0,)
    (fill,) = events.fills
    assert fill.closeout and fill.units == -U
    close_bid = THRESHOLD - 1e-4
    assert fill.price == pytest.approx(close_bid - 1e-5, abs=1e-12)  # normal slippage
    assert state.positions == (0,)
    assert state.target_levels == (0.0,) and state.target_units == (0,)
    assert state.equity == pytest.approx(10_000.0 + U * (fill.price - FILL), abs=1e-6)
    assert events.used_margin > 0  # the margin that triggered it is what is recorded
    assert len(events.closed_lots) == 1


def test_after_a_closeout_the_account_stays_flat() -> None:
    state, _ = long_then(THRESHOLD - 1e-4)
    state, events = step(state, CONFIG, mkt(), 0.0)
    assert events.fills == () and not events.closeout


def test_closeout_is_checked_after_rollover() -> None:
    from shadow_trader.environment import InstrumentSpec

    config = EngineConfig(instruments=(InstrumentSpec(rollover_long=0.5),))
    state, _ = step(initial_state(config), config, mkt(), 1.0)
    rolled = Market(1.1, 0.0001, 1.1, 0.0001, rollover_days=1)
    _, events = step(state, config, rolled, 1.0)
    assert events.rollover_charge > 0 and events.closeout


# --- ledger: flags and verify_ledger ---------------------------------------------


def record(markets: list[Market], targets: list[float], state=None, config=CONFIG):
    builder = LedgerBuilder(config)
    state = state or initial_state(config)
    for i, (m, t) in enumerate(zip(markets, targets, strict=True)):
        state, events = step(state, config, m, t)
        builder.add_step(T0 + timedelta(minutes=5 * i), events)
    return builder.build()


def closeout_ledger():
    return record(
        [mkt(), mkt(1.1, THRESHOLD - 1e-4)],
        [1.0, 1.0],
    )


def test_ledger_carries_clip_and_closeout_flags() -> None:
    small = EngineConfig(starting_equity=5_000.0)
    clipped = record(
        [mkt(), mkt()], [1.0, 1.0], state=pending_long(5_000.0), config=small
    )
    assert clipped.fills["clipped"].to_list() == [True]
    assert clipped.steps["clipped"].to_list() == [True, True]
    assert clipped.steps["shortfall_units"].to_list()[0] == 45_455
    verify_ledger(clipped)

    ledger = closeout_ledger()
    assert ledger.steps["closeout"].to_list() == [False, True]
    assert ledger.fills["closeout"].to_list() == [False, True]
    assert ledger.steps["position_units"][-1] == 0
    verify_ledger(ledger)


def with_steps(ledger, **cols: list):
    return ledger.__class__(
        steps=ledger.steps.with_columns(**{k: pl.Series(v) for k, v in cols.items()}),
        fills=ledger.fills,
        starting_equity=ledger.starting_equity,
        closed_lots=ledger.closed_lots,
        leverage_cap=ledger.leverage_cap,
    )


def test_verify_rejects_a_closeout_that_is_not_the_last_row() -> None:
    ledger = record(
        [mkt(), mkt(1.1, THRESHOLD - 1e-4), mkt()],
        [1.0, 1.0, 0.0],
    )
    with pytest.raises(LedgerError, match="closeout"):
        verify_ledger(ledger)


def test_verify_rejects_a_closeout_row_without_a_closeout_fill() -> None:
    ledger = closeout_ledger()
    bad = ledger.__class__(
        steps=ledger.steps,
        fills=ledger.fills.with_columns(closeout=pl.lit(False)),
        starting_equity=ledger.starting_equity,
        closed_lots=ledger.closed_lots,
        leverage_cap=ledger.leverage_cap,
    )
    with pytest.raises(LedgerError, match="closeout"):
        verify_ledger(bad)


def test_verify_rejects_a_closeout_row_that_is_not_flat() -> None:
    ledger = closeout_ledger()
    with pytest.raises(LedgerError, match="closeout"):  # fill flagged, row not
        verify_ledger(with_steps(ledger, closeout=[False, False]))
    # flag the opening row instead: a closeout row that holds a position
    with pytest.raises(LedgerError, match="closeout"):
        verify_ledger(with_steps(ledger, closeout=[True, True]))


def test_verify_rejects_exposure_added_beyond_the_leverage_limit() -> None:
    ledger = record([mkt(), mkt()], [1.0, 1.0])
    verify_ledger(ledger)
    # a fill that adds more than equity x cap allows must be rejected
    tight = ledger.__class__(
        steps=ledger.steps,
        fills=ledger.fills,
        starting_equity=ledger.starting_equity,
        closed_lots=ledger.closed_lots,
        leverage_cap=5.0,
    )
    with pytest.raises(LedgerError, match="leverage"):
        verify_ledger(tight)
