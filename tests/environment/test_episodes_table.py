"""The ledger ``episodes`` table and the episode checks of ``verify_ledger``."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

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

CONFIG = EngineConfig()
T0 = datetime(2024, 1, 2, tzinfo=UTC)
BAR = timedelta(minutes=5)
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


def sweep(*, chain: bool = True) -> Ledger:
    """Two Sessions of (long, final) Bars, the second carrying equity in."""
    builder = LedgerBuilder(CONFIG)
    equity = CONFIG.starting_equity
    step_no = 0
    for episode in range(2):
        state = initial_state(CONFIG, equity)
        builder.begin_episode(
            episode,
            session_id=episode,
            start_time=T0 + step_no * BAR,
            start_equity=equity,
            chained=chain and episode > 0,
        )
        for market, target in ((OPEN, 1.0), (FINAL, 1.0)):
            state, events = step(state, CONFIG, market, target)
            builder.add_step(
                T0 + step_no * BAR, events, episode_id=episode, session_id=episode
            )
            step_no += 1
        builder.end_episode("session_end")
        equity = state.equity
    return builder.build()


def tampered(ledger: Ledger, episode: int, column: str, value: object) -> Ledger:
    episodes = ledger.episodes.with_columns(
        pl.when(pl.col("episode_id") == episode)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    return replace(ledger, episodes=episodes)


def test_a_chained_sweep_verifies_and_records_one_row_per_episode() -> None:
    ledger = sweep()

    verify_ledger(ledger)
    episodes = ledger.episodes
    assert episodes.columns == [
        "episode_id",
        "session_id",
        "start_time",
        "start_equity",
        "end_equity",
        "ending",
        "bars",
        "chained",
    ]
    assert episodes["bars"].to_list() == [2, 2]
    assert episodes["ending"].to_list() == ["session_end", "session_end"]
    first_end = ledger.steps["equity"][1]
    assert episodes["end_equity"][0] == first_end
    assert episodes["start_equity"][1] == first_end  # chained
    assert episodes["end_equity"][1] == ledger.steps["equity"][3]
    assert ledger.fills["forced_flat"].to_list() == [False, True, False, True]


def test_every_episode_ends_flat_at_session_end() -> None:
    ledger = sweep()
    ends = ledger.steps.filter(pl.col("step").is_in([1, 3]))
    assert ends["position_units"].to_list() == [0, 0]


def test_end_equity_must_match_the_last_steps_row() -> None:
    bad = tampered(sweep(), 0, "end_equity", 123.0)
    with pytest.raises(LedgerError, match="end equity"):
        verify_ledger(bad)


def test_start_equity_must_match_the_first_steps_row() -> None:
    bad = tampered(sweep(chain=False), 0, "start_equity", 10_005.0)
    with pytest.raises(LedgerError, match="equity"):
        verify_ledger(bad)


def test_a_chained_episode_must_start_where_the_last_one_ended() -> None:
    bad = tampered(sweep(), 1, "start_equity", 10_005.0)
    with pytest.raises(LedgerError, match="chain"):
        verify_ledger(bad)


def test_bar_count_must_match_the_steps() -> None:
    bad = tampered(sweep(), 0, "bars", 3)
    with pytest.raises(LedgerError, match="bars"):
        verify_ledger(bad)


def test_a_session_end_episode_that_ends_in_a_position_is_rejected() -> None:
    builder = LedgerBuilder(CONFIG)
    builder.begin_episode(0, 0, T0, CONFIG.starting_equity)
    _, events = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    builder.add_step(T0, events, episode_id=0, session_id=0)
    builder.end_episode("session_end")
    with pytest.raises(LedgerError, match="flat"):
        verify_ledger(builder.build())


def test_nothing_chains_after_a_closeout() -> None:
    bad = tampered(sweep(), 0, "ending", "closeout")
    with pytest.raises(LedgerError, match="closeout"):
        verify_ledger(bad)


def test_steps_must_belong_to_a_listed_episode() -> None:
    ledger = sweep()
    bad = replace(ledger, episodes=ledger.episodes.filter(pl.col("episode_id") == 0))
    with pytest.raises(LedgerError, match="episode"):
        verify_ledger(bad)


def test_a_ledger_without_episodes_still_verifies_from_starting_equity() -> None:
    _, events = step(initial_state(CONFIG), CONFIG, OPEN, 1.0)
    builder = LedgerBuilder(CONFIG)
    builder.add_step(T0, events)
    verify_ledger(builder.build())
