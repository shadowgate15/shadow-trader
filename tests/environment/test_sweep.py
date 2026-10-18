"""Session-end force-flat in the env, explicit reset options and the sweep."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from shadow_trader.environment import (
    ConfigError,
    EngineConfig,
    EnvConfig,
    EpisodeConfig,
    verify_ledger,
)
from shadow_trader.environment.env import TradingEnv, make_env
from shadow_trader.environment.sweep import SweepResult, run_sweep

from .fixture import FINAL_SESSION_BARS, SESSION_BARS, WARM_UP, load_fixture

LONG, FLAT = 2, 1  # indices into the default levels (-1, 0, 1)
HAIR_TRIGGER = EngineConfig(leverage_cap=50.0, closeout_fraction=1.0)


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


def build(prepared, **overrides) -> TradingEnv:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20), **overrides)
    return make_env(dataset, scaler, config, seed=3, record_ledger=True)


def always(action: int) -> Callable[[object], int]:
    return lambda _obs: action


def play(env: TradingEnv, action: int) -> dict:
    done = False
    info: dict = {}
    while not done:
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


# --- reset(options) ---------------------------------------------------------------


def test_reset_options_pick_the_session_start_bar_and_equity(prepared) -> None:
    env = build(prepared)

    _, info = env.reset(options={"session": 1, "start_bar": 100, "equity": 7_500.0})

    assert info["session"] == 1
    assert info["start_bar_in_session"] == 100
    assert env.equity == 7_500.0
    _, _, _, _, step_info = env.step(FLAT)
    assert step_info["equity_delta"] == 0.0


def test_default_start_is_the_first_usable_bar(prepared) -> None:
    env = build(prepared)
    _, info = env.reset(options={"session": 0})
    assert info["start_bar_in_session"] == WARM_UP


def test_reset_options_ignore_the_rng_state(prepared) -> None:
    env = build(prepared)
    options = {"session": 2, "start_bar": 50, "equity": 9_000.0}

    first_obs, first_info = env.reset(seed=1, options=options)
    play(env, LONG)
    first = env.ledger()
    env.reset(seed=99)  # perturb the generator with random windows
    env.reset()
    second_obs, second_info = env.reset(seed=5, options=options)
    play(env, LONG)

    assert first_info == second_info
    for key in first_obs:
        np.testing.assert_array_equal(first_obs[key], second_obs[key])
    assert env.ledger().steps.drop("episode_id").equals(first.steps.drop("episode_id"))


@pytest.mark.parametrize(
    "options",
    [
        {"session": 99},
        {"session": 3},  # no usable Bar
        {"session": 0, "start_bar": 2},  # still in warm-up
        {"session": 0, "start_bar": SESSION_BARS},  # past the Session
        {"start_bar": 10},  # a start without a Session
        {"session": 0, "equity": -1.0},
        {"session": 0, "bogus": 1},
    ],
)
def test_invalid_reset_options_raise(prepared, options) -> None:
    with pytest.raises(ValueError):
        build(prepared).reset(options=options)


# --- Session-end force-flat in the env ------------------------------------------------


def test_the_agent_is_forced_flat_on_the_final_bar_and_the_episode_ends(
    prepared,
) -> None:
    env = build(prepared)
    env.reset(options={"session": 4})

    info = play(env, LONG)  # asks for long on every Bar, including the last

    ledger = env.ledger()
    verify_ledger(ledger)
    assert info["ending"] == "session_end"
    assert info["forced_flat"] is True
    last = ledger.steps.height - 1
    assert ledger.steps["position_units"][last] == 0
    forced = ledger.fills.filter(pl.col("forced_flat"))
    assert forced["step"].to_list() == [last]  # the cost lands in the last step
    assert forced["units"][0] < 0
    assert ledger.episodes["ending"].to_list() == ["session_end"]
    assert ledger.episodes["bars"].to_list() == [FINAL_SESSION_BARS - WARM_UP - 1]


def test_session_end_truncates_rather_than_terminates(prepared) -> None:
    env = build(prepared)
    env.reset(options={"session": 4})
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(FLAT)
        done = terminated or truncated
    assert truncated and not terminated


def test_a_session_whose_data_stops_before_the_final_bar_ends_as_dataset_end(
    prepared,
) -> None:
    env = build(prepared)
    env.reset(options={"session": 0})
    info = play(env, FLAT)
    assert info["ending"] == "dataset_end"
    assert env.ledger().episodes["ending"].to_list() == ["dataset_end"]


def test_a_training_window_that_reaches_the_final_bar_is_forced_flat(prepared) -> None:
    dataset, scaler = prepared
    config = EnvConfig(
        episode=EpisodeConfig(window_bars=FINAL_SESSION_BARS - WARM_UP - 1)
    )
    env = make_env(dataset, scaler, config, seed=0, record_ledger=True)
    # only the start on each short final Session's first usable Bar ends on its final
    # Bar; the seed is fixed, so sampling until one is drawn is deterministic
    for _ in range(5_000):
        _, info = env.reset()
        if info["session"] in (4, 5):
            break
    else:
        pytest.fail("never sampled a final Session")
    info = play(env, LONG)
    assert info["ending"] == "session_end"
    verify_ledger(env.ledger())


# --- the sweep -----------------------------------------------------------------------


def test_a_sweep_runs_one_episode_per_session_in_order_and_chains(prepared) -> None:
    env = build(prepared)

    result = run_sweep(env, always(LONG))

    assert isinstance(result, SweepResult)
    ledger = result.ledger
    verify_ledger(ledger)
    assert result.sessions_run == [0, 1, 2, 4, 5]
    assert result.skipped == [3]  # the Session with no usable Bar
    assert result.closeout is False
    episodes = ledger.episodes
    assert episodes["session_id"].to_list() == [0, 1, 2, 4, 5]
    assert episodes["chained"].to_list() == [False, True, True, True, True]
    ends, starts = episodes["end_equity"].to_list(), episodes["start_equity"].to_list()
    assert starts[1:] == ends[:-1]
    assert starts[0] == EngineConfig().starting_equity
    assert episodes["ending"].to_list() == [
        "dataset_end",
        "dataset_end",
        "dataset_end",
        "session_end",
        "session_end",
    ]
    # every Session-final episode ends flat
    final_rows = (
        ledger.steps.filter(pl.col("episode_id").is_in([3, 4]))
        .group_by("episode_id", maintain_order=True)
        .last()
    )
    assert final_rows["position_units"].to_list() == [0, 0]


def test_a_sweep_can_cover_chosen_sessions(prepared) -> None:
    result = run_sweep(build(prepared), always(FLAT), sessions=[4, 5])
    assert result.sessions_run == [4, 5]
    assert result.ledger.episodes["chained"].to_list() == [False, True]


def test_a_flat_policy_sweep_changes_nothing(prepared) -> None:
    result = run_sweep(build(prepared), always(FLAT))
    equity = EngineConfig().starting_equity
    assert set(result.ledger.steps["equity"].to_list()) == {equity}
    assert result.ledger.fills.height == 0


def test_a_closeout_stops_the_sweep_and_records_the_stop(prepared) -> None:
    env = build(prepared, engine=HAIR_TRIGGER)

    result = run_sweep(env, always(LONG))

    assert result.closeout is True
    assert result.sessions_run == [0]  # the rest were not run
    ledger = result.ledger
    verify_ledger(ledger)
    assert ledger.episodes["ending"].to_list() == ["closeout"]
    assert ledger.episodes.height == 1


def test_skipped_sessions_are_logged(prepared, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        run_sweep(build(prepared), always(FLAT))
    assert any("no usable" in r.message for r in caplog.records)


def test_a_scripted_baseline_goes_through_the_same_engine_and_ledger(prepared) -> None:
    # a baseline needs no model: only a function from observation to level index
    def buy_and_hold_morning(obs: object) -> int:
        return LONG

    agent = run_sweep(build(prepared), buy_and_hold_morning)
    again = run_sweep(build(prepared), buy_and_hold_morning)

    verify_ledger(agent.ledger)
    for table in ("steps", "fills", "closed_lots", "episodes"):
        assert getattr(agent.ledger, table).equals(getattr(again.ledger, table))


def test_a_sweep_needs_a_recording_env(prepared) -> None:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    env = make_env(dataset, scaler, config, seed=0)
    with pytest.raises(ValueError, match="record_ledger"):
        run_sweep(env, always(FLAT))


def test_hold_across_weekend_in_a_config_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "env.toml"
    path.write_text("[episode]\nhold_across_weekend = true\n")
    with pytest.raises(ConfigError, match="weekend"):
        EnvConfig.from_toml(path)
