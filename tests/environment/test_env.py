"""TradingEnv skeleton, driven through ``make_env`` with scripted actions.

Everything is asserted on the Ledger, ``info`` and observations. The fixture Dataset
(see ``fixture.py``) has three Sessions of 280 Bars (first usable Bar 6) and a fourth
Session of 5 Bars that has no usable Bar. Windows are short (N=50) so they fit.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import polars as pl
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from shadow_trader.environment import Ledger, verify_ledger
from shadow_trader.environment.config import ConfigError, EngineConfig
from shadow_trader.environment.env import TradingEnv, make_env
from shadow_trader.environment.env_config import (
    EnvConfig,
    EpisodeConfig,
    RewardConfig,
)

from .fixture import SESSION_OPENS, WARM_UP, load_fixture

N = 50
SHORT = SESSION_OPENS[3]


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


def build(prepared, seed: int = 0, *, record: bool = True, **overrides) -> TradingEnv:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=N), **overrides)
    return make_env(dataset, scaler, config, seed=seed, record_ledger=record)


def run(env: TradingEnv, actions: Sequence[int], seed: int | None = None):
    """Reset, play ``actions`` (cycled) to the end; return (rewards, infos, info0)."""
    _, info0 = env.reset(seed=seed)
    rewards, infos = [], []
    done, i = False, 0
    while not done:
        _, reward, terminated, truncated, info = env.step(actions[i % len(actions)])
        rewards.append(reward)
        infos.append(info)
        done = terminated or truncated
        i += 1
    return rewards, infos, info0


def test_spaces_mirror_the_observation_and_levels(prepared) -> None:
    env = build(prepared)

    assert env.action_space == spaces.Discrete(3)
    assert isinstance(env.observation_space, spaces.Dict)
    assert set(env.observation_space.spaces) == {
        "features",
        "valid",
        "position",
        "account",
    }
    assert env.observation_space["features"].shape == (1, 1, 3)
    assert env.observation_space["position"].shape == (1, 3)
    assert env.observation_space["account"].shape == (2,)


def test_level_set_is_configurable(prepared) -> None:
    levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    env = build(prepared, engine=EngineConfig(levels=levels))

    assert env.action_space == spaces.Discrete(5)
    # index 3 is +0.5: from flat it buys half the +1 amount
    run(env, [3, 3], seed=1)
    half = env.ledger().fills["units"][0]
    run(env, [4, 4], seed=1)
    full = env.ledger().fills["units"][0]
    assert half == pytest.approx(full / 2, abs=1)


def test_check_env_passes(prepared) -> None:
    check_env(build(prepared, record=False), skip_render_check=True)


def test_reset_observation_is_in_space(prepared) -> None:
    env = build(prepared)

    obs, _ = env.reset(seed=3)

    assert env.observation_space.contains(obs)
    assert obs["features"].dtype == np.float32
    assert np.isfinite(obs["features"]).all()
    assert obs["position"][0].tolist() == [0.0, 0.0, 0.0]


def test_scripted_run_produces_a_valid_ledger(prepared) -> None:
    env = build(prepared)

    run(env, [2, 2, 1, 0, 0, 1, 2], seed=1)
    ledger = env.ledger()

    assert isinstance(ledger, Ledger)
    verify_ledger(ledger)
    assert ledger.steps.height == N
    assert ledger.fills.height > 3
    assert ledger.closed_lots.height > 0
    assert ledger.steps["episode_id"].n_unique() == 1
    assert "reward" not in "".join(ledger.steps.columns + ledger.fills.columns)


def test_same_seed_same_windows_and_ledger_different_seed_other_windows(
    prepared,
) -> None:
    def starts(seed: int) -> list[datetime]:
        env = build(prepared, seed)
        out = []
        for _ in range(8):
            _, info = env.reset()
            out.append(info["start_time"])
        return out

    assert starts(5) == starts(5)
    assert starts(5) != starts(6)

    first = build(prepared, 5)
    second = build(prepared, 5)
    run(first, [2, 1, 0], seed=9)
    run(second, [2, 1, 0], seed=9)
    assert first.ledger().steps.equals(second.ledger().steps)
    assert first.ledger().fills.equals(second.ledger().fills)


def test_a_different_seed_with_the_same_start_changes_nothing_else(prepared) -> None:
    env = build(prepared, record=True)
    by_start: dict[datetime, tuple[int, pl.DataFrame]] = {}
    for seed in range(300):
        _, _, info = run(env, [2, 0, 1], seed=seed)
        steps = env.ledger().steps.drop("episode_id")
        previous = by_start.setdefault(info["start_time"], (seed, steps))
        if previous[0] != seed:
            assert previous[1].equals(steps)
            return
    pytest.fail("no two seeds shared a window start")


def test_reward_is_scaled_log_equity_return(prepared) -> None:
    scale = 10_000.0
    env = build(prepared, reward=RewardConfig(reward_scale=scale))

    rewards, _, _ = run(env, [2, 2, 0, 0, 1], seed=4)
    equity = [10_000.0, *env.ledger().steps["equity"].to_list()]

    expected = [
        math.log(after / before) * scale for before, after in itertools.pairwise(equity)
    ]
    assert rewards == pytest.approx(expected, rel=1e-12)
    # rewards sum to the episode's log P&L
    assert sum(rewards) == pytest.approx(math.log(equity[-1] / equity[0]) * scale)
    assert any(r != 0 for r in rewards)


def test_flat_episode_earns_nothing_and_costs_nothing(prepared) -> None:
    env = build(prepared)

    rewards, _, _ = run(env, [1])

    assert rewards == [0.0] * N
    assert env.ledger().fills.height == 0
    assert env.ledger().steps["equity"].unique().to_list() == [10_000.0]


def test_reward_scale_only_rescales(prepared) -> None:
    small = build(prepared, reward=RewardConfig(reward_scale=1.0))
    large = build(prepared, reward=RewardConfig(reward_scale=100.0))

    a, _, _ = run(small, [2, 1, 0], seed=2)
    b, _, _ = run(large, [2, 1, 0], seed=2)

    assert b == pytest.approx([100.0 * r for r in a])


def test_episodes_stay_in_one_session_and_start_on_usable_bars(
    prepared, caplog: pytest.LogCaptureFixture
) -> None:
    dataset, _scaler = prepared
    frame = dataset.read("EURUSD")
    usable = dict(zip(frame["open_time"], frame["usable"], strict=True))
    session_of = dict(zip(frame["open_time"], frame["session_id"], strict=True))

    with caplog.at_level(logging.WARNING):
        env = build(prepared)
    skipped = [r.getMessage() for r in caplog.records]
    assert any(str(SHORT.date()) in message for message in skipped)

    seen_sessions = set()
    for seed in range(40):
        _, _, info = run(env, [2, 0], seed=seed)
        steps = env.ledger().steps
        start = info["start_time"]
        assert usable[start]
        assert steps.height == N
        sessions = {session_of[t] for t in steps["open_time"].to_list()}
        assert sessions == {session_of[start]}
        assert steps["open_time"][0] > start
        assert session_of[start] != SHORT
        seen_sessions.add(session_of[start])
    assert len(seen_sessions) == 3
    assert info["start_bar_in_session"] >= WARM_UP


def test_last_step_truncates_with_window_end_and_equity_delta(prepared) -> None:
    env = build(prepared)
    env.reset(seed=1)

    results = []
    done = False
    while not done:
        results.append(env.step(2))
        done = results[-1][2] or results[-1][3]

    assert len(results) == N
    for _, _, terminated, truncated, info in results[:-1]:
        assert not terminated
        assert not truncated
        assert info["ending"] is None
    _, _, terminated, truncated, info = results[-1]
    assert truncated
    assert not terminated
    assert info["ending"] == "window_end"
    equity = env.ledger().steps["equity"].to_list()
    for n, (_, _, _, _, info) in enumerate(results):
        before = 10_000.0 if n == 0 else equity[n - 1]
        assert info["equity_delta"] == pytest.approx(equity[n] - before, abs=1e-9)
    with pytest.raises(RuntimeError):
        env.step(2)


def test_choosing_the_current_target_is_a_no_op_and_units_stay_frozen(
    prepared,
) -> None:
    env = build(prepared)

    run(env, [2], seed=7)  # go long on step 0 and keep choosing +1
    ledger = env.ledger()

    assert ledger.fills.height == 1
    assert ledger.fills["step"].to_list() == [0]
    assert set(ledger.steps["position_units"].to_list()) == {ledger.fills["units"][0]}


def test_a_level_change_trades_and_flat_exits(prepared) -> None:
    env = build(prepared)

    run(env, [2, 2] + [1] * N, seed=7)  # long, hold, then flat
    ledger = env.ledger()

    assert ledger.fills.height == 2
    assert ledger.fills["step"].to_list() == [0, 2]
    assert ledger.fills["units"][1] == -ledger.fills["units"][0]
    assert ledger.closed_lots.height == 1
    assert ledger.steps["position_units"].to_list()[2:] == [0] * (N - 2)


def test_no_reward_anywhere_in_info(prepared) -> None:
    env = build(prepared)

    _, infos, _ = run(env, [2, 0], seed=1)

    assert all("reward" not in key for info in infos for key in info)


def test_a_dataset_without_eligible_windows_is_rejected(prepared) -> None:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=10_000))

    with pytest.raises(ValueError, match="window"):
        make_env(dataset, scaler, config, seed=0)


def test_a_scaler_from_another_spec_is_rejected(prepared) -> None:
    from shadow_trader.preparation.scaler import Scaler

    dataset, scaler = prepared
    other = Scaler({**scaler.artifact, "spec_hash": "0" * 64})

    with pytest.raises(Exception, match="scaler"):
        make_env(dataset, other, EnvConfig(), seed=0)


def test_bar_size_mismatch_is_rejected(prepared) -> None:
    from shadow_trader.environment import ObservationConfig

    dataset, scaler = prepared
    config = EnvConfig(observation=ObservationConfig(bar_minutes=15))

    with pytest.raises(ConfigError, match="bar"):
        make_env(dataset, scaler, config, seed=0)
