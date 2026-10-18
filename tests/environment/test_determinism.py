"""Golden-hash replay, seeding, ``make_env_fn`` and the ``regen-golden`` guards."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from polars.testing import assert_frame_equal

from shadow_trader import main
from shadow_trader.environment import EnvConfig, EpisodeConfig
from shadow_trader.environment.determinism import (
    current_platform,
    hash_parts,
    make_env_fn,
    make_env_fns,
)

from . import golden
from .fixture import load_fixture

TABLES = ("steps", "fills", "closed_lots", "episodes")


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


@pytest.fixture(scope="module")
def run():
    return golden.run_golden(np.random.SeedSequence(golden.GOLDEN_ENTROPY))


def test_golden_scenario_covers_gaps_rollover_closeout_and_forced_flat(run) -> None:
    steps = pl.concat([p.ledger.steps for p in run.parts])
    fills = pl.concat([p.ledger.fills for p in run.parts])
    assert steps["carried_mark"].any()  # a Gap
    assert (steps["rollover_charge"] != 0).any()
    assert steps["closeout"].any()
    assert fills["forced_flat"].any()
    assert fills["unfilled_forced_flat"].any()


def test_same_seed_sequence_gives_the_same_hash(run) -> None:
    again = golden.run_golden(np.random.SeedSequence(golden.GOLDEN_ENTROPY))
    assert hash_parts(again.parts) == hash_parts(run.parts)


def test_a_different_seed_changes_only_window_starts(run) -> None:
    other = golden.run_golden(np.random.SeedSequence(golden.GOLDEN_ENTROPY + 1))
    assert hash_parts(other.parts) != hash_parts(run.parts)
    for mine, theirs in zip(run.parts, other.parts, strict=True):
        if not mine.name.startswith("random"):
            assert hash_parts([mine]) == hash_parts([theirs])
    mine_eps = [p for p in run.parts if p.name.startswith("random")]
    theirs_eps = [p for p in other.parts if p.name.startswith("random")]
    assert [p.ledger.episodes["start_time"][0] for p in mine_eps] != [
        p.ledger.episodes["start_time"][0] for p in theirs_eps
    ]
    for part in mine_eps + theirs_eps:  # the window length is not a seed matter
        assert part.ledger.episodes["bars"].to_list() == [golden.WINDOW]


def test_the_hash_is_sensitive_to_a_single_table_cell(run) -> None:
    part = run.parts[0]
    steps = part.ledger.steps.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("equity") + 1e-9)
        .otherwise(pl.col("equity"))
        .alias("equity")
    )
    changed = golden.replace_steps(part, steps)
    assert hash_parts([changed]) != hash_parts([part])


def test_the_hash_is_sensitive_to_observations(run) -> None:
    part = run.parts[0]
    features = part.observations["features"].copy()
    features.flat[0] += np.float32(1e-3)
    changed = golden.replace_observations(
        part, {**part.observations, "features": features}
    )
    assert hash_parts([changed]) != hash_parts([part])


def test_the_replay_matches_the_checked_in_golden(run) -> None:
    record = golden.load_record()
    for part in run.parts:
        for table in TABLES:
            stored = pl.read_parquet(golden.GOLDEN_DIR / part.name / f"{table}.parquet")
            assert_frame_equal(getattr(part.ledger, table), stored)
    if record["platform"] != current_platform():
        pytest.skip(f"golden hash was recorded on {record['platform']}")
    assert hash_parts(run.parts) == record["hash"]


# --- make_env_fn --------------------------------------------------------------


def config() -> EnvConfig:
    return EnvConfig(episode=EpisodeConfig(window_bars=20))


def first_start(env_fn) -> object:
    env = env_fn()
    _, info = env.reset()
    return info["start_time"]


def test_make_env_fn_is_zero_argument_and_picklable(prepared) -> None:
    dataset, scaler = prepared
    env_fn = make_env_fn(dataset, scaler, config(), seed=np.random.SeedSequence(3))
    clone = pickle.loads(pickle.dumps(env_fn))
    assert first_start(clone) == first_start(env_fn)


def test_spawned_workers_are_independent_and_reproducible(prepared) -> None:
    dataset, scaler = prepared

    def starts(entropy: int) -> list:
        fns = make_env_fns(dataset, scaler, config(), 6, seed=entropy)
        return [first_start(fn) for fn in fns]

    assert starts(7) == starts(7)
    assert len(set(starts(7))) > 1
    entropies = {
        fn().seed_entropy for fn in make_env_fns(dataset, scaler, config(), 6, seed=7)
    }
    assert len(entropies) == 6


def drive(vec, steps: int = 60):
    vec.reset()
    rewards, dones = [], 0
    for _ in range(steps):
        _, reward, terminated, truncated, _ = vec.step(vec.action_space.sample())
        rewards.append(reward.copy())
        dones += int((terminated | truncated).sum())
    vec.close()
    return np.array(rewards), dones


def test_make_env_fn_works_with_sync_and_async_vector_envs(prepared) -> None:
    dataset, scaler = prepared

    def fns():
        return make_env_fns(dataset, scaler, config(), 2, seed=11)

    sync = SyncVectorEnv(fns(), autoreset_mode=AutoresetMode.NEXT_STEP)
    sync.action_space.seed(0)
    sync_rewards, sync_dones = drive(sync)
    pool = AsyncVectorEnv(
        fns(), context="spawn", autoreset_mode=AutoresetMode.NEXT_STEP
    )
    pool.action_space.seed(0)
    async_rewards, async_dones = drive(pool)
    assert sync_dones > 0  # episodes ended, and the vector env reset them
    assert sync_dones == async_dones
    np.testing.assert_array_equal(sync_rewards, async_rewards)


# --- regen-golden -------------------------------------------------------------


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_regen_golden_refuses_a_dirty_tree(monkeypatch, capsys) -> None:
    before = snapshot(golden.GOLDEN_DIR)
    monkeypatch.setattr("shadow_trader.environment.cli.tree_is_dirty", lambda: True)
    assert main(["env", "regen-golden"]) == 1
    assert "dirty" in capsys.readouterr().err
    assert snapshot(golden.GOLDEN_DIR) == before


def test_regen_golden_refuses_an_off_platform_run(monkeypatch, capsys) -> None:
    before = snapshot(golden.GOLDEN_DIR)
    monkeypatch.setattr("shadow_trader.environment.cli.tree_is_dirty", lambda: False)
    monkeypatch.setattr(
        "shadow_trader.environment.cli.current_platform", lambda: "plan9-mips"
    )
    assert main(["env", "regen-golden"]) == 1
    assert "platform" in capsys.readouterr().err
    assert snapshot(golden.GOLDEN_DIR) == before
