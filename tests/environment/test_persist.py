"""Ledger persistence, the provenance manifest and ``read_ledger``."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from shadow_trader.environment import (
    EngineConfig,
    EnvConfig,
    EpisodeConfig,
    LedgerError,
    RewardConfig,
    persist,
)
from shadow_trader.environment.env import make_env
from shadow_trader.environment.persist import (
    MANIFEST_FILE,
    SCHEMA_VERSION,
    ManifestError,
    persist_sweep,
    read_ledger,
    read_manifest,
    write_ledger,
)

from .fixture import load_fixture

LONG = 2
SEAM = 7


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


def recorded_env(prepared, config=None):
    dataset, scaler = prepared
    config = config or EnvConfig(episode=EpisodeConfig(window_bars=20))
    return make_env(dataset, scaler, config, seed=11, record_ledger=True)


def play_one(env) -> None:
    env.reset(options={"session": 0})
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(LONG)
        done = term or trunc


@pytest.fixture
def persisted(prepared, tmp_path) -> tuple[Path, object]:
    env = recorded_env(prepared)
    play_one(env)
    path = tmp_path / "ledger"
    write_ledger(env, path, episode_mode="eval", seam_version=SEAM)
    return path, env.ledger()


def test_round_trips_through_read_ledger(persisted) -> None:
    path, ledger = persisted
    back = read_ledger(path)
    assert back.steps.equals(ledger.steps)
    assert back.fills.equals(ledger.fills)
    assert back.closed_lots.equals(ledger.closed_lots)
    assert back.episodes.equals(ledger.episodes)
    assert back.starting_equity == ledger.starting_equity
    assert back.leverage_cap == ledger.leverage_cap
    assert sorted(p.name for p in path.glob("*.parquet")) == [
        "closed_lots.parquet",
        "episodes.parquet",
        "fills.parquet",
        "steps.parquet",
    ]


def test_manifest_records_provenance(persisted, prepared) -> None:
    path, _ = persisted
    dataset, scaler = prepared
    m = read_manifest(path)
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["dataset_id"] == dataset.dataset_id
    assert m["scaler_hash"] == scaler.scaler_id
    assert m["seam_version"] == SEAM
    assert m["episode_mode"] == "eval"
    assert m["env_config_hash"] == config.config_hash()
    assert m["package_version"]
    assert m["seed_entropy"] == "11"
    assert m["configs"]["reward"]["hash"] == config.reward.config_hash()
    assert m["configs"]["reward"]["json"] == {"reward_scale": 10_000.0}
    assert set(m["configs"]) == {"engine", "reward", "episode", "observation"}


def test_reward_config_is_recorded_though_ledger_has_no_reward(
    prepared, tmp_path
) -> None:
    config = EnvConfig(
        reward=RewardConfig(reward_scale=5.0), episode=EpisodeConfig(window_bars=20)
    )
    env = recorded_env(prepared, config)
    play_one(env)
    write_ledger(env, tmp_path / "l", episode_mode="eval", seam_version=SEAM)
    assert read_manifest(tmp_path / "l")["configs"]["reward"]["json"] == {
        "reward_scale": 5.0
    }
    assert not any("reward" in c for c in env.ledger().steps.columns)


def test_sub_configs_hash_separately_and_change_the_env_hash() -> None:
    base = EnvConfig()
    variants = [
        replace(base, engine=EngineConfig(leverage_cap=3.0)),
        replace(base, reward=RewardConfig(reward_scale=1.0)),
        replace(base, episode=EpisodeConfig(window_bars=10)),
        replace(base, observation=replace(base.observation, window_bars=4)),
    ]
    hashes = [v.config_hash() for v in variants]
    assert base.config_hash() not in hashes
    assert len(set(hashes)) == 4
    assert base.sub_config_hashes().keys() == {
        "engine",
        "reward",
        "episode",
        "observation",
    }
    assert base.sub_config_hashes()["reward"] == base.reward.config_hash()
    assert variants[0].sub_config_hashes()["reward"] == base.reward.config_hash()


def test_missing_manifest_is_rejected(persisted) -> None:
    path, _ = persisted
    (path / MANIFEST_FILE).unlink()
    with pytest.raises(ManifestError, match="missing"):
        read_ledger(path)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda m: m.update(dataset_id="other"),
        lambda m: m["configs"]["reward"]["json"].update(reward_scale=1.0),
        lambda m: m.update(seam_version=99),
    ],
)
def test_tampered_manifest_is_rejected(persisted, tamper) -> None:
    path, _ = persisted
    doc = json.loads((path / MANIFEST_FILE).read_text())
    tamper(doc)
    (path / MANIFEST_FILE).write_text(json.dumps(doc))
    with pytest.raises(ManifestError):
        read_ledger(path)


def _bump_equity(path: Path) -> None:
    steps = pl.read_parquet(path / "steps.parquet")
    steps.with_columns(pl.col("equity") + 1.0).write_parquet(path / "steps.parquet")


def test_table_that_disagrees_with_manifest_is_rejected(persisted) -> None:
    path, _ = persisted
    _bump_equity(path)
    with pytest.raises(ManifestError, match="steps"):
        read_ledger(path)


def test_read_ledger_verifies_the_ledger(persisted) -> None:
    """A re-sealed manifest over an invalid ledger still fails verify_ledger."""
    path, _ = persisted
    _bump_equity(path)
    doc = json.loads((path / MANIFEST_FILE).read_text())
    doc["tables"]["steps"] = persist._file_hash(path / "steps.parquet")
    doc["manifest_hash"] = persist._manifest_hash(doc)
    (path / MANIFEST_FILE).write_text(json.dumps(doc))
    with pytest.raises(LedgerError):
        read_ledger(path)


def test_training_episodes_write_no_ledger_by_default(prepared, tmp_path) -> None:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    env = make_env(dataset, scaler, config, seed=0)
    env.reset()
    assert not env.records_ledger
    with pytest.raises(RuntimeError):
        write_ledger(env, tmp_path / "x", episode_mode="train", seam_version=SEAM)
    assert not (tmp_path / "x").exists()


def test_evaluation_sweep_always_persists(prepared, tmp_path) -> None:
    env = recorded_env(prepared)
    result = persist_sweep(env, lambda _o: LONG, tmp_path / "sweep", seam_version=SEAM)
    back = read_ledger(tmp_path / "sweep")
    assert back.steps.equals(result.ledger.steps)
    assert read_manifest(tmp_path / "sweep")["episode_mode"] == "eval_sweep"
