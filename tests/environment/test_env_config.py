"""EnvConfig composition, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from shadow_trader.environment.config import ConfigError, EngineConfig
from shadow_trader.environment.env_config import (
    EnvConfig,
    EpisodeConfig,
    RewardConfig,
)


def test_defaults() -> None:
    config = EnvConfig()

    assert config.reward.reward_scale == 10_000.0
    assert config.episode.window_bars == 576
    assert config.engine == EngineConfig()
    assert config.observation.window_bars == 1


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_reward_scale_must_be_positive_and_finite(scale: float) -> None:
    with pytest.raises(ConfigError):
        RewardConfig(reward_scale=scale)


def test_window_bars_must_be_positive() -> None:
    with pytest.raises(ConfigError):
        EpisodeConfig(window_bars=0)


def test_hold_across_weekend_is_rejected_in_v1() -> None:
    with pytest.raises(ConfigError, match="weekend"):
        EpisodeConfig(hold_across_weekend=True)


def test_config_hash_is_stable_and_sensitive() -> None:
    base = EnvConfig()

    assert base.config_hash() == EnvConfig().config_hash()
    assert len(base.config_hash()) == 64
    assert EnvConfig(reward=RewardConfig(reward_scale=1.0)).config_hash() != (
        base.config_hash()
    )
    assert base.reward.config_hash() != base.episode.config_hash()


def test_from_toml_overrides_only_what_it_names(tmp_path: Path) -> None:
    path = tmp_path / "env.toml"
    path.write_text(
        "[reward]\nreward_scale = 100.0\n"
        "[episode]\nwindow_bars = 50\n"
        "[engine]\nlevels = [-1.0, -0.5, 0.0, 0.5, 1.0]\n"
        "[observation]\nwindow_bars = 3\n"
    )

    config = EnvConfig.from_toml(path)

    assert config.reward.reward_scale == 100.0
    assert config.episode.window_bars == 50
    assert config.engine.levels == (-1.0, -0.5, 0.0, 0.5, 1.0)
    assert config.engine.starting_equity == 10_000.0
    assert config.observation.window_bars == 3


def test_from_toml_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "env.toml"
    path.write_text("[reward]\nbonus = 1\n")

    with pytest.raises(ConfigError, match="bonus"):
        EnvConfig.from_toml(path)
