"""Training-only environment configuration (Gymnasium-free).

``EnvConfig`` composes the seam configs (``EngineConfig``, ``ObservationConfig``)
with the env-layer ones: ``RewardConfig`` and ``EpisodeConfig``. Construction
matches ``EngineConfig``: frozen dataclasses, defaults in code, validation in
``__post_init__``, and a TOML file that only overrides the defaults, with one
table per part (``[engine]``, ``[reward]``, ``[episode]``, ``[observation]``).
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ._common import HashableConfig, canonical_hash, check_int, check_number
from .config import ConfigError, EngineConfig
from .observation import ObservationConfig

__all__ = ["EnvConfig", "EpisodeConfig", "RewardConfig"]


@dataclass(frozen=True)
class RewardConfig(HashableConfig):
    """``reward_scale`` multiplies the per-step log-return of equity."""

    reward_scale: float = 10_000.0

    def __post_init__(self) -> None:
        if check_number(self.reward_scale, "reward_scale", ConfigError) <= 0:
            raise ConfigError("reward_scale must be positive and finite")


@dataclass(frozen=True)
class EpisodeConfig(HashableConfig):
    """Training episodes are random windows of ``window_bars`` stepped Bars in one
    Session. The start Bar, observed at reset, comes on top: a window needs
    ``window_bars + 1`` Bars (see ``eligible_starts``).

    ``hold_across_weekend`` exists only to be rejected: v1 has no cross-Session
    episode mode.
    """

    window_bars: int = 576
    hold_across_weekend: bool = False

    def __post_init__(self) -> None:
        if check_int(self.window_bars, "window_bars", ConfigError) < 1:
            raise ConfigError("window_bars must be a positive integer")
        if self.hold_across_weekend:
            raise ConfigError("hold_across_weekend is not supported in v1")


@dataclass(frozen=True)
class EnvConfig(HashableConfig):
    engine: EngineConfig = field(default_factory=EngineConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def sub_configs(self) -> dict[str, Any]:
        """Each sub-config as plain data, keyed by its field name."""
        return asdict(self)

    def sub_config_hashes(self) -> dict[str, str]:
        """SHA-256 of each sub-config's canonical JSON; independent of the others."""
        return {
            name: canonical_hash(value) for name, value in self.sub_configs().items()
        }

    @classmethod
    def from_toml(cls, path: Path) -> EnvConfig:
        """Load a TOML file over the code defaults; raise ConfigError on any problem."""
        try:
            raw = tomllib.loads(Path(path).read_text())
        except OSError as exc:
            raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {path}: {exc}") from None
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ConfigError(f"unknown table(s) in config: {', '.join(unknown)}")
        return cls(
            engine=EngineConfig._from_mapping(raw.get("engine", {})),
            reward=_section(RewardConfig, raw, "reward"),
            episode=_section(EpisodeConfig, raw, "episode"),
            observation=_section(ObservationConfig, raw, "observation"),
        )


def _section(cls: Any, raw: Mapping[str, Any], name: str) -> Any:
    table = raw.get(name, {})
    if not isinstance(table, dict):
        raise ConfigError(f"[{name}] must be a table")
    unknown = sorted(set(table) - {f.name for f in fields(cls)})
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")
    return cls(**table)
