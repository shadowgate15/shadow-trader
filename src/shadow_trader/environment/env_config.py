"""Training-only environment configuration (Gymnasium-free).

``EnvConfig`` composes the seam configs (``EngineConfig``, ``ObservationConfig``)
with the env-layer ones: ``RewardConfig`` and ``EpisodeConfig``. Construction
matches ``EngineConfig``: frozen dataclasses, defaults in code, validation in
``__post_init__``, and a TOML file that only overrides the defaults, with one
table per part (``[engine]``, ``[reward]``, ``[episode]``, ``[observation]``).
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from shadow_trader.preparation._io import canonical_json

from .config import ConfigError, EngineConfig
from .observation import ObservationConfig

__all__ = ["EnvConfig", "EpisodeConfig", "RewardConfig"]


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class RewardConfig:
    """``reward_scale`` multiplies the per-step log-return of equity."""

    reward_scale: float = 10_000.0

    def __post_init__(self) -> None:
        scale = self.reward_scale
        if isinstance(scale, bool) or not isinstance(scale, int | float):
            raise ConfigError("reward_scale must be a number")
        if not math.isfinite(scale) or scale <= 0:
            raise ConfigError("reward_scale must be positive and finite")

    def config_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class EpisodeConfig:
    """Training episodes are random windows of ``window_bars`` Bars in one Session.

    ``hold_across_weekend`` exists only to be rejected: v1 has no cross-Session
    episode mode.
    """

    window_bars: int = 576
    hold_across_weekend: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_bars, bool)
            or not isinstance(self.window_bars, int)
            or self.window_bars < 1
        ):
            raise ConfigError("window_bars must be a positive integer")
        if self.hold_across_weekend:
            raise ConfigError("hold_across_weekend is not supported in v1")

    def config_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class EnvConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def canonical_json(self) -> bytes:
        return canonical_json(asdict(self))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

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
