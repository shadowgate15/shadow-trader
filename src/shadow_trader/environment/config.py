"""Engine configuration: frozen, validated, hashable.

``EngineConfig`` (account level plus the per-Instrument ``InstrumentSpec`` table)
is the only configuration the pure engine reads. Construction matches ingestion:
frozen dataclasses, defaults in code, validation in ``__post_init__``, and a TOML
file (``EngineConfig.from_toml``) that only overrides the code defaults.

The TOML shape is top-level account keys plus one ``[instruments.<NAME>]`` table
per Instrument; omitted keys keep their code defaults and unknown keys are errors.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from shadow_trader.ingestion.config import ConfigError

from ._common import HashableConfig, check_number

__all__ = ["ConfigError", "EngineConfig", "InstrumentSpec"]


def _finite(value: float, name: str) -> float:
    return check_number(value, name, ConfigError)


@dataclass(frozen=True)
class InstrumentSpec(HashableConfig):
    """Per-Instrument trading terms.

    ``rollover_long`` / ``rollover_short`` are fractions of notional charged per
    rollover day (positive is a charge, negative a credit), read by ``step``.
    v1 supports USD-quote pairs only, so P&L is already in the account currency.
    """

    name: str = "EURUSD"
    pip_size: float = 0.0001
    margin_rate: float = 0.02
    rollover_long: float = 0.0
    rollover_short: float = 0.0
    slippage_pips: float = 0.1
    regulatory_cap: float = 50.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("instrument name must not be empty")
        if not self.name.endswith("USD") or self.name.startswith("USD"):
            raise ConfigError(
                f"{self.name}: only USD-quote pairs are supported (USD-base and "
                "cross pairs need conversion support)"
            )
        for field_name in (
            "pip_size",
            "margin_rate",
            "rollover_long",
            "rollover_short",
            "slippage_pips",
            "regulatory_cap",
        ):
            _finite(getattr(self, field_name), f"{self.name}.{field_name}")
        if self.pip_size <= 0:
            raise ConfigError(f"{self.name}.pip_size must be positive")
        if not 0 < self.margin_rate <= 1:
            raise ConfigError(f"{self.name}.margin_rate must be in (0, 1]")
        if self.slippage_pips < 0:
            raise ConfigError(f"{self.name}.slippage_pips must not be negative")
        if self.regulatory_cap <= 0:
            raise ConfigError(f"{self.name}.regulatory_cap must be positive")
        if self.margin_rate < 1.0 / self.regulatory_cap:
            raise ConfigError(
                f"{self.name}.margin_rate {self.margin_rate} is below the "
                f"regulatory minimum 1/{self.regulatory_cap}"
            )

    @property
    def slippage(self) -> float:
        """Constant adverse slippage as a price amount."""
        return self.slippage_pips * self.pip_size


@dataclass(frozen=True)
class EngineConfig(HashableConfig):
    """Account-level settings plus the Instrument table (order is significant).

    ``leverage_cap`` is the effective cap used for sizing; it may be lower than
    every Instrument's regulatory cap and is rejected if higher. ``levels`` is the
    action level set (net-position fractions in [-1, 1]); the env maps indexes to
    it and the engine only validates targets.
    """

    instruments: tuple[InstrumentSpec, ...] = (InstrumentSpec(),)
    starting_equity: float = 10_000.0
    closeout_fraction: float = 0.5
    leverage_cap: float = 10.0
    levels: tuple[float, ...] = field(default=(-1.0, 0.0, 1.0))

    def __post_init__(self) -> None:
        if not self.instruments:
            raise ConfigError("at least one instrument is required")
        names = [spec.name for spec in self.instruments]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate instruments: {names}")
        if _finite(self.starting_equity, "starting_equity") <= 0:
            raise ConfigError("starting_equity must be positive")
        if not 0 < _finite(self.closeout_fraction, "closeout_fraction") <= 1:
            raise ConfigError("closeout_fraction must be in (0, 1]")
        if _finite(self.leverage_cap, "leverage_cap") <= 0:
            raise ConfigError("leverage_cap must be positive")
        for spec in self.instruments:
            if self.leverage_cap > spec.regulatory_cap:
                raise ConfigError(
                    f"effective leverage cap {self.leverage_cap} exceeds the "
                    f"regulatory cap {spec.regulatory_cap} for {spec.name}"
                )
        if not self.levels:
            raise ConfigError("levels must not be empty")
        for level in self.levels:
            if not -1.0 <= _finite(level, "levels") <= 1.0:
                raise ConfigError(f"level {level} is outside [-1, 1]")
        if len(set(self.levels)) != len(self.levels):
            raise ConfigError("levels must be distinct")

    @property
    def names(self) -> tuple[str, ...]:
        """The Instruments' names, in table order."""
        return tuple(spec.name for spec in self.instruments)

    @classmethod
    def from_toml(cls, path: Path) -> EngineConfig:
        """Load a TOML file over the code defaults; raise ConfigError on any problem."""
        try:
            raw = tomllib.loads(Path(path).read_text())
        except OSError as exc:
            raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {path}: {exc}") from None
        return cls._from_mapping(raw)

    @classmethod
    def _from_mapping(cls, raw: Mapping[str, Any]) -> EngineConfig:
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(f"unknown key(s) in config: {', '.join(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "instruments":
                kwargs[key] = _instruments(value)
            elif key == "levels":
                if not isinstance(value, list):
                    raise ConfigError("levels must be an array")
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _instruments(value: Any) -> tuple[InstrumentSpec, ...]:
    if not isinstance(value, dict):
        raise ConfigError("[instruments] must be a table of [instruments.NAME]")
    allowed = {f.name for f in fields(InstrumentSpec)} - {"name"}
    specs = []
    for name, table in value.items():
        if not isinstance(table, dict):
            raise ConfigError(f"[instruments.{name}] must be a table")
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise ConfigError(
                f"unknown key(s) in [instruments.{name}]: {', '.join(unknown)}"
            )
        specs.append(InstrumentSpec(name=name, **table))
    return tuple(specs)
