"""Engine configuration: frozen, validated, hashable.

``EngineConfig`` (account level plus the per-Instrument ``InstrumentSpec`` table)
is the only configuration the pure engine reads. Construction matches ingestion:
frozen dataclasses, defaults in code, validation in ``__post_init__``, and a TOML
file (``EngineConfig.from_toml``) that only overrides the code defaults.

The TOML shape is top-level account keys plus one ``[instruments.<SYMBOL>]`` table
per Instrument; omitted keys keep their code defaults and unknown keys are errors.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from shadow_trader.ingestion.config import ConfigError
from shadow_trader.preparation._io import canonical_json

__all__ = ["ConfigError", "EngineConfig", "InstrumentSpec"]


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class InstrumentSpec:
    """Per-Instrument trading terms.

    ``rollover_long`` / ``rollover_short`` are fractions of notional charged per
    rollover day (positive is a charge, negative a credit), read by ``step``.
    v1 supports USD-quote pairs only, so P&L is already in the account currency.
    """

    symbol: str = "EURUSD"
    pip_size: float = 0.0001
    margin_rate: float = 0.02
    rollover_long: float = 0.0
    rollover_short: float = 0.0
    slippage_pips: float = 0.1
    regulatory_cap: float = 50.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ConfigError("instrument symbol must not be empty")
        if not self.symbol.endswith("USD") or self.symbol.startswith("USD"):
            raise ConfigError(
                f"{self.symbol}: only USD-quote pairs are supported (USD-base and "
                "cross pairs need conversion support)"
            )
        for name in (
            "pip_size",
            "margin_rate",
            "rollover_long",
            "rollover_short",
            "slippage_pips",
            "regulatory_cap",
        ):
            _finite(getattr(self, name), f"{self.symbol}.{name}")
        if self.pip_size <= 0:
            raise ConfigError(f"{self.symbol}.pip_size must be positive")
        if not 0 < self.margin_rate <= 1:
            raise ConfigError(f"{self.symbol}.margin_rate must be in (0, 1]")
        if self.slippage_pips < 0:
            raise ConfigError(f"{self.symbol}.slippage_pips must not be negative")
        if self.regulatory_cap <= 0:
            raise ConfigError(f"{self.symbol}.regulatory_cap must be positive")
        if self.margin_rate < 1.0 / self.regulatory_cap:
            raise ConfigError(
                f"{self.symbol}.margin_rate {self.margin_rate} is below the "
                f"regulatory minimum 1/{self.regulatory_cap}"
            )

    @property
    def slippage(self) -> float:
        """Constant adverse slippage as a price amount."""
        return self.slippage_pips * self.pip_size

    def config_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class EngineConfig:
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
        symbols = [spec.symbol for spec in self.instruments]
        if len(set(symbols)) != len(symbols):
            raise ConfigError(f"duplicate instruments: {symbols}")
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
                    f"regulatory cap {spec.regulatory_cap} for {spec.symbol}"
                )
        if not self.levels:
            raise ConfigError("levels must not be empty")
        for level in self.levels:
            if not -1.0 <= _finite(level, "levels") <= 1.0:
                raise ConfigError(f"level {level} is outside [-1, 1]")
        if len(set(self.levels)) != len(self.levels):
            raise ConfigError("levels must be distinct")

    @property
    def effective_leverage_cap(self) -> float:
        return self.leverage_cap

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(spec.symbol for spec in self.instruments)

    def canonical_json(self) -> bytes:
        """Key-sorted compact JSON of the full config (hash input, manifest)."""
        return canonical_json(asdict(self))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

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
        raise ConfigError("[instruments] must be a table of [instruments.SYMBOL]")
    allowed = {f.name for f in fields(InstrumentSpec)} - {"symbol"}
    specs = []
    for symbol, table in value.items():
        if not isinstance(table, dict):
            raise ConfigError(f"[instruments.{symbol}] must be a table")
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise ConfigError(
                f"unknown key(s) in [instruments.{symbol}]: {', '.join(unknown)}"
            )
        specs.append(InstrumentSpec(symbol=symbol, **table))
    return tuple(specs)
