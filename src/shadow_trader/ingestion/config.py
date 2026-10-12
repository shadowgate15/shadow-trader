"""Ingestion configuration: a TOML file validated into frozen dataclasses.

Every config value is either content-affecting (it changes Dataset rows or is
recorded as an input to them, so it feeds the Dataset hash) or reporting-only
(it only changes warnings and reports). Reporting-only fields are declared with
``reporting_only()`` so the hashing code can tell them apart.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

SUPPORTED_INSTRUMENTS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
)
SUPPORTED_BAR_SIZES = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
SUPPORTED_LIVE_ADAPTERS = ("replay",)
DEFAULT_START = "2009-01"
_MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


def month_parts(month: str) -> tuple[int, int]:
    """(year, month number) of a ``YYYY-MM`` string."""
    return int(month[:4]), int(month[5:])


def to_period(month: str) -> str:
    """``YYYY-MM`` to the raw-cache ``YYYYMM`` period."""
    return month.replace("-", "")


def to_month(period: str) -> str:
    """Raw-cache ``YYYYMM`` period to ``YYYY-MM``."""
    return f"{period[:4]}-{period[4:]}"


def pip_size(instrument: str) -> Decimal:
    """Price size of one pip: 0.01 for JPY pairs, 0.0001 otherwise."""
    return Decimal("0.01") if instrument.endswith("JPY") else Decimal("0.0001")


def last_complete_month(now: datetime) -> str:
    """The last fully elapsed UTC month before ``now``, as ``YYYY-MM``."""
    year, month = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    return f"{year:04d}-{month:02d}"


class ConfigError(Exception):
    """The config is missing, malformed or invalid (CLI exit code 2)."""


def reporting_only[T](default: T) -> T:
    """Declare a field whose value never affects Dataset content."""
    return field(default=default, metadata={"reporting_only": True})  # type: ignore[return-value]


def _content_view(obj: Any) -> dict[str, Any]:
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if not f.metadata.get("reporting_only")
    }


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path


@dataclass(frozen=True)
class AcknowledgedMissing:
    instrument: str
    month: str
    reason: str | None = None


@dataclass(frozen=True)
class BuildConfig:
    instruments: tuple[str, ...]
    bar_sizes: tuple[int, ...] = (5,)
    start: str = DEFAULT_START
    end: str | None = None
    acknowledged_missing: tuple[AcknowledgedMissing, ...] = ()


@dataclass(frozen=True)
class InstrumentConfig:
    price_scale: int


@dataclass(frozen=True)
class GapRuleConfig:
    outage_threshold_minutes: int = 30


@dataclass(frozen=True)
class AnomalyRuleConfig:
    crossed_quote: bool = True
    zero_spread: bool = True
    price_spike: bool = True
    price_spike_window_ticks: int = 50
    price_spike_k: float = 10.0
    price_spike_floor_pips: float = 10.0
    stale_quote: bool = True
    stale_quote_min_ticks: int = 20
    stale_quote_min_minutes: int = 5
    spread_spike: bool = True
    spread_spike_k: float = 10.0
    spread_spike_floor_pips: float = 2.0
    spread_spike_lookback_hours: int = 24
    alert_fraction: float = reporting_only(0.01)


@dataclass(frozen=True)
class ReconciliationConfig:
    """Diff tolerances; reporting-only, recorded in the diff report."""

    price_pips: float = 0.5
    spread_pips: float = 0.3
    spread_relative: float = 0.3
    tick_ratio_min: float = 0.5
    tick_ratio_max: float = 2.0


@dataclass(frozen=True)
class LiveConfig:
    adapter: str = "replay"
    grace_seconds: int = 1
    repair_window_minutes: int = 60


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    build: BuildConfig
    instruments: Mapping[str, InstrumentConfig]
    gap_rule: GapRuleConfig
    anomaly_rule: AnomalyRuleConfig
    reconciliation: ReconciliationConfig
    live: LiveConfig

    def content_affecting(self) -> dict[str, Any]:
        """The values that feed the Dataset hash (reporting-only ones omitted).

        Reconciliation and live settings are reporting-only as a whole.
        """
        return {
            "instruments": {
                pair: dataclasses.asdict(self.instruments[pair])
                for pair in sorted(self.instruments)
            },
            "acknowledged_missing": [
                dataclasses.asdict(a) for a in self.build.acknowledged_missing
            ],
            "rules": {
                "gap": _content_view(self.gap_rule),
                "anomaly": _content_view(self.anomaly_rule),
            },
        }

    def with_overrides(
        self,
        *,
        instruments: tuple[str, ...] | None = None,
        bar_sizes: tuple[int, ...] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> Config:
        """Apply CLI overrides. Only instruments, time range and bar sizes."""
        build = dataclasses.replace(
            self.build,
            instruments=instruments or self.build.instruments,
            bar_sizes=bar_sizes or self.build.bar_sizes,
            start=start or self.build.start,
            end=end or self.build.end,
        )
        _validate_build(build, self.instruments)
        return dataclasses.replace(self, build=build)


def _check_keys(table: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _table(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _checked(value: Any, kind: type, name: str) -> Any:
    """Type-check one scalar; ints are accepted where floats are expected."""
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(f"{name} must be of type {kind.__name__}")
    return value


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def _validate_month(value: str, name: str) -> None:
    if not _MONTH.fullmatch(value):
        raise ConfigError(f"{name} must be a month like 2009-01, got {value!r}")


def validate_bar_sizes(bar_sizes: tuple[int, ...]) -> None:
    if not bar_sizes:
        raise ConfigError("build.bar_sizes must not be empty")
    for size in bar_sizes:
        if size not in SUPPORTED_BAR_SIZES:
            raise ConfigError(
                f"unsupported bar size {size}; must be a whole-minute divisor "
                f"of 60: {', '.join(map(str, SUPPORTED_BAR_SIZES))}"
            )


def _validate_build(build: BuildConfig, instruments: Mapping[str, Any]) -> None:
    if not build.instruments:
        raise ConfigError("build.instruments must list at least one instrument")
    for pair in (
        *build.instruments,
        *(a.instrument for a in build.acknowledged_missing),
    ):
        if pair not in instruments:
            raise ConfigError(
                f"unsupported instrument {pair}; supported: "
                f"{', '.join(SUPPORTED_INSTRUMENTS)}"
            )
    validate_bar_sizes(build.bar_sizes)
    _validate_month(build.start, "build.start")
    if build.end is not None:
        _validate_month(build.end, "build.end")
        if build.end < build.start:
            raise ConfigError(
                f"build.start {build.start} is after build.end {build.end}"
            )


def _default_scale(pair: str) -> int:
    return 3 if pair.endswith("JPY") else 5


def _parse_build(raw: Mapping[str, Any]) -> BuildConfig:
    _check_keys(
        raw,
        {"instruments", "bar_sizes", "start", "end", "acknowledged_missing"},
        "[build]",
    )
    instruments = tuple(_checked(raw.get("instruments", []), list, "build.instruments"))
    bar_sizes = tuple(_checked(raw.get("bar_sizes", [5]), list, "build.bar_sizes"))
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in bar_sizes):
        raise ConfigError("build.bar_sizes must be a list of integers")
    acknowledged = []
    for entry in _checked(
        raw.get("acknowledged_missing", []), list, "build.acknowledged_missing"
    ):
        if not isinstance(entry, dict):
            raise ConfigError("build.acknowledged_missing entries must be tables")
        _check_keys(entry, {"instrument", "month", "reason"}, "acknowledged_missing")
        for key in ("instrument", "month"):
            if key not in entry:
                raise ConfigError(f"acknowledged_missing entry lacks {key}")
        _validate_month(entry["month"], "acknowledged_missing.month")
        acknowledged.append(
            AcknowledgedMissing(
                entry["instrument"], entry["month"], entry.get("reason")
            )
        )
    end = raw.get("end")
    return BuildConfig(
        instruments=instruments,
        bar_sizes=bar_sizes,
        start=_checked(raw.get("start", DEFAULT_START), str, "build.start"),
        end=None if end is None else _checked(end, str, "build.end"),
        acknowledged_missing=tuple(acknowledged),
    )


def _parse_flat[T](cls: type[T], raw: Mapping[str, Any], where: str) -> T:
    """Build a flat dataclass of scalar fields, type-checked against defaults."""
    defaults = {
        f.name: f.default
        for f in dataclasses.fields(cls)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    }
    _check_keys(raw, set(defaults), where)
    values = {
        name: _checked(value, type(defaults[name]), f"{where}.{name}")
        for name, value in raw.items()
    }
    return cls(**values)


def load_config(path: Path) -> Config:
    """Load and validate a config file; raise ConfigError on any problem."""
    try:
        raw = tomllib.loads(path.read_text())
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None

    _check_keys(
        raw,
        {"paths", "build", "instruments", "rules", "reconciliation", "live"},
        "config",
    )
    paths = _table(raw, "paths")
    _check_keys(paths, {"data_root"}, "[paths]")
    if "data_root" not in paths:
        raise ConfigError("[paths] data_root is required")
    data_root = Path(_checked(paths["data_root"], str, "paths.data_root"))

    build = _parse_build(_table(raw, "build"))

    instruments = {
        pair: InstrumentConfig(_default_scale(pair)) for pair in SUPPORTED_INSTRUMENTS
    }
    for pair, table in _table(raw, "instruments").items():
        if pair not in SUPPORTED_INSTRUMENTS:
            raise ConfigError(f"unsupported instrument {pair} in [instruments]")
        _check_keys(table, {"price_scale"}, f"[instruments.{pair}]")
        scale = _checked(
            table.get("price_scale", _default_scale(pair)),
            int,
            f"instruments.{pair}.price_scale",
        )
        _positive(scale, f"instruments.{pair}.price_scale")
        instruments[pair] = InstrumentConfig(scale)

    rules = _table(raw, "rules")
    _check_keys(rules, {"gap", "anomaly"}, "[rules]")
    gap = _parse_flat(GapRuleConfig, _table(rules, "gap"), "rules.gap")
    _positive(gap.outage_threshold_minutes, "rules.gap.outage_threshold_minutes")
    anomaly = _parse_flat(AnomalyRuleConfig, _table(rules, "anomaly"), "rules.anomaly")
    if not 0 < anomaly.alert_fraction <= 1:
        raise ConfigError("rules.anomaly.alert_fraction must be in (0, 1]")
    reconciliation = _parse_flat(
        ReconciliationConfig, _table(raw, "reconciliation"), "reconciliation"
    )
    live = _parse_flat(LiveConfig, _table(raw, "live"), "live")
    if live.adapter not in SUPPORTED_LIVE_ADAPTERS:
        raise ConfigError(
            f"unsupported live adapter {live.adapter!r}; "
            f"supported: {', '.join(SUPPORTED_LIVE_ADAPTERS)}"
        )
    _positive(live.grace_seconds, "live.grace_seconds")
    _positive(live.repair_window_minutes, "live.repair_window_minutes")

    _validate_build(build, instruments)
    return Config(
        paths=PathsConfig(data_root),
        build=build,
        instruments=instruments,
        gap_rule=gap,
        anomaly_rule=anomaly,
        reconciliation=reconciliation,
        live=live,
    )
