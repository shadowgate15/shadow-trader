"""Preparation configuration: a TOML file validated into frozen dataclasses.

Tables: ``[paths]`` (ingestion ``data_root`` and a separate ``prepared_root``),
``[input]`` (the explicit parent ingestion ``dataset_id``, never "latest"),
``[spec]`` (the feature spec version), the optional ``[policy]`` (the anomaly
policy's ``distrusted_flags``), ``[split]`` (the chronological scheme and
its explicit UTC bounds), ``[embargo]`` (the required ``episode_bars``) and the
optional ``[report]`` (advisory warning thresholds; reporting only, never hashed).
Unknown keys are errors. CLI flags may
override only the config path.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shadow_trader.ingestion.config import ConfigError

from .anomaly import DEFAULT_POLICY, AnomalyPolicy
from .spec import SPECS, FeatureSpec

__all__ = ["ConfigError", "PreparedConfig", "load_prepared_config"]


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path  # the ingestion root, read through open_dataset
    prepared_root: Path  # where prepared Datasets live, never under data_root


@dataclass(frozen=True)
class InputConfig:
    dataset_id: str


@dataclass(frozen=True)
class SpecConfig:
    version: int


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split: half-open ranges from each start to the next one.

    Bounds are nominal; the build snaps each forward to the next Session open.
    ``test_end`` is optional (the test range then runs to the end of the parent).
    """

    scheme: str
    train_start: datetime
    validation_start: datetime
    test_start: datetime
    test_end: datetime | None = None


@dataclass(frozen=True)
class EmbargoConfig:
    episode_bars: int  # required, >= 1, no default


@dataclass(frozen=True)
class ReportConfig:
    """Advisory warning thresholds: reporting only, never hashed, never fatal."""

    min_usable_fraction: float = 0.5  # low_usable_fraction below this, per role
    max_distrusted_flag_fraction: float = 0.01  # high_distrusted_flag_fraction
    max_scaler_clip_fraction: float = 0.05  # scaler_clip_fraction_high
    min_rows: int = 2000  # short_dataset: fewer rows than this per Instrument


@dataclass(frozen=True)
class PreparedConfig:
    paths: PathsConfig
    input: InputConfig
    spec: SpecConfig
    split: SplitConfig
    embargo: EmbargoConfig
    anomaly: AnomalyPolicy = DEFAULT_POLICY
    report: ReportConfig = ReportConfig()

    @property
    def feature_spec(self) -> FeatureSpec:
        return SPECS[self.spec.version]


def _table(raw: Mapping[str, Any], key: str, required: set[str]) -> Mapping[str, Any]:
    table = raw.get(key)
    if not isinstance(table, dict):
        raise ConfigError(f"[{key}] is required and must be a table")
    missing = sorted(required - set(table))
    if missing:
        raise ConfigError(f"[{key}] is missing: {', '.join(missing)}")
    return table


def _check_keys(table: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _checked[T](value: Any, kind: type[T], name: str) -> T:
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(f"{name} must be of type {kind.__name__}")
    return value


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ConfigError(f"{name} must be a TOML datetime with a UTC offset (Z)")
    return value.astimezone(UTC)


def load_prepared_config(path: Path) -> PreparedConfig:
    """Load and validate a config file; raise ConfigError on any problem."""
    try:
        raw = tomllib.loads(Path(path).read_text())
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None

    _check_keys(
        raw,
        {"paths", "input", "spec", "policy", "split", "embargo", "report"},
        "config",
    )
    paths = _table(raw, "paths", {"data_root", "prepared_root"})
    _check_keys(paths, {"data_root", "prepared_root"}, "[paths]")
    data_root = Path(_checked(paths["data_root"], str, "paths.data_root"))
    prepared_root = Path(_checked(paths["prepared_root"], str, "paths.prepared_root"))
    resolved_data = data_root.resolve()
    resolved_prepared = prepared_root.resolve()
    if resolved_prepared == resolved_data or resolved_data in resolved_prepared.parents:
        raise ConfigError(
            "paths.prepared_root must be separate from paths.data_root "
            "so ingestion scans never see prepared Datasets"
        )

    input_table = _table(raw, "input", {"dataset_id"})
    _check_keys(input_table, {"dataset_id"}, "[input]")
    dataset_id = _checked(input_table["dataset_id"], str, "input.dataset_id")
    if not dataset_id or dataset_id == "latest":
        raise ConfigError("input.dataset_id must name a Dataset explicitly")

    spec_table = _table(raw, "spec", {"version"})
    _check_keys(spec_table, {"version"}, "[spec]")
    version = _checked(spec_table["version"], int, "spec.version")
    if version not in SPECS:
        known = ", ".join(map(str, sorted(SPECS)))
        raise ConfigError(f"unknown spec.version {version}; known: {known}")

    split_keys = {"scheme", "train_start", "validation_start", "test_start", "test_end"}
    split_table = _table(
        raw, "split", {"scheme", "train_start", "validation_start", "test_start"}
    )
    _check_keys(split_table, split_keys, "[split]")
    scheme = _checked(split_table["scheme"], str, "split.scheme")
    if scheme != "chronological":
        raise ConfigError(f"unknown split.scheme {scheme!r}; known: chronological")
    test_end = split_table.get("test_end")
    split = SplitConfig(
        scheme,
        _utc(split_table["train_start"], "split.train_start"),
        _utc(split_table["validation_start"], "split.validation_start"),
        _utc(split_table["test_start"], "split.test_start"),
        None if test_end is None else _utc(test_end, "split.test_end"),
    )

    embargo_table = _table(raw, "embargo", {"episode_bars"})
    _check_keys(embargo_table, {"episode_bars"}, "[embargo]")
    episode_bars = _checked(embargo_table["episode_bars"], int, "embargo.episode_bars")
    if episode_bars < 1:
        raise ConfigError("embargo.episode_bars must be >= 1")

    policy = DEFAULT_POLICY
    if "policy" in raw:
        table = _table(raw, "policy", set())
        _check_keys(table, {"distrusted_flags"}, "[policy]")
        if "distrusted_flags" in table:
            flags = table["distrusted_flags"]
            if not isinstance(flags, list) or not all(
                isinstance(f, str) for f in flags
            ):
                raise ConfigError("policy.distrusted_flags must be a list of strings")
            policy = AnomalyPolicy(tuple(flags))

    report = ReportConfig()
    if "report" in raw:
        table = _table(raw, "report", set())
        _check_keys(table, {f.name for f in fields(ReportConfig)}, "[report]")
        values: dict[str, Any] = {}
        for key, value in table.items():
            if key == "min_rows":
                values[key] = _checked(value, int, f"report.{key}")
            elif isinstance(value, int | float) and not isinstance(value, bool):
                values[key] = float(value)
            else:
                raise ConfigError(f"report.{key} must be a number")
        report = ReportConfig(**values)

    return PreparedConfig(
        PathsConfig(data_root, prepared_root),
        InputConfig(dataset_id),
        SpecConfig(version),
        split,
        EmbargoConfig(episode_bars),
        policy,
        report,
    )
