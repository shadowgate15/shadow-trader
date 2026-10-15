"""Preparation configuration: a TOML file validated into frozen dataclasses.

Tables: ``[paths]`` (ingestion ``data_root`` and a separate ``prepared_root``),
``[input]`` (the explicit parent ingestion ``dataset_id``, never "latest") and
``[spec]`` (the feature spec version). Unknown keys are errors. CLI flags may
override only the config path.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadow_trader.ingestion.config import ConfigError

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
class PreparedConfig:
    paths: PathsConfig
    input: InputConfig
    spec: SpecConfig

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


def load_prepared_config(path: Path) -> PreparedConfig:
    """Load and validate a config file; raise ConfigError on any problem."""
    try:
        raw = tomllib.loads(Path(path).read_text())
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None

    _check_keys(raw, {"paths", "input", "spec"}, "config")
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

    return PreparedConfig(
        PathsConfig(data_root, prepared_root),
        InputConfig(dataset_id),
        SpecConfig(version),
    )
