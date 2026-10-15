"""Prepared Dataset storage: the writer, hasher and reader.

A prepared Dataset is an immutable directory
``<prepared_root>/datasets/<dataset_id>/`` with Parquet partitions per
Instrument and year and a ``manifest.json`` written last by atomic rename (no
manifest means incomplete, never read). ``dataset_id`` is a Merkle hash over
canonical prepared rows plus the parent ingestion id, spec hash, bar size and
schema version. The polars version, ``created_at`` and builder revision are
recorded in the manifest but never hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from shadow_trader.ingestion.dataset import BAR_SCHEMA

from .spec import FeatureSpec

SCHEMA_VERSION = 1
KIND = "prepared"


class PreparedDatasetError(Exception):
    """A prepared Dataset is missing, incomplete or has an unknown schema_version."""


def prepared_schema(spec: FeatureSpec) -> pa.Schema:
    """Ingestion Bar columns followed by the spec's unscaled ``feat_*`` columns."""
    return pa.schema(
        [*BAR_SCHEMA, *(pa.field(name, pa.float64()) for name in spec.names)]
    )


@dataclass(frozen=True)
class PreparedRef:
    dataset_id: str
    path: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedDataset:
    """An opened prepared Dataset: its manifest and its rows."""

    dataset_id: str
    path: Path
    manifest: Mapping[str, Any]

    def read(self, instrument: str) -> pl.DataFrame:
        """One Instrument's rows ordered by ``open_time``; nulls stay nulls."""
        parts = [
            _read_partition(self.path / part["path"])
            for part in self.manifest["partitions"]
            if part["instrument"] == instrument
        ]
        if not parts:
            raise PreparedDatasetError(
                f"dataset {self.dataset_id} has no Instrument {instrument}"
            )
        return pl.concat(parts).sort("open_time")


def _read_partition(path: Path) -> pl.DataFrame:
    table = pq.read_table(path)
    _check_version((table.schema.metadata or {}).get(b"schema_version", b""), path)
    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)
    return frame


def _check_version(value: object, where: object) -> None:
    text = value.decode() if isinstance(value, bytes) else str(value)
    if text != str(SCHEMA_VERSION):
        raise PreparedDatasetError(
            f"unknown schema_version {text!r} in {where}; this reader "
            f"supports {SCHEMA_VERSION}."
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_isoformat
    ).encode()


def _isoformat(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    raise TypeError(f"cannot canonicalise {value!r}")


def _partition_hash(table: pa.Table) -> str:
    digest = hashlib.sha256()
    for row in table.to_pylist():  # canonical order: sorted by open_time
        digest.update(_canonical_json(list(row.values())) + b"\n")
    return digest.hexdigest()


def _dataset_hash(
    parent: str, spec_hash: str, bar_size: int, partitions: Mapping[str, str]
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "kind": KIND,
                "schema_version": SCHEMA_VERSION,
                "parent_ingestion": parent,
                "spec_hash": spec_hash,
                "bar_size": bar_size,
                # Merkle leaves combined in sorted partition-key order.
                "partitions": sorted(partitions.items()),
            }
        )
    ).hexdigest()


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _builder_version() -> str | None:
    try:
        return metadata.version("shadow-trader")
    except metadata.PackageNotFoundError:
        return None


def write_prepared(
    prepared_root: Path,
    frames: Mapping[str, pl.DataFrame],
    *,
    parent_ingestion: str,
    spec: FeatureSpec,
    created_at: datetime | None = None,
) -> PreparedRef:
    """Write an immutable prepared Dataset of ``frames`` (per Instrument).

    Each frame holds the Bar columns plus the spec's ``feat_*`` columns.
    """
    schema = prepared_schema(spec)
    parts: dict[str, tuple[str, str, int, pa.Table]] = {}
    for pair, frame in sorted(frames.items()):
        ordered = frame.sort("open_time")
        years = ordered["open_time"].dt.year()
        for year in sorted(years.unique().to_list()):
            table = ordered.filter(years == year).to_arrow().cast(schema)
            table = table.replace_schema_metadata(
                {"schema_version": str(SCHEMA_VERSION), "bar_size": str(spec.bar_size)}
            )
            key = f"instrument={pair}/year={year}"
            parts[key] = (pair, _partition_hash(table), table.num_rows, table)

    hashes = {key: part[1] for key, part in parts.items()}
    content_hash = _dataset_hash(parent_ingestion, spec.hash, spec.bar_size, hashes)

    target = Path(prepared_root) / "datasets" / content_hash
    if (target / "manifest.json").exists():  # identical inputs: already built
        manifest = json.loads((target / "manifest.json").read_text())
        return PreparedRef(content_hash, target, manifest)
    if target.exists():  # incomplete earlier attempt: never read, safe to discard
        shutil.rmtree(target)

    entries: list[dict[str, Any]] = []
    for key, (pair, digest, rows, table) in sorted(parts.items()):
        rel = f"bars/{key}/part-0.parquet"
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        entries.append({"instrument": pair, "path": rel, "rows": rows, "hash": digest})

    manifest = {
        "kind": KIND,
        "dataset_id": content_hash,
        "content_hash": content_hash,
        "schema_version": SCHEMA_VERSION,
        "bar_size": spec.bar_size,
        "instruments": sorted(frames),
        "parent_ingestion": parent_ingestion,
        "feature_spec": {
            "version": spec.version,
            "hash": spec.hash,
            "columns": list(spec.names),
        },
        "partitions": entries,
        "row_count": sum(e["rows"] for e in entries),
        # Provenance: recorded, never hashed.
        "polars_version": pl.__version__,
        "builder_version": _builder_version(),
        "builder_git_revision": _git_revision(),
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
    }
    _write_manifest(target, manifest)
    return PreparedRef(content_hash, target, manifest)


def _write_manifest(target: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest last, by atomic rename."""
    tmp = target / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, target / "manifest.json")


def open_prepared(data_root: Path, dataset_id: str) -> PreparedDataset:
    """Open a prepared Dataset; refuse a missing manifest or unknown schema_version.

    ``data_root`` is the prepared root (``paths.prepared_root``).
    """
    path = Path(data_root) / "datasets" / dataset_id
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise PreparedDatasetError(f"dataset {dataset_id} has no manifest (incomplete)")
    manifest = json.loads(manifest_path.read_text())
    _check_version(manifest.get("schema_version"), manifest_path)
    if manifest.get("kind") != KIND:
        raise PreparedDatasetError(f"dataset {dataset_id} is not a prepared Dataset")
    return PreparedDataset(dataset_id, path, manifest)
