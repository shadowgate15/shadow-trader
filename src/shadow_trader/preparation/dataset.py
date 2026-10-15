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

from .anomaly import DEFAULT_POLICY, AnomalyPolicy, policy_summary
from .spec import FeatureSpec

SCHEMA_VERSION = 1
KIND = "prepared"


class PreparedDatasetError(Exception):
    """A prepared Dataset is missing, incomplete or has an unknown schema_version."""


def prepared_schema(spec: FeatureSpec) -> pa.Schema:
    """Ingestion Bar columns followed by the spec's unscaled ``feat_*`` columns."""
    return pa.schema(
        [
            *BAR_SCHEMA,
            *(pa.field(name, pa.float64()) for name in spec.names),
            pa.field("usable", pa.bool_()),
            pa.field("unusable_reason", pa.string()),
        ]
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
    parent: str,
    spec_hash: str,
    policy_hash: str,
    bar_size: int,
    partitions: Mapping[str, str],
    identity: Mapping[str, Any] | None = None,
) -> str:
    # ``identity`` (split scheme and bounds, Embargo E, role) joins the hash only
    # when given, so a Dataset written without a split keeps its plain hash.
    extra = {} if identity is None else {"identity": identity}
    return hashlib.sha256(
        _canonical_json(
            {
                **extra,
                "kind": KIND,
                "schema_version": SCHEMA_VERSION,
                "parent_ingestion": parent,
                "spec_hash": spec_hash,
                "anomaly_policy_hash": policy_hash,
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


def _split_series(manifest: Mapping[str, Any]) -> tuple[object, object] | None:
    identity = manifest.get("identity")
    if identity is None:
        return None
    return identity.get("scheme"), identity.get("bounds", {}).get("train_start")


def _previous_same_role(
    prepared_root: Path,
    dataset_id: str,
    manifest: Mapping[str, Any],
    role: object,
) -> Mapping[str, Any] | None:
    """The latest other complete Dataset of the same role and series, if any.

    Same role, bar size, Instruments, spec hash and policy hash, and the same
    split series (scheme and nominal ``train_start``; both absent for a Dataset
    written without a split), so an unrelated split is never a lineage parent.
    Only a source of lineage and of linkable partitions: a link still needs a
    matching row hash.
    """
    series = _split_series(manifest)
    found: list[tuple[str, Mapping[str, Any]]] = []
    for path in (prepared_root / "datasets").glob("*/manifest.json"):
        other = json.loads(path.read_text())
        if (
            other.get("kind") == KIND
            and other["dataset_id"] != dataset_id
            and other.get("role") == role
            and other["bar_size"] == manifest["bar_size"]
            and other["instruments"] == manifest["instruments"]
            and other["feature_spec"]["hash"] == manifest["feature_spec"]["hash"]
            and other["anomaly_policy"]["hash"] == manifest["anomaly_policy"]["hash"]
            and _split_series(other) == series
        ):
            found.append((other["created_at"], other))
    return max(found, key=lambda item: item[0])[1] if found else None


@dataclass(frozen=True)
class PreparedPlan:
    """A prepared Dataset whose content hash is known but which is not written yet.

    Planning first lets a build learn all three Dataset ids (and so its own id)
    before any manifest, which names the siblings and the build, is written.
    """

    dataset_id: str
    _frames: Mapping[str, pl.DataFrame]
    _parts: Mapping[str, tuple[str, str, int, pa.Table]]
    _manifest: Mapping[str, Any]

    def write(
        self,
        prepared_root: Path,
        manifest_extra: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> PreparedRef:
        """Write the immutable Dataset; ``manifest_extra`` is recorded, never hashed."""
        target = Path(prepared_root) / "datasets" / self.dataset_id
        if (target / "manifest.json").exists():  # identical inputs: already built
            manifest = json.loads((target / "manifest.json").read_text())
            return PreparedRef(self.dataset_id, target, manifest)
        if target.exists():  # incomplete earlier attempt: never read, safe to discard
            shutil.rmtree(target)

        extra = dict(manifest_extra or {})
        prepared_parent = _previous_same_role(
            Path(prepared_root), self.dataset_id, self._manifest, extra.get("role")
        )
        parent_hashes: dict[str, str] = {}
        if prepared_parent is not None:
            parent_hashes = {
                p["path"]: p["hash"] for p in prepared_parent["partitions"]
            }
        open_year: dict[str, int] = {}
        for key, (pair, *_rest) in self._parts.items():
            year = int(key.rsplit("=", 1)[1])
            open_year[pair] = max(open_year.get(pair, year), year)

        entries: list[dict[str, Any]] = []
        for key, (pair, digest, rows, table) in sorted(self._parts.items()):
            rel = f"bars/{key}/part-0.parquet"
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            closed = int(key.rsplit("=", 1)[1]) < open_year[pair]
            # Link only a closed partition whose row hash matches the previous
            # same-role Dataset's; anything else (open year, mismatch) is rewritten.
            if (
                closed
                and prepared_parent is not None
                and parent_hashes.get(rel) == digest
            ):
                source = (
                    Path(prepared_root) / "datasets" / prepared_parent["dataset_id"]
                )
                try:
                    # Rehash the bytes actually on disk: a manifest hash alone
                    # would let a corrupted earlier partition propagate.
                    if _partition_hash(pq.read_table(source / rel)) != digest:
                        raise OSError("stored partition does not match its hash")
                    os.link(source / rel, dest)
                except (OSError, pa.ArrowException):  # unlinkable or damaged
                    pq.write_table(table, dest)
            else:
                pq.write_table(table, dest)
            entries.append(
                {"instrument": pair, "path": rel, "rows": rows, "hash": digest}
            )

        manifest = {
            **self._manifest,
            **extra,
            # Lineage (recorded, never hashed): the previous same-role Dataset.
            "prepared_parent": (
                prepared_parent["dataset_id"] if prepared_parent else None
            ),
            "partitions": entries,
            "row_count": sum(e["rows"] for e in entries),
            # Provenance: recorded, never hashed.
            "polars_version": pl.__version__,
            "builder_version": _builder_version(),
            "builder_git_revision": _git_revision(),
            "created_at": (created_at or datetime.now(UTC)).isoformat(),
        }
        _write_manifest(target, manifest)
        return PreparedRef(self.dataset_id, target, manifest)


def plan_prepared(
    frames: Mapping[str, pl.DataFrame],
    *,
    parent_ingestion: str,
    spec: FeatureSpec,
    policy: AnomalyPolicy = DEFAULT_POLICY,
    identity: Mapping[str, Any] | None = None,
) -> PreparedPlan:
    """Hash ``frames`` (per Instrument) into a prepared Dataset id, writing nothing.

    Each frame holds the Bar columns, the spec's ``feat_*`` columns and the
    ``usable`` / ``unusable_reason`` columns. ``identity`` (split scheme and
    bounds, Embargo E, role) is hashed and recorded in the manifest.
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
    content_hash = _dataset_hash(
        parent_ingestion, spec.hash, policy.hash, spec.bar_size, hashes, identity
    )
    manifest: dict[str, Any] = {
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
        "anomaly_policy": policy_summary(policy, frames.values()),
    }
    if identity is not None:
        manifest["identity"] = dict(identity)
    return PreparedPlan(content_hash, frames, parts, manifest)


def write_prepared(
    prepared_root: Path,
    frames: Mapping[str, pl.DataFrame],
    *,
    parent_ingestion: str,
    spec: FeatureSpec,
    policy: AnomalyPolicy = DEFAULT_POLICY,
    created_at: datetime | None = None,
    identity: Mapping[str, Any] | None = None,
) -> PreparedRef:
    """Write an immutable prepared Dataset of ``frames`` (per Instrument)."""
    plan = plan_prepared(
        frames,
        parent_ingestion=parent_ingestion,
        spec=spec,
        policy=policy,
        identity=identity,
    )
    return plan.write(prepared_root, created_at=created_at)


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


@dataclass(frozen=True)
class BuildRecord:
    """The hash-addressed build record: three Dataset ids and the shared refs."""

    build_id: str
    path: Path
    record: Mapping[str, Any]


def _build_hashed(
    shared: Mapping[str, Any], dataset_ids: Mapping[str, str]
) -> dict[str, Any]:
    return {"kind": "build", "datasets": dict(dataset_ids), **shared}


def build_id_of(shared: Mapping[str, Any], dataset_ids: Mapping[str, str]) -> str:
    """Hash of the three Dataset ids and the shared refs."""
    return hashlib.sha256(
        _canonical_json(_build_hashed(shared, dataset_ids))
    ).hexdigest()


def write_build_record(
    prepared_root: Path,
    shared: Mapping[str, Any],
    dataset_ids: Mapping[str, str],
    created_at: datetime | None = None,
) -> BuildRecord:
    """Write ``<prepared_root>/builds/<build_id>.json`` by atomic rename.

    ``build_id`` hashes the Dataset ids and ``shared`` refs; the polars version
    and ``created_at`` are recorded but never hashed.
    """
    hashed = _build_hashed(shared, dataset_ids)
    build_id = build_id_of(shared, dataset_ids)
    path = Path(prepared_root) / "builds" / f"{build_id}.json"
    if path.is_file():  # identical inputs: already built
        return BuildRecord(build_id, path, json.loads(path.read_text()))
    record = {
        **hashed,
        "build_id": build_id,
        "polars_version": pl.__version__,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return BuildRecord(build_id, path, record)
