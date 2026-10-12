"""Dataset storage: the Dataset writer, hasher and reader, and the raw cache.

A Dataset is an immutable directory ``<data_root>/datasets/<dataset_id>/`` with
Parquet partitions per Instrument and year and a ``manifest.json`` written last
by atomic rename (no manifest means incomplete, never read). ``dataset_id`` is a
Merkle hash over logical rows plus ``schema_version``, bar size and the
content-affecting rules; provenance is never hashed.

Raw cache: write-once, byte-exact.

Raw cache layout: ``<data_root>/raw/<source>/<instrument>/<period>.<ext>``
with a ``<file>.meta.json`` sidecar. Files are byte-exact and never
overwritten or pruned; a corrected re-download is a new file
(``<period>.r2.<ext>``, ``<period>.r3.<ext>``, ...) whose sidecar names the
file it supersedes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .bars import Bar
from .config import Config

SCHEMA_VERSION = 1
VIEW = "latest"
SOURCE = "histdata"

BAR_SCHEMA = pa.schema(
    [
        ("instrument", pa.string()),
        ("open_time", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("spread_close", pa.float64()),
        ("spread_mean", pa.float64()),
        ("tick_count", pa.int64()),
        ("is_gap", pa.bool_()),
        ("session_id", pa.timestamp("us", tz="UTC")),
        ("is_rollover", pa.bool_()),
        ("quality_flags", pa.list_(pa.string())),
    ]
)
_COLUMNS = tuple(BAR_SCHEMA.names)


class RawCache:
    def __init__(self, data_root: Path) -> None:
        self._root = Path(data_root) / "raw"

    def path_for(
        self, source: str, instrument: str, period: str, ext: str, revision: int = 1
    ) -> Path:
        suffix = "" if revision == 1 else f".r{revision}"
        return self._root / source / instrument / f"{period}{suffix}.{ext}"

    def has(self, source: str, instrument: str, period: str, ext: str) -> bool:
        return self.path_for(source, instrument, period, ext).exists()

    def store(
        self,
        source: str,
        instrument: str,
        period: str,
        ext: str,
        data: bytes,
        *,
        request: Mapping[str, Any],
        fetched_at: datetime,
        http: Mapping[str, Any],
        supersede: bool = False,
    ) -> Path:
        """Write ``data`` and its sidecar; never replace an existing file.

        Raises ``FileExistsError`` if the period is cached and ``supersede``
        is false. With ``supersede`` the bytes go to the next free revision
        and the sidecar records the file it supersedes.
        """
        revision = 1
        path = self.path_for(source, instrument, period, ext)
        superseded: str | None = None
        if path.exists():
            if not supersede:
                raise FileExistsError(f"{path} is already cached")
            while path.exists():
                superseded = path.name
                revision += 1
                path = self.path_for(source, instrument, period, ext, revision)
        path.parent.mkdir(parents=True, exist_ok=True)

        sidecar = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "request": dict(request),
            "fetched_at": fetched_at.isoformat(),
            "http": dict(http),
            "supersedes": superseded,
        }
        # Sidecar first, data last: a data file without a sidecar never exists.
        meta_path = path.with_name(path.name + ".meta.json")
        _write_new(meta_path, json.dumps(sidecar, indent=2, sort_keys=True).encode())
        _write_new(path, data)
        return path


def _write_new(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data`` atomically; fail if it already exists."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.link(tmp, path)  # raises FileExistsError rather than replacing
    finally:
        os.unlink(tmp)


class DatasetError(Exception):
    """A Dataset is missing, incomplete or has an unknown schema_version."""


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    path: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class Dataset:
    """An opened Dataset: its manifest and its Bars."""

    dataset_id: str
    path: Path
    manifest: Mapping[str, Any]

    def read(self, instrument: str) -> list[Bar]:
        bars: list[Bar] = []
        for part in self.manifest["partitions"]:
            if part["instrument"] != instrument:
                continue
            table = pq.read_table(self.path / part["path"])
            _check_version(
                (table.schema.metadata or {}).get(b"schema_version", b"").decode(),
                part["path"],
            )
            for row in table.to_pylist():
                row["quality_flags"] = tuple(row["quality_flags"])
                bars.append(Bar(**row))
        return bars


def _check_version(value: object, where: str) -> None:
    if str(value) != str(SCHEMA_VERSION):
        raise DatasetError(
            f"unknown schema_version {value!r} in {where}; this reader "
            f"supports {SCHEMA_VERSION}. Re-ingest from the raw cache."
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _row(bar: Bar) -> list[Any]:
    return [
        bar.instrument,
        bar.open_time.astimezone(UTC).isoformat(),
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.spread_close,
        bar.spread_mean,
        bar.tick_count,
        bar.is_gap,
        bar.session_id.astimezone(UTC).isoformat(),
        bar.is_rollover,
        list(bar.quality_flags),
    ]


def _partition_hash(bars: Sequence[Bar]) -> str:
    digest = hashlib.sha256()
    for bar in bars:  # canonical order: sorted by open_time, fixed column order
        digest.update(_canonical_json(_row(bar)) + b"\n")
    return digest.hexdigest()


def _dataset_hash(
    source: str,
    bar_size: int,
    rules: Mapping[str, Any],
    partitions: Mapping[str, str],
) -> str:
    # The source is hashed so a live and a backfill Dataset with identical rows
    # stay distinct Datasets (they are never merged and share no lineage).
    return hashlib.sha256(
        _canonical_json(
            {
                "source": source,
                "schema_version": SCHEMA_VERSION,
                "bar_size": bar_size,
                "rules": rules,
                "partitions": sorted(partitions.items()),
            }
        )
    ).hexdigest()


def _table(bars: Sequence[Bar], bar_size: int) -> pa.Table:
    columns = {name: [getattr(b, name) for b in bars] for name in _COLUMNS}
    columns["quality_flags"] = [list(f) for f in columns["quality_flags"]]
    table = pa.table(columns, schema=BAR_SCHEMA)
    return table.replace_schema_metadata(
        {
            "schema_version": str(SCHEMA_VERSION),
            "view": VIEW,
            "bar_size": str(bar_size),
        }
    )


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:12]


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


def _raw_inputs(data_root: Path, instruments: Iterable[str]) -> dict[str, str]:
    """sha256 of every cached raw file for the instruments, keyed by path."""
    found: dict[str, str] = {}
    for instrument in instruments:
        folder = data_root / "raw" / SOURCE / instrument
        for sidecar in sorted(folder.glob("*.meta.json")):
            name = sidecar.name.removesuffix(".meta.json")
            found[f"{SOURCE}/{instrument}/{name}"] = json.loads(sidecar.read_text())[
                "sha256"
            ]
    return found


def write_dataset(
    config: Config,
    bar_size: int,
    bars: Mapping[str, Sequence[Bar]],
    *,
    parent: str | None = None,
    created_at: datetime | None = None,
    builder_revision: str | None = None,
    tick_flag_counts: Mapping[str, Mapping[str, int]] | None = None,
    source: str = SOURCE,
    raw_inputs: Mapping[str, str] | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> DatasetRef:
    """Write an immutable Dataset of ``bars`` (per Instrument) and return it.

    Unchanged closed partitions are hardlinked from ``parent``; only the open
    (latest) year of each Instrument is rewritten.
    """
    data_root = Path(config.paths.data_root)
    rules = config.content_affecting(bars)
    ordered = {
        pair: sorted(series, key=lambda b: b.open_time)
        for pair, series in sorted(bars.items())
    }
    parts: dict[str, list[Bar]] = {}
    for pair, series in ordered.items():
        for bar in series:
            key = f"instrument={pair}/year={bar.open_time.astimezone(UTC).year}"
            parts.setdefault(key, []).append(bar)
    hashes = {key: _partition_hash(rows) for key, rows in parts.items()}
    content_hash = _dataset_hash(source, bar_size, rules, hashes)

    target = data_root / "datasets" / content_hash
    if (target / "manifest.json").exists():
        manifest = json.loads((target / "manifest.json").read_text())
        return DatasetRef(content_hash, target, manifest)
    if target.exists():  # incomplete earlier attempt: never read, safe to discard
        shutil.rmtree(target)

    parent_hashes: dict[str, str] = {}
    if parent:
        parent_manifest = _read_manifest(data_root, parent)
        parent_hashes = {p["path"]: p["hash"] for p in parent_manifest["partitions"]}
    open_year = {
        pair: max(b.open_time.astimezone(UTC).year for b in series)
        for pair, series in ordered.items()
        if series
    }
    entries = []
    for key, rows in sorted(parts.items()):
        rel = f"bars/{key}/part-0.parquet"
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pair = rows[0].instrument
        year = int(key.rsplit("=", 1)[1])
        closed = year < open_year[pair]
        if closed and parent and parent_hashes.get(rel) == hashes[key]:
            os.link(data_root / "datasets" / parent / rel, dest)
        else:
            pq.write_table(_table(rows, bar_size), dest)
        entries.append(
            {
                "instrument": pair,
                "path": rel,
                "rows": len(rows),
                "gap_count": sum(b.is_gap for b in rows),
                "hash": hashes[key],
            }
        )

    all_rows = [b for series in ordered.values() for b in series]
    gap_rules = rules["rules"]["gap"]
    anomaly_rules = rules["rules"]["anomaly"]
    first = min((b.open_time for b in all_rows), default=None)
    last = max((b.open_time for b in all_rows), default=None)
    manifest = {
        "source": source,
        "bar_size": bar_size,
        "instruments": sorted(ordered),
        "time_range": {
            "start": first.isoformat() if first else None,
            "end": (last + timedelta(minutes=bar_size)).isoformat() if last else None,
        },
        "row_count": len(all_rows),
        "content_hash": content_hash,
        "dataset_id": content_hash,
        "parent": parent,
        "schema_version": SCHEMA_VERSION,
        "view": VIEW,
        "partitions": entries,
        "raw_inputs": (
            dict(raw_inputs)
            if raw_inputs is not None
            else _raw_inputs(data_root, ordered)
        ),
        "acknowledged_missing": [
            dataclasses.asdict(a)
            for a in config.build.acknowledged_missing
            if a.instrument in ordered
        ],
        "builder_version": _builder_version(),
        "builder_git_revision": builder_revision or _git_revision(),
        "gap_rules": gap_rules,
        "gap_rule_set_version": _short_hash(gap_rules),
        "anomaly_rules": anomaly_rules,
        "anomaly_rule_set_version": _short_hash(anomaly_rules),
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "coverage": {
            pair: _coverage(series, (tick_flag_counts or {}).get(pair, {}))
            for pair, series in ordered.items()
        },
        **(extra_manifest or {}),
    }
    _write_manifest(target, manifest)
    return DatasetRef(content_hash, target, manifest)


def latest_dataset(
    data_root: Path, source: str, bar_size: int, instruments: Sequence[str]
) -> str | None:
    """The most recently created complete Dataset of one series, if any."""
    found: list[tuple[str, str]] = []
    for path in (Path(data_root) / "datasets").glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if (
            manifest["source"] == source
            and manifest["bar_size"] == bar_size
            and manifest["instruments"] == sorted(instruments)
        ):
            found.append((manifest["created_at"], manifest["dataset_id"]))
    return max(found)[1] if found else None


def _coverage(series: Sequence[Bar], tick_flags: Mapping[str, int]) -> dict[str, Any]:
    flag_counts: dict[str, int] = {}
    for bar in series:
        for flag in bar.quality_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "sessions": len({b.session_id for b in series}),
        "gaps": sum(b.is_gap for b in series),
        "flagged_bars": sum(bool(b.quality_flags) for b in series),
        "flag_counts": dict(sorted(flag_counts.items())),
        "tick_flag_counts": dict(sorted(tick_flags.items())),
    }


def _write_manifest(target: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest last, by atomic rename."""
    tmp = target / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, target / "manifest.json")


def _read_manifest(data_root: Path, dataset_id: str) -> dict[str, Any]:
    path = Path(data_root) / "datasets" / dataset_id / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"dataset {dataset_id} has no manifest (incomplete)")
    manifest = json.loads(path.read_text())
    _check_version(manifest.get("schema_version"), str(path))
    return manifest


def open_dataset(data_root: Path, dataset_id: str) -> Dataset:
    """Open a Dataset; refuse a missing manifest or an unknown schema_version."""
    manifest = _read_manifest(Path(data_root), dataset_id)
    return Dataset(dataset_id, Path(data_root) / "datasets" / dataset_id, manifest)
