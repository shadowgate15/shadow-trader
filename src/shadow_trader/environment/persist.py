"""Ledger persistence: Parquet tables plus a provenance manifest.

``write_ledger(env, path, ...)`` writes one Parquet file per Ledger table and a
``manifest.json`` into the directory ``path``. The manifest records the schema
version, Dataset id, Scaler artifact hash, the ``EnvConfig`` hash and each
sub-config's hash with its full canonical JSON (``RewardConfig`` included though
the Ledger has no reward column), ``SEAM_VERSION``, package version, seed entropy
and episode mode. It also carries the Ledger's account facts (starting equity,
leverage cap) and a SHA-256 of each table file, and seals itself with a
``manifest_hash`` over everything else.

``read_ledger(path)`` rejects a missing or inconsistent manifest (``ManifestError``)
and then runs ``verify_ledger`` on what it loaded.

Training episodes write nothing by default: the env must have been made with
``record_ledger=True``, and ``write_ledger`` is an explicit request. Evaluation
sweeps always persist, through ``persist_sweep``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from shadow_trader.preparation._io import canonical_json

from .env import TradingEnv
from .env_config import EnvConfig
from .ledger import (
    CLOSED_LOTS_SCHEMA,
    EPISODES_SCHEMA,
    FILLS_SCHEMA,
    STEPS_SCHEMA,
    Ledger,
    LedgerError,
    verify_ledger,
)
from .report import summarize, summary_document
from .sweep import SweepResult, run_sweep

__all__ = [
    "MANIFEST_FILE",
    "SCHEMA_VERSION",
    "SUMMARY_FILE",
    "ManifestError",
    "build_manifest",
    "persist_sweep",
    "read_ledger",
    "read_manifest",
    "verify_summary",
    "write_ledger",
]

SCHEMA_VERSION = 2
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
EPISODE_MODE_SWEEP = "eval_sweep"

_TABLES: dict[str, dict[str, pl.DataType]] = {
    "steps": STEPS_SCHEMA,
    "fills": FILLS_SCHEMA,
    "closed_lots": CLOSED_LOTS_SCHEMA,
    "episodes": EPISODES_SCHEMA,
}


class ManifestError(LedgerError):
    """A persisted Ledger's manifest is missing, malformed or does not match."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha(path.read_bytes())


def _manifest_hash(doc: dict[str, Any]) -> str:
    return _sha(canonical_json({k: v for k, v in doc.items() if k != "manifest_hash"}))


def _summary_bytes(ledger: Ledger) -> bytes:
    return canonical_json(summary_document(summarize(ledger)))


def verify_summary(path: Path, ledger: Ledger) -> None:
    """Reject a missing, edited or stale ``summary.json`` beside ``ledger``'s tables."""
    file = Path(path) / SUMMARY_FILE
    if not file.is_file():
        raise ManifestError(f"summary missing: {file}")
    if file.read_bytes() != _summary_bytes(ledger):
        raise ManifestError("summary.json does not match the recomputed summary")


def build_manifest(
    ledger: Ledger,
    config: EnvConfig,
    *,
    dataset_id: str,
    scaler_hash: str,
    seam_version: int,
    seed_entropy: int | None,
    episode_mode: str,
    package_version: str | None = None,
) -> dict[str, Any]:
    """The manifest body for ``ledger`` (without table file hashes or the seal)."""
    subs = config.sub_configs()
    hashes = config.sub_config_hashes()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "scaler_hash": scaler_hash,
        "env_config_hash": config.config_hash(),
        "configs": {
            name: {"hash": hashes[name], "json": json.loads(canonical_json(value))}
            for name, value in subs.items()
        },
        "seam_version": seam_version,
        "package_version": package_version or metadata.version("shadow-trader"),
        "seed_entropy": None if seed_entropy is None else str(seed_entropy),
        "episode_mode": episode_mode,
        "ledger": {
            "starting_equity": ledger.starting_equity,
            "leverage_cap": ledger.leverage_cap,
        },
    }


def write_ledger(
    env: TradingEnv,
    path: Path,
    *,
    episode_mode: str,
    seam_version: int,
) -> Path:
    """Persist ``env``'s Ledger to directory ``path``; returns the manifest path.

    Raises ``RuntimeError`` (writing nothing) when the env records no Ledger.
    """
    ledger = env.ledger()  # raises when the env was not made with record_ledger
    path = Path(path)
    manifest = build_manifest(
        ledger,
        env.config,
        dataset_id=env.provenance["dataset_id"],
        scaler_hash=env.provenance["scaler_hash"],
        seam_version=seam_version,
        seed_entropy=env.seed_entropy,
        episode_mode=episode_mode,
    )
    path.mkdir(parents=True, exist_ok=True)
    tables: dict[str, str] = {}
    for name in _TABLES:
        file = path / f"{name}.parquet"
        getattr(ledger, name).write_parquet(file)
        tables[name] = _file_hash(file)
    manifest["tables"] = tables
    (path / SUMMARY_FILE).write_bytes(_summary_bytes(ledger))
    manifest["manifest_hash"] = _manifest_hash(manifest)
    target = path / MANIFEST_FILE
    tmp = target.with_name(MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(target)
    return target


def persist_sweep(
    env: TradingEnv,
    policy: Callable[[Any], int],
    path: Path,
    *,
    seam_version: int,
    sessions: Iterable[int] | None = None,
) -> SweepResult:
    """Run an evaluation sweep and always persist its Ledger to ``path``."""
    result = run_sweep(env, policy, sessions=sessions)
    write_ledger(env, path, episode_mode=EPISODE_MODE_SWEEP, seam_version=seam_version)
    return result


def read_manifest(path: Path) -> dict[str, Any]:
    """Load and self-check the manifest of the persisted Ledger at ``path``."""
    file = Path(path) / MANIFEST_FILE
    if not file.is_file():
        raise ManifestError(f"manifest missing: {file}")
    try:
        doc = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest unreadable: {file}: {exc}") from None
    if not isinstance(doc, dict) or doc.get("manifest_hash") != _manifest_hash(doc):
        raise ManifestError("manifest hash mismatch: the manifest was altered")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"unknown schema_version {doc.get('schema_version')!r}; "
            f"this reader supports {SCHEMA_VERSION}"
        )
    configs = doc.get("configs", {})
    for name, entry in configs.items():
        if _sha(canonical_json(entry["json"])) != entry["hash"]:
            raise ManifestError(f"config {name!r} json does not match its hash")
    combined = {name: entry["json"] for name, entry in configs.items()}
    if _sha(canonical_json(combined)) != doc.get("env_config_hash"):
        raise ManifestError("env_config_hash does not match the sub-configs")
    return doc


def read_ledger(path: Path) -> Ledger:
    """Load a persisted Ledger, checking the manifest and running ``verify_ledger``."""
    path = Path(path)
    doc = read_manifest(path)
    frames: dict[str, pl.DataFrame] = {}
    for name, schema in _TABLES.items():
        file = path / f"{name}.parquet"
        if not file.is_file():
            raise ManifestError(f"table {name} missing: {file}")
        if _file_hash(file) != doc["tables"].get(name):
            raise ManifestError(f"table {name} does not match the manifest")
        frames[name] = pl.read_parquet(file)
        if dict(frames[name].schema) != schema:
            raise ManifestError(f"table {name} has an unexpected schema")
    facts = doc["ledger"]
    cap = facts["leverage_cap"]
    equity = facts["starting_equity"]
    if not isinstance(equity, int | float) or not math.isfinite(equity):
        raise ManifestError("manifest starting_equity is not a finite number")
    ledger = Ledger(
        steps=frames["steps"],
        fills=frames["fills"],
        starting_equity=float(equity),
        closed_lots=frames["closed_lots"],
        leverage_cap=None if cap is None else float(cap),
        episodes=frames["episodes"],
    )
    verify_ledger(ledger)
    verify_summary(path, ledger)
    return ledger
