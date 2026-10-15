"""``prepare verify``: audit stored prepared output without writing anything.

Given a build id it rehashes the three Datasets against their manifests, reruns
every ``boundary_check`` check against the stored rows and confirms the scaler
ref resolves. Each check is reported individually (``VerifyCheck``); nothing is
written, repaired or modified, and no report file is produced.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from shadow_trader.ingestion.dataset import open_dataset

from .boundary import BUILT_IN, BoundaryContext
from .config import PreparedConfig
from .dataset import _dataset_hash, _partition_hash, build_id_of, open_prepared
from .scaler import Scaler, load_scaler, scaler_fit_range_check
from .spec import SPECS
from .split import BOUND_NAMES, ROLES, RoleSlice, SplitBounds


@dataclass(frozen=True)
class VerifyCheck:
    name: str
    detail: str | None = None  # None means the check passed

    @property
    def passed(self) -> bool:
        return self.detail is None


def _guard(name: str, fn: Callable[[], str | None]) -> VerifyCheck:
    try:
        return VerifyCheck(name, fn())
    except Exception as exc:  # corrupt or missing stored files are failures
        return VerifyCheck(name, f"{type(exc).__name__}: {exc}")


def _parse(bounds: dict[str, str | None]) -> dict[str, datetime | None]:
    # Manifests are stored key-sorted; the ordering check needs chronological order.
    return {
        name: None if (text := bounds[name]) is None else datetime.fromisoformat(text)
        for name in BOUND_NAMES
    }


def verify_build(config: PreparedConfig, build_id: str) -> list[VerifyCheck]:
    root = config.paths.prepared_root
    results: list[VerifyCheck] = []

    record_path = root / "builds" / f"{build_id}.json"
    try:
        record: dict[str, Any] = json.loads(record_path.read_text())
        shared = {
            k: v
            for k, v in record.items()
            if k not in {"kind", "datasets", "build_id", "polars_version", "created_at"}
        }
        if build_id_of(shared, record["datasets"]) != build_id:
            raise ValueError("record content does not hash to its build id")
        results.append(VerifyCheck("build_record"))
    except Exception as exc:
        return [VerifyCheck("build_record", f"{type(exc).__name__}: {exc}")]

    datasets = {}
    for role in ROLES:
        dataset_id = record["datasets"][role]

        def open_and_hash(role: str = role, dataset_id: str = dataset_id) -> str | None:
            dataset = open_prepared(root, dataset_id)
            datasets[role] = dataset
            manifest = dataset.manifest
            hashes: dict[str, str] = {}
            bad = []
            for part in manifest["partitions"]:
                digest = _partition_hash(pq.read_table(dataset.path / part["path"]))
                if digest != part["hash"]:
                    bad.append(part["path"])
                hashes[part["path"].rsplit("/", 1)[0].removeprefix("bars/")] = digest
            if bad:
                return f"{role}: partition hash mismatch: {', '.join(bad)}"
            spec = SPECS[manifest["feature_spec"]["version"]]
            if spec.hash != manifest["feature_spec"]["hash"]:
                return f"{role}: feature spec hash differs from the recorded one"
            expected = _dataset_hash(
                manifest["parent_ingestion"],
                spec.hash,
                manifest["anomaly_policy"]["hash"],
                manifest["bar_size"],
                hashes,
                manifest.get("identity"),
            )
            if expected != dataset_id:
                return f"{role}: content hashes to {expected}, not {dataset_id}"
            return None

        results.append(_guard(f"dataset.{role}", open_and_hash))

    scaler_holder: list[Scaler] = []

    def resolve_scaler() -> str | None:
        scaler_holder.append(load_scaler(root, record["scaler"]))
        return None

    results.append(_guard("scaler.resolves", resolve_scaler))

    def make_context() -> dict[str, BoundaryContext]:
        train = datasets["train"].manifest
        bounds = SplitBounds(
            _parse(train["split"]["bounds"]), _parse(train["split"]["snapped_bounds"])
        )
        spec = SPECS[train["feature_spec"]["version"]]
        parent = open_dataset(config.paths.data_root, train["parent_ingestion"])
        contexts = {}
        for pair in train["instruments"]:
            roles = {}
            for role in ROLES:
                frame = datasets[role].read(pair)
                embargo = datasets[role].manifest["embargo"]
                trimmed = embargo["trimmed_bars"]
                roles[role] = RoleSlice(
                    role,
                    frame,
                    frame.height + trimmed,
                    trimmed,
                    None,
                    None,
                )
            opens = pl.Series([bar.open_time for bar in parent.read(pair)])
            contexts[pair] = BoundaryContext(
                bounds, roles, opens, record["embargo_bars"], spec
            )
        return contexts

    contexts: dict[str, BoundaryContext] = {}
    try:
        contexts = make_context()
        context_error = None
    except Exception as exc:
        context_error = (
            f"cannot rebuild the split from stored rows: {type(exc).__name__}: {exc}"
        )

    checks = list(BUILT_IN)
    for name, check in checks:
        label = f"boundary_check.{name}"
        if context_error:
            results.append(VerifyCheck(label, context_error))
            continue
        results.append(_guard(label, lambda c=check: _first_detail(contexts, c)))

    label = "boundary_check.scaler_fit_range"
    if context_error or not scaler_holder:
        results.append(VerifyCheck(label, context_error or "scaler did not resolve"))
    else:
        fit = scaler_fit_range_check(scaler_holder[0])
        results.append(_guard(label, lambda: _first_detail(contexts, fit)))
    return results


def _first_detail(
    contexts: dict[str, BoundaryContext], check: Callable[[BoundaryContext], str | None]
) -> str | None:
    for pair, ctx in contexts.items():
        detail = check(ctx)
        if detail is not None:
            return f"{pair}: {detail}"
    return None
