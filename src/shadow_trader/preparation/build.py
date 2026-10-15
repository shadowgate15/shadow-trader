"""The preparation build: read one parent ingestion Dataset, add features, split, write.

Reads the parent only through ingestion's public ``open_dataset``. Build order:
open parent, check bar size, compute features per Instrument, check alignment
(coverage, bar size, Session geometry), split + Embargo,
``boundary_check``, then write the three Datasets and the build record. Every
check runs before the first write, so a failed build leaves nothing behind.

Between the split and ``boundary_check`` the train Dataset is hashed and the
scaler fit on the train role only; its fit-range check joins ``boundary_check``
and its artifact is written before the three Datasets (whose manifests name it).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from shadow_trader.ingestion.bars import Bar
from shadow_trader.ingestion.dataset import (
    BAR_SCHEMA,
    DatasetError,
    open_dataset,
)
from shadow_trader.ingestion.progress import NULL_PROGRESS, ProgressSink

from .align import AlignmentFailure, check_alignment
from .anomaly import DEFAULT_POLICY, AnomalyPolicy, apply_anomaly_policy
from .boundary import BoundaryCheckFailure, BoundaryContext, Check, boundary_check
from .config import PreparedConfig
from .dataset import PreparedRef, build_id_of, plan_prepared, write_build_record
from .report import (
    REPORT_VERSION,
    advisory_warnings,
    max_clip_fraction,
    report_path,
    role_summary,
    write_failed_report,
    write_json,
)
from .scaler import (
    ScalerFitError,
    fit_scaler,
    scaler_fit_range_check,
    write_scaler,
)
from .spec import FeatureSpec, compute_features
from .split import (
    ROLES,
    SplitFailure,
    dataset_identity,
    embargo_bars,
    snap_bounds,
    split_frame,
)

# Checks run after the built-in boundary checks (the scaler's fit-range check is
# appended per build, since it closes over that build's fitted scaler).
EXTRA_CHECKS: tuple[tuple[str, Check], ...] = ()


class BuildFailure(Exception):
    """The build cannot proceed (CLI exit code 1); nothing is written.

    ``code`` is the closed failure code (``boundary_check.*``,
    ``embargo_exceeds_dataset``, ``no_usable_bars``) when there is one.
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.report_path: Path | None = None  # the failed-build report, once written


@dataclass(frozen=True)
class BuildRef:
    """The result of one build: its id, record and the three Datasets by role."""

    build_id: str
    path: Path
    record: Mapping[str, Any]
    datasets: Mapping[str, PreparedRef]
    report: Mapping[str, Any]  # the non-authoritative build report
    report_path: Path


def _bars_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    columns = {name: [row[name] for row in rows] for name in BAR_SCHEMA.names}
    table = pa.table(columns, schema=BAR_SCHEMA)
    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)
    return frame


def featured_frame(
    bars: Sequence[Bar], spec: FeatureSpec, policy: AnomalyPolicy = DEFAULT_POLICY
) -> pl.DataFrame:
    """One Instrument's Bars with ``feat_*``, ``usable`` and ``unusable_reason``."""
    rows = [{**vars(bar), "quality_flags": list(bar.quality_flags)} for bar in bars]
    featured = compute_features(_bars_frame(rows), spec).collect()
    return apply_anomaly_policy(featured, spec, policy)


def _span(frame: pl.DataFrame, bar_size: int) -> dict[str, str]:
    """The actual ``[start, end)`` of one Instrument's Bars in a Dataset."""
    first, last = frame.select(
        pl.col("open_time").min().alias("first"),
        pl.col("open_time").max().alias("last"),
    ).row(0)
    return {
        "start": first.isoformat(),
        "end": (last + timedelta(minutes=bar_size)).isoformat(),
    }


def build_prepared(
    config: PreparedConfig, progress: ProgressSink = NULL_PROGRESS
) -> BuildRef:
    """Build the train, validation and test Datasets and return the build's ref.

    Writes the build report beside the build record. A failure writes a small
    failed-build report (outside the immutable store) and records its path on
    the raised ``BuildFailure``.
    """
    counts: dict[str, Any] = {"instruments": 0, "rows": {}}
    try:
        return _build(config, progress, counts)
    except BuildFailure as exc:
        exc.report_path = write_failed_report(
            config, code=exc.code, message=str(exc), counts=counts
        )
        raise


def _build(
    config: PreparedConfig, progress: ProgressSink, counts: dict[str, Any]
) -> BuildRef:
    began = time.perf_counter()
    spec = config.feature_spec
    parent_id = config.input.dataset_id
    try:
        parent = open_dataset(config.paths.data_root, parent_id)
    except (DatasetError, OSError) as exc:
        raise BuildFailure(
            f"cannot open parent Dataset {parent_id}: {exc}", "parent_unavailable"
        ) from exc

    bar_size = parent.manifest["bar_size"]
    if bar_size != spec.bar_size:
        raise BuildFailure(
            f"bar size mismatch: parent Dataset {parent_id} has {bar_size}-minute "
            f"Bars but feature spec v{spec.version} is bound to "
            f"{spec.bar_size}-minute Bars",
            "bar_size_mismatch",
        )
    instruments = list(parent.manifest["instruments"])

    progress.start("features", len(instruments))
    frames: dict[str, pl.DataFrame] = {}
    for instrument in instruments:
        frames[instrument] = featured_frame(
            parent.read(instrument), spec, config.anomaly
        )
        counts["instruments"] += 1
        counts["rows"][instrument] = frames[instrument].height
        progress.advance(instrument)
    progress.done()
    features_seconds = time.perf_counter() - began

    embargo = embargo_bars(spec, config.embargo)
    try:
        opens = sorted(
            set().union(*(set(f["session_id"].to_list()) for f in frames.values()))
        )
        bounds = snap_bounds(config.split, opens)
        check_alignment(
            frames,
            bar_size=spec.bar_size,
            start=bounds.start("train"),
            end=bounds.snapped["test_end"],
        )
        roles = {pair: split_frame(f, bounds, embargo) for pair, f in frames.items()}
        plans = {
            role: plan_prepared(
                {pair: roles[pair][role].frame for pair in instruments},
                parent_ingestion=parent_id,
                spec=spec,
                policy=config.anomaly,
                identity=dataset_identity(config.split, embargo, role),
            )
            for role in ROLES
        }
        contexts = {
            pair: BoundaryContext(
                bounds, roles[pair], frames[pair]["open_time"], embargo, spec
            )
            for pair in instruments
        }

        def run_checks(extra: Sequence[tuple[str, Check]]) -> list[dict[str, Any]]:
            return [boundary_check(contexts[pair], extra=extra) for pair in instruments]

        try:
            scaler = fit_scaler(
                pl.concat([roles[pair]["train"].frame for pair in instruments]),
                spec,
                policy_hash=config.anomaly.hash,
                fit_dataset=plans["train"].dataset_id,
            )
        except ScalerFitError:
            # A degenerate split (empty or unusable train) is reported by its own
            # boundary check first; only then is the fit failure the cause.
            run_checks(EXTRA_CHECKS)
            raise
        checks = run_checks(
            (*EXTRA_CHECKS, ("scaler_fit_range", scaler_fit_range_check(scaler)))
        )
        checked = checks[0]  # every Instrument passed, on the same grid
    except ScalerFitError as exc:
        raise BuildFailure(str(exc), exc.code) from exc
    except AlignmentFailure as exc:
        raise BuildFailure(str(exc), exc.code) from exc
    except (SplitFailure, BoundaryCheckFailure) as exc:
        raise BuildFailure(f"{exc.code}: {exc.detail}", exc.code) from exc

    ids = {role: plan.dataset_id for role, plan in plans.items()}
    split_block = {"scheme": config.split.scheme, **bounds.block()}
    shared: dict[str, Any] = {
        "parent_ingestion": parent_id,
        "spec_hash": spec.hash,
        "policy_hash": config.anomaly.hash,
        "split": split_block,
        "embargo_bars": embargo,
        "boundary_check": checked,
        "scaler": scaler.scaler_id,
    }
    build_id = build_id_of(shared, ids)

    role_summaries = {
        role: role_summary({pair: roles[pair][role].frame for pair in instruments})
        for role in ROLES
    }
    clips = {
        role: max_clip_fraction(
            scaler, {pair: roles[pair][role].frame for pair in instruments}
        )
        for role in ROLES
    }
    warnings = advisory_warnings(role_summaries, clips, config.report)

    write_scaler(config.paths.prepared_root, scaler)
    progress.start("write", len(ROLES))
    datasets: dict[str, PreparedRef] = {}
    for role in ROLES:
        datasets[role] = plans[role].write(
            config.paths.prepared_root,
            {
                "role": role,
                "build_id": build_id,
                "siblings": {r: i for r, i in ids.items() if r != role},
                "split": split_block,
                "embargo": roles[instruments[0]][role].embargo_block(
                    embargo, spec, config.embargo.episode_bars
                ),
                "instrument_spans": {
                    pair: _span(roles[pair][role].frame, spec.bar_size)
                    for pair in instruments
                },
                "boundary_check": checked,
                "scaler": {"id": scaler.scaler_id, "fit": role == "train"},
            },
        )
        progress.advance(role)
    record = write_build_record(config.paths.prepared_root, shared, ids)
    progress.done()

    total_seconds = time.perf_counter() - began
    rows = sum(s["rows"] for s in role_summaries.values())
    path = report_path(config.paths.prepared_root, record.build_id)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "build_id": record.build_id,
        "created_at": datetime.now(UTC).isoformat(),
        "durations": {
            "features_seconds": features_seconds,
            "total_seconds": total_seconds,
        },
        "throughput": {"rows": rows, "rows_per_second": rows / total_seconds},
        "threads": pl.thread_pool_size(),
        "roles": role_summaries,
        "scaler": scaler.scaler_id,
        "manifests": {
            role: str(ref.path / "manifest.json") for role, ref in datasets.items()
        },
        "build_record": str(record.path),
        "warnings": warnings,
    }
    write_json(path, report)
    return BuildRef(record.build_id, record.path, record.record, datasets, report, path)
