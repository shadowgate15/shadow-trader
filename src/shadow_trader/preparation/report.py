"""Build reports: the build report, the failed-build report and advisory warnings.

Reports are versioned JSON (``report_version``). They are non-authoritative:
never hashed into a Dataset or build id and never read by consumers. The build
report sits beside the build record, paired by build id only; a failed-build
report goes to the sibling ``<prepared_root>.failed`` directory, outside the immutable
store. Warnings are advisory with an additive-only code set and never fail a
build; their thresholds come from the unhashed ``[report]`` config table.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .config import PreparedConfig, ReportConfig
from .scaler import ROBUST, Scaler

REPORT_VERSION = 1

# Closed, additive-only vocabulary of warning codes.
WARNING_CODES = (
    "low_usable_fraction",
    "all_unusable_instrument",
    "high_distrusted_flag_fraction",
    "scaler_clip_fraction_high",
    "short_dataset",
)

_STAMP = "%Y-%m-%dT%H:%M:%SZ"


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` by atomic rename (overwriting any earlier report)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True))
    os.replace(tmp, path)


def report_path(prepared_root: Path, build_id: str) -> Path:
    return Path(prepared_root) / "builds" / f"{build_id}.report.json"


def role_summary(frames: Mapping[str, pl.DataFrame]) -> dict[str, Any]:
    """Row, usable, unusable-reason and null counts of one role's frames."""
    reasons: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    instruments: dict[str, Any] = {}
    for pair, frame in sorted(frames.items()):
        rows, usable = frame.height, int(frame["usable"].sum())
        instruments[pair] = {
            "rows": rows,
            "usable": usable,
            "usable_fraction": usable / rows if rows else 0.0,
        }
        reasons.update(
            dict(frame["unusable_reason"].drop_nulls().value_counts().iter_rows())
        )
        for name in frame.columns:
            if name.startswith("feat_"):
                nulls[name] += frame[name].null_count()
    rows = sum(i["rows"] for i in instruments.values())
    usable = sum(i["usable"] for i in instruments.values())
    return {
        "rows": rows,
        "usable": usable,
        "usable_fraction": usable / rows if rows else 0.0,
        "unusable_reasons": dict(sorted(reasons.items())),
        "null_counts": dict(sorted(nulls.items())),
        "instruments": instruments,
    }


def max_clip_fraction(scaler: Scaler, frames: Mapping[str, pl.DataFrame]) -> float:
    """The largest share of any Instrument column's values that scaling would clip."""
    clip = scaler.artifact["clip"]
    worst = 0.0
    for pair, frame in frames.items():
        for name, stats in scaler.artifact["instruments"][pair]["columns"].items():
            if stats["method"] != ROBUST:
                continue
            values = frame[name].drop_nulls()
            if len(values):
                over = ((values - stats["median"]).abs() / stats["iqr"] > clip).sum()
                worst = max(worst, over / len(values))
    return worst


def advisory_warnings(
    roles: Mapping[str, Mapping[str, Any]],
    clip_fractions: Mapping[str, float],
    thresholds: ReportConfig,
) -> list[dict[str, str]]:
    """Advisory warnings from per-role summaries; they never fail a build."""
    found: list[dict[str, str]] = []

    def warn(code: str, role: str, message: str) -> None:
        found.append({"code": code, "role": role, "message": message})

    for role, summary in roles.items():
        rows = summary["rows"]
        fraction = summary["usable_fraction"]
        if fraction < thresholds.min_usable_fraction:
            warn(
                "low_usable_fraction",
                role,
                f"{role}: usable fraction {fraction:.3f} is below "
                f"{thresholds.min_usable_fraction}",
            )
        for pair, part in summary["instruments"].items():
            if part["rows"] and part["usable"] == 0:
                warn(
                    "all_unusable_instrument",
                    role,
                    f"{role}: {pair} has no usable Bars",
                )
            if part["rows"] < thresholds.min_rows:
                warn(
                    "short_dataset",
                    role,
                    f"{role}: {pair} has {part['rows']} rows, "
                    f"fewer than {thresholds.min_rows}",
                )
        distrusted = summary["unusable_reasons"].get("distrusted_flag", 0)
        if rows and distrusted / rows > thresholds.max_distrusted_flag_fraction:
            warn(
                "high_distrusted_flag_fraction",
                role,
                f"{role}: {distrusted / rows:.3f} of Bars carry a distrusted flag, "
                f"above {thresholds.max_distrusted_flag_fraction}",
            )
        clipped = clip_fractions.get(role, 0.0)
        if clipped > thresholds.max_scaler_clip_fraction:
            warn(
                "scaler_clip_fraction_high",
                role,
                f"{role}: up to {clipped:.3f} of a column's values are clipped, "
                f"above {thresholds.max_scaler_clip_fraction}",
            )
    return found


def write_failed_report(
    config: PreparedConfig,
    *,
    code: str | None,
    message: str,
    counts: Mapping[str, Any],
    now: datetime | None = None,
) -> Path:
    """Record a failed build outside the immutable store; return its path."""
    now = now or datetime.now(UTC)
    split = config.split
    root = Path(config.paths.prepared_root)
    path = root.with_name(f"{root.name}.failed") / f"{now:%Y%m%dT%H%M%S%fZ}.json"
    write_json(
        path,
        {
            "report_version": REPORT_VERSION,
            "status": "failed",
            "code": code,
            "message": message,
            "failed_at": now.isoformat(),
            "inputs": {
                "parent_dataset": config.input.dataset_id,
                "spec_version": config.spec.version,
                "embargo_episode_bars": config.embargo.episode_bars,
                "split": {
                    "scheme": split.scheme,
                    "train_start": split.train_start.strftime(_STAMP),
                    "validation_start": split.validation_start.strftime(_STAMP),
                    "test_start": split.test_start.strftime(_STAMP),
                    "test_end": split.test_end and split.test_end.strftime(_STAMP),
                },
            },
            "counts": dict(counts),
        },
    )
    return path


def console_summary(
    build_id: str,
    dataset_ids: Mapping[str, str],
    scaler_id: str,
    roles: Mapping[str, Mapping[str, Any]],
    boundary_check: Mapping[str, Any],
    warnings: Sequence[Mapping[str, Any]],
    path: Path,
) -> list[str]:
    """The one-screen build summary, as lines."""
    lines = [f"built prepared build {build_id}"]
    for role, dataset_id in dataset_ids.items():
        summary = roles[role]
        lines.append(
            f"{role} {dataset_id} rows {summary['rows']} "
            f"usable {summary['usable_fraction']:.1%}"
        )
    lines.append(f"scaler {scaler_id}")
    lines.append(
        "boundary_check pass" if boundary_check.get("passed") else "boundary_check FAIL"
    )
    codes = sorted({w["code"] for w in warnings})
    lines.append(
        f"warnings {len(warnings)}" + (f": {', '.join(codes)}" if codes else "")
    )
    lines.append(f"report {path}")
    return lines
