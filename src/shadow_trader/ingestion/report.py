"""Run reports: the build report, the failed-build report and the console summary.

Reports are versioned JSON (``report_version``). They are non-authoritative:
never hashed into a Dataset and never read by consumers.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from .anomaly import summarize_flags
from .bars import Bar
from .config import Config

# Closed vocabulary of warning codes (spec #28, section 11). Additive-only.
WARNING_CODES = (
    "out_of_session_ticks",
    "session_alignment",
    "thin_data",
    "anomaly_alert_fraction_exceeded",
    "live_quote_after_seal",
)

REPORT_VERSION = 1
BUILD_REPORT = "build-report.json"
OUTAGE_RUN_CAP = 200


class WarningEntry(Protocol):
    code: str
    message: str
    locator: str | None


@dataclass(frozen=True)
class Alignment:
    """A week whose first tick is off the 17:00 New York open."""

    week: str
    locator: str
    offset_minutes: int  # signed: positive is after the open
    wrong_offset_signature: bool


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` by atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _counts(bars: Sequence[Bar], dropped: int) -> dict[str, Any]:
    gaps = [b for b in bars if b.is_gap]
    outage = sum("suspected_outage" in b.quality_flags for b in gaps)
    missing = sum("source_missing" in b.quality_flags for b in gaps)
    return {
        "sessions": len({b.session_id for b in bars}),
        "bars": len(bars),
        "gap_rows": {
            "plain": len(gaps) - outage - missing,
            "suspected_outage": outage,
            "source_missing": missing,
        },
        "out_of_session_ticks": dropped,
    }


def _outage_runs(bars: Sequence[Bar], bar_size: int) -> dict[str, Any]:
    """Runs of consecutive ``suspected_outage`` Bars, listed up to the cap."""
    step = timedelta(minutes=bar_size)
    runs: list[list[Bar]] = []
    for bar in sorted(bars, key=lambda b: b.open_time):
        if "suspected_outage" not in bar.quality_flags:
            continue
        last = runs[-1][-1] if runs else None
        if (
            last is not None
            and last.session_id == bar.session_id
            and bar.open_time == last.open_time + step
        ):
            runs[-1].append(bar)
        else:
            runs.append([bar])
    return {
        "total": len(runs),
        "listed": [
            {
                "start": run[0].open_time.isoformat(),
                "end": (run[-1].open_time + step).isoformat(),
                "duration_minutes": len(run) * bar_size,
            }
            for run in runs[:OUTAGE_RUN_CAP]
        ],
    }


def resolved_config(config: Config) -> dict[str, Any]:
    """The whole config as plain JSON-able values, defaults included."""
    return json.loads(
        json.dumps(
            {
                "paths": dataclasses.asdict(config.paths),
                "build": dataclasses.asdict(config.build),
                "instruments": {
                    pair: dataclasses.asdict(value)
                    for pair, value in sorted(config.instruments.items())
                },
                "rules": {
                    "gap": dataclasses.asdict(config.gap_rule),
                    "anomaly": dataclasses.asdict(config.anomaly_rule),
                },
                "reconciliation": dataclasses.asdict(config.reconciliation),
                "live": dataclasses.asdict(config.live),
            },
            default=str,
        )
    )


def build_report(
    *,
    dataset_id: str,
    bar_size: int,
    series: Mapping[str, Sequence[Bar]],
    dropped_out_of_session: Mapping[str, int],
    tick_flag_counts: Mapping[str, Mapping[str, int]] | None = None,
    alignments: Mapping[str, Sequence[Alignment]] | None = None,
    warnings: Sequence[WarningEntry] = (),
    raw_inputs: Mapping[str, str] | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """The build report of one successful Dataset build (a plain JSON document)."""
    alert_fraction = config.anomaly_rule.alert_fraction if config else 0.01
    instruments: dict[str, Any] = {}
    for pair, bars in sorted(series.items()):
        summary = summarize_flags(
            bars, (tick_flag_counts or {}).get(pair, {}), alert_fraction
        )
        instruments[pair] = {
            **_counts(bars, dropped_out_of_session.get(pair, 0)),
            "flags": {
                "bar_counts": summary.bar_counts,
                "tick_counts": summary.tick_counts,
                "non_gap_bars": summary.non_gap_bars,
                "alerted_bars": summary.alerted_bars,
                "alert_fraction": alert_fraction,
                "alert_exceeded": bool(summary.warnings),
            },
            "session_alignment": [
                dataclasses.asdict(a) for a in (alignments or {}).get(pair, ())
            ],
            "suspected_outage_runs": _outage_runs(bars, bar_size),
        }
    return {
        "report_version": REPORT_VERSION,
        "status": "ok",
        "dataset_id": dataset_id,
        "bar_size": bar_size,
        "instruments": instruments,
        "warnings": [
            {"code": w.code, "message": w.message, "locator": w.locator}
            for w in warnings
        ],
        "raw_inputs": dict(raw_inputs or {}),
        "config": resolved_config(config) if config else None,
    }


def write_build_report(dataset_path: Path, report: Mapping[str, Any]) -> Path:
    path = dataset_path / BUILD_REPORT
    _write_json(path, report)
    return path
