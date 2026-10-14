"""Run reports: the build report, the failed-build report and the console summary.

Reports are versioned JSON (``report_version``). They are non-authoritative:
never hashed into a Dataset and never read by consumers.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    "late_ticks_dropped",
)

REPORT_VERSION = 1
BUILD_REPORT = "build-report.json"
OUTAGE_RUN_CAP = 200
DROPPED_EXAMPLE_CAP = 20


class WarningEntry(Protocol):
    @property
    def code(self) -> str: ...
    @property
    def message(self) -> str: ...
    @property
    def locator(self) -> str | None: ...


@dataclass(frozen=True)
class Alignment:
    """A week whose first tick is off the 17:00 New York open."""

    week: str
    locator: str
    offset_minutes: int  # signed: positive is after the open
    wrong_offset_signature: bool


def write_json(path: Path, document: Mapping[str, Any]) -> None:
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
    session_clocks: Mapping[str, Mapping[str, Any]] | None = None,
    warnings: Sequence[WarningEntry] = (),
    raw_inputs: Mapping[str, str] | None = None,
    config: Config | None = None,
    timings: Mapping[str, float] | None = None,
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
            "session_clock": dict((session_clocks or {}).get(pair, {})),
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
        # wall-clock seconds per phase; never part of the Dataset content hash
        "timings": dict(timings or {}),
    }


PHASE_LABELS = {
    "clock_pre_pass": "clock pre-pass",
    "bar_pass": "bar pass",
    "write_dataset": "write dataset",
}


def console_summary(report: Mapping[str, Any], path: Path) -> list[str]:
    """Console lines derived from a report's JSON (never stored).

    One line per warning code with its count, then the artifact path; a failed
    report leads with the failing condition and locator.
    """
    lines: list[str] = []
    if report["status"] == "failed":
        where = f" ({report['locator']})" if report.get("locator") else ""
        lines.append(f"build failed: {report['condition']}{where}")
    else:
        counts: dict[str, int] = {}
        for warning in report["warnings"]:
            counts[warning["code"]] = counts.get(warning["code"], 0) + 1
        lines.extend(f"warning: {code} x{counts[code]}" for code in counts)
        if timings := report.get("timings"):
            lines.append(
                "timings: "
                + ", ".join(
                    f"{PHASE_LABELS.get(phase, phase)} {seconds:.1f} s"
                    for phase, seconds in timings.items()
                )
            )
    lines.append(f"report: {path}")
    return lines


def write_failed_report(
    data_root: Path, bar_size: int, condition: str, locator: str | None
) -> Path:
    """Record a failed build under ``<data_root>/reports/failed``."""
    now = datetime.now(UTC)
    path = Path(data_root) / "reports" / "failed" / f"{now:%Y%m%dT%H%M%S%fZ}.json"
    write_json(
        path,
        {
            "report_version": REPORT_VERSION,
            "status": "failed",
            "bar_size": bar_size,
            "condition": condition,
            "locator": locator,
            "failed_at": now.isoformat(),
        },
    )
    return path


def live_report_dir(data_root: Path, source: str, instrument: str) -> Path:
    return Path(data_root) / "reports" / "live" / source / instrument


def nearest_rank(ordered: Sequence[float], p: float) -> float:
    """Nearest-rank percentile ``p`` (0-1) of an ascending, non-empty sequence."""
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def lag_distribution(lags: Sequence[float]) -> dict[str, Any]:
    """Count, p50, p95 and max (seconds) of late-quote lags; nearest-rank."""
    if not lags:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(lags)
    return {
        "count": len(ordered),
        "p50": nearest_rank(ordered, 0.50),
        "p95": nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def live_run_report(
    *,
    run_id: str,
    status: str,
    source: str,
    instrument: str,
    bar_size: int,
    started_at: datetime,
    ended_at: datetime | None,
    last_watermark: datetime | None,
    source_down: Sequence[Mapping[str, Any]],
    bars_by_revision: Mapping[int, int],
    gap_bars: int,
    late_accepted: int,
    dropped_after_seal: int,
    out_of_session: int,
    late_lags: Sequence[float],
    catch_ups: Sequence[Mapping[str, Any]],
    dropped_examples: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The live run report of one ingester process lifetime (a plain document)."""
    return {
        "report_version": REPORT_VERSION,
        "run_id": run_id,
        "status": status,
        "source": source,
        "instrument": instrument,
        "bar_size": bar_size,
        "run": {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat() if ended_at else None,
            "last_watermark": last_watermark.isoformat() if last_watermark else None,
        },
        "source_down": [dict(i) for i in source_down],
        "counts": {
            "bars": sum(bars_by_revision.values()),
            "bars_by_revision": {
                str(r): n for r, n in sorted(bars_by_revision.items())
            },
            "gap_bars": gap_bars,
            "late_quotes_accepted": late_accepted,
            "quotes_dropped_after_seal": dropped_after_seal,
            "quotes_dropped_out_of_session": out_of_session,
        },
        "late_quote_lag_seconds": lag_distribution(late_lags),
        "catch_up": [dict(c) for c in catch_ups],
        "dropped_after_seal_examples": [
            dict(e) for e in dropped_examples[:DROPPED_EXAMPLE_CAP]
        ],
        "warnings": (
            [
                {
                    "code": "live_quote_after_seal",
                    "message": f"{dropped_after_seal} quotes arrived after their "
                    "Bar was sealed and were dropped",
                    "locator": None,
                }
            ]
            if dropped_after_seal
            else []
        ),
        "config": dict(resolved) if resolved is not None else None,
    }


def write_live_report(
    data_root: Path, source: str, instrument: str, document: Mapping[str, Any]
) -> Path:
    """Atomically (re)write one run's report."""
    directory = live_report_dir(data_root, source, instrument)
    path = directory / f"{document['run_id']}.json"
    write_json(path, document)
    return path


def mark_stale_runs_crashed(
    data_root: Path,
    source: str,
    instrument: str,
    restarted_at: datetime,
) -> list[dict[str, Any]]:
    """Mark reports still ``running`` as ``crashed``; return their downtime.

    A run whose report never left ``running`` died without a clean shutdown.
    The downtime is from its last watermark (or its start, if it never saw an
    event) to ``restarted_at``, as a source-down interval of reason
    ``between_runs``.
    """
    downtime: list[dict[str, Any]] = []
    directory = live_report_dir(data_root, source, instrument)
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        text = path.read_text()
        if '"status": "running"' not in text:
            continue  # cheap pre-check: most reports are long finished
        document = json.loads(text)
        run = document["run"]
        began = run["last_watermark"] or run["started_at"]
        document["status"] = "crashed"
        write_json(path, document)
        downtime.append({"start": began, "end": restarted_at.isoformat()})
    return downtime


def write_build_report(dataset_path: Path, report: Mapping[str, Any]) -> Path:
    path = dataset_path / BUILD_REPORT
    write_json(path, report)
    return path
