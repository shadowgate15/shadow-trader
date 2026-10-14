"""``build-live``: a live Dataset from the raw quote log (ADR 0003).

The raw log is replayed in arrival order through the same incremental builder
the live ingester used, so late and catch-up quotes land as the same
Revisions. Only sealed slots (past their repair window) enter the Dataset, at
their latest Revision. The run-length outage rule is re-applied across the
series and unioned with the flags live emitted, and Gap Bars overlapping a
source-down interval (adapter-reported or between runs) are flagged too.

A live Dataset has a different ``source`` from a backfill Dataset; the two
never share a parent edge and are never merged.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import dataset
from .bars import Bar
from .config import Config, validate_bar_sizes
from .dataset import DatasetRef
from .gaps import flag_suspected_outages, with_flag
from .live import Event, LiveIngester, overlaps_interval
from .progress import NULL_PROGRESS, ProgressSink
from .rawlog import raw_log_dir, read_raw_log
from .replay import ReplayAdapter
from .report import live_report_dir


def _runs(data_root: Path, source: str, instrument: str) -> list[dict[str, Any]]:
    directory = live_report_dir(data_root, source, instrument)
    reports = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    return sorted(reports, key=lambda r: r["run"]["started_at"])


def _down_intervals(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reported source-down intervals plus the downtime between clean runs."""
    down = [dict(i) for run in runs for i in run["source_down"]]
    for before, after in itertools.pairwise(runs):
        if before["status"] == "crashed":
            continue  # the next run already recorded this downtime
        began = before["run"]["last_watermark"] or before["run"]["started_at"]
        resumed = after["run"]["started_at"]
        if began < resumed:
            down.append({"start": began, "end": resumed, "reason": "between_runs"})
    return down


def _overlaps(bar: Bar, size: timedelta, down: list[dict[str, Any]]) -> bool:
    return any(
        overlaps_interval(
            bar.open_time,
            size,
            datetime.fromisoformat(i["start"]),
            datetime.fromisoformat(i["end"]) if i["end"] else None,
        )
        for i in down
    )


def _log_files(data_root: Path, source: str, instrument: str) -> dict[str, str]:
    """sha256 of every raw log file, from its sidecar (or its bytes if open)."""
    found: dict[str, str] = {}
    directory = raw_log_dir(data_root, source, instrument)
    for path in sorted(directory.iterdir()) if directory.exists() else []:
        if path.name.endswith(".meta.json"):
            continue
        sidecar = path.with_name(path.name + ".meta.json")
        meta: dict[str, Any] = (
            json.loads(sidecar.read_text()) if sidecar.exists() else {"open": True}
        )
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if meta.get("open")  # provisional sidecar: the bytes are the truth
            else meta["sha256"]
        )
        found[f"live/{source}/{instrument}/{path.name}"] = digest
    return found


def _events(data_root: Path, source: str, instrument: str) -> Iterator[Event]:
    for quote, _ in read_raw_log(data_root, source, instrument):
        yield quote


def build_live(
    config: Config,
    bar_size: int,
    start: datetime,
    end: datetime,
    *,
    progress: ProgressSink = NULL_PROGRESS,
) -> DatasetRef:
    """Build a live Dataset over the closed range ``[start, end)``."""
    validate_bar_sizes((bar_size,))
    data_root = Path(config.paths.data_root)
    source = config.live.adapter
    size = timedelta(minutes=bar_size)
    series: dict[str, list[Bar]] = {}
    raw_inputs: dict[str, str] = {}
    run_intervals: dict[str, list[dict[str, Any]]] = {}
    down_intervals: dict[str, list[dict[str, Any]]] = {}
    instruments = config.build.instruments
    progress.start("replaying raw logs", len(instruments))
    for instrument in instruments:
        runs = _runs(data_root, source, instrument)
        down = _down_intervals(runs)
        floor = start
        if runs:  # slots before the first run ever started are not Gaps
            floor = max(start, datetime.fromisoformat(runs[0]["run"]["started_at"]))
        ingester = LiveIngester(
            config,
            ReplayAdapter(iter(()), instrument=instrument, source=source),
            sink=lambda bar, revision: None,
            bar_size=bar_size,
            start=floor,
            record=False,
        )
        sealed = ingester.replay_sealed(_events(data_root, source, instrument))
        bars = [
            with_flag(b, "suspected_outage")
            if b.is_gap and _overlaps(b, size, down)
            else b
            for b in sealed
            if start <= b.open_time < end
        ]
        series[instrument] = flag_suspected_outages(
            bars, bar_size, config.gap_rule.outage_threshold_minutes
        )
        raw_inputs.update(_log_files(data_root, source, instrument))
        progress.advance(instrument)
        down_intervals[instrument] = down
        run_intervals[instrument] = [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "start": r["run"]["started_at"],
                "end": r["run"]["ended_at"],
            }
            for r in runs
        ]
    progress.done()
    parent = dataset.latest_dataset(
        data_root, source, bar_size, config.build.instruments
    )
    progress.start("writing dataset")
    ref = dataset.write_dataset(
        config,
        bar_size,
        series,
        parent=parent,
        source=source,
        raw_inputs=raw_inputs,
        extra_manifest={
            "live": {
                "range": {
                    "start": start.astimezone(UTC).isoformat(),
                    "end": end.astimezone(UTC).isoformat(),
                },
                "runs": run_intervals,
                "source_down": down_intervals,
                "timing": {
                    "grace_seconds": config.live.grace_seconds,
                    "repair_window_minutes": config.live.repair_window_minutes,
                },
            }
        },
    )
    progress.done()
    return ref
