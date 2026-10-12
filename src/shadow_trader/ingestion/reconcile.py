"""Advisory live-vs-backfill reconciliation diff (ADR 0003, spec #28 section 10).

Compares a live and a backfill Dataset of the same Instruments, bar size and
``schema_version`` over their overlapping range, matched on ``open_time`` with
the live side at its latest Revision. It is advisory only: it never blocks,
never rewrites and never touches either Dataset, and it is the only link
between the two (they share no parent edge and are never merged).

Tolerances come from the ``[reconciliation]`` config block and are recorded in
the report. The report lists the 500 worst Bars by price deviation; Bars that
are a Gap on one side and priced on the other are always listed, outside the
cap.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .bars import NY, Bar
from .config import Config
from .dataset import Dataset, DatasetError, open_dataset
from .report import REPORT_VERSION, _write_json

WORST_BAR_CAP = 500
_PRICE_FIELDS = ("open", "high", "low", "close")
_DRIFT_FIELDS = (*_PRICE_FIELDS, "spread_close", "spread_mean")


class ReconciliationRefused(Exception):
    """The two Datasets are not comparable (CLI exit code 1)."""


def _pip(instrument: str) -> float:
    return 0.01 if instrument.endswith("JPY") else 0.0001


def _open(config: Config, dataset_id: str) -> Dataset:
    try:
        return open_dataset(config.paths.data_root, dataset_id)
    except (DatasetError, OSError) as exc:
        raise ReconciliationRefused(f"cannot open dataset {dataset_id}: {exc}") from exc


def _check_comparable(live: Dataset, backfill: Dataset) -> None:
    for key, label in (
        ("instruments", "Instruments"),
        ("bar_size", "bar size"),
        ("schema_version", "schema_version"),
    ):
        if live.manifest[key] != backfill.manifest[key]:
            raise ReconciliationRefused(
                f"{label} differ: live {live.manifest[key]!r} vs "
                f"backfill {backfill.manifest[key]!r}"
            )


def _dev(a: float, b: float, pip: float) -> float:
    return round(abs(a - b) / pip, 6)


def _rank(ordered: Sequence[float], p: float) -> float:
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


@dataclass
class _Week:
    matched: int = 0
    within: int = 0
    beyond: int = 0
    live_only: int = 0
    backfill_only: int = 0
    gap_in_one_only: int = 0


@dataclass
class _Tally:
    counts: dict[str, int] = field(default_factory=dict)
    weeks: dict[str, _Week] = field(default_factory=dict)
    drift: dict[str, list[float]] = field(
        default_factory=lambda: {f: [] for f in (*_DRIFT_FIELDS, "tick_count")}
    )
    ratios: list[float] = field(default_factory=list)
    ratio_outside: int = 0
    deviating: list[tuple[float, datetime, str, dict[str, Any]]] = field(
        default_factory=list
    )
    one_sided: list[dict[str, Any]] = field(default_factory=list)

    def bump(self, key: str, week: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        week_counts = self.weeks.setdefault(week, _Week())
        if hasattr(week_counts, key):
            setattr(week_counts, key, getattr(week_counts, key) + 1)


def _week(bar: Bar) -> str:
    return bar.session_id.astimezone(NY).date().isoformat()


def _values(bar: Bar) -> dict[str, Any]:
    return {f: getattr(bar, f) for f in (*_DRIFT_FIELDS, "tick_count")}


def _compare(
    tally: _Tally, live: Bar, backfill: Bar, config: Config, week: str
) -> None:
    tol = config.reconciliation
    pip = _pip(live.instrument)
    assert None not in (live.open, backfill.open)
    price_devs = {
        f: _dev(getattr(live, f), getattr(backfill, f), pip) for f in _PRICE_FIELDS
    }
    assert live.spread_mean is not None and backfill.spread_mean is not None
    spread_allowed = max(
        tol.spread_pips * pip, tol.spread_relative * backfill.spread_mean
    )
    spread_beyond = abs(live.spread_mean - backfill.spread_mean) > spread_allowed
    worst = max(price_devs.values())
    beyond = worst > tol.price_pips or spread_beyond
    tally.bump("matched", week)
    tally.bump("beyond" if beyond else "within", week)

    for name in _DRIFT_FIELDS:
        scale = pip
        tally.drift[name].append(
            (getattr(live, name) - getattr(backfill, name)) / scale
        )
    assert live.tick_count is not None and backfill.tick_count is not None
    tally.drift["tick_count"].append(float(live.tick_count - backfill.tick_count))
    ratio = live.tick_count / backfill.tick_count if backfill.tick_count else math.inf
    tally.ratios.append(ratio)
    if not tol.tick_ratio_min <= ratio <= tol.tick_ratio_max:
        tally.ratio_outside += 1

    if worst > 0:
        tally.deviating.append(
            (
                worst,
                live.open_time,
                live.instrument,
                {
                    "instrument": live.instrument,
                    "open_time": live.open_time.isoformat(),
                    "price_deviation_pips": worst,
                    "deviation_pips": price_devs,
                    "beyond_tolerance": beyond,
                    "live": _values(live),
                    "backfill": _values(backfill),
                },
            )
        )


def _drift_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p95_abs": None, "max_abs": None}
    magnitudes = sorted(abs(v) for v in values)
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "p95_abs": round(_rank(magnitudes, 0.95), 6),
        "max_abs": round(magnitudes[-1], 6),
    }


def _index(dataset: Dataset, start: datetime, end: datetime) -> dict[Any, Bar]:
    return {
        (bar.instrument, bar.open_time): bar
        for instrument in dataset.manifest["instruments"]
        for bar in dataset.read(instrument)
        if start <= bar.open_time < end
    }


def _overlap(live: Dataset, backfill: Dataset) -> tuple[datetime, datetime] | None:
    ranges = [d.manifest["time_range"] for d in (live, backfill)]
    if any(r["start"] is None for r in ranges):
        return None
    start = max(datetime.fromisoformat(r["start"]) for r in ranges)
    end = min(datetime.fromisoformat(r["end"]) for r in ranges)
    return (start, end) if start < end else None


def build_reconciliation(
    config: Config, live: Dataset, backfill: Dataset
) -> dict[str, Any]:
    """The reconciliation report document of two comparable Datasets."""
    _check_comparable(live, backfill)
    tally = _Tally()
    overlap = _overlap(live, backfill)
    if overlap is not None:
        live_bars = _index(live, *overlap)
        back_bars = _index(backfill, *overlap)
        for key in sorted(live_bars.keys() | back_bars.keys()):
            mine, theirs = live_bars.get(key), back_bars.get(key)
            if mine is None or theirs is None:
                present = mine or theirs
                assert present is not None
                tally.bump(
                    "live_only" if theirs is None else "backfill_only", _week(present)
                )
            elif mine.is_gap and theirs.is_gap:
                tally.bump("both_gap", _week(mine))
            elif mine.is_gap or theirs.is_gap:
                tally.bump("gap_in_one_only", _week(mine))
                tally.one_sided.append(
                    {
                        "instrument": mine.instrument,
                        "open_time": mine.open_time.isoformat(),
                        "gap_side": "live" if mine.is_gap else "backfill",
                        "live": _values(mine) | {"flags": list(mine.quality_flags)},
                        "backfill": _values(theirs)
                        | {"flags": list(theirs.quality_flags)},
                    }
                )
            else:
                _compare(tally, mine, theirs, config, _week(mine))

    worst = sorted(tally.deviating, key=lambda d: (-d[0], d[1], d[2]))
    counts = tally.counts
    ratios = sorted(tally.ratios)
    return {
        "report_version": REPORT_VERSION,
        "status": "ok",
        "advisory": True,
        "live_dataset_id": live.dataset_id,
        "backfill_dataset_id": backfill.dataset_id,
        "instruments": live.manifest["instruments"],
        "bar_size": live.manifest["bar_size"],
        "schema_version": live.manifest["schema_version"],
        "overlap": (
            {"start": overlap[0].isoformat(), "end": overlap[1].isoformat()}
            if overlap
            else None
        ),
        "tolerances": dataclasses.asdict(config.reconciliation),
        "summary": {
            key: counts.get(key, 0)
            for key in (
                "matched",
                "within",
                "beyond",
                "live_only",
                "backfill_only",
                "gap_in_one_only",
                "both_gap",
            )
        },
        "drift": {
            "unit": "pips (live - backfill); tick_count in ticks",
            **{name: _drift_summary(v) for name, v in tally.drift.items()},
        },
        "tick_ratio": {
            "outside_tolerance": tally.ratio_outside,
            "min": ratios[0] if ratios else None,
            "max": ratios[-1] if ratios else None,
        },
        "weeks": {
            week: dataclasses.asdict(w) for week, w in sorted(tally.weeks.items())
        },
        "worst_bars": {
            "cap": WORST_BAR_CAP,
            "total_with_deviation": len(worst),
            "listed": [entry for _, _, _, entry in worst[:WORST_BAR_CAP]],
        },
        "one_sided_gaps": sorted(
            tally.one_sided, key=lambda g: (g["open_time"], g["instrument"])
        ),
    }


def reconciliation_path(data_root: Path, live_id: str, backfill_id: str) -> Path:
    return (
        Path(data_root)
        / "reports"
        / "reconciliation"
        / f"reconciliation-{live_id}-{backfill_id}.json"
    )


def reconcile(config: Config, live_id: str, backfill_id: str) -> Path:
    """Write the reconciliation report for two Datasets; return its path.

    Raises ``ReconciliationRefused`` if the Datasets are not comparable.
    Neither Dataset is modified.
    """
    live = _open(config, live_id)
    backfill = _open(config, backfill_id)
    document = build_reconciliation(config, live, backfill)
    path = reconciliation_path(config.paths.data_root, live_id, backfill_id)
    _write_json(path, document)
    return path


__all__: Sequence[str] = [
    "WORST_BAR_CAP",
    "ReconciliationRefused",
    "build_reconciliation",
    "reconcile",
    "reconciliation_path",
]
