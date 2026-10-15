"""Chronological split with an Embargo.

Bounds are explicit UTC datetimes that snap forward to the next Session open, so
no Session straddles a split. Ranges are half-open and share one set of bounds
across Instruments: train ``[train_start, validation_start)``, validation
``[validation_start, test_start)`` and test ``[test_start, test_end)`` (to the end
of the parent when ``test_end`` is unset). The Embargo ``E`` (longest feature
lookback plus ``embargo.episode_bars``, in index Bars) is cut off the tail of
train and validation only; the cut Bars stay in the parent and are simply absent
from the split's Dataset. There is no ``embargoed`` flag column.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from .config import EmbargoConfig, SplitConfig
from .spec import FeatureSpec

ROLES = ("train", "validation", "test")
BOUND_NAMES = ("train_start", "validation_start", "test_start", "test_end")


class SplitFailure(Exception):
    """A split bound cannot be resolved; ``code`` is the closed failure code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def longest_lookback(spec: FeatureSpec) -> int:
    return max((c.lookback for c in spec.columns), default=0)


def embargo_bars(spec: FeatureSpec, embargo: EmbargoConfig) -> int:
    """``E = longest feature lookback + embargo.episode_bars`` in index Bars."""
    return longest_lookback(spec) + embargo.episode_bars


@dataclass(frozen=True)
class SplitBounds:
    nominal: Mapping[str, datetime | None]  # as configured
    snapped: Mapping[str, datetime | None]  # forward to the next Session open

    def block(self) -> dict[str, dict[str, str | None]]:
        return {
            "bounds": _iso(self.nominal),
            "snapped_bounds": _iso(self.snapped),
        }

    def start(self, role: str) -> datetime:
        start = self.snapped[f"{role}_start"]
        assert start is not None
        return start

    def end(self, role: str) -> datetime | None:
        later = {"train": "validation_start", "validation": "test_start"}
        return self.snapped[later[role]] if role in later else self.snapped["test_end"]


def _iso(bounds: Mapping[str, datetime | None]) -> dict[str, Any]:
    return {k: None if v is None else v.isoformat() for k, v in bounds.items()}


def snap_bounds(split: SplitConfig, session_opens: Sequence[datetime]) -> SplitBounds:
    """Snap each configured bound forward to the next Session open.

    ``session_opens`` are the parent's Session opens, ascending. A bound later
    than every Session open is a failure, except an unreachable ``test_end``,
    which simply leaves the test range open-ended.
    """
    nominal: dict[str, datetime | None] = {
        name: getattr(split, name) for name in BOUND_NAMES
    }
    snapped: dict[str, datetime | None] = {}
    for name, bound in nominal.items():
        if bound is None:
            snapped[name] = None
            continue
        index = bisect_left(session_opens, bound)
        if index == len(session_opens):
            if name == "test_end":
                snapped[name] = None
                continue
            raise SplitFailure(
                "boundary_check.session_open",
                f"{name} {bound.isoformat()} is after the last Session open "
                f"{session_opens[-1].isoformat()}",
            )
        snapped[name] = session_opens[index]
    return SplitBounds(nominal, snapped)


@dataclass(frozen=True)
class RoleSlice:
    """One split's Bars for one Instrument, after the Embargo trim."""

    role: str
    frame: pl.DataFrame
    raw_height: int  # Bars in the range before the trim
    trimmed_bars: int
    nominal_end: datetime | None
    effective_end: datetime | None  # open_time of the first trimmed Bar

    def embargo_block(
        self, bars: int, spec: FeatureSpec, episode: int
    ) -> dict[str, Any]:
        return {
            "episode_bars": episode,
            "longest_lookback": longest_lookback(spec),
            "bars": bars,
            "nominal_end": _maybe_iso(self.nominal_end),
            "effective_end": _maybe_iso(self.effective_end),
            "trimmed_bars": self.trimmed_bars,
        }


def _maybe_iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _relabel_dataset_start(frame: pl.DataFrame) -> pl.DataFrame:
    """Name the warm-up of a split's first Session ``dataset_start``.

    The anomaly policy ran over the whole parent, where only the parent's first
    Session is a Dataset start; a split's own first Session is one too.
    """
    if frame.height == 0:
        return frame
    first = frame["session_id"][0]
    return frame.with_columns(
        pl.when(
            (pl.col("unusable_reason") == "session_warm_up")
            & (pl.col("session_id") == first)
        )
        .then(pl.lit("dataset_start"))
        .otherwise(pl.col("unusable_reason"))
        .alias("unusable_reason")
    )


def split_frame(
    frame: pl.DataFrame, bounds: SplitBounds, embargo: int
) -> dict[str, RoleSlice]:
    """Cut one Instrument's featured parent frame into the three trimmed splits."""
    ordered = frame.sort("open_time")
    slices: dict[str, RoleSlice] = {}
    for role in ROLES:
        start, end = bounds.start(role), bounds.end(role)
        in_range = pl.col("open_time") >= start
        if end is not None:
            in_range = in_range & (pl.col("open_time") < end)
        selected = _relabel_dataset_start(ordered.filter(in_range))
        raw = selected.height
        trim = embargo if role != "test" else 0
        kept = selected.head(max(raw - trim, 0))
        trimmed = raw - kept.height
        effective = end
        if trimmed:
            effective = selected["open_time"][kept.height]
        slices[role] = RoleSlice(role, kept, raw, trimmed, end, effective)
    return slices


def dataset_identity(
    split: SplitConfig, bounds: SplitBounds, embargo: int, role: str
) -> dict[str, Any]:
    """What the Dataset hash covers beyond rows: scheme, bounds, E and role.

    Both the nominal bounds as configured and the bounds snapped to Session
    opens are covered.
    """
    return {
        "scheme": split.scheme,
        "bounds": _iso(bounds.nominal),
        "snapped_bounds": _iso(bounds.snapped),
        "embargo_bars": embargo,
        "role": role,
    }
