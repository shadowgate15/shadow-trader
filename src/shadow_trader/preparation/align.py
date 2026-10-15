"""Alignment: the shared Session grid, checked, never constructed.

Ingestion already puts every Instrument on one Session grid (one row per slot,
Gap rows included), so alignment is a check, not a step: no joins, forward-fills,
dropped or invented rows, and no resampling. Each Instrument must cover the
requested span, share the bar size and have the same Session geometry (the same
slots and Session ids) within it. A single Instrument is the N=1 case.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta

import polars as pl


class AlignmentFailure(Exception):
    """The Instruments do not share the grid; ``code`` is the closed failure code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def check_alignment(
    frames: Mapping[str, pl.DataFrame],
    *,
    bar_size: int,
    start: datetime,
    end: datetime | None,
) -> None:
    """Raise ``AlignmentFailure`` unless every frame is on the same grid.

    ``frames`` hold each Instrument's ``open_time`` and ``session_id``. The
    requested span is ``[start, end)``; ``end=None`` means up to the latest Bar
    any Instrument has.
    """
    step = timedelta(minutes=bar_size)
    for pair, frame in sorted(frames.items()):
        steps = frame["open_time"].sort().diff().drop_nulls()
        if steps.len() and steps.min() != step:
            raise AlignmentFailure(
                "bar_size_mismatch",
                f"{pair} has a {steps.min()} step between Bars, "
                f"not {bar_size} minutes; nothing is resampled",
            )

    firsts: dict[str, datetime] = {}
    ends: dict[str, datetime] = {}
    for pair, frame in frames.items():
        first, last = frame.select(
            pl.col("open_time").min().alias("first"),
            pl.col("open_time").max().alias("last"),
        ).row(0)
        firsts[pair], ends[pair] = first, last + step
    wanted_end = end if end is not None else max(ends.values())
    for pair in sorted(frames):
        first = firsts[pair]
        if first > start:
            raise AlignmentFailure(
                "coverage",
                f"{pair} does not cover {start.isoformat()} to "
                f"{first.isoformat()} of the requested span",
            )
        if ends[pair] < wanted_end:
            raise AlignmentFailure(
                "coverage",
                f"{pair} does not cover {ends[pair].isoformat()} to "
                f"{wanted_end.isoformat()} of the requested span",
            )

    in_span = pl.col("open_time") >= start
    if end is not None:
        in_span = in_span & (pl.col("open_time") < end)
    grids = {
        pair: frame.filter(in_span).sort("open_time").select("open_time", "session_id")
        for pair, frame in sorted(frames.items())
    }
    (reference, expected), *rest = grids.items()
    for pair, grid in rest:
        if not grid.equals(expected):
            raise AlignmentFailure(
                "session_geometry",
                f"{pair} has a different Session grid from {reference} "
                f"({grid.height} vs {expected.height} slots in the span)",
            )
