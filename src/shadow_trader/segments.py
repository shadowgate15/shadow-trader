"""Segments and the chronological Split, derived once from the stored Bars.

``train`` and ``evaluate`` both call :func:`load`; nothing downstream re-derives
either. Everything is expressed as half-open Bar-index ranges ``[first, stop)``
into the stored Bars, so the Split and the Segments can be intersected exactly.

A Segment ends at any Gap of at least ``config.GAP_THRESHOLD`` (time with no
Bar). Smaller Gaps stay inside their Segment. The Split cuts by Bar count, so a
Segment may straddle a Period boundary; :meth:`Structure.segments_within` clips.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

import pyarrow.parquet as pq

from shadow_trader import bars, config


@dataclass(frozen=True)
class Segment:
    first: int  # index of its first Bar
    stop: int  # one past its last Bar
    start_time: datetime
    end_time: datetime  # open time of its last Bar
    gap_after: timedelta | None  # the Gap that closed it; None for the last

    def __len__(self) -> int:
        return self.stop - self.first


@dataclass(frozen=True)
class Period:
    name: str
    first: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.first


@dataclass(frozen=True)
class Split:
    train: Period
    validation: Period
    test: Period

    @property
    def periods(self) -> tuple[Period, Period, Period]:
        return (self.train, self.validation, self.test)


@dataclass(frozen=True)
class Structure:
    times: tuple[datetime, ...]
    segments: tuple[Segment, ...]
    split: Split

    @property
    def bar_count(self) -> int:
        return len(self.times)

    def segments_within(self, period: Period) -> list[Segment]:
        """Segments intersected with ``period``, clipped to its Bar range."""
        out = []
        for s in self.segments:
            first, stop = max(s.first, period.first), min(s.stop, period.stop)
            if first >= stop:
                continue
            out.append(
                Segment(
                    first,
                    stop,
                    self.times[first],
                    self.times[stop - 1],
                    s.gap_after if stop == s.stop else None,
                )
            )
        return out

    def render(self) -> str:
        """Human-readable figures to sanity-check against the ingest summary."""
        lines = [
            f"bars: {self.bar_count}",
            f"gap threshold: {config.GAP_THRESHOLD}",
            f"segments: {len(self.segments)}",
        ]
        for i, s in enumerate(self.segments):
            closed = f"closed by gap {s.gap_after}" if s.gap_after else "last"
            lines.append(
                f"  {i:>4}: bars {s.first}-{s.stop} ({len(s)}) "
                f"{s.start_time:%Y-%m-%d %H:%M} to {s.end_time:%Y-%m-%d %H:%M}"
                f" {closed}"
            )
        lines.append("split:")
        for p in self.split.periods:
            lines.append(
                f"  {p.name:>10}: bars {p.first}-{p.stop} ({len(p)}) "
                f"{self.times[p.first]:%Y-%m-%d %H:%M} to "
                f"{self.times[p.stop - 1]:%Y-%m-%d %H:%M}"
            )
        return "\n".join(lines)


def derive_segments(times: list[datetime]) -> list[Segment]:
    """Split sorted Bar times at every Gap of at least ``config.GAP_THRESHOLD``."""
    cuts = [
        (i + 1, (b - a) - bars.BAR_INTERVAL)
        for i, (a, b) in enumerate(pairwise(times))
        if (b - a) - bars.BAR_INTERVAL >= config.GAP_THRESHOLD
    ]
    starts = [0, *(i for i, _ in cuts)]
    stops = [*(i for i, _ in cuts), len(times)]
    gaps = [*(g for _, g in cuts), None]
    return [
        Segment(a, b, times[a], times[b - 1], g)
        for a, b, g in zip(starts, stops, gaps, strict=True)
        if b > a
    ]


def derive_split(bar_count: int) -> Split:
    """Chronological train / validation / test by Bar count (``config.SPLIT``)."""
    train_frac, val_frac, _ = config.SPLIT
    a = round(bar_count * train_frac)
    b = round(bar_count * (train_frac + val_frac))
    return Split(
        Period("train", 0, a),
        Period("validation", a, b),
        Period("test", b, bar_count),
    )


def derive(times: list[datetime]) -> Structure:
    return Structure(
        tuple(times), tuple(derive_segments(times)), derive_split(len(times))
    )


def load() -> Structure:
    """The one shared derivation, read from the stored Bars."""
    table = pq.read_table(bars.bars_path(), columns=["time"])
    return derive(table["time"].to_pylist())
