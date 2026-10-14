"""Gap versus outage semantics.

``is_gap`` stays the only no-data bit and Gap rows stay null-priced. Two
Gap-only quality flags explain why a Gap exists:

* ``suspected_outage``: a run of consecutive Gap Bars inside one Session lasting
  at least the configured duration (independent of bar size). Runs never cross
  Sessions. Gaps already flagged ``source_missing`` do not join runs.
* ``source_missing``: the Bar lies in a whole month listed in
  ``acknowledged_missing``; those Bars are synthesized, never read from a file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta

from .anomaly import VOCABULARY
from .bars import Bar, gap_bar
from .config import month_parts
from .session import NY, open_of, slots_per_session, week_sunday


def with_flag(bar: Bar, flag: str) -> Bar:
    """``bar`` with ``flag`` added, in vocabulary order."""
    flags = {*bar.quality_flags, flag}
    return dataclasses.replace(
        bar, quality_flags=tuple(f for f in VOCABULARY if f in flags)
    )


def flag_suspected_outages(
    bars: Sequence[Bar], bar_minutes: int, threshold_minutes: int
) -> list[Bar]:
    """Flag Gap runs of at least ``threshold_minutes`` within one Session."""
    result = list(bars)
    start = 0
    while start < len(result):
        if not _in_run(result[start]):
            start += 1
            continue
        end = start
        while (
            end + 1 < len(result)
            and _in_run(result[end + 1])
            and result[end + 1].session_id == result[start].session_id
            and result[end + 1].open_time
            == result[end].open_time + timedelta(minutes=bar_minutes)
        ):
            end += 1
        if (end - start + 1) * bar_minutes >= threshold_minutes:
            for i in range(start, end + 1):
                result[i] = with_flag(result[i], "suspected_outage")
        start = end + 1
    return result


def _in_run(bar: Bar) -> bool:
    return bar.is_gap and "source_missing" not in bar.quality_flags


def month_bounds(month: str) -> tuple[datetime, datetime]:
    """UTC span of a New York calendar month (``YYYY-MM``), as HistData files cut it."""
    year, number = month_parts(month)
    following = (year + number // 12, number % 12 + 1)
    return (
        datetime(year, number, 1, tzinfo=NY).astimezone(UTC),
        datetime(*following, 1, tzinfo=NY).astimezone(UTC),
    )


def source_missing_bars(instrument: str, month: str, bar_minutes: int) -> Iterator[Bar]:
    """Gap rows flagged ``source_missing`` for every in-Session slot of a month."""
    begin, end = month_bounds(month)
    size = timedelta(minutes=bar_minutes)
    sunday = week_sunday(begin.astimezone(NY).date()) - timedelta(days=7)
    while True:
        opened = open_of(sunday)
        if opened >= end:
            return
        for slot in range(slots_per_session(bar_minutes)):
            start = opened + slot * size
            if begin <= start < end:
                yield gap_bar(instrument, start, opened, ("source_missing",))
        sunday += timedelta(days=7)
