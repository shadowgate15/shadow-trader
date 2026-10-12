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
from datetime import UTC, datetime, time, timedelta

from .anomaly import VOCABULARY
from .bars import NY, SESSION_HOURS, Bar

_ROLLOVER = time(17, 0)


def _with_flag(bar: Bar, flag: str) -> Bar:
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
                result[i] = _with_flag(result[i], "suspected_outage")
        start = end + 1
    return result


def _in_run(bar: Bar) -> bool:
    return bar.is_gap and "source_missing" not in bar.quality_flags


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    """UTC span of a New York calendar month (``YYYY-MM``), as HistData files cut it."""
    year, number = int(month[:4]), int(month[5:])
    following = (year + number // 12, number % 12 + 1)
    return (
        datetime(year, number, 1, tzinfo=NY).astimezone(UTC),
        datetime(*following, 1, tzinfo=NY).astimezone(UTC),
    )


def source_missing_bars(instrument: str, month: str, bar_minutes: int) -> Iterator[Bar]:
    """Gap rows flagged ``source_missing`` for every in-Session slot of a month."""
    begin, end = _month_bounds(month)
    size = timedelta(minutes=bar_minutes)
    sunday = begin.astimezone(NY).date()
    sunday -= timedelta(days=(sunday.weekday() + 1) % 7 + 7)
    while True:
        opened = datetime.combine(sunday, _ROLLOVER, tzinfo=NY).astimezone(UTC)
        if opened >= end:
            return
        for slot in range(SESSION_HOURS * 60 // bar_minutes):
            start = opened + slot * size
            if begin <= start < end:
                local = start.astimezone(NY)
                yield Bar(
                    instrument=instrument,
                    open_time=start,
                    open=None,
                    high=None,
                    low=None,
                    close=None,
                    spread_close=None,
                    spread_mean=None,
                    tick_count=None,
                    is_gap=True,
                    session_id=opened,
                    is_rollover=local.time() == _ROLLOVER and local.weekday() < 4,
                    quality_flags=("source_missing",),
                )
        sunday += timedelta(days=7)
