"""Session geometry: the one place that knows when the FX week opens and closes.

A Session runs from Sunday 17:00 America/New_York (weekly open) to Friday
17:00 (weekly close), exactly 120 hours; the US clock change happens while
closed, so each Session has a single UTC offset.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SESSION_HOURS = 120
ROLLOVER = time(17, 0)


def slots_per_session(bar_minutes: int) -> int:
    return SESSION_HOURS * 60 // bar_minutes


def week_sunday(day: date) -> date:
    """The Sunday on or before ``day`` (a New York calendar date)."""
    return day - timedelta(days=(day.weekday() + 1) % 7)


def open_of(sunday: date) -> datetime:
    """UTC weekly open of the Session that starts on ``sunday`` (New York)."""
    return datetime.combine(sunday, ROLLOVER, tzinfo=NY).astimezone(UTC)


def session_of(ts: datetime) -> tuple[datetime, datetime] | None:
    """The (open, close) UTC pair of the Session containing ``ts``, if any."""
    utc = ts.astimezone(UTC)
    sunday = week_sunday(utc.astimezone(NY).date())
    opened = open_of(sunday)
    if utc < opened:
        sunday -= timedelta(days=7)
        opened = open_of(sunday)
    closed = open_of(sunday + timedelta(days=5))
    return (opened, closed) if utc < closed else None


def next_open(opened: datetime) -> datetime:
    """Open of the Session after the one that opened at ``opened``."""
    return open_of(opened.astimezone(NY).date() + timedelta(days=7))


def upcoming_open(ts: datetime) -> datetime:
    """The first weekly open at or after ``ts`` when ``ts`` is in the closure."""
    local = ts.astimezone(NY)
    return open_of(local.date() + timedelta(days=(6 - local.weekday()) % 7))


def is_rollover(slot_start: datetime) -> bool:
    """Whether the slot starts at a daily 17:00 New York rollover (Mon-Fri open)."""
    local = slot_start.astimezone(NY)
    return local.time() == ROLLOVER and local.weekday() < 4
