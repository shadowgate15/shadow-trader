"""Append-only record of every Revision a live run emitted (ADR 0003).

Earlier Revisions are never overwritten or deleted. Two views are derived:
the as-seen view (the Revision current at a moment, for the decision audit
trail) and the latest view (highest Revision per slot).
"""

from __future__ import annotations

from datetime import datetime

from .bars import Bar


class RevisionLog:
    def __init__(self) -> None:
        self._entries: list[tuple[int, Bar, datetime]] = []  # append order
        self._next: dict[datetime, int] = {}

    def append(self, bar: Bar, revision: int, emitted_at: datetime) -> None:
        """Record a Revision; it must be the next one for its slot."""
        expected = self._next.get(bar.open_time, 0)
        if revision != expected:
            raise ValueError(
                f"slot {bar.open_time.isoformat()} expects revision {expected}, "
                f"got {revision}"
            )
        self._next[bar.open_time] = revision + 1
        self._entries.append((revision, bar, emitted_at))

    def history(self, open_time: datetime) -> list[tuple[int, Bar, datetime]]:
        """Every Revision of one slot as (revision, Bar, emitted_at), oldest first."""
        return [e for e in self._entries if e[1].open_time == open_time]

    def as_seen(self, at: datetime) -> list[Bar]:
        """The Bars as they stood at ``at``: per slot, the Revision then current."""
        current: dict[datetime, Bar] = {}
        for _, bar, emitted_at in self._entries:
            if emitted_at <= at:
                current[bar.open_time] = bar
        return [current[t] for t in sorted(current)]

    def latest(self) -> list[Bar]:
        """The highest Revision of every slot, in time order."""
        current = {bar.open_time: bar for _, bar, _ in self._entries}
        return [current[t] for t in sorted(current)]


__all__ = ["RevisionLog"]
