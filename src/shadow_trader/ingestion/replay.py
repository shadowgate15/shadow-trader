"""Replay adapter: feeds recorded ticks through the live path, no network."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta

from .backfill import parse_ticks, read_raw_month
from .config import Config, last_complete_month
from .dataset import RawCache
from .gaps import month_bounds
from .live import Event, Heartbeat
from .session import session_of
from .sources import months_between


def live_adapter(config: Config, instrument: str) -> ReplayAdapter:
    """The adapter named by ``[live].adapter`` (v1 supports only ``replay``)."""
    return ReplayAdapter.from_raw_cache(config, instrument)


class ReplayAdapter:
    """The only v1 live adapter. Replays an event iterable in order."""

    def __init__(
        self,
        events: Iterable[Event],
        *,
        instrument: str,
        source: str = "replay",
    ) -> None:
        self._events = events
        self.instrument = instrument
        self.source = source

    def events(self) -> Iterator[Event]:
        yield from self._events

    @classmethod
    def from_raw_cache(
        cls, config: Config, instrument: str, *, now: datetime | None = None
    ) -> ReplayAdapter:
        """Replay the configured range of HistData ticks from the raw cache.

        Reads only the write-once raw cache, never the network. The range ends
        at the configured end month (default: the last complete UTC month before
        ``now``), and a closing heartbeat at the end of the last tick's Session
        (clipped to the range) lets the ingester close the trailing slots.
        """
        return cls(
            _cache_events(config, instrument, now or datetime.now(UTC)),
            instrument=instrument,
        )


def _cache_events(config: Config, instrument: str, now: datetime) -> Iterator[Event]:
    cache = RawCache(config.paths.data_root)
    build = config.build
    scale = config.instruments[instrument].price_scale
    end = build.end or last_complete_month(now)
    last: datetime | None = None
    grace = timedelta(seconds=config.live.grace_seconds)
    for period in months_between(build.start, end):
        text = read_raw_month(cache, instrument, period)
        for quote in parse_ticks(text, f"{instrument}/{period}", scale):
            last = quote.ts
            yield quote
    if last is not None and (session := session_of(last)) is not None:
        yield Heartbeat(min(session[1], month_bounds(end)[1]) + grace)
