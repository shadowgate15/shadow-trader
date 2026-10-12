"""Replay adapter: feeds recorded ticks through the live path, no network."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .backfill import parse_ticks, read_raw_month
from .config import Config
from .dataset import RawCache
from .live import Event
from .sources import months_between


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
    def from_raw_cache(cls, config: Config, instrument: str) -> ReplayAdapter:
        """Replay the configured range of HistData ticks from the raw cache.

        Reads only the write-once raw cache, never the network.
        """
        return cls(_cache_events(config, instrument), instrument=instrument)


def _cache_events(config: Config, instrument: str) -> Iterator[Event]:
    cache = RawCache(config.paths.data_root)
    build = config.build
    scale = config.instruments[instrument].price_scale
    for period in months_between(build.start, build.end or build.start):
        text = read_raw_month(cache, instrument, period)
        yield from parse_ticks(text, f"{instrument}/{period}", scale)
