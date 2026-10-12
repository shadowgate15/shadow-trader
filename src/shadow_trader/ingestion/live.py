"""Live ingester core: a quote stream in, closed Bars with Revisions out.

Adapters speak a four-event interface (``Quote``, ``Heartbeat``,
``SourceDown``, ``SourceUp``). ``LiveIngester`` drives the same incremental
Bar builder and anomaly detector as backfill, in-process, and hands every Bar
to a sink callback together with its Revision (ADR 0003).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from .anomaly import FlaggingBarBuilder
from .bars import Bar, Quote, _next_open, _session_of
from .config import Config
from .revisions import RevisionLog


@dataclass(frozen=True)
class Heartbeat:
    """The source is alive at ``ts``; advances the watermark without a tick."""

    ts: datetime


@dataclass(frozen=True)
class SourceDown:
    """The source went away at ``start``."""

    start: datetime


@dataclass(frozen=True)
class SourceUp:
    """The source came back at ``ts``."""

    ts: datetime


Event = Quote | Heartbeat | SourceDown | SourceUp
BarSink = Callable[[Bar, int], None]


class QuoteStream(Protocol):
    """What a live source adapter provides.

    ``events`` yields source-UTC-stamped events, non-decreasing in time except
    for catch-up output. An adapter may also offer
    ``fetch_quotes(start, end) -> Iterable[Quote]`` to repair missed quotes.
    """

    source: str
    instrument: str

    def events(self) -> Iterator[Event]: ...


log = logging.getLogger(__name__)


@dataclass
class _ClosedSlot:
    """A closed, not yet sealed slot: what is needed to re-emit it as a Revision."""

    snapshot: FlaggingBarBuilder  # builder state just before this slot's quotes
    quotes: list[Quote] = field(default_factory=list)  # arrival order
    bar: Bar | None = None
    revision: int = 0


class LiveIngester:
    """Turns one Instrument's quote stream into closed Bars at one bar size."""

    def __init__(
        self,
        config: Config,
        adapter: QuoteStream,
        *,
        sink: BarSink,
        bar_size: int | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._adapter = adapter
        self._sink = sink
        self._clock = clock
        self.revisions = RevisionLog()
        minutes = bar_size if bar_size is not None else config.build.bar_sizes[0]
        instrument = adapter.instrument
        self._size = timedelta(minutes=minutes)
        self._grace = timedelta(seconds=config.live.grace_seconds)
        self._repair = timedelta(minutes=config.live.repair_window_minutes)
        self._builder = FlaggingBarBuilder(
            instrument,
            minutes,
            config.instruments[instrument].price_scale,
            config.anomaly_rule,
        )
        self._watermark: datetime | None = None
        self._cursor: datetime | None = None  # start of the next slot to close
        self._pending: dict[datetime, list[Quote]] = {}
        self._last_session_close: datetime | None = None
        self._closed: dict[datetime, _ClosedSlot] = {}  # unsealed, in slot order
        self._down_since: datetime | None = None
        self.late_dropped = 0
        self.dropped_out_of_session = 0

    def run(self) -> None:
        """Consume the adapter's stream to its end, emitting Bars to the sink."""
        for event in self._adapter.events():
            self._handle(event)
        self._finish()

    def _handle(self, event: Event) -> None:
        if isinstance(event, Quote):
            self._on_quote(event)
        elif isinstance(event, Heartbeat):
            self._advance_watermark(event.ts)
        elif isinstance(event, SourceDown):
            self._down_since = event.start
        elif isinstance(event, SourceUp):
            self._advance_watermark(event.ts)
            self._catch_up(event.ts)

    def _catch_up(self, up: datetime) -> None:
        """Feed the quotes missed while down through the same path as late ones."""
        down, self._down_since = self._down_since, None
        fetch = getattr(self._adapter, "fetch_quotes", None)
        if down is None or fetch is None:
            return
        for quote in fetch(down, up):
            self._on_quote(quote)

    def _on_quote(self, quote: Quote) -> None:
        session = _session_of(quote.ts)
        if session is None:
            self.dropped_out_of_session += 1
        else:
            if self._cursor is None:
                self._cursor = session[0]
            self._last_session_close = session[1]
            slot = self._slot_of(quote.ts)
            if slot >= self._cursor:
                self._pending.setdefault(slot, []).append(quote)
            elif slot in self._closed:
                self._closed[slot].quotes.append(quote)
                self._revise(slot)
            else:
                self.late_dropped += 1
                log.warning(
                    "dropped quote at %s: its Bar is sealed", quote.ts.isoformat()
                )
        self._advance_watermark(quote.ts)

    def _advance_watermark(self, ts: datetime) -> None:
        ts = ts.astimezone(UTC)
        if self._watermark is None or ts > self._watermark:
            self._watermark = ts
        self._close_ready()
        self._seal()

    def _seal(self) -> None:
        """Seal slots whose repair window the watermark has passed."""
        assert self._watermark is not None
        while self._closed:
            slot = next(iter(self._closed))
            if slot + self._size + self._repair > self._watermark:
                return
            del self._closed[slot]

    def _slot_of(self, ts: datetime) -> datetime:
        session = _session_of(ts)
        assert session is not None
        utc = ts.astimezone(UTC)
        return session[0] + (utc - session[0]) // self._size * self._size

    def _close_ready(self) -> None:
        """Close, in order, every slot whose end plus grace the watermark passed."""
        assert self._watermark is not None
        while self._cursor is not None:
            slot = self._cursor
            end = slot + self._size
            if end + self._grace > self._watermark:
                return
            self._close(slot)

    def _finish(self) -> None:
        """End of stream: close the rest of the last Session, as a backfill does."""
        while (
            self._cursor is not None
            and self._last_session_close is not None
            and self._cursor < self._last_session_close
        ):
            self._close(self._cursor, repairable=False)

    def _close(self, slot: datetime, *, repairable: bool = True) -> None:
        snapshot = copy.deepcopy(self._builder) if repairable else self._builder
        closed = _ClosedSlot(snapshot, self._pending.pop(slot, []))
        closed.bar = self._build(slot, closed)
        if repairable:
            self._closed[slot] = closed
        self._emit(closed.bar, 0)
        self._cursor = self._after(slot)

    def _emit(self, bar: Bar, revision: int) -> None:
        self.revisions.append(bar, revision, self._clock())
        self._sink(bar, revision)

    def _build(self, slot: datetime, closed: _ClosedSlot) -> Bar:
        """Feed the slot's quotes (timestamp, then arrival) and close the slot."""
        for quote in sorted(closed.quotes, key=lambda x: x.ts):
            self._builder.push(quote)
        (bar,) = self._builder.advance(slot + self._size)
        return bar

    def _revise(self, slot: datetime) -> None:
        """Replay from ``slot`` on; re-emit each Bar that changed as the next Revision.

        A late quote can also move later slots through the look-back anomaly
        state, so every closed slot from here on is rebuilt and compared.
        """
        self._builder = copy.deepcopy(self._closed[slot].snapshot)
        for t, closed in self._closed.items():
            if t < slot:
                continue
            closed.snapshot = copy.deepcopy(self._builder)
            bar = self._build(t, closed)
            if bar != closed.bar:
                closed.bar = bar
                closed.revision += 1
                self._emit(bar, closed.revision)

    def _after(self, slot: datetime) -> datetime:
        session = _session_of(slot)
        assert session is not None
        nxt = slot + self._size
        return nxt if nxt < session[1] else _next_open(session[0])


__all__ = [
    "BarSink",
    "Event",
    "Heartbeat",
    "LiveIngester",
    "QuoteStream",
    "SourceDown",
    "SourceUp",
]
