"""Live ingester core: a quote stream in, closed Bars with Revisions out.

Adapters speak a four-event interface (``Quote``, ``Heartbeat``,
``SourceDown``, ``SourceUp``). ``LiveIngester`` drives the same incremental
Bar builder and anomaly detector as backfill, in-process, and hands every Bar
to a sink callback together with its Revision (ADR 0003).
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from .anomaly import FlaggingBarBuilder
from .bars import Bar, Quote
from .config import Config
from .gaps import with_flag
from .rawlog import RawQuoteLog
from .report import (
    DROPPED_EXAMPLE_CAP,
    live_run_report,
    mark_stale_runs_crashed,
    resolved_config,
    write_live_report,
)
from .revisions import RevisionLog
from .session import next_open, session_of, upcoming_open


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


@dataclass(frozen=True)
class SourceInterval:
    """A span with no usable feed; ``end`` is None while it is still open."""

    start: datetime
    end: datetime | None
    reason: str  # "adapter" (reported by the adapter) or "between_runs"


def overlaps_interval(
    open_time: datetime,
    size: timedelta,
    start: datetime,
    end: datetime | None,
) -> bool:
    """Whether the slot at ``open_time`` overlaps ``[start, end)`` (None: open)."""
    return start < open_time + size and (end is None or end > open_time)


log = logging.getLogger(__name__)


@dataclass
class _ClosedSlot:
    """A closed, not yet sealed slot: what is needed to re-emit it as a Revision."""

    snapshot: FlaggingBarBuilder  # builder state just before this slot's quotes
    quotes: list[Quote] = field(default_factory=list)  # arrival order
    bar: Bar | None = None
    revision: int = 0


def _discard(bar: Bar, revision: int) -> None:
    """The default sink: Bars are still kept in ``LiveIngester.revisions``."""


class LiveIngester:
    """Turns one Instrument's quote stream into closed Bars at one bar size."""

    def __init__(
        self,
        config: Config,
        adapter: QuoteStream,
        *,
        sink: BarSink = _discard,
        bar_size: int | None = None,
        start: datetime | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        record: bool = True,
    ) -> None:
        self._record = record  # False: a rebuild from the raw log writes nothing
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
        self._closed: dict[datetime, _ClosedSlot] = {}  # unsealed, in slot order
        self._down_since: datetime | None = None
        self.late_dropped = 0
        self.dropped_out_of_session = 0
        self._rawlog = RawQuoteLog(
            config.paths.data_root, adapter.source, adapter.instrument
        )
        self._stopped = False
        self._config = config
        self._resolved = resolved_config(config)
        self._start = start
        self._started_at: datetime | None = None
        self._run_id = ""
        self._bars_by_revision: Counter[int] = Counter()
        self._gap_bars = 0
        self._late_accepted = 0
        self._late_lags: list[float] = []
        self._dropped_examples: list[dict[str, str]] = []
        self._catch_ups: list[dict[str, object]] = []
        self.outages: list[SourceInterval] = []
        self._first_slot: datetime | None = None
        if start is not None:
            self._first_slot = self._cursor = self._slot_at_or_after(start)
            self._builder.start_at(self._first_slot)

    def _slot_at_or_after(self, start: datetime) -> datetime:
        """The first slot opening at or after ``start``, skipping the closure."""
        start = start.astimezone(UTC)
        session = session_of(start)
        if session is not None:
            first = session[0] + -((session[0] - start) // self._size) * self._size
            if first < session[1]:
                return first
            return next_open(session[0])
        return upcoming_open(start)

    def run(self) -> None:
        """Consume the adapter's stream to its end, emitting Bars to the sink.

        Only slots the watermark has closed are emitted: the end of a stream is
        not time passing, so it never invents the rest of the Session.
        """
        self._begin_run()
        try:
            for event in self._adapter.events():
                self._handle(event)
                if self._stopped:
                    self._report("stopped")  # clean shutdown: closed slots only
                    return
            self._report("finished")
        finally:
            self._rawlog.close()

    def _begin_run(self) -> None:
        """Mark any crashed earlier run, record its downtime, open this report."""
        self._started_at = self._start or self._clock()
        self._run_id = f"{self._started_at:%Y%m%dT%H%M%S%f}Z"
        for gap in mark_stale_runs_crashed(
            self._config.paths.data_root,
            self._adapter.source,
            self._adapter.instrument,
            self._started_at,
        ):
            self.outages.append(
                SourceInterval(
                    datetime.fromisoformat(gap["start"]),
                    datetime.fromisoformat(gap["end"]),
                    "between_runs",
                )
            )
        self._report("running")

    def _report(self, status: str) -> None:
        if not self._record:
            return
        assert self._started_at is not None
        write_live_report(
            self._config.paths.data_root,
            self._adapter.source,
            self._adapter.instrument,
            live_run_report(
                run_id=self._run_id,
                status=status,
                source=self._adapter.source,
                instrument=self._adapter.instrument,
                bar_size=self._size // timedelta(minutes=1),
                started_at=self._started_at,
                ended_at=None if status == "running" else self._clock(),
                last_watermark=self._watermark,
                source_down=[
                    {
                        "start": i.start.isoformat(),
                        "end": i.end.isoformat() if i.end else None,
                        "reason": i.reason,
                    }
                    for i in self.outages
                ],
                bars_by_revision=self._bars_by_revision,
                gap_bars=self._gap_bars,
                late_accepted=self._late_accepted,
                dropped_after_seal=self.late_dropped,
                out_of_session=self.dropped_out_of_session,
                late_lags=self._late_lags,
                catch_ups=self._catch_ups,
                dropped_examples=self._dropped_examples,
                resolved=self._resolved,
            ),
        )

    def replay_sealed(self, events: Iterable[Event]) -> list[Bar]:
        """Rebuild from logged events (arrival order); return the sealed Bars.

        Only for an ingester made with ``record=False``. Sealed means past its
        repair window: open or repairable slots are left out, and nothing is
        flushed. Each Bar is its slot's latest Revision.
        """
        assert not self._record
        for event in events:
            self._handle(event)
        return [b for b in self.revisions.latest() if b.open_time not in self._closed]

    def stop(self) -> None:
        """Ask a running ingester to shut down cleanly after the current event.

        Nothing beyond the slots the watermark has already closed is flushed.
        """
        self._stopped = True

    def _handle(self, event: Event) -> None:
        if isinstance(event, Quote):
            self._on_quote(event)
        elif isinstance(event, Heartbeat):
            self._advance_watermark(event.ts)
        elif isinstance(event, SourceDown):
            self._down_since = event.start
            self.outages.append(SourceInterval(event.start, None, "adapter"))
        elif isinstance(event, SourceUp):
            self._end_outage(event.ts)
            self._advance_watermark(event.ts)
            self._catch_up(event.ts)

    def _end_outage(self, ts: datetime) -> None:
        if self.outages and self.outages[-1].end is None:
            last = self.outages[-1]
            self.outages[-1] = SourceInterval(last.start, ts, last.reason)

    def _catch_up(self, up: datetime) -> None:
        """Feed the quotes missed while down through the same path as late ones."""
        down, self._down_since = self._down_since, None
        fetch = getattr(self._adapter, "fetch_quotes", None)
        if down is None or fetch is None:
            return
        accepted, dropped = self._late_accepted, self.late_dropped
        quotes = list(fetch(down, up))
        for quote in quotes:
            self._on_quote(quote)
        self._catch_ups.append(
            {
                "start": down.isoformat(),
                "end": up.isoformat(),
                "quotes": len(quotes),
                "accepted_as_revisions": self._late_accepted - accepted,
                "dropped_after_seal": self.late_dropped - dropped,
            }
        )

    def _on_quote(self, quote: Quote) -> None:
        if self._record:
            self._rawlog.append(quote, self._clock())
        session = session_of(quote.ts)
        if session is None:
            self.dropped_out_of_session += 1
        else:
            if self._cursor is None:
                self._cursor = session[0]
            slot = self._slot_of(quote.ts)
            if self._first_slot is not None and slot < self._first_slot:
                pass  # before the start: not part of this run's grid
            elif slot >= self._cursor:
                self._pending.setdefault(slot, []).append(quote)
            elif slot in self._closed:
                self._closed[slot].quotes.append(quote)
                self._late_accepted += 1
                self._late_lags.append(self._lag(quote))
                self._revise(slot)
            else:
                self.late_dropped += 1
                if len(self._dropped_examples) < DROPPED_EXAMPLE_CAP:
                    self._dropped_examples.append(
                        {
                            "source_ts": quote.ts.astimezone(UTC).isoformat(),
                            "bid": str(quote.bid),
                            "ask": str(quote.ask),
                            "lag_seconds": str(self._lag(quote)),
                        }
                    )
                log.warning(
                    "dropped quote at %s: its Bar is sealed", quote.ts.isoformat()
                )
        self._advance_watermark(quote.ts)

    def _lag(self, quote: Quote) -> float:
        """How far the watermark had moved past a late quote's own time."""
        assert self._watermark is not None
        return max(0.0, (self._watermark - quote.ts).total_seconds())

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
        session = session_of(ts)
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

    def _close(self, slot: datetime, *, repairable: bool = True) -> None:
        snapshot = copy.deepcopy(self._builder) if repairable else self._builder
        closed = _ClosedSlot(snapshot, self._pending.pop(slot, []))
        closed.bar = self._build(slot, closed)
        if repairable:
            self._closed[slot] = closed
        self._emit(closed.bar, 0)
        self._cursor = self._after(slot)
        if repairable:
            self._report("running")

    def _emit(self, bar: Bar, revision: int) -> None:
        self.revisions.append(bar, revision, self._clock())
        self._bars_by_revision[revision] += 1
        self._gap_bars += bar.is_gap
        self._sink(bar, revision)

    def _build(self, slot: datetime, closed: _ClosedSlot) -> Bar:
        """Feed the slot's quotes (timestamp, then arrival) and close the slot."""
        for quote in sorted(closed.quotes, key=lambda x: x.ts):
            self._builder.push(quote)
        (bar,) = self._builder.advance(slot + self._size)
        return self._flag_outage(bar)

    def _flag_outage(self, bar: Bar) -> Bar:
        """A Gap Bar overlapping a source-down interval is a suspected outage.

        Applies regardless of the run-length rule; priced Bars stay unflagged.
        """
        if not bar.is_gap:
            return bar
        if any(
            overlaps_interval(bar.open_time, self._size, i.start, i.end)
            for i in self.outages
        ):
            return with_flag(bar, "suspected_outage")
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
        session = session_of(slot)
        assert session is not None
        nxt = slot + self._size
        return nxt if nxt < session[1] else next_open(session[0])


__all__ = [
    "BarSink",
    "Event",
    "Heartbeat",
    "LiveIngester",
    "QuoteStream",
    "SourceDown",
    "SourceUp",
    "overlaps_interval",
]
