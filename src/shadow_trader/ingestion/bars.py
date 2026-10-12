"""Incremental Bar builder: push a quote, pull closed Bars, flush at the end.

One core serves backfill (feed raw ticks, then ``flush``) and, later, live
ingestion. It builds the dense Session grid (Sunday 17:00 to Friday 17:00 New
York, 120 h), aggregates in scaled integers and converts to float64 once per
value. It never cleans: every in-Session tick enters the statistics. A slot
with no ticks inside a Session is a Gap row, never carried forward.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .config import ConfigError, validate_bar_sizes

NY = ZoneInfo("America/New_York")
SESSION_HOURS = 120
_ROLLOVER = time(17, 0)


@dataclass(frozen=True)
class Quote:
    """One bid/ask tick. ``ts`` is timezone-aware."""

    ts: datetime
    bid: Decimal
    ask: Decimal


@dataclass(frozen=True)
class Bar:
    """One row of the Data contract schema (no ``revision``; Datasets hold latest)."""

    instrument: str
    open_time: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    spread_close: float | None
    spread_mean: float | None
    tick_count: int | None
    is_gap: bool
    session_id: datetime
    is_rollover: bool
    quality_flags: tuple[str, ...] = ()


def _session_of(ts: datetime) -> tuple[datetime, datetime] | None:
    """The (open, close) UTC pair of the Session containing ``ts``, if any."""
    utc = ts.astimezone(UTC)
    local = utc.astimezone(NY)
    sunday = local.date() - timedelta(days=(local.weekday() + 1) % 7)
    opened = datetime.combine(sunday, _ROLLOVER, tzinfo=NY).astimezone(UTC)
    if utc < opened:
        sunday -= timedelta(days=7)
        opened = datetime.combine(sunday, _ROLLOVER, tzinfo=NY).astimezone(UTC)
    closed = datetime.combine(sunday + timedelta(days=5), _ROLLOVER, tzinfo=NY)
    closed = closed.astimezone(UTC)
    return (opened, closed) if utc < closed else None


def _next_open(opened: datetime) -> datetime:
    sunday = opened.astimezone(NY).date() + timedelta(days=7)
    return datetime.combine(sunday, _ROLLOVER, tzinfo=NY).astimezone(UTC)


class BarBuilder:
    """Builds the Bars of one Instrument at one bar size.

    Ticks arrive in non-decreasing slot order; within the open slot they may
    arrive out of order and are ordered by timestamp, then arrival. A tick for
    a slot that has already closed is rejected. The grid starts at the open of
    the first Session that receives a tick.
    """

    def __init__(self, instrument: str, bar_minutes: int, price_scale: int) -> None:
        validate_bar_sizes((bar_minutes,))
        if price_scale < 0:
            raise ConfigError("price_scale must not be negative")
        self.instrument = instrument
        self._size = timedelta(minutes=bar_minutes)
        self._slots = SESSION_HOURS * 60 // bar_minutes
        self._factor = 10**price_scale
        self._open: datetime | None = None  # open of the cursor Session
        self._slot = 0  # next slot to close
        self._pending: list[tuple[datetime, int, int]] = []
        self.dropped_out_of_session = 0

    def push(self, quote: Quote) -> list[Bar]:
        """Add one quote; return the Bars it caused to close, in order."""
        if quote.ts.tzinfo is None:
            raise ValueError("quote timestamp must be timezone-aware")
        bid = self._scaled(quote.bid)
        ask = self._scaled(quote.ask)
        session = _session_of(quote.ts)
        if session is None:
            self.dropped_out_of_session += 1
            return []
        opened = session[0]
        slot = int((quote.ts.astimezone(UTC) - opened) // self._size)
        out = self._move_to(opened, slot, quote.ts)
        self._pending.append((quote.ts.astimezone(UTC), bid, ask))
        return out

    def start_at(self, slot_start: datetime) -> None:
        """Begin the grid at the slot opening at ``slot_start`` (mid-Session start).

        Earlier slots of the Session are never emitted, so they are not Gaps.
        Only valid before any quote or clock has moved the builder.
        """
        session = _session_of(slot_start)
        if self._open is not None or session is None:
            raise ValueError("start_at needs a fresh builder and an in-Session slot")
        self._open = session[0]
        self._slot = int((slot_start.astimezone(UTC) - session[0]) // self._size)

    def advance(self, ts: datetime) -> list[Bar]:
        """Close every slot that ends at or before ``ts``; no tick is added.

        This is how a clock (a heartbeat or watermark) closes empty slots as
        Gaps. A ``ts`` in the weekly closure closes the rest of the Session.
        """
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        session = _session_of(ts)
        if session is None:
            out: list[Bar] = []
            if self._open is not None:
                self._close_through(self._slots, out)
            return out
        opened = session[0]
        slot = int((ts.astimezone(UTC) - opened) // self._size)
        if self._open is not None and (opened, slot) < (self._open, self._slot):
            return []  # a clock running behind the grid is a no-op
        return self._move_to(opened, slot, ts)

    def _move_to(self, opened: datetime, slot: int, ts: datetime) -> list[Bar]:
        out: list[Bar] = []
        if self._open is None:
            self._open = opened
        elif (opened, slot) < (self._open, self._slot):
            raise ValueError(
                f"quote at {ts.isoformat()} falls in an already closed slot"
            )
        while self._open != opened:
            self._close_through(self._slots, out)
            self._open = _next_open(self._open)
            self._slot = 0
        self._close_through(slot, out)
        return out

    def flush(self) -> list[Bar]:
        """Close the open slot and the rest of the current Session."""
        out: list[Bar] = []
        if self._open is not None and self._slot < self._slots:
            self._close_through(self._slots, out)
        return out

    def _scaled(self, price: Decimal) -> int:
        scaled = price * self._factor
        if scaled != scaled.to_integral_value():
            raise ValueError(f"price {price} has more precision than the price scale")
        return int(scaled)

    def _close_through(self, slot: int, out: list[Bar]) -> None:
        while self._slot < slot:
            out.append(self._emit())

    def _emit(self) -> Bar:
        assert self._open is not None
        start = self._open + self._slot * self._size
        local = start.astimezone(NY)
        rollover = local.time() == _ROLLOVER and local.weekday() < 4
        ticks = sorted(self._pending, key=lambda t: t[0])  # stable: arrival ties
        self._pending = []
        self._slot += 1
        if not ticks:
            return Bar(
                open=None,
                high=None,
                low=None,
                close=None,
                spread_close=None,
                spread_mean=None,
                tick_count=None,
                is_gap=True,
                instrument=self.instrument,
                open_time=start,
                session_id=self._open,
                is_rollover=rollover,
            )
        bids = [b for _, b, _ in ticks]
        spreads = [a - b for _, b, a in ticks]
        return Bar(
            open=bids[0] / self._factor,
            high=max(bids) / self._factor,
            low=min(bids) / self._factor,
            close=bids[-1] / self._factor,
            spread_close=spreads[-1] / self._factor,
            spread_mean=sum(spreads) / (len(spreads) * self._factor),
            tick_count=len(ticks),
            is_gap=False,
            instrument=self.instrument,
            open_time=start,
            session_id=self._open,
            is_rollover=rollover,
        )


def build_bars(
    instrument: str, quotes: Iterable[Quote], bar_minutes: int, price_scale: int
) -> tuple[list[Bar], int]:
    """Batch helper: feed all quotes, flush; return (Bars, dropped tick count)."""
    builder = BarBuilder(instrument, bar_minutes, price_scale)
    bars: list[Bar] = []
    for quote in quotes:
        bars.extend(builder.push(quote))
    bars.extend(builder.flush())
    return bars, builder.dropped_out_of_session


__all__ = ["Bar", "BarBuilder", "Quote", "build_bars"]
