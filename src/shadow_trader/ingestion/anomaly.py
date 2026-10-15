"""Anomaly detector: quality flags on Bars, never altering or dropping values.

A pass separate from the Bar builder over the same ticks, in two tiers. Tick
rules look at each quote as it arrives; the Bar rule looks at each closed Bar.
Every rule is look-back-only and its state carries across Sessions, so backfill
and live flag the same Bar the same way. Baseline-dependent rules stay silent
until their baseline is warm.

``FlaggingBarBuilder`` is the incremental entry point: it has the same
``push``/``flush`` shape as ``BarBuilder`` and returns Bars with
``quality_flags`` filled in.
"""

from __future__ import annotations

import dataclasses
import itertools
import statistics
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from .bars import Bar, BarBuilder, Quote
from .config import AnomalyRuleConfig, pip_size
from .session import session_of

# Closed, additive-only vocabulary. Gap-only flags are written by Gap handling.
TICK_FLAGS = ("crossed_quote", "zero_spread", "price_spike", "stale_quote")
BAR_FLAGS = ("spread_spike",)
GAP_FLAGS = ("suspected_outage", "source_missing")
VOCABULARY = (*TICK_FLAGS, *BAR_FLAGS, *GAP_FLAGS)
# Flags that feed the alert warning; zero_spread and spread_spike are excluded.
ALERT_FLAGS = ("price_spike", "crossed_quote", "stale_quote")

_MAD_TO_SIGMA = 1.4826  # scales MAD to a standard deviation for normal data


@dataclasses.dataclass(frozen=True)
class AnomalyWarning:
    """A structured warning entry (closed ``code``, human ``message``)."""

    code: str
    message: str


@dataclasses.dataclass(frozen=True)
class FlagSummary:
    """Per-flag counts (Bars and ticks) and the alert-fraction check."""

    bar_counts: dict[str, int]
    tick_counts: dict[str, int]
    non_gap_bars: int
    alerted_bars: int
    warnings: list[AnomalyWarning]


def summarize_flags(
    bars: Sequence[Bar], tick_counts: Mapping[str, int], alert_fraction: float
) -> FlagSummary:
    """Count flags and warn if too many non-Gap Bars carry an alert flag.

    Only ``price_spike``, ``crossed_quote`` and ``stale_quote`` count toward the
    alert; ``zero_spread`` and ``spread_spike`` are reported but never warn.
    """
    counts: Counter[str] = Counter(f for b in bars for f in b.quality_flags)
    priced = [b for b in bars if not b.is_gap]
    alerted = sum(any(f in ALERT_FLAGS for f in b.quality_flags) for b in priced)
    warnings = []
    if priced and alerted / len(priced) > alert_fraction:
        warnings.append(
            AnomalyWarning(
                "anomaly_alert_fraction_exceeded",
                f"{alerted} of {len(priced)} non-Gap Bars carry "
                f"{'/'.join(ALERT_FLAGS)} (> {alert_fraction:.2%})",
            )
        )
    return FlagSummary(
        bar_counts=dict(sorted(counts.items())),
        tick_counts=dict(sorted(tick_counts.items())),
        non_gap_bars=len(priced),
        alerted_bars=alerted,
        warnings=warnings,
    )


class TickDetector:
    """Tick-tier rules over one Instrument's in-Session quotes."""

    def __init__(
        self, instrument: str, rules: AnomalyRuleConfig, price_scale: int
    ) -> None:
        self._rules = rules
        self._factor = 10**price_scale
        self._pip = self._factor * pip_size(instrument)
        self._bids: deque[int] = deque(maxlen=rules.price_spike_window_ticks)
        self._run_quote: tuple[Decimal, Decimal] | None = None
        self._run_ticks = 0
        self._run_start: datetime | None = None

    def observe(self, quote: Quote) -> set[str]:
        """Return the tick flags this quote trips."""
        flags: set[str] = set()
        if self._rules.crossed_quote and quote.ask < quote.bid:
            flags.add("crossed_quote")
        if self._rules.zero_spread and quote.ask == quote.bid:
            flags.add("zero_spread")
        bid = int(quote.bid * self._factor)
        if self._rules.price_spike and self._is_price_spike(bid):
            flags.add("price_spike")
        self._bids.append(bid)
        if self._rules.stale_quote and self._is_stale(quote):
            flags.add("stale_quote")
        return flags

    def _is_stale(self, quote: Quote) -> bool:
        """Part of an identical-quote run that has reached both thresholds."""
        if (quote.bid, quote.ask) != self._run_quote:
            self._run_quote = (quote.bid, quote.ask)
            self._run_ticks = 0
            self._run_start = quote.ts
        self._run_ticks += 1
        assert self._run_start is not None
        return self._run_ticks >= self._rules.stale_quote_min_ticks and (
            quote.ts - self._run_start
            >= timedelta(minutes=self._rules.stale_quote_min_minutes)
        )

    def _is_price_spike(self, bid: int) -> bool:
        """Bid far from the median of the previous N ticks (warm windows only)."""
        window = self._bids
        if len(window) < (window.maxlen or 0) or len(window) < 2:
            return False
        changes = [b - a for a, b in itertools.pairwise(window)]
        centre = statistics.median(changes)
        scale = _MAD_TO_SIGMA * statistics.median(abs(c - centre) for c in changes)
        floor = float(self._rules.price_spike_floor_pips) * float(self._pip)
        limit = max(self._rules.price_spike_k * scale, floor)
        return abs(bid - statistics.median(window)) > limit


class SpreadSpikeDetector:
    """Bar-tier rule: ``spread_mean`` far above the recent median spread.

    The baseline is the median ``spread_mean`` of non-Gap Bars whose
    ``open_time`` falls in the lookback duration before the Bar, so it does not
    depend on bar size. It is warm once a full lookback has elapsed since the
    first non-Gap Bar seen; until then the rule is silent. Flagged Bars still
    join the baseline (no cleaning).
    """

    def __init__(self, instrument: str, rules: AnomalyRuleConfig) -> None:
        self._rules = rules
        self._lookback = timedelta(hours=rules.spread_spike_lookback_hours)
        self._pip = float(pip_size(instrument))
        self._history: deque[tuple[datetime, float]] = deque()
        self._first: datetime | None = None

    def observe(self, bar: Bar) -> bool:
        """Return whether this closed Bar is a spread spike, then learn from it."""
        if bar.is_gap or bar.spread_mean is None:
            return False
        while self._history and self._history[0][0] < bar.open_time - self._lookback:
            self._history.popleft()
        if self._first is None:
            self._first = bar.open_time
        spike = False
        warm = bar.open_time - self._first >= self._lookback
        if self._rules.spread_spike and warm and self._history:
            median = statistics.median(s for _, s in self._history)
            floor = self._rules.spread_spike_floor_pips * self._pip
            limit = max(self._rules.spread_spike_k * median, median + floor)
            spike = bar.spread_mean > limit
        self._history.append((bar.open_time, bar.spread_mean))
        return spike


class FlaggingBarBuilder:
    """A ``BarBuilder`` whose closed Bars carry anomaly flags."""

    def __init__(
        self,
        instrument: str,
        bar_minutes: int,
        price_scale: int,
        rules: AnomalyRuleConfig,
    ) -> None:
        self._builder = BarBuilder(instrument, bar_minutes, price_scale)
        self._ticks = TickDetector(instrument, rules, price_scale)
        self._spreads = SpreadSpikeDetector(instrument, rules)
        self._size = timedelta(minutes=bar_minutes)
        self._slot_flags: dict[datetime, set[str]] = {}
        self.tick_flag_counts: Counter[str] = Counter()

    @property
    def dropped_out_of_session(self) -> int:
        return self._builder.dropped_out_of_session

    def push(self, quote: Quote) -> list[Bar]:
        if session_of(quote.ts) is not None:
            tick_flags = self._ticks.observe(quote)
            if tick_flags:
                self._slot_flags.setdefault(self._slot_of(quote.ts), set()).update(
                    tick_flags
                )
                self.tick_flag_counts.update(tick_flags)
        return [self._flag(bar) for bar in self._builder.push(quote)]

    def start_at(self, slot_start: datetime) -> None:
        self._builder.start_at(slot_start)

    def advance(self, ts: datetime) -> list[Bar]:
        """Close slots ending at or before ``ts`` (a clock, not a tick)."""
        return [self._flag(bar) for bar in self._builder.advance(ts)]

    def flush(self) -> list[Bar]:
        return [self._flag(bar) for bar in self._builder.flush()]

    def end_warm_up(self) -> None:
        """Finish a replay that only primed the detectors; keep their state.

        Closes the replayed Session (so the Bar detector sees its Bars), then
        forgets everything the replay produced: Bars, flag counts and drops.
        """
        self.flush()
        self._slot_flags.clear()
        self.tick_flag_counts.clear()
        self._builder.dropped_out_of_session = 0

    def _slot_of(self, ts: datetime) -> datetime:
        utc = ts.astimezone(UTC)
        return utc - (utc - datetime(1970, 1, 1, tzinfo=UTC)) % self._size

    def _flag(self, bar: Bar) -> Bar:
        flags = self._slot_flags.pop(bar.open_time, set())
        if self._spreads.observe(bar):
            flags.add("spread_spike")
        ordered = tuple(f for f in VOCABULARY if f in flags)
        return dataclasses.replace(bar, quality_flags=ordered) if ordered else bar


__all__ = [
    "AnomalyWarning",
    "FlagSummary",
    "FlaggingBarBuilder",
    "SpreadSpikeDetector",
    "TickDetector",
    "summarize_flags",
]
