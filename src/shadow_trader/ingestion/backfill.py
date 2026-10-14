"""Backfill build: raw cache of HistData ticks to an immutable Dataset.

Reads only the write-once raw cache (never the network). The raw clock of each
Session week is inferred (New York local time or fixed UTC-5, see ``clock``);
ticks are converted to UTC with their week's offset and fed to the shared
incremental ``BarBuilder``. A week that fits no clock rule, or a tick with more
precision than the Instrument's price scale, fails the build naming week or
file and row. Everything else suspect (thin data, ticks outside a Session, a
misaligned weekly open) only warns.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import signal
import time
import zipfile
from collections import Counter
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import dataset
from .anomaly import FlaggingBarBuilder, summarize_flags
from .bars import Bar, Quote, gap_bar
from .clock import (
    BACKWARD_TOLERANCE,
    MonthEvidence,
    UnexplainedJump,
    WeekClock,
    WeekClockError,
    WeekEvidence,
    month_evidence,
)
from .config import (
    DEFAULT_START,
    AnomalyRuleConfig,
    Config,
    ConfigError,
    last_complete_month,
    to_month,
    validate_bar_sizes,
)
from .dataset import DatasetRef, RawCache
from .gaps import flag_suspected_outages, month_bounds, source_missing_bars
from .progress import NULL_PROGRESS, ProgressSink
from .report import (
    WARNING_CODES,
    Alignment,
    build_report,
    write_build_report,
    write_failed_report,
)
from .session import NY, next_open, open_of, slots_per_session, week_sunday
from .sources import SOURCE, months_between


class BuildFailure(Exception):
    """The backfill cannot produce a trustworthy Dataset (CLI exit code 1).

    ``locator`` names where it failed (file and row, or Instrument and period).
    """

    def __init__(self, message: str, locator: str | None = None) -> None:
        super().__init__(message)
        self.locator = locator
        self.report_path: Path | None = None  # the failure report, once written

    def __reduce__(self):  # keeps ``locator`` when raised in a pool worker
        return (BuildFailure, (str(self), self.locator))


@dataclass(frozen=True)
class BuildWarning:
    code: str
    message: str
    locator: str | None = None

    def __post_init__(self) -> None:
        if self.code not in WARNING_CODES:
            raise ValueError(f"unknown warning code {self.code!r}")


@dataclass(frozen=True)
class BackfillResult:
    ref: DatasetRef
    warnings: tuple[BuildWarning, ...] = field(default=())
    dropped_out_of_session: Mapping[str, int] = field(default_factory=dict)
    report_path: Path | None = None


def read_raw_month(cache: RawCache, instrument: str, period: str) -> str:
    """Text of one raw month; a missing, corrupt or empty month fails the build."""
    path = cache.path_for(SOURCE, instrument, period, "zip")
    if not path.exists():
        raise BuildFailure(
            f"raw month missing: {instrument} {period}", f"{instrument}/{period}"
        )
    member = f"DAT_ASCII_{instrument}_T_{period}.csv"
    try:
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
            text = archive.read(member).decode()
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError):
        raise BuildFailure(
            f"raw month corrupt: {instrument} {period}", f"{instrument}/{period}"
        ) from None
    if not text.strip():
        raise BuildFailure(
            f"raw month empty: {instrument} {period}", f"{instrument}/{period}"
        )
    return text


_EVENTS: Any = None  # in a pool worker: the queue of month events to the parent


def _init_worker(events: Any) -> None:
    """Pool workers leave Ctrl-C to the parent, which shuts the pool down; they
    keep the queue their progress events travel on."""
    global _EVENTS
    _EVENTS = events
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _drain(events: Any, on_event: Callable[[str, str, str | None], None]) -> None:
    """Hand every queued worker event to the parent's progress callback."""
    while not events.empty():
        on_event(*events.get())


@contextlib.contextmanager
def worker_pool(jobs: int, events: Any = None) -> Generator[ProcessPoolExecutor | None]:
    """A process pool of ``jobs`` workers, or ``None`` for in-process (``jobs`` 1).

    Any exit through an exception (Ctrl-C included) terminates the workers, so
    none outlives the build. ``events`` is a queue the workers can send progress
    events on (see ``_chunk_task``).
    """
    if jobs <= 1:
        yield None
        return
    pool = ProcessPoolExecutor(
        jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(events,),
    )
    try:
        yield pool
    except BaseException:
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run_tasks[T, R](
    pool: ProcessPoolExecutor | None,
    task: Callable[[T], R],
    items: Sequence[T],
    on_done: Callable[[int, R], None] = lambda index, result: None,
    poll: Callable[[], None] | None = None,
) -> list[R]:
    """``task`` over ``items``, results in item order whatever the finish order.

    ``on_done`` sees each result as it arrives. On a failure, queued items after
    the failed one are cancelled and running ones finish; the failure raised is
    the earliest by item order, which is what a serial run hits first. ``poll``
    runs while the pool works and once more after it, to relay worker events.
    """
    if pool is None:
        results = []
        for index, item in enumerate(items):
            results.append(task(item))
            on_done(index, results[-1])
        return results
    futures = [pool.submit(task, item) for item in items]
    position = {future: index for index, future in enumerate(futures)}
    pending: set[Future[R]] = set(futures)
    failed: int | None = None
    while pending:
        finished, pending = wait(
            pending, timeout=0.05 if poll else None, return_when=FIRST_COMPLETED
        )
        if poll:
            poll()
        for future in sorted(finished, key=position.__getitem__):
            if future.cancelled():
                continue
            index = position[future]
            if future.exception() is not None:
                if failed is None or index < failed:
                    failed = index
                for other in pending:
                    if position[other] > failed:
                        other.cancel()
            elif failed is None:
                on_done(index, future.result())
    if poll:
        poll()
    if failed is not None:
        error = futures[failed].exception()
        assert error is not None
        raise error
    return [future.result() for future in futures]


def _month_evidence(task: tuple[RawCache, str, str]) -> MonthEvidence:
    """One clock pre-pass task: unzip a raw month and return its week-clock evidence."""
    cache, instrument, period = task
    label = f"{instrument}/{period}"
    try:
        return month_evidence(read_raw_month(cache, instrument, period), label)
    except UnexplainedJump as jump:
        raise BuildFailure(
            f"ticks out of order at {jump}", f"{jump.label} row {jump.row}"
        ) from None


def infer_week_clock(
    cache: RawCache,
    instrument: str,
    periods: Iterable[str],
    *,
    progress: ProgressSink = NULL_PROGRESS,
    pool: ProcessPoolExecutor | None = None,
) -> WeekClock:
    """The raw-clock offset of every Session week in ``periods`` (a pre-pass).

    With a ``pool`` each month is scanned in a worker; the evidence merges in
    month order either way.
    """
    evidence = WeekEvidence()
    months = list(periods)
    progress.start("scanning clocks", len(months))
    scanned = run_tasks(
        pool,
        _month_evidence,
        [(cache, instrument, period) for period in months],
        lambda index, _: progress.advance(f"{instrument} {months[index]}"),
    )
    for month in scanned:
        evidence.merge(month)
    try:
        clock = evidence.infer()
    except WeekClockError as error:
        week = next(iter(error.failures))
        raise BuildFailure(
            f"{instrument} {error}", f"{instrument}/week-{week}"
        ) from None
    progress.done()
    _announce_fallbacks(instrument, clock, progress)
    return clock


def _announce_fallbacks(
    instrument: str, clock: WeekClock, progress: ProgressSink
) -> None:
    """Surface the tolerated clock decisions (they are also in the build report)."""
    for replays in clock.replays.values():
        for r in replays:
            progress.warn(
                f"{instrument}: dropped replayed block in {r.label} row {r.row} "
                f"({r.rows} rows, {r.first} to {r.last})"
            )
    for weeks, what in (
        (clock.inherited, "inherited the adjacent weeks' clock rule"),
        (clock.single_end, "decided from one end of the week only"),
        (clock.calendar, "fell back to the calendar clock rule"),
    ):
        if weeks:
            listed = ", ".join(w.isoformat() for w in sorted(weeks)[:5])
            more = f" (+{len(weeks) - 5} more)" if len(weeks) > 5 else ""
            progress.warn(f"{instrument}: {len(weeks)} week(s) {what}: {listed}{more}")


ALIGNMENT_TOLERANCE = timedelta(hours=1)
# Offsets (minutes) typical of a mis-converted timestamp convention: a 1 h or
# 2 h shift, e.g. DST applied when it should not be or the reverse.
WRONG_OFFSET_SIGNATURES = ((55, 65), (115, 125))
_WEEK_LEAD = timedelta(hours=6)  # ticks this close before the open join the week


def _week_open(ts: datetime) -> datetime:
    """UTC weekly open (Sunday 17:00 New York) of the week ``ts`` belongs to."""
    sunday = week_sunday(ts.astimezone(NY).date())
    opened = open_of(sunday)
    if ts < opened - _WEEK_LEAD:
        opened = open_of(sunday - timedelta(days=7))
    return opened


def _alignment_warning(
    instrument: str, opened: datetime, first: datetime
) -> tuple[BuildWarning, Alignment] | None:
    offset = first - opened
    minutes = abs(offset) / timedelta(minutes=1)
    signature = any(low <= minutes <= high for low, high in WRONG_OFFSET_SIGNATURES)
    if abs(offset) <= ALIGNMENT_TOLERANCE and not signature:
        return None
    week = opened.astimezone(NY).date().isoformat()
    message = (
        f"{instrument} week of {week}: first tick is {minutes:.0f} min "
        f"{'after' if offset >= timedelta(0) else 'before'} the 17:00 New York open"
    )
    if signature:
        message += "; matches a wrong-offset signature (timestamp convention?)"
    locator = f"{instrument}/week-{week}"
    return (
        BuildWarning("session_alignment", message, locator),
        Alignment(week, locator, round(offset / timedelta(minutes=1)), signature),
    )


def parse_ticks(
    text: str, label: str, price_scale: int, clock: WeekClock
) -> Iterator[Quote]:
    dropped = {r.row: r.rows for r in clock.replays.get(label, ())}
    skip_until = 0
    for row, line in enumerate(text.splitlines(), start=1):
        skip_until = row + dropped[row] if row in dropped else skip_until
        if row < skip_until:
            continue  # a replayed block (see ``clock.Replay``)
        where = f"{label} row {row}"
        try:
            stamp, bid, ask, *_ = line.split(",")
            naive = datetime.strptime(stamp[:15], "%Y%m%d %H%M%S").replace(
                microsecond=int(stamp[15:18]) * 1000
            )
            quote_bid, quote_ask = Decimal(bid), Decimal(ask)
        except (ValueError, InvalidOperation):
            raise BuildFailure(
                f"unparseable tick at {where}: {line!r}", where
            ) from None
        if any((price * 10**price_scale) % 1 != 0 for price in (quote_bid, quote_ask)):
            raise BuildFailure(
                f"tick has more precision than price scale {price_scale} "
                f"at {where}: {line!r}",
                where,
            )
        yield Quote(clock.to_utc(naive), quote_bid, quote_ask)


# Whole Session weeks per bar-pass chunk. Internal: not a CLI or config setting
# (``build_backfill`` takes a test-only override).
CHUNK_WEEKS = 4
# A chunk replays this many Session weeks before its own to prime the detectors.
# Known limit: a stale-quote run that began more than one week before a chunk
# seam is not reproduced there (it needs a whole week of unchanged quotes). The
# price-spike window (ticks) and the spread-spike lookback (hours) fit easily.
WARM_UP_WEEKS = 1


@dataclass(frozen=True)
class WeekSpan:
    """Whole Session weeks of a chunk, by weekly open (UTC); ``None`` is unbounded.

    ``first`` is the open of the first week, ``end`` the open of the week after
    the last. A tick belongs to the span by its Session week (``_week_open``),
    so the pre-open lead window and the weekend closure go with their week.
    """

    first: datetime | None = None
    end: datetime | None = None


def session_weeks(lo: datetime, hi: datetime) -> list[datetime]:
    """Weekly opens (UTC) of every Session week that can hold ticks in [lo, hi)."""
    opens = [_week_open(lo)]
    # raw files run up to hours past ``hi`` in UTC, so look a day further
    while (following := _week_open(opens[-1] + timedelta(days=8))) < hi + timedelta(
        days=1
    ):
        opens.append(following)
    return opens


def week_chunks(opens: list[datetime], weeks: int) -> list[WeekSpan]:
    """Spans of ``weeks`` whole Session weeks; the ends are unbounded."""
    starts = range(0, len(opens), weeks)
    return [
        WeekSpan(
            opens[start] if start else None,
            opens[start + weeks] if start + weeks < len(opens) else None,
        )
        for start in starts
    ]


def span_periods(periods: list[str], span: WeekSpan) -> list[str]:
    """The monthly raw files a span (and its warm-up week) can touch."""
    pad = timedelta(days=1)
    lo = span.first - timedelta(days=7 * WARM_UP_WEEKS) - pad if span.first else None
    hi = span.end + pad if span.end else None
    return [
        period
        for period in periods
        if (lo is None or to_month(period) >= f"{lo.astimezone(NY):%Y-%m}")
        and (hi is None or to_month(period) <= f"{hi.astimezone(NY):%Y-%m}")
    ]


@dataclass(frozen=True)
class ChunkResult:
    """Plain, picklable output of one bar-pass chunk (no writes, no I/O handles)."""

    bars: tuple[Bar, ...]
    missing: dict[datetime, Bar]  # source-missing Gap bars of acknowledged months
    tick_flag_counts: dict[str, int]
    dropped_out_of_session: int
    late_dropped: int
    first_ticks: dict[datetime, datetime]  # weekly open -> first tick


def build_chunk(
    cache: RawCache,
    instrument: str,
    bar_size: int,
    price_scale: int,
    anomaly_rule: AnomalyRuleConfig,
    clock: WeekClock,
    periods: Iterable[str],
    acknowledged: frozenset[str] = frozenset(),
    span: WeekSpan | None = None,
    *,
    on_month: Callable[[str, str | None], None] | None = None,
) -> ChunkResult:
    """Bar pass over the Session weeks of ``span`` for one Instrument.

    ``periods`` are the monthly raw files to read (see ``span_periods``). The
    week before ``span.first`` is replayed first to prime the detectors and its
    output is discarded. Ticks outside the span count nowhere, so chunks that
    tile the history sum to the single-chunk result.

    Returns plain data only; per-Instrument post-processing (outage flagging,
    flag summaries, alignment and thin-data warnings) is the merge step's job.
    """
    span = span or WeekSpan()
    warm_from = (
        _week_open(span.first - timedelta(days=1)) if span.first is not None else None
    )
    builder = FlaggingBarBuilder(instrument, bar_size, price_scale, anomaly_rule)
    bars: list[Bar] = []
    first_ticks: dict[datetime, datetime] = {}
    missing: dict[datetime, Bar] = {}
    warming = warm_from is not None
    newest: datetime | None = None
    late_dropped = 0
    for period in periods:
        month = to_month(period)
        if month in acknowledged:
            for gap in source_missing_bars(instrument, month, bar_size):
                missing[gap.open_time] = gap
            if on_month:
                on_month(period, "acknowledged missing")
            continue
        text = read_raw_month(cache, instrument, period)
        label = f"{instrument}/{period}"
        for row, quote in enumerate(
            parse_ticks(text, label, price_scale, clock), start=1
        ):
            opened = _week_open(quote.ts)
            if span.end is not None and opened >= span.end:
                if quote.ts >= span.end:
                    break  # ticks are chronological, apart from late ones
                continue
            if span.first is not None and opened < span.first:
                if opened != warm_from:
                    continue
            elif warming:
                builder.end_warm_up()
                warming = False
            if not warming:
                first_ticks.setdefault(opened, quote.ts)
            try:
                pushed = builder.push(quote)
                if not warming:
                    bars.extend(pushed)
            except ValueError as exc:
                if warming:
                    continue
                # a stale tick a second or so behind a closed slot (HistData's
                # 2026 files); the pre-pass already failed any bigger step back
                if newest is None or newest - quote.ts >= BACKWARD_TOLERANCE:
                    where = f"{label} row {row}"
                    raise BuildFailure(
                        f"ticks out of order at {where}: {exc}", where
                    ) from None
                late_dropped += 1
                continue
            newest = quote.ts if newest is None else max(newest, quote.ts)
        if on_month:
            on_month(period, None)
    if warming:
        builder.end_warm_up()
    bars.extend(builder.flush())
    return ChunkResult(
        tuple(bars),
        missing,
        dict(builder.tick_flag_counts),
        builder.dropped_out_of_session,
        late_dropped,
        first_ticks,
    )


@dataclass(frozen=True)
class ChunkTask:
    """The arguments of one bar-pass chunk, as one picklable pool-task item."""

    cache: RawCache
    instrument: str
    bar_size: int
    price_scale: int
    anomaly_rule: AnomalyRuleConfig
    clock: WeekClock
    periods: list[str]
    acknowledged: frozenset[str]
    span: WeekSpan


def _send_month(instrument: str) -> Callable[[str, str | None], None] | None:
    """Month callback of a pool worker: queue the event for the parent."""
    if _EVENTS is None:
        return None
    events = _EVENTS
    return lambda period, note: events.put((instrument, period, note))


def _chunk_task(task: ChunkTask) -> ChunkResult:
    """Run one chunk; in a pool worker each month read is sent to the parent as
    it finishes, so progress is live inside a long chunk."""
    return build_chunk(
        task.cache,
        task.instrument,
        task.bar_size,
        task.price_scale,
        task.anomaly_rule,
        task.clock,
        task.periods,
        task.acknowledged,
        task.span,
        on_month=_send_month(task.instrument),
    )


def _once_per_month(
    progress: ProgressSink, instrument: str
) -> Callable[[str, str | None], None]:
    """Month callback for chunks, which share months: advance once per month."""
    advanced: set[str] = set()

    def on_month(period: str, note: str | None = None) -> None:
        if period not in advanced:
            advanced.add(period)
            progress.advance(f"{instrument} {period}", note)

    return on_month


def _chunk_months(
    tasks: Sequence[ChunkTask],
    on_month: Callable[[str, str | None], None],
    acknowledged: frozenset[str],
) -> Callable[[int, ChunkResult], None]:
    """Completion callback: a chunk's months advance (once each) when it returns,
    so progress stays right when chunks finish out of order."""

    def chunk_done(index: int, _: ChunkResult) -> None:
        for period in tasks[index].periods:
            note = "acknowledged missing" if to_month(period) in acknowledged else None
            on_month(period, note)

    return chunk_done


def _relay(
    events: Any, progress: ProgressSink, on_month: Callable[[str, str | None], None]
) -> Callable[[], None] | None:
    """Parent side of the worker queue: feed worker month events to ``on_month``
    (which advances each month once, whichever chunk reports it first)."""
    if events is None:
        return None
    return lambda: _drain(events, lambda _, period, note: on_month(period, note))


def _empty_sessions(last: Bar, following: Bar, bar_size: int) -> list[Bar]:
    """Gap Bars of the Sessions between two Bars that no chunk saw a tick for.

    A serial pass closes every Session that passed between two ticks as Gaps; a
    chunk whose weeks hold no ticks has no builder to do it.
    """
    gaps: list[Bar] = []
    opened = next_open(last.session_id)
    while opened < following.session_id:
        step = timedelta(minutes=bar_size)
        gaps.extend(
            gap_bar(last.instrument, opened + slot * step, opened)
            for slot in range(slots_per_session(bar_size))
        )
        opened = next_open(opened)
    return gaps


def merge_chunks(chunks: list[ChunkResult], bar_size: int) -> ChunkResult:
    """Combine chunks of consecutive week spans into the single-chunk result."""
    tick_flags: Counter[str] = Counter()
    merged = ChunkResult((), {}, {}, 0, 0, {})
    bars: list[Bar] = []
    for chunk in chunks:
        if bars and chunk.bars:
            bars.extend(_empty_sessions(bars[-1], chunk.bars[0], bar_size))
        bars.extend(chunk.bars)
        merged.missing.update(chunk.missing)
        merged.first_ticks.update(chunk.first_ticks)
        tick_flags.update(chunk.tick_flag_counts)
        merged = replace(
            merged,
            dropped_out_of_session=merged.dropped_out_of_session
            + chunk.dropped_out_of_session,
            late_dropped=merged.late_dropped + chunk.late_dropped,
        )
    return replace(merged, bars=tuple(bars), tick_flag_counts=dict(tick_flags))


def build_backfill(
    config: Config,
    bar_size: int,
    *,
    now: datetime | None = None,
    progress: ProgressSink = NULL_PROGRESS,
    chunk_weeks: int = CHUNK_WEEKS,
) -> BackfillResult:
    """Build one Dataset (all configured Instruments) at ``bar_size`` minutes.

    A failed build writes a failure report under ``<data_root>/reports/failed``
    (no Dataset directory exists) and re-raises ``BuildFailure``. Without a
    configured end month the range runs to the last complete UTC month before
    ``now`` (default: the current time). ``chunk_weeks`` is a test-only override
    of the bar-pass chunk size; the Dataset never depends on it. With
    ``config.build.jobs`` above 1 the clock pre-pass months and bar-pass chunks
    run in a process pool; the parent alone merges and writes, so the Dataset,
    report and warnings are the same at any job count.
    """
    validate_bar_sizes((bar_size,))
    events = (
        multiprocessing.get_context("spawn").SimpleQueue()
        if config.build.jobs > 1
        else None
    )
    try:
        with worker_pool(config.build.jobs, events) as pool:
            return _build(
                config,
                bar_size,
                now or datetime.now(UTC),
                progress,
                chunk_weeks,
                pool,
                events,
            )
    except BuildFailure as failure:
        # an unwritable report must not mask the failure itself
        with contextlib.suppress(OSError):
            failure.report_path = write_failed_report(
                config.paths.data_root, bar_size, str(failure), failure.locator
            )
        raise


def build_dataset(config: Config, bar_size: int) -> DatasetRef:
    """Build a backfill Dataset at ``bar_size`` from the raw cache (spec API).

    The CLI ``build`` command and this share ``build_backfill``; use that for
    the warnings and the build report.
    """
    return build_backfill(config, bar_size).ref


def _build(
    config: Config,
    bar_size: int,
    now: datetime,
    progress: ProgressSink,
    chunk_weeks: int,
    pool: ProcessPoolExecutor | None,
    events: Any = None,
) -> BackfillResult:
    cache = RawCache(config.paths.data_root)
    build = config.build
    end = build.end or last_complete_month(now)
    if end < build.start:
        raise ConfigError(
            f"build.start {build.start} is after the last complete month {end}"
        )
    lo, hi = month_bounds(build.start)[0], month_bounds(end)[1]
    series: dict[str, list[Bar]] = {}
    warnings: list[BuildWarning] = []

    def warn(warning: BuildWarning) -> None:
        warnings.append(warning)
        progress.warn(f"{warning.code}: {warning.message}")

    dropped: dict[str, int] = {}
    tick_flags: dict[str, dict[str, int]] = {}
    alignments: dict[str, list[Alignment]] = {}
    clocks: dict[str, WeekClock] = {}
    timings = {"clock_pre_pass": 0.0, "bar_pass": 0.0, "write_dataset": 0.0}
    for instrument in build.instruments:
        instrument_began = time.perf_counter()
        scale = config.instruments[instrument].price_scale
        acknowledged = frozenset(
            a.month for a in build.acknowledged_missing if a.instrument == instrument
        )
        pre_pass_began = time.perf_counter()
        clocks[instrument] = clock = infer_week_clock(
            cache,
            instrument,
            [
                period
                for period in months_between(build.start, end)
                if to_month(period) not in acknowledged
            ],
            progress=progress,
            pool=pool,
        )
        pre_pass = time.perf_counter() - pre_pass_began
        timings["clock_pre_pass"] += pre_pass
        periods = months_between(build.start, end)
        for period in periods:
            if (month := to_month(period)) < DEFAULT_START and (
                month not in acknowledged
            ):
                warn(
                    BuildWarning(
                        "thin_data",
                        f"{instrument} {period} is before the history floor "
                        f"{DEFAULT_START}; ticks are sparse",
                        f"{instrument}/{period}",
                    )
                )
        progress.start("building months", len(periods))
        on_month = _once_per_month(progress, instrument)

        spans = week_chunks(session_weeks(lo, hi), chunk_weeks)
        tasks = [
            ChunkTask(
                cache,
                instrument,
                bar_size,
                scale,
                config.anomaly_rule,
                clock,
                span_periods(periods, span),
                acknowledged,
                span,
            )
            for span in spans
        ]

        chunks = run_tasks(
            pool,
            _chunk_task,
            tasks,
            _chunk_months(tasks, on_month, acknowledged),
            poll=_relay(events, progress, on_month),
        )
        chunk = merge_chunks(chunks, bar_size)
        progress.done()
        timings["bar_pass"] += time.perf_counter() - instrument_began - pre_pass
        bars = [bar for bar in chunk.bars if lo <= bar.open_time < hi]
        for opened, first in chunk.first_ticks.items():
            if (off := _alignment_warning(instrument, opened, first)) is not None:
                warn(off[0])
                alignments.setdefault(instrument, []).append(off[1])
        merged = {bar.open_time: bar for bar in bars} | chunk.missing
        series[instrument] = flag_suspected_outages(
            sorted(merged.values(), key=lambda bar: bar.open_time),
            bar_size,
            config.gap_rule.outage_threshold_minutes,
        )
        tick_flags[instrument] = dict(chunk.tick_flag_counts)
        summary = summarize_flags(
            series[instrument],
            chunk.tick_flag_counts,
            config.anomaly_rule.alert_fraction,
        )
        for w in summary.warnings:
            warn(BuildWarning(w.code, w.message, instrument))
        if chunk.late_dropped:
            warn(
                BuildWarning(
                    "late_ticks_dropped",
                    f"{chunk.late_dropped} {instrument} tick(s) arrived up to "
                    f"{BACKWARD_TOLERANCE.seconds} s behind a closed slot and "
                    "were dropped",
                    instrument,
                )
            )
        if chunk.dropped_out_of_session:
            dropped[instrument] = chunk.dropped_out_of_session
            warn(
                BuildWarning(
                    "out_of_session_ticks",
                    f"{chunk.dropped_out_of_session} {instrument} ticks fell "
                    "outside any Session and were dropped",
                    instrument,
                )
            )
    progress.start("writing dataset")
    write_began = time.perf_counter()
    ref = dataset.write_dataset(
        config,
        bar_size,
        series,
        parent=dataset.latest_dataset(
            config.paths.data_root, SOURCE, bar_size, build.instruments
        ),
        tick_flag_counts=tick_flags,
        extra_manifest={"session_clock": _clock_manifest(clocks)},
    )
    timings["write_dataset"] = time.perf_counter() - write_began
    report_path = write_build_report(
        ref.path,
        build_report(
            dataset_id=ref.dataset_id,
            bar_size=bar_size,
            series=series,
            dropped_out_of_session=dropped,
            tick_flag_counts=tick_flags,
            alignments=alignments,
            session_clocks={pair: c.summary() for pair, c in clocks.items()},
            warnings=warnings,
            raw_inputs=ref.manifest["raw_inputs"],
            config=config,
            timings=timings,
        ),
    )
    progress.done()
    return BackfillResult(ref, tuple(warnings), dropped, report_path)


def _clock_manifest(clocks: Mapping[str, WeekClock]) -> dict[str, object]:
    """Provenance of the inferred raw-clock offsets (not hashed into the Dataset)."""
    return {pair: clock.provenance() for pair, clock in clocks.items()}
