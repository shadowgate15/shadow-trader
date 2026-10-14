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
import time
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import dataset
from .anomaly import FlaggingBarBuilder, summarize_flags
from .bars import Bar, Quote
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
from .session import NY, open_of, week_sunday
from .sources import SOURCE, months_between


class BuildFailure(Exception):
    """The backfill cannot produce a trustworthy Dataset (CLI exit code 1).

    ``locator`` names where it failed (file and row, or Instrument and period).
    """

    def __init__(self, message: str, locator: str | None = None) -> None:
        super().__init__(message)
        self.locator = locator
        self.report_path: Path | None = None  # the failure report, once written


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


def _month_evidence(cache: RawCache, instrument: str, period: str) -> MonthEvidence:
    """One clock pre-pass task: unzip a raw month and return its week-clock evidence."""
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
) -> WeekClock:
    """The raw-clock offset of every Session week in ``periods`` (a pre-pass)."""
    evidence = WeekEvidence()
    months = list(periods)
    progress.start("scanning clocks", len(months))
    for period in months:
        evidence.merge(_month_evidence(cache, instrument, period))
        progress.advance(f"{instrument} {period}")
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


def build_backfill(
    config: Config,
    bar_size: int,
    *,
    now: datetime | None = None,
    progress: ProgressSink = NULL_PROGRESS,
) -> BackfillResult:
    """Build one Dataset (all configured Instruments) at ``bar_size`` minutes.

    A failed build writes a failure report under ``<data_root>/reports/failed``
    (no Dataset directory exists) and re-raises ``BuildFailure``. Without a
    configured end month the range runs to the last complete UTC month before
    ``now`` (default: the current time).
    """
    validate_bar_sizes((bar_size,))
    try:
        return _build(config, bar_size, now or datetime.now(UTC), progress)
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
    config: Config, bar_size: int, now: datetime, progress: ProgressSink
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
        builder = FlaggingBarBuilder(instrument, bar_size, scale, config.anomaly_rule)
        bars: list[Bar] = []
        first_ticks: dict[datetime, datetime] = {}  # weekly open -> first tick
        acknowledged = {
            a.month for a in build.acknowledged_missing if a.instrument == instrument
        }
        missing: dict[datetime, Bar] = {}
        newest: datetime | None = None
        late_dropped = 0
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
        )
        pre_pass = time.perf_counter() - pre_pass_began
        timings["clock_pre_pass"] += pre_pass
        periods = months_between(build.start, end)
        progress.start("building months", len(periods))
        for period in periods:
            month = to_month(period)
            if month in acknowledged:
                for gap in source_missing_bars(instrument, month, bar_size):
                    missing[gap.open_time] = gap
                progress.advance(f"{instrument} {period}", "acknowledged missing")
                continue
            text = read_raw_month(cache, instrument, period)
            if month < DEFAULT_START:
                warn(
                    BuildWarning(
                        "thin_data",
                        f"{instrument} {period} is before the history floor "
                        f"{DEFAULT_START}; ticks are sparse",
                        f"{instrument}/{period}",
                    )
                )
            label = f"{instrument}/{period}"
            for row, quote in enumerate(
                parse_ticks(text, label, scale, clock), start=1
            ):
                first_ticks.setdefault(_week_open(quote.ts), quote.ts)
                try:
                    bars.extend(builder.push(quote))
                except ValueError as exc:
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
            progress.advance(f"{instrument} {period}")
        bars.extend(builder.flush())
        progress.done()
        timings["bar_pass"] += time.perf_counter() - instrument_began - pre_pass
        bars = [bar for bar in bars if lo <= bar.open_time < hi]
        for opened, first in first_ticks.items():
            if (off := _alignment_warning(instrument, opened, first)) is not None:
                warn(off[0])
                alignments.setdefault(instrument, []).append(off[1])
        merged = {bar.open_time: bar for bar in bars} | missing
        series[instrument] = flag_suspected_outages(
            sorted(merged.values(), key=lambda bar: bar.open_time),
            bar_size,
            config.gap_rule.outage_threshold_minutes,
        )
        tick_flags[instrument] = dict(builder.tick_flag_counts)
        summary = summarize_flags(
            series[instrument],
            builder.tick_flag_counts,
            config.anomaly_rule.alert_fraction,
        )
        for w in summary.warnings:
            warn(BuildWarning(w.code, w.message, instrument))
        if late_dropped:
            warn(
                BuildWarning(
                    "late_ticks_dropped",
                    f"{late_dropped} {instrument} tick(s) arrived up to "
                    f"{BACKWARD_TOLERANCE.seconds} s behind a closed slot and "
                    "were dropped",
                    instrument,
                )
            )
        if builder.dropped_out_of_session:
            dropped[instrument] = builder.dropped_out_of_session
            warn(
                BuildWarning(
                    "out_of_session_ticks",
                    f"{builder.dropped_out_of_session} {instrument} ticks fell "
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
