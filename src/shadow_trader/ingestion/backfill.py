"""Backfill build: raw cache of HistData ticks to an immutable Dataset.

Reads only the write-once raw cache (never the network). Ticks are New York
local time with DST; each is converted to UTC through ``America/New_York`` and
fed to the shared incremental ``BarBuilder``. A tick whose local time is
ambiguous or nonexistent, or that has more precision than the Instrument's
price scale, fails the build naming file and row. Everything else suspect
(thin data, ticks outside a Session, a misaligned weekly open) only warns.
"""

from __future__ import annotations

import contextlib
import io
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import dataset
from .anomaly import FlaggingBarBuilder, summarize_flags
from .bars import Bar, Quote
from .config import DEFAULT_START, Config, to_month, validate_bar_sizes
from .dataset import DatasetRef, RawCache
from .gaps import flag_suspected_outages, source_missing_bars
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


def _to_utc(naive: datetime) -> datetime | None:
    """UTC instant of a New York local time, or None if ambiguous/nonexistent."""
    first = naive.replace(tzinfo=NY, fold=0)
    second = naive.replace(tzinfo=NY, fold=1)
    if first.utcoffset() != second.utcoffset():
        return None
    return first.astimezone(UTC)


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


def parse_ticks(text: str, label: str, price_scale: int) -> Iterator[Quote]:
    for row, line in enumerate(text.splitlines(), start=1):
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
        ts = _to_utc(naive)
        if ts is None:
            raise BuildFailure(
                f"ambiguous or nonexistent New York local time {stamp} at {where}",
                where,
            )
        yield Quote(ts, quote_bid, quote_ask)


def build_backfill(config: Config, bar_size: int) -> BackfillResult:
    """Build one Dataset (all configured Instruments) at ``bar_size`` minutes.

    A failed build writes a failure report under ``<data_root>/reports/failed``
    (no Dataset directory exists) and re-raises ``BuildFailure``.
    """
    validate_bar_sizes((bar_size,))
    try:
        return _build(config, bar_size)
    except BuildFailure as failure:
        # an unwritable report must not mask the failure itself
        with contextlib.suppress(OSError):
            failure.report_path = write_failed_report(
                config.paths.data_root, bar_size, str(failure), failure.locator
            )
        raise


def _build(config: Config, bar_size: int) -> BackfillResult:
    cache = RawCache(config.paths.data_root)
    build = config.build
    end = build.end or build.start
    series: dict[str, list[Bar]] = {}
    warnings: list[BuildWarning] = []
    dropped: dict[str, int] = {}
    tick_flags: dict[str, dict[str, int]] = {}
    alignments: dict[str, list[Alignment]] = {}
    for instrument in build.instruments:
        scale = config.instruments[instrument].price_scale
        builder = FlaggingBarBuilder(instrument, bar_size, scale, config.anomaly_rule)
        bars: list[Bar] = []
        first_ticks: dict[datetime, datetime] = {}  # weekly open -> first tick
        acknowledged = {
            a.month for a in build.acknowledged_missing if a.instrument == instrument
        }
        missing: dict[datetime, Bar] = {}
        for period in months_between(build.start, end):
            month = to_month(period)
            if month in acknowledged:
                for gap in source_missing_bars(instrument, month, bar_size):
                    missing[gap.open_time] = gap
                continue
            text = read_raw_month(cache, instrument, period)
            if month < DEFAULT_START:
                warnings.append(
                    BuildWarning(
                        "thin_data",
                        f"{instrument} {period} is before the history floor "
                        f"{DEFAULT_START}; ticks are sparse",
                        f"{instrument}/{period}",
                    )
                )
            for quote in parse_ticks(text, f"{instrument}/{period}", scale):
                first_ticks.setdefault(_week_open(quote.ts), quote.ts)
                bars.extend(builder.push(quote))
        bars.extend(builder.flush())
        for opened, first in first_ticks.items():
            if (off := _alignment_warning(instrument, opened, first)) is not None:
                warnings.append(off[0])
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
        warnings.extend(
            BuildWarning(w.code, w.message, instrument) for w in summary.warnings
        )
        if builder.dropped_out_of_session:
            dropped[instrument] = builder.dropped_out_of_session
            warnings.append(
                BuildWarning(
                    "out_of_session_ticks",
                    f"{builder.dropped_out_of_session} {instrument} ticks fell "
                    "outside any Session and were dropped",
                    instrument,
                )
            )
    ref = dataset.build_dataset(config, bar_size, series, tick_flag_counts=tick_flags)
    report_path = write_build_report(
        ref.path,
        build_report(
            dataset_id=ref.dataset_id,
            bar_size=bar_size,
            series=series,
            dropped_out_of_session=dropped,
            tick_flag_counts=tick_flags,
            alignments=alignments,
            warnings=warnings,
            raw_inputs=ref.manifest["raw_inputs"],
            config=config,
        ),
    )
    return BackfillResult(ref, tuple(warnings), dropped, report_path)
