"""Backfill build: raw cache of HistData ticks to an immutable Dataset.

Reads only the write-once raw cache (never the network). Ticks are New York
local time with DST; each is converted to UTC through ``America/New_York`` and
fed to the shared incremental ``BarBuilder``. A tick whose local time is
ambiguous or nonexistent, or that has more precision than the Instrument's
price scale, fails the build naming file and row. Everything else suspect
(thin data, ticks outside a Session, a misaligned weekly open) only warns.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from . import dataset
from .bars import Bar, BarBuilder, Quote
from .config import DEFAULT_START, Config, validate_bar_sizes
from .dataset import DatasetRef, RawCache
from .sources import SOURCE, months_between

NY = ZoneInfo("America/New_York")


class BuildFailure(Exception):
    """The backfill cannot produce a trustworthy Dataset (CLI exit code 1)."""


@dataclass(frozen=True)
class BuildWarning:
    code: str
    message: str
    locator: str | None = None


@dataclass(frozen=True)
class BackfillResult:
    ref: DatasetRef
    warnings: tuple[BuildWarning, ...] = field(default=())
    dropped_out_of_session: Mapping[str, int] = field(default_factory=dict)


def _raw_zip(cache: RawCache, instrument: str, period: str) -> zipfile.ZipFile:
    path = cache.path_for(SOURCE, instrument, period, "zip")
    if not path.exists():
        raise BuildFailure(f"raw month missing: {instrument} {period}")
    try:
        return zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    except zipfile.BadZipFile:
        raise BuildFailure(f"raw month corrupt: {instrument} {period}") from None


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
    local = ts.astimezone(NY)
    sunday = local.date() - timedelta(days=(local.weekday() + 1) % 7)
    opened = datetime.combine(sunday, time(17, 0), tzinfo=NY).astimezone(UTC)
    if ts < opened - _WEEK_LEAD:
        sunday -= timedelta(days=7)
        opened = datetime.combine(sunday, time(17, 0), tzinfo=NY).astimezone(UTC)
    return opened


def _alignment_warning(
    instrument: str, opened: datetime, first: datetime
) -> BuildWarning | None:
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
    return BuildWarning("session_alignment", message, f"{instrument}/week-{week}")


def _parse(text: str, label: str, price_scale: int) -> Iterator[Quote]:
    for row, line in enumerate(text.splitlines(), start=1):
        where = f"{label} row {row}"
        try:
            stamp, bid, ask, *_ = line.split(",")
            naive = datetime.strptime(stamp[:15], "%Y%m%d %H%M%S").replace(
                microsecond=int(stamp[15:18]) * 1000
            )
            quote_bid, quote_ask = Decimal(bid), Decimal(ask)
        except (ValueError, InvalidOperation):
            raise BuildFailure(f"unparseable tick at {where}: {line!r}") from None
        if any((price * 10**price_scale) % 1 != 0 for price in (quote_bid, quote_ask)):
            raise BuildFailure(
                f"tick has more precision than price scale {price_scale} "
                f"at {where}: {line!r}"
            )
        ts = _to_utc(naive)
        if ts is None:
            raise BuildFailure(
                f"ambiguous or nonexistent New York local time {stamp} at {where}"
            )
        yield Quote(ts, quote_bid, quote_ask)


def build_backfill(config: Config, bar_size: int) -> BackfillResult:
    """Build one Dataset (all configured Instruments) at ``bar_size`` minutes."""
    validate_bar_sizes((bar_size,))
    cache = RawCache(config.paths.data_root)
    build = config.build
    end = build.end or build.start
    series: dict[str, list[Bar]] = {}
    warnings: list[BuildWarning] = []
    dropped: dict[str, int] = {}
    for instrument in build.instruments:
        scale = config.instruments[instrument].price_scale
        builder = BarBuilder(instrument, bar_size, scale)
        bars: list[Bar] = []
        first_ticks: dict[datetime, datetime] = {}  # weekly open -> first tick
        for period in months_between(build.start, end):
            archive = _raw_zip(cache, instrument, period)
            if f"{period[:4]}-{period[4:]}" < DEFAULT_START:
                warnings.append(
                    BuildWarning(
                        "thin_data",
                        f"{instrument} {period} is before the history floor "
                        f"{DEFAULT_START}; ticks are sparse",
                        f"{instrument}/{period}",
                    )
                )
            member = f"DAT_ASCII_{instrument}_T_{period}.csv"
            text = archive.read(member).decode()
            for quote in _parse(text, f"{instrument}/{period}", scale):
                first_ticks.setdefault(_week_open(quote.ts), quote.ts)
                bars.extend(builder.push(quote))
        bars.extend(builder.flush())
        for opened, first in first_ticks.items():
            if (warning := _alignment_warning(instrument, opened, first)) is not None:
                warnings.append(warning)
        series[instrument] = bars
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
    ref = dataset.build_dataset(config, bar_size, series)
    return BackfillResult(ref, tuple(warnings), dropped)
