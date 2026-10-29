"""Acquire the raw weekly History from the FXCM free candle repository.

Source: ``{BASE_URL}/m1/EURUSD/{year}/{index}.csv.gz``, one gzipped CSV per
trading week, bid and ask in one file, timestamps in UTC. This is the only
source: weeks the vendor does not publish are Gaps, never patched from
elsewhere.

On disk, ``<raw_dir>/m1/EURUSD/<year>/<index>.csv.gz`` mirrors the vendor's
path, holding the downloaded bytes verbatim.

Weeks are *discovered*, never computed. The vendor's index is not reliably the
ISO week, so no filename is derived from a date: candidate indices are probed
and each file's coverage is read from its own timestamp column.
"""

import csv
import gzip
import io
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from shadow_trader import config

BASE_URL = "https://candledata.fxcorporate.com"
PERIODICITY = "m1"
INSTRUMENT = "EURUSD"
# The vendor host refuses urllib's default agent.
USER_AGENT = "shadow-trader"
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S.%f"

MAX_INDEX = 53
# A vendor index near a year boundary may sit in the neighbouring year's
# directory, so the tail of the preceding year is probed too.
PREVIOUS_YEAR_PROBE_FROM = 50
# Consecutive weekly files are ~49h apart (the weekend); anything near a week
# longer means a whole week is missing.
MAX_BREAK_BETWEEN_WEEKS = timedelta(days=4)


class HistoryError(Exception):
    """The acquired data violates an invariant; ingest must not continue."""


class Source(Protocol):
    def get(self, year: int, index: int) -> bytes | None:
        """The file's raw bytes, or None if the vendor does not publish it."""
        ...


class HttpSource:
    def get(self, year: int, index: int) -> bytes | None:
        url = f"{BASE_URL}/{PERIODICITY}/{INSTRUMENT}/{year}/{index}.csv.gz"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise


def default_source() -> Source:
    return HttpSource()


@dataclass(frozen=True)
class Week:
    """A cached weekly file and the span its own timestamps cover."""

    year: int
    index: int
    path: Path
    first: datetime
    last: datetime


@dataclass(frozen=True)
class FetchResult:
    weeks: list[Week]  # the weeks making up History, in time order


def week_path(raw_dir: Path, year: int, index: int) -> Path:
    return raw_dir / PERIODICITY / INSTRUMENT / str(year) / f"{index}.csv.gz"


def read_coverage(raw: bytes, label: str) -> tuple[datetime, datetime] | None:
    """First and last timestamp in a weekly file, read from the data itself.

    None for a header-only file: the vendor serves an empty placeholder for a
    week it has no data for, which is a missing week, not corrupt input.
    """
    try:
        text = gzip.decompress(raw).decode()
        rows = csv.reader(io.StringIO(text))
        next(rows)  # header
        stamps = [
            datetime.strptime(row[0], TIMESTAMP_FORMAT).replace(tzinfo=UTC)
            for row in rows
            if row
        ]
    except (OSError, EOFError, ValueError, IndexError, StopIteration) as error:
        raise HistoryError(f"{label}: unreadable weekly file ({error})") from error
    if not stamps:
        return None
    return min(stamps), max(stamps)


def candidates(start: datetime, end: datetime):
    """(year, index) pairs to probe: indices are enumerated, dates never mapped."""
    for index in range(PREVIOUS_YEAR_PROBE_FROM, MAX_INDEX + 1):
        yield start.year - 1, index
    for year in range(start.year, end.year + 1):
        for index in range(0, MAX_INDEX + 1):
            yield year, index


def fetch_history(
    source: Source,
    raw_dir: Path,
    *,
    start: datetime,
    end: datetime,
    progress: Callable[[str], None] = print,
) -> FetchResult:
    """Download (once) every weekly file spanning [start, end] and validate it."""
    found: list[Week] = []
    for year, index in candidates(start, end):
        if found and year == end.year and found[-1].first > end:
            break  # past the pinned range in the last year: nothing more needed
        path = week_path(raw_dir, year, index)
        label = f"{year}/{index}"
        if path.exists():
            raw = path.read_bytes()
            state = "cached"
        else:
            fetched = source.get(year, index)
            if fetched is None:
                continue
            raw, state = fetched, "fetched"
        coverage = read_coverage(raw, label)
        if coverage is None:
            progress(f"empty {label} (vendor placeholder, treated as unpublished)")
            continue
        first, last = coverage
        if state == "fetched":
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            partial.write_bytes(raw)
            partial.replace(path)
        progress(f"{state} {label} ({first:%Y-%m-%d} to {last:%Y-%m-%d})")
        found.append(Week(year, index, path, first, last))

    weeks = sorted(
        (w for w in found if w.last >= start and w.first <= end),
        key=lambda w: w.first,
    )
    check_complete(weeks, start, end)
    progress(
        f"history: {len(weeks)} weekly files, no week missing beyond known vendor gaps"
    )
    return FetchResult(weeks)


def is_known_missing_week(before_last: datetime, after_first: datetime) -> bool:
    """True if the break is exactly one configured vendor-unpublished week."""
    return any(
        before_last < opens
        and after_first > closes
        and after_first - before_last <= MAX_BREAK_BETWEEN_WEEKS + timedelta(days=7)
        for opens, closes in config.KNOWN_MISSING_WEEKS
    )


def check_complete(weeks: list[Week], start: datetime, end: datetime) -> None:
    if not weeks:
        raise HistoryError("no weekly files found in the pinned range")
    if weeks[0].first > start:
        raise HistoryError(f"history does not reach the start {start:%Y-%m-%d}")
    if weeks[-1].last < end:
        raise HistoryError(f"history does not reach the end {end:%Y-%m-%d}")
    breaks = [
        f"{before.last:%Y-%m-%d %H:%M} to {after.first:%Y-%m-%d %H:%M}"
        for before, after in pairwise(weeks)
        if after.first - before.last > MAX_BREAK_BETWEEN_WEEKS
        and not is_known_missing_week(before.last, after.first)
    ]
    if breaks:
        raise HistoryError("week missing between " + "; ".join(breaks))


def run() -> FetchResult:
    return fetch_history(
        default_source(),
        config.DATA_DIR / "raw",
        start=config.HISTORY_START,
        end=config.HISTORY_END,
    )
