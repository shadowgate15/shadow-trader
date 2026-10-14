"""Quote sources: the HistData tick fetcher.

WARNING: FRAGILE AND UNDOCUMENTED. HistData has no API and publishes no
terms. This module scrapes the same web form a browser uses: it reads a
hidden token (``tk``) from the month page and POSTs it to ``get.php`` with the
month page as ``Referer``. Any change to that form, the token, the file naming
or the 0-byte-reply behaviour can break it without notice. That is accepted
(ADR 0002): the write-once raw cache (``dataset.RawCache``) absorbs the
fragility, because everything downstream reads only the cache, and a source
change later means a new Dataset ``source``, not an edit here.

Observed behaviour this code relies on:

* A month page whose form carries an empty ``tk`` token means "no such month";
  that is final. A 0-byte HTTP 200 to the POST means either that or a
  transient failure, so it is retried before the month is recorded as absent.
  Year pages list all 12 months regardless of coverage; only a download proves
  coverage.
* Every attempt reloads the month page for a fresh token, and a failed attempt
  is retried with exponential backoff (``BACKOFF_SECONDS``).
* A month's file may still be revised just after the month ends, so a month is
  only fetched once it has settled (``SETTLE_DAYS`` after its end). The raw
  cache is write-once, so a premature download would be frozen as final; an
  unsettled month is reported and left for a later run.
* No checksums are published, so the zip is validated structurally.
* Months are fetched concurrently by a pool of worker threads (``workers``),
  one task per (Instrument, month); a month's token-page and zip requests run
  in the same worker, so the token never crosses threads. There is no fixed
  pacing: the worker count is the only speed control. An earlier sequential
  run saw no rate limit at ~400 requests spaced 0.2-0.5 s; behaviour under
  concurrent load is unmeasured, so keep the worker count modest.

This is the only code in the project that touches the network, and it does so
only through the ``Transport`` seam so tests run on stubbed responses.
"""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from .config import last_complete_month, month_parts
from .dataset import SOURCE, RawCache
from .progress import NULL_PROGRESS, ProgressSink

BASE_URL = "https://www.histdata.com"
POST_URL = f"{BASE_URL}/get.php"
ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 30.0)  # first wait between attempts, and the cap; doubles
DEFAULT_WORKERS = 8
SETTLE_DAYS = 7
USER_AGENT = (
    "shadow-trader/0.1 raw-cache fetcher "
    "(+https://github.com/shadowgate15/shadow-trader)"
)
_INPUT = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
_NAME_TK = re.compile(r"""\bname=["']tk["']""", re.IGNORECASE)
_VALUE = re.compile(r"""\bvalue=["']([^"']*)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: Mapping[str, str]


class Transport(Protocol):
    """The network seam. Tests supply a stub; production uses urllib."""

    def get(self, url: str, headers: Mapping[str, str]) -> Response: ...

    def post(
        self, url: str, form: Mapping[str, str], headers: Mapping[str, str]
    ) -> Response: ...


class UrllibTransport:
    """The real network transport (stdlib only; never used in tests)."""

    def _send(self, request: urllib.request.Request) -> Response:
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                return Response(reply.status, reply.read(), dict(reply.headers))
        except urllib.error.HTTPError as exc:
            return Response(exc.code, exc.read(), dict(exc.headers))

    def get(self, url: str, headers: Mapping[str, str]) -> Response:
        return self._send(urllib.request.Request(url, headers=dict(headers)))

    def post(
        self, url: str, form: Mapping[str, str], headers: Mapping[str, str]
    ) -> Response:
        body = urllib.parse.urlencode(form).encode()
        return self._send(urllib.request.Request(url, body, dict(headers)))


Month = tuple[str, str]  # (instrument, YYYYMM)


@dataclass
class FetchReport:
    fetched: list[Month] = field(default_factory=list)
    skipped: list[Month] = field(default_factory=list)
    absent: list[Month] = field(default_factory=list)
    rejected: list[tuple[Month, str]] = field(default_factory=list)
    unsettled: list[Month] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Absent and unsettled months are warnings; rejected downloads fail."""
        return not self.rejected


def month_page_url(instrument: str, period: str) -> str:
    year, month = int(period[:4]), int(period[4:])
    return (
        f"{BASE_URL}/download-free-forex-historical-data/"
        f"?/ascii/tick-data-quotes/{instrument.lower()}/{year}/{month}"
    )


def months_between(start: str, end: str) -> list[str]:
    """Inclusive ``YYYYMM`` periods from ``YYYY-MM`` start to ``YYYY-MM`` end."""
    year, month = month_parts(start)
    last = month_parts(end)
    periods = []
    while (year, month) <= last:
        periods.append(f"{year:04d}{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return periods


def is_settled(period: str, now: datetime) -> bool:
    """Whether the month's file is final: ``SETTLE_DAYS`` past the month's end."""
    year, month = int(period[:4]), int(period[4:])
    end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
    return now >= end + timedelta(days=SETTLE_DAYS)


def last_settled_month(now: datetime) -> str:
    """The last month whose file has settled by ``now``, as ``YYYY-MM``."""
    month = last_complete_month(now)
    if is_settled(month.replace("-", ""), now):
        return month
    year, number = month_parts(month)
    year, number = (year - 1, 12) if number == 1 else (year, number - 1)
    return f"{year:04d}-{number:02d}"


def _token(page: str) -> str | None:
    """The download form's ``tk`` value; ``""`` for an empty one, None if absent."""
    for tag in _INPUT.findall(page):
        if _NAME_TK.search(tag):
            value = _VALUE.search(tag)
            return value.group(1) if value else ""
    return None


def _validate_zip(data: bytes, instrument: str, period: str) -> str | None:
    """Return a reason the zip is bad, or None if it is good."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            bad = archive.testzip()
            if bad is not None:
                return f"corrupt member {bad}"
            expected = f"DAT_ASCII_{instrument}_T_{period}.csv"
            if expected not in archive.namelist():
                return f"missing {expected}"
    except zipfile.BadZipFile as exc:
        return f"invalid zip: {exc}"
    return None


@dataclass(frozen=True)
class _Outcome:
    """What one (Instrument, month) task did: its label and any rejection reason."""

    key: Month
    label: str  # fetched, cached, absent, unsettled or rejected
    reason: str = ""


def fetch_months(
    cache: RawCache,
    transport: Transport,
    instruments: Sequence[str],
    start: str,
    end: str | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    supersede: bool = False,
    progress: ProgressSink = NULL_PROGRESS,
    workers: int = DEFAULT_WORKERS,
) -> FetchReport:
    """Fetch HistData tick months into the raw cache.

    Months already cached are skipped without any network call unless
    ``supersede`` is set, in which case the download becomes a new revision
    file (the old one is kept). ``end`` defaults to the last settled month, and
    a month in range that has not settled yet is not fetched (``unsettled``).

    Months are fetched by ``workers`` threads draining one flat queue across all
    Instruments. Only the calling thread touches ``progress``; the report lists
    are in (Instrument, month) order whatever order the months finish in.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    clock = now()
    periods = months_between(start, end or last_settled_month(clock))
    tasks = [(i, p) for i in instruments for p in periods]
    progress.start("fetching months", len(tasks))
    outcomes: dict[Month, _Outcome] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _fetch_task, cache, transport, sleep, now, clock, supersede, *task
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes[outcome.key] = outcome
            instrument, period = outcome.key
            if outcome.label == "rejected":
                progress.warn(f"{instrument} {period} rejected: {outcome.reason}")
            progress.advance(f"{instrument} {period}", outcome.label)
    progress.done()
    report = FetchReport()
    for task in tasks:
        outcome = outcomes[task]
        match outcome.label:
            case "fetched":
                report.fetched.append(task)
            case "cached":
                report.skipped.append(task)
            case "absent":
                report.absent.append(task)
            case "unsettled":
                report.unsettled.append(task)
            case _:
                report.rejected.append((task, outcome.reason))
    return report


def _fetch_task(
    cache: RawCache,
    transport: Transport,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
    clock: datetime,
    supersede: bool,
    instrument: str,
    period: str,
) -> _Outcome:
    """One self-contained (Instrument, month) task; returns its outcome."""
    key = (instrument, period)
    if not supersede and cache.has(SOURCE, instrument, period, "zip"):
        return _Outcome(key, "cached")
    if not is_settled(period, clock):
        return _Outcome(key, "unsettled")
    return _fetch_one(cache, transport, sleep, instrument, period, now, supersede)


def _fetch_one(
    cache: RawCache,
    transport: Transport,
    sleep: Callable[[float], None],
    instrument: str,
    period: str,
    now: Callable[[], datetime],
    supersede: bool,
) -> _Outcome:
    """Fetch one month; its outcome is fetched, absent or rejected."""
    key = (instrument, period)
    page_url = month_page_url(instrument, period)
    headers = {"User-Agent": USER_AGENT}
    delay, cap = BACKOFF_SECONDS
    problem = ""  # why the last attempt failed; empty after only 0-byte replies
    for attempt in range(1, ATTEMPTS + 1):
        if attempt > 1:
            sleep(delay)
            delay = min(delay * 2, cap)
        try:
            page = transport.get(page_url, headers)
            token = _token(page.body.decode("utf-8", "replace"))
            if page.status != 200 or token is None:
                problem = f"no token on month page (HTTP {page.status})"
                continue
            if not token:
                return _Outcome(key, "absent")  # an empty token: no such month
            form = {
                "tk": token,
                "date": period[:4],
                "datemonth": period,
                "platform": "ASCII",
                "timeframe": "T",
                "fxpair": instrument,
            }
            reply = transport.post(POST_URL, form, {**headers, "Referer": page_url})
        except (OSError, urllib.error.URLError) as exc:
            problem = f"network error: {exc}"
            continue
        if reply.status != 200:
            problem = f"HTTP {reply.status}"
            continue
        if not reply.body:
            problem = ""
            continue
        bad = _validate_zip(reply.body, instrument, period)
        if bad is not None:
            problem = bad
            continue
        cache.store(
            SOURCE,
            instrument,
            period,
            "zip",
            reply.body,
            request={**form, "url": POST_URL, "referer": page_url},
            fetched_at=now(),
            http={
                "status": reply.status,
                "headers": dict(reply.headers),
                "content_length": len(reply.body),
                "attempts": attempt,
            },
            supersede=supersede,
        )
        return _Outcome(key, "fetched")
    if problem:
        return _Outcome(key, "rejected", problem)
    return _Outcome(key, "absent")
