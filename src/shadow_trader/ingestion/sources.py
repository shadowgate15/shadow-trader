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
* HTTP 429 is the throttle signal (and the only one, for now: see
  ``THROTTLE_STATUS``). Any worker's 429 starts a global cooldown that every
  worker waits out before its next request: ``Retry-After`` (integer seconds or
  an HTTP date) when usable, else exponential backoff, always capped at 5
  minutes. A 429 does not consume one of a month's ``ATTEMPTS``; the same month
  is retried, and after ``MAX_THROTTLES`` throttles it is rejected as throttled.
* Each 429 also halves the number of requests allowed in flight (minimum 1);
  every ``GROW_AFTER`` consecutive successful requests raise it by one, never
  above ``workers``, and a 429 restarts that run. Only a genuinely good reply
  (a month page with a form, a valid zip) counts towards that run; any other
  failure (HTTP 500, network error, bad or empty reply) restarts it too.
* 429s from requests that were already in flight when a cooldown began are one
  throttle, not several: they do not escalate the backoff or halve the limit
  again. A good reply restarts the backoff.
* Cooldown and backoff waits end early on Ctrl-C, so the run stops promptly.

This is the only code in the project that touches the network, and it does so
only through the ``Transport`` seam so tests run on stubbed responses.
"""

from __future__ import annotations

import io
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Protocol

from .config import DEFAULT_WORKERS, last_complete_month, month_parts
from .dataset import SOURCE, RawCache
from .progress import NULL_PROGRESS, ProgressSink

BASE_URL = "https://www.histdata.com"
POST_URL = f"{BASE_URL}/get.php"
ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 30.0)  # first wait between attempts, and the cap; doubles
THROTTLE_STATUS = 429  # the only throttle signal for now
MAX_THROTTLES = 10  # per month, then it is rejected as throttled
GROW_AFTER = 10  # consecutive successful requests per +1 active worker
MAX_COOLDOWN_SECONDS = 300.0
MAX_BACKOFF_EXPONENT = 10  # 2**10 s already exceeds the cooldown cap
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
    interrupted: bool = False

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


def _retry_after(headers: Mapping[str, str], now: datetime) -> float | None:
    """Seconds a ``Retry-After`` header asks for (integer or HTTP date), if usable."""
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - now).total_seconds())


class _Aborted(Exception):
    """A wait was cut short because the run was interrupted."""


class _Limiter:
    """Shared by all workers: a global cooldown and an adaptive request limit.

    ``slot()`` admits at most ``allowed`` requests at once. A throttle halves
    ``allowed`` (floor 1); every ``GROW_AFTER`` consecutive good replies raise it
    by one, up to ``maximum``. ``abort`` ends waits early (``pause`` raises).
    """

    def __init__(
        self,
        sleep: Callable[[float], None],
        now: Callable[[], datetime],
        maximum: int = 1,
        abort: threading.Event | None = None,
    ) -> None:
        self._sleep = sleep
        self._now = now
        self._abort = abort or threading.Event()
        self._cond = threading.Condition()
        self._until: datetime | None = None  # the cooldown ends here
        self._started: datetime | None = None  # when the active cooldown began
        self._streak = 0  # consecutive throttles, for the backoff fallback
        self._maximum = maximum
        self._allowed = maximum
        self._active = 0  # requests currently holding a slot
        self._good_run = 0  # consecutive good replies

    def pause(self, seconds: float) -> None:
        """Wait ``seconds`` (a cooldown or retry backoff); raise if interrupted."""
        self._sleep(seconds)
        if self._abort.is_set():
            raise _Aborted

    @contextmanager
    def slot(self) -> Generator[datetime]:
        """Hold a request slot after any cooldown; yields when the request began."""
        while True:
            with self._cond:
                remaining = (
                    0.0
                    if self._until is None
                    else (self._until - self._now()).total_seconds()
                )
                if remaining <= 0:
                    if self._active < self._allowed:
                        self._active += 1
                        sent = self._now()
                        break
                    self._cond.wait()
                    continue
            self.pause(remaining)
        try:
            yield sent
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()

    def throttled(self, headers: Mapping[str, str], sent: datetime) -> None:
        """A 429 to a request sent at ``sent``: start or extend the cooldown."""
        with self._cond:
            now = self._now()
            retry_after = _retry_after(headers, now)
            if self._started is not None and sent <= self._started:
                # Already in flight when the last cooldown began: same throttle.
                if retry_after is not None:
                    self._extend(now, retry_after)
                return
            if retry_after is None:
                exponent = min(self._streak, MAX_BACKOFF_EXPONENT)
                retry_after = BACKOFF_SECONDS[0] * 2**exponent
            self._streak = min(self._streak + 1, MAX_BACKOFF_EXPONENT)
            self._started = now
            self._extend(now, retry_after)
            self._allowed = max(1, self._allowed // 2)
            self._good_run = 0

    def _extend(self, now: datetime, seconds: float) -> None:
        until = now + timedelta(seconds=min(seconds, MAX_COOLDOWN_SECONDS))
        if self._until is None or until > self._until:
            self._until = until

    def observe(self, response: Response, sent: datetime) -> None:
        """Account for a reply: 429 eases off, any other bad status is a failure."""
        if response.status == THROTTLE_STATUS:
            self.throttled(response.headers, sent)
        elif response.status != 200:
            self.failed()

    def good(self) -> None:
        """A genuinely good reply: restart the backoff; a good run speeds up."""
        with self._cond:
            self._streak = 0
            self._good_run += 1
            if self._good_run >= GROW_AFTER:
                self._good_run = 0
                if self._allowed < self._maximum:
                    self._allowed += 1
                    self._cond.notify_all()

    def failed(self) -> None:
        """A non-throttle failure: the run of good replies starts over."""
        with self._cond:
            self._good_run = 0


class _Label(StrEnum):
    FETCHED = "fetched"
    CACHED = "cached"
    ABSENT = "absent"
    UNSETTLED = "unsettled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class _Outcome:
    """What one (Instrument, month) task did: its label and any rejection reason."""

    key: Month
    label: _Label
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
    abort = threading.Event()  # set on Ctrl-C: ends cooldown and backoff waits

    def wait(seconds: float) -> None:
        if sleep is time.sleep:
            abort.wait(seconds)
        else:
            sleep(seconds)

    limiter = _Limiter(wait, now, workers, abort)
    interrupted = False

    def record(outcome: _Outcome) -> None:
        outcomes[outcome.key] = outcome
        instrument, period = outcome.key
        if outcome.label is _Label.REJECTED:
            progress.warn(f"{instrument} {period} rejected: {outcome.reason}")
        progress.advance(f"{instrument} {period}", outcome.label)

    queue = iter(tasks)
    pending: set[Future[_Outcome]] = set()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Tasks are handed out lazily (at most ``workers`` in flight), so
            # after an interrupt no further month ever starts.
            try:
                while True:
                    while len(pending) < workers and (task := next(queue, None)):
                        pending.add(
                            pool.submit(
                                _fetch_task,
                                cache,
                                transport,
                                now,
                                clock,
                                supersede,
                                limiter,
                                *task,
                            )
                        )
                    if not pending:
                        break
                    finished, _ = wait_futures(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        pending.discard(future)
                        record(future.result())
            except KeyboardInterrupt:
                interrupted = True
                abort.set()
            # In-flight months finish and are stored; their outcomes are kept.
            # Nothing here may lose the partial report: a worker that fails or
            # was aborted just leaves its month for a re-run, and a second
            # Ctrl-C stops the draining.
            for future in list(pending):
                try:
                    record(future.result())
                except KeyboardInterrupt:
                    abort.set()
                    break
                except _Aborted:
                    pass
                except Exception as exc:
                    progress.warn(f"a month failed while stopping: {exc!r}")
    finally:
        progress.done()
    report = FetchReport(interrupted=interrupted)
    for task in tasks:
        if task not in outcomes:
            continue  # never started: the run was interrupted
        outcome = outcomes[task]
        match outcome.label:
            case _Label.FETCHED:
                report.fetched.append(task)
            case _Label.CACHED:
                report.skipped.append(task)
            case _Label.ABSENT:
                report.absent.append(task)
            case _Label.UNSETTLED:
                report.unsettled.append(task)
            case _Label.REJECTED:
                report.rejected.append((task, outcome.reason))
    return report


def _fetch_task(
    cache: RawCache,
    transport: Transport,
    now: Callable[[], datetime],
    clock: datetime,
    supersede: bool,
    limiter: _Limiter,
    instrument: str,
    period: str,
) -> _Outcome:
    """One self-contained (Instrument, month) task; returns its outcome."""
    key = (instrument, period)
    if not supersede and cache.has(SOURCE, instrument, period, "zip"):
        return _Outcome(key, _Label.CACHED)
    if not is_settled(period, clock):
        return _Outcome(key, _Label.UNSETTLED)
    return _fetch_one(cache, transport, instrument, period, now, supersede, limiter)


def _fetch_one(
    cache: RawCache,
    transport: Transport,
    instrument: str,
    period: str,
    now: Callable[[], datetime],
    supersede: bool,
    limiter: _Limiter,
) -> _Outcome:
    """Fetch one month; its outcome is fetched, absent or rejected."""
    key = (instrument, period)
    page_url = month_page_url(instrument, period)
    headers = {"User-Agent": USER_AGENT}
    delay, cap = BACKOFF_SECONDS
    problem = ""  # why the last attempt failed; empty after only 0-byte replies
    attempt = 0  # failed-or-current non-throttle attempts; a 429 does not count
    throttles = 0
    backoff = False  # a non-throttle failure is awaiting its backoff sleep
    while attempt < ATTEMPTS:
        if throttles >= MAX_THROTTLES:
            return _Outcome(key, _Label.REJECTED, f"throttled {throttles} times")
        if backoff:
            limiter.pause(delay)
            delay = min(delay * 2, cap)
            backoff = False
        attempt += 1
        backoff = True  # unless this attempt is throttled, a failure backs off
        try:
            with limiter.slot() as sent:
                page = transport.get(page_url, headers)
                limiter.observe(page, sent)
            if page.status == THROTTLE_STATUS:
                throttles, attempt, backoff = throttles + 1, attempt - 1, False
                continue
            token = _token(page.body.decode("utf-8", "replace"))
            if page.status != 200 or token is None:
                limiter.failed()
                problem = f"no token on month page (HTTP {page.status})"
                continue
            limiter.good()
            if not token:
                return _Outcome(key, _Label.ABSENT)  # an empty token: no such month
            form = {
                "tk": token,
                "date": period[:4],
                "datemonth": period,
                "platform": "ASCII",
                "timeframe": "T",
                "fxpair": instrument,
            }
            with limiter.slot() as sent:
                download = transport.post(
                    POST_URL, form, {**headers, "Referer": page_url}
                )
                limiter.observe(download, sent)
        except (OSError, urllib.error.URLError) as exc:
            limiter.failed()
            problem = f"network error: {exc}"
            continue
        if download.status == THROTTLE_STATUS:
            throttles, attempt, backoff = throttles + 1, attempt - 1, False
            continue
        if download.status != 200:
            problem = f"HTTP {download.status}"
            continue
        if not download.body:
            limiter.failed()
            problem = ""
            continue
        bad = _validate_zip(download.body, instrument, period)
        if bad is not None:
            limiter.failed()
            problem = bad
            continue
        limiter.good()
        cache.store(
            SOURCE,
            instrument,
            period,
            "zip",
            download.body,
            request={**form, "url": POST_URL, "referer": page_url},
            fetched_at=now(),
            http={
                "status": download.status,
                "headers": dict(download.headers),
                "content_length": len(download.body),
                "attempts": attempt,
            },
            supersede=supersede,
        )
        return _Outcome(key, _Label.FETCHED)
    if problem:
        return _Outcome(key, _Label.REJECTED, problem)
    return _Outcome(key, _Label.ABSENT)
