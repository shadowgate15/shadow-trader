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

* A 0-byte HTTP 200 means either "no such month" or a transient failure, so it
  is retried before a month is recorded as absent. Year pages list all 12
  months regardless of coverage; only a download proves coverage.
* No checksums are published, so the zip is validated structurally.
* No rate limit was hit at ~400 requests spaced 0.2-0.5 s; we keep that pace.

This is the only code in the project that touches the network, and it does so
only through the ``Transport`` seam so tests run on stubbed responses.
"""

from __future__ import annotations

import io
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from .config import last_complete_month, month_parts
from .dataset import SOURCE, RawCache

BASE_URL = "https://www.histdata.com"
POST_URL = f"{BASE_URL}/get.php"
ZERO_BYTE_ATTEMPTS = 3
PACE_SECONDS = (0.2, 0.5)
_TOKEN = re.compile(
    r"""<input[^>]*\bname=["']tk["'][^>]*\bvalue=["']([^"']+)["']"""
    r"""|<input[^>]*\bvalue=["']([^"']+)["'][^>]*\bname=["']tk["']""",
    re.IGNORECASE,
)


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

    USER_AGENT = "Mozilla/5.0 (compatible; shadow-trader raw-cache fetcher)"

    def _send(self, request: urllib.request.Request) -> Response:
        request.add_header("User-Agent", self.USER_AGENT)
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

    @property
    def ok(self) -> bool:
        """Absent months are warnings; rejected downloads are failures."""
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
) -> FetchReport:
    """Fetch HistData tick months into the raw cache.

    Months already cached are skipped without any network call unless
    ``supersede`` is set, in which case the download becomes a new revision
    file (the old one is kept). ``end`` defaults to the last complete month.
    """
    report = FetchReport()
    first_request = True

    def paced[T](call: Callable[[], T]) -> T:
        nonlocal first_request
        if not first_request:
            sleep(random.uniform(*PACE_SECONDS))
        first_request = False
        return call()

    clock = now()
    for instrument in instruments:
        for period in months_between(start, end or last_complete_month(clock)):
            key = (instrument, period)
            if not supersede and cache.has(SOURCE, instrument, period, "zip"):
                report.skipped.append(key)
                continue
            try:
                _fetch_one(
                    cache, transport, paced, report, instrument, period, now, supersede
                )
            except (OSError, urllib.error.URLError) as exc:
                report.rejected.append((key, f"network error: {exc}"))
    return report


def _fetch_one(
    cache: RawCache,
    transport: Transport,
    paced: Callable,
    report: FetchReport,
    instrument: str,
    period: str,
    now: Callable[[], datetime],
    supersede: bool,
) -> None:
    key = (instrument, period)
    page_url = month_page_url(instrument, period)
    page = paced(lambda: transport.get(page_url, {}))
    match = _TOKEN.search(page.body.decode("utf-8", "replace"))
    if page.status != 200 or match is None:
        report.rejected.append((key, f"no token on month page (HTTP {page.status})"))
        return
    form = {
        "tk": match.group(1) or match.group(2),
        "date": period[:4],
        "datemonth": period,
        "platform": "ASCII",
        "timeframe": "T",
        "fxpair": instrument,
    }
    reply = None
    attempts = 0
    for attempts in range(1, ZERO_BYTE_ATTEMPTS + 1):  # noqa: B007
        reply = paced(lambda: transport.post(POST_URL, form, {"Referer": page_url}))
        if reply.status != 200 or reply.body:
            break
    assert reply is not None
    if reply.status != 200:
        report.rejected.append((key, f"HTTP {reply.status}"))
        return
    if not reply.body:
        report.absent.append(key)
        return
    problem = _validate_zip(reply.body, instrument, period)
    if problem is not None:
        report.rejected.append((key, problem))
        return
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
            "attempts": attempts,
        },
        supersede=supersede,
    )
    report.fetched.append(key)
