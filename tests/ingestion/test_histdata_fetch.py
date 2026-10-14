import hashlib
import io
import json
import threading
import time
import zipfile
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from shadow_trader.ingestion.dataset import RawCache
from shadow_trader.ingestion.sources import Response, fetch_months

NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
PAGE = (
    '<html><form id="file_down" action="/get.php">'
    '<input type="hidden" name="tk" id="tk" value="TOKEN123" />'
    "</form></html>"
)


def make_zip(pair: str, month: str, csv: bytes = b"20240301 000000000,1.1,1.2,0\n"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"DAT_ASCII_{pair}_T_{month}.csv", csv)
        zf.writestr(f"DAT_ASCII_{pair}_T_{month}.txt", b"status")
    return buf.getvalue()


class StubTransport:
    """Recorded-response stand-in for the HistData site. Never touches a network."""

    def __init__(self, posts, pages=None):
        # posts: {datemonth: [Response, ...]}; the last response repeats, and so
        # does the last of ``pages`` (month-page replies, default one good page).
        self.posts = {k: list(v) for k, v in posts.items()}
        self.pages = list(pages or [Response(200, PAGE.encode(), {})])
        self.calls: list[tuple] = []
        self._lock = threading.Lock()

    def get(self, url, headers):
        with self._lock:
            self.calls.append(("GET", url, dict(headers)))
            page = self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]
        if isinstance(page, Exception):
            raise page
        return page

    def gets(self):
        return [c for c in self.calls if c[0] == "GET"]

    def post(self, url, form, headers):
        with self._lock:
            self.calls.append(("POST", url, dict(form), dict(headers)))
            queue = self.posts[form["datemonth"]]
            return queue.pop(0) if len(queue) > 1 else queue[0]

    def posts_for(self, month):
        return [c for c in self.calls if c[0] == "POST" and c[2]["datemonth"] == month]


def run(
    tmp_path,
    transport,
    *,
    instruments=("EURUSD",),
    start="2024-03",
    end="2024-03",
    now=NOW,
    workers=8,
    progress=None,
):
    sleeps: list[float] = []
    report = fetch_months(
        RawCache(tmp_path),
        transport,
        instruments,
        start,
        end,
        sleep=sleeps.append,
        now=lambda: now,
        workers=workers,
        **({"progress": progress} if progress else {}),
    )
    return report, sleeps


def ok(pair="EURUSD", month="202403"):
    return Response(200, make_zip(pair, month), {"Content-Type": "application/zip"})


def test_fetch_mimics_the_site_form_and_caches_byte_exact(tmp_path):
    transport = StubTransport({"202403": [ok()]})

    report, _ = run(tmp_path, transport)

    page_url = (
        "https://www.histdata.com/download-free-forex-historical-data/"
        "?/ascii/tick-data-quotes/eurusd/2024/3"
    )
    post = transport.posts_for("202403")[0]
    assert post[1] == "https://www.histdata.com/get.php"
    assert post[2] == {
        "tk": "TOKEN123",
        "date": "2024",
        "datemonth": "202403",
        "platform": "ASCII",
        "timeframe": "T",
        "fxpair": "EURUSD",
    }
    assert post[3]["Referer"] == page_url
    assert transport.calls[0][:2] == ("GET", page_url)
    cached = tmp_path / "raw" / "histdata" / "EURUSD" / "202403.zip"
    assert cached.read_bytes() == make_zip("EURUSD", "202403")
    assert report.fetched == [("EURUSD", "202403")]
    assert report.ok


def test_sidecar_matches_stored_bytes_and_records_request(tmp_path):
    run(tmp_path, StubTransport({"202403": [ok()]}))

    cached = tmp_path / "raw" / "histdata" / "EURUSD" / "202403.zip"
    sidecar = json.loads(cached.with_name("202403.zip.meta.json").read_text())
    assert sidecar["sha256"] == hashlib.sha256(cached.read_bytes()).hexdigest()
    assert sidecar["request"]["datemonth"] == "202403"
    assert sidecar["request"]["fxpair"] == "EURUSD"
    assert sidecar["fetched_at"] == NOW.isoformat()
    assert sidecar["http"]["status"] == 200
    assert sidecar["http"]["attempts"] == 1
    assert sidecar["http"]["headers"]["Content-Type"] == "application/zip"


def test_zero_byte_reply_is_retried_before_the_month_is_absent(tmp_path):
    transport = StubTransport({"202403": [Response(200, b"", {}), ok()]})

    report, _ = run(tmp_path, transport)

    assert len(transport.posts_for("202403")) == 2
    assert report.fetched == [("EURUSD", "202403")]
    assert report.absent == []


def test_persistent_zero_byte_reply_records_absent_and_caches_nothing(tmp_path):
    transport = StubTransport({"202403": [Response(200, b"", {})]})

    report, _ = run(tmp_path, transport)

    assert len(transport.posts_for("202403")) == 3
    assert report.absent == [("EURUSD", "202403")]
    assert report.ok  # absence is a warning, not a failure
    assert not (tmp_path / "raw" / "histdata" / "EURUSD").exists()


def test_truncated_or_invalid_zip_is_rejected_and_not_cached(tmp_path):
    good = make_zip("EURUSD", "202403")
    transport = StubTransport({"202403": [Response(200, good[: len(good) // 2], {})]})

    report, _ = run(tmp_path, transport)

    assert [m for m, _ in report.rejected] == [("EURUSD", "202403")]
    assert not report.ok
    assert not (tmp_path / "raw" / "histdata" / "EURUSD").exists()


def test_zip_without_the_expected_csv_is_rejected(tmp_path):
    transport = StubTransport({"202403": [ok("GBPUSD", "202403")]})

    report, _ = run(tmp_path, transport)

    assert len(report.rejected) == 1
    assert report.fetched == []


def test_non_200_reply_is_rejected(tmp_path):
    transport = StubTransport({"202403": [Response(503, b"busy", {})]})

    report, _ = run(tmp_path, transport)

    assert len(report.rejected) == 1


def test_rerun_skips_cached_months_without_any_network_or_rewrite(tmp_path):
    run(tmp_path, StubTransport({"202403": [ok()]}))
    cached = tmp_path / "raw" / "histdata" / "EURUSD" / "202403.zip"
    before = (cached.stat().st_mtime_ns, cached.read_bytes())
    transport = StubTransport({"202403": [ok()]})

    report, _ = run(tmp_path, transport)

    assert transport.calls == []
    assert report.skipped == [("EURUSD", "202403")]
    assert (cached.stat().st_mtime_ns, cached.read_bytes()) == before


def test_no_fixed_pacing_sleep_between_requests(tmp_path):
    transport = StubTransport({"202403": [ok()], "202404": [ok(month="202404")]})

    report, sleeps = run(tmp_path, transport, end="2024-04")

    assert report.fetched == [("EURUSD", "202403"), ("EURUSD", "202404")]
    assert sleeps == []


def test_range_spans_instruments_and_defaults_end_to_last_complete_month(tmp_path):
    months = {f"2026{m:02d}": [ok(month=f"2026{m:02d}")] for m in (8, 9)}
    transport = StubTransport(months)

    report, _ = run(
        tmp_path, transport, instruments=("EURUSD",), start="2026-08", end=None
    )

    # NOW is 2026-10-12, so the last complete month is 2026-09.
    assert [m for _, m in report.fetched] == ["202608", "202609"]


def page_with(token: str) -> Response:
    return Response(200, PAGE.replace("TOKEN123", token).encode(), {})


def backoffs(sleeps):
    """The waits between attempts (pacing waits are all under a second)."""
    return [s for s in sleeps if s >= 1]


def test_failed_attempts_back_off_exponentially_before_succeeding(tmp_path):
    bad = Response(503, b"busy", {})
    transport = StubTransport({"202403": [bad, bad, ok()]})

    report, sleeps = run(tmp_path, transport)

    assert report.fetched == [("EURUSD", "202403")]
    assert backoffs(sleeps) == [1.0, 2.0]


def test_every_attempt_fetches_a_fresh_form_token(tmp_path):
    transport = StubTransport(
        {"202403": [Response(200, b"", {}), ok()]},
        pages=[page_with("FIRST"), page_with("SECOND")],
    )

    run(tmp_path, transport)

    assert [c[2]["tk"] for c in transport.posts_for("202403")] == ["FIRST", "SECOND"]
    assert len(transport.gets()) == 2


def test_empty_token_means_no_such_month_without_posting_or_retrying(tmp_path):
    transport = StubTransport({"202403": [ok()]}, pages=[page_with("")])

    report, sleeps = run(tmp_path, transport)

    assert report.absent == [("EURUSD", "202403")]
    assert report.ok
    assert transport.posts_for("202403") == []
    assert len(transport.gets()) == 1
    assert backoffs(sleeps) == []
    assert not (tmp_path / "raw" / "histdata" / "EURUSD").exists()


def test_month_page_without_a_token_field_is_retried_then_rejected(tmp_path):
    transport = StubTransport(
        {"202403": [ok()]}, pages=[Response(200, b"<html></html>", {})]
    )

    report, sleeps = run(tmp_path, transport)

    assert [m for m, _ in report.rejected] == [("EURUSD", "202403")]
    assert len(transport.gets()) == 3
    assert transport.posts_for("202403") == []
    assert backoffs(sleeps) == [1.0, 2.0]


def test_network_error_is_retried(tmp_path):
    transport = StubTransport(
        {"202403": [ok()]}, pages=[OSError("reset"), Response(200, PAGE.encode(), {})]
    )

    report, _ = run(tmp_path, transport)

    assert report.fetched == [("EURUSD", "202403")]


def test_requests_carry_a_descriptive_user_agent(tmp_path):
    transport = StubTransport({"202403": [ok()]})

    run(tmp_path, transport)

    agents = [c[-1].get("User-Agent", "") for c in transport.calls]
    assert len(agents) == 2
    assert all(a.startswith("shadow-trader/") and "github.com" in a for a in agents)


def test_month_not_yet_settled_is_not_fetched_or_cached_as_final(tmp_path):
    early = datetime(2026, 10, 6, 23, 0, tzinfo=UTC)  # Sept ended 6 d 23 h ago
    transport = StubTransport({"202609": [ok(month="202609")]})

    report, _ = run(tmp_path, transport, start="2026-09", end="2026-09", now=early)

    assert report.unsettled == [("EURUSD", "202609")]
    assert report.fetched == []
    assert report.ok  # waiting is not a failure
    assert transport.calls == []
    assert not (tmp_path / "raw" / "histdata" / "EURUSD").exists()


def test_month_is_fetched_once_seven_days_after_it_ended(tmp_path):
    settled = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    transport = StubTransport({"202609": [ok(month="202609")]})

    report, _ = run(tmp_path, transport, start="2026-09", end="2026-09", now=settled)

    assert report.fetched == [("EURUSD", "202609")]
    assert report.unsettled == []


def test_default_end_is_the_last_settled_month(tmp_path):
    early = datetime(2026, 10, 5, tzinfo=UTC)  # September has not settled yet
    transport = StubTransport({"202608": [ok(month="202608")]})

    report, _ = run(tmp_path, transport, start="2026-08", end=None, now=early)

    assert [m for _, m in report.fetched] == ["202608"]
    assert report.unsettled == []


class GateTransport:
    """Blocks every month-page request until ``size`` are in flight at once."""

    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=10)
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0

    def get(self, url, headers):
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            self.barrier.wait()
        finally:
            with self.lock:
                self.inflight -= 1
        return Response(200, PAGE.encode(), {})

    def post(self, url, form, headers):
        month = form["datemonth"]
        return ok(month=month)


def test_up_to_n_requests_are_in_flight_and_never_more(tmp_path):
    transport = GateTransport(3)

    report, _ = run(tmp_path, transport, start="2024-01", end="2024-06", workers=3)

    assert len(report.fetched) == 6
    assert transport.peak == 3  # the barrier only opens with 3 in flight


class Recorder:
    def __init__(self, on_advance=None):
        self.threads: set[int] = set()
        self.events: list[tuple] = []
        self.on_advance = on_advance

    def _note(self, *event):
        self.threads.add(threading.get_ident())
        self.events.append(event)

    def start(self, phase, total=None):
        self._note("start", phase, total)

    def advance(self, detail="", outcome=None):
        self._note("advance", detail, outcome)
        if self.on_advance:
            self.on_advance(detail)

    def warn(self, message):
        self._note("warn", message)

    def status(self, text):
        self._note("status", text)

    def done(self):
        self._note("done")


class LaggingTransport(StubTransport):
    """The 202403 download only finishes once 202404 has been reported."""

    def __init__(self, posts):
        super().__init__(posts)
        self.april_reported = threading.Event()

    def post(self, url, form, headers):
        if form["datemonth"] == "202403":
            assert self.april_reported.wait(10)
        return super().post(url, form, headers)


def test_report_keeps_input_order_and_progress_stays_on_the_calling_thread(tmp_path):
    transport = LaggingTransport({"202403": [ok()], "202404": [ok(month="202404")]})
    sink = Recorder(
        on_advance=lambda d: transport.april_reported.set() if "202404" in d else None
    )

    report, _ = run(tmp_path, transport, end="2024-04", workers=2, progress=sink)

    advanced = [e[1] for e in sink.events if e[0] == "advance"]
    assert advanced == ["EURUSD 202404", "EURUSD 202403"]  # completion order
    assert report.fetched == [("EURUSD", "202403"), ("EURUSD", "202404")]
    assert sink.threads == {threading.get_ident()}
    assert sink.events[0] == ("start", "fetching months", 2)
    assert sink.events[-1] == ("done",)


def test_rejected_month_warns_from_the_calling_thread(tmp_path):
    sink = Recorder()
    transport = StubTransport({"202403": [Response(200, b"junk", {})]})

    run(tmp_path, transport, progress=sink)

    assert [e for e in sink.events if e[0] == "warn"]
    assert sink.threads == {threading.get_ident()}


def test_workers_must_be_positive(tmp_path):
    import pytest

    for bad in (0, -1):
        with pytest.raises(ValueError, match="workers"):
            run(tmp_path, StubTransport({"202403": [ok()]}), workers=bad)


class FakeClock:
    """Injected ``now``/``sleep``: sleeping advances time; nothing really waits."""

    def __init__(self, start=NOW):
        self.t = start
        self.slept: list[float] = []
        self._lock = threading.Lock()

    def now(self):
        with self._lock:
            return self.t

    def sleep(self, seconds):
        with self._lock:
            self.slept.append(seconds)
            self.t += timedelta(seconds=seconds)


class TimedTransport(StubTransport):
    """Records the fake-clock time of every request."""

    def __init__(self, clock, posts, pages=None):
        super().__init__(posts, pages)
        self.clock = clock
        self.times: list[tuple[str, datetime]] = []

    def get(self, url, headers):
        self.times.append(("GET", self.clock.now()))
        return super().get(url, headers)

    def post(self, url, form, headers):
        self.times.append(("POST", self.clock.now()))
        return super().post(url, form, headers)


def run_clocked(tmp_path, transport, clock, *, workers=8, **kwargs):
    return fetch_months(
        RawCache(tmp_path),
        transport,
        kwargs.pop("instruments", ("EURUSD",)),
        kwargs.pop("start", "2024-03"),
        kwargs.pop("end", "2024-03"),
        sleep=clock.sleep,
        now=clock.now,
        workers=workers,
    )


def throttled(retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return Response(429, b"", headers)


def test_retry_after_seconds_pauses_every_worker(tmp_path):
    clock = FakeClock()
    transport = TimedTransport(
        clock, {"202403": [throttled("30"), ok()], "202404": [ok(month="202404")]}
    )

    report = run_clocked(tmp_path, transport, clock, start="2024-03", end="2024-04")

    assert report.ok and len(report.fetched) == 2
    # Every request after the first 429 reply happens >= 30 s later.
    throttled_at = min(t for kind, t in transport.times if kind == "POST")
    later = [t for _, t in transport.times if t > throttled_at]
    assert later and all(t >= throttled_at + timedelta(seconds=30) for t in later)


def test_retry_after_http_date_is_honoured(tmp_path):
    clock = FakeClock()
    date = format_datetime(NOW + timedelta(seconds=45), usegmt=True)
    transport = TimedTransport(clock, {"202403": [throttled(date), ok()]})

    run_clocked(tmp_path, transport, clock)

    assert 45 <= sum(clock.slept) < 60


def test_missing_retry_after_backs_off_exponentially(tmp_path):
    clock = FakeClock()
    transport = TimedTransport(
        clock, {"202403": [throttled(), throttled(), throttled(), ok()]}
    )

    report = run_clocked(tmp_path, transport, clock)

    assert report.fetched == [("EURUSD", "202403")]
    waits = clock.slept
    assert len(waits) == 3
    assert waits[0] < waits[1] < waits[2]


def test_cooldown_is_capped_at_five_minutes(tmp_path):
    clock = FakeClock()
    transport = TimedTransport(clock, {"202403": [throttled("86400"), ok()]})

    report = run_clocked(tmp_path, transport, clock)

    assert report.fetched == [("EURUSD", "202403")]
    assert sum(clock.slept) == 300


def test_throttles_do_not_use_up_attempts(tmp_path):
    clock = FakeClock()
    transport = TimedTransport(clock, {"202403": [*[throttled("1")] * 5, ok()]})

    report = run_clocked(tmp_path, transport, clock)

    assert report.fetched == [("EURUSD", "202403")]


def test_other_failures_still_use_up_attempts(tmp_path):
    clock = FakeClock()
    boom = Response(500, b"", {})
    transport = TimedTransport(clock, {"202403": [boom, boom, boom, ok()]})

    report = run_clocked(tmp_path, transport, clock)

    assert report.rejected == [(("EURUSD", "202403"), "HTTP 500")]


def test_429_on_the_month_page_is_a_throttle_too(tmp_path):
    clock = FakeClock()
    pages = [throttled("5")] * 5 + [Response(200, PAGE.encode(), {})]
    transport = TimedTransport(clock, {"202403": [ok()]}, pages=pages)

    report = run_clocked(tmp_path, transport, clock)

    assert report.fetched == [("EURUSD", "202403")]


def test_endlessly_throttled_month_is_rejected_as_throttled(tmp_path):
    clock = FakeClock()
    transport = TimedTransport(clock, {"202403": [throttled("1")]})

    report = run_clocked(tmp_path, transport, clock)

    ((key, reason),) = report.rejected
    assert key == ("EURUSD", "202403")
    assert "throttled" in reason
    assert 9 <= len(transport.posts_for("202403")) <= 11


class InterruptingSink:
    """Releases the in-flight month, then raises Ctrl-C when March is reported."""

    def __init__(self, release):
        self.release = release

    def start(self, phase, total=None):
        pass

    def advance(self, detail="", outcome=None):
        if "202403" in detail:
            self.release.set()
            raise KeyboardInterrupt

    def warn(self, message):
        pass

    def status(self, text):
        pass

    def done(self):
        pass


class GatedTransport(StubTransport):
    """April's download blocks until released, so it is in flight at the interrupt."""

    def __init__(self, posts, gate):
        super().__init__(posts)
        self.gate = gate

    def post(self, url, form, headers):
        if form["datemonth"] == "202404":
            assert self.gate.wait(timeout=10)
        return super().post(url, form, headers)


def test_interrupt_stops_dispatch_finishes_in_flight_and_rerun_fetches_the_rest(
    tmp_path,
):
    gate = threading.Event()
    posts = {m: [ok(month=m)] for m in ("202403", "202404", "202405", "202406")}
    transport = GatedTransport(posts, gate)

    report, _ = run(
        tmp_path,
        transport,
        end="2024-06",
        workers=2,
        progress=InterruptingSink(gate),
    )

    assert report.interrupted
    assert report.fetched == [("EURUSD", "202403"), ("EURUSD", "202404")]
    assert transport.posts_for("202405") == []
    assert transport.posts_for("202406") == []
    assert (tmp_path / "raw" / "histdata" / "EURUSD" / "202404.zip").exists()

    again = StubTransport(posts)
    rerun, _ = run(tmp_path, again, end="2024-06", workers=2)

    assert not rerun.interrupted
    assert rerun.skipped == [("EURUSD", "202403"), ("EURUSD", "202404")]
    assert rerun.fetched == [("EURUSD", "202405"), ("EURUSD", "202406")]


def test_an_uninterrupted_run_is_not_marked_interrupted(tmp_path):
    report, _ = run(tmp_path, StubTransport({"202403": [ok()]}))

    assert not report.interrupted


class AdaptiveClock(FakeClock):
    """FakeClock that flags the first cooldown wait: the throttle is fully handled."""

    def __init__(self):
        super().__init__()
        self.cooling = threading.Event()

    def sleep(self, seconds):
        super().sleep(seconds)
        self.cooling.set()


class ScriptedTransport:
    """Requests are numbered in start order.

    The first ``workers`` requests are the initial group: request 1 is throttled
    and the rest stay in flight until the cooldown begins, then succeed. Later
    requests (numbered from 1 as ``m``) are "new": ``throttle_at`` makes some
    throttled, ``barriers`` makes some wait for company, and ``at_start`` records
    how many new requests were in flight (itself included) when each started.
    """

    def __init__(self, clock, workers, throttle_at=(), barriers=None):
        self.clock = clock
        self.workers = workers
        self.throttle_at = set(throttle_at)
        self.barriers = barriers or {}
        self.lock = threading.Lock()
        self.count = 0
        self.new_inflight = 0
        self.at_start: dict[int, int] = {}

    def get(self, url, headers):
        if self._request():
            return throttled("1")
        return Response(200, PAGE.encode(), {})

    def post(self, url, form, headers):
        if self._request():
            return throttled("1")
        return ok(month=form["datemonth"])

    def _request(self) -> bool:
        """Block as scripted; True when this request is to be throttled."""
        with self.lock:
            self.count += 1
            n = self.count
            m = n - self.workers
            if m > 0:
                self.new_inflight += 1
                self.at_start[m] = self.new_inflight
        try:
            if n == 1:
                return True
            if m <= 0:
                assert self.clock.cooling.wait(timeout=10)
                return False
            if m in self.barriers:
                self.barriers[m].wait()
            else:
                time.sleep(0.005)  # give any permitted overlap a chance to occur
            return m in self.throttle_at
        finally:
            if m > 0:
                with self.lock:
                    self.new_inflight -= 1

    def peak(self, lo, hi):
        return max(v for m, v in self.at_start.items() if lo <= m <= hi)


def run_adaptive(tmp_path, transport, clock, workers):
    return run_clocked(
        tmp_path, transport, clock, workers=workers, start="2023-01", end="2024-04"
    )


def test_a_throttle_halves_the_requests_in_flight_with_a_floor_of_one(tmp_path):
    clock = AdaptiveClock()
    transport = ScriptedTransport(clock, workers=4)

    report = run_adaptive(tmp_path, transport, clock, workers=4)

    assert report.ok
    # 4 -> 2. Three initial successes plus seven more are needed to regrow.
    assert transport.peak(1, 7) <= 2


def test_a_throttle_at_one_active_worker_keeps_one(tmp_path):
    clock = AdaptiveClock()
    transport = ScriptedTransport(clock, workers=2, throttle_at={3})

    report = run_adaptive(tmp_path, transport, clock, workers=2)

    assert report.ok
    assert transport.peak(1, 9) == 1  # 2 -> 1, then another 429 leaves 1


def test_active_workers_grow_by_one_per_ten_consecutive_successes(tmp_path):
    clock = AdaptiveClock()
    # 3 workers, throttled to 1. The 2 initial successes count towards the run,
    # so the 8th new success is the 10th (limit 2) and the 18th is the 20th (3).
    pair, trio = threading.Barrier(2, timeout=10), threading.Barrier(3, timeout=10)
    barriers = {9: pair, 10: pair, 19: trio, 20: trio, 21: trio}
    transport = ScriptedTransport(clock, workers=3, barriers=barriers)

    report = run_adaptive(tmp_path, transport, clock, workers=3)

    assert report.ok
    assert transport.peak(1, 8) == 1
    assert transport.peak(9, 10) == 2  # barrier: two ran together
    assert transport.peak(11, 18) <= 2
    assert transport.peak(19, 21) == 3  # barrier: three ran together
    assert transport.peak(1, 60) == 3  # never above the configured maximum


def test_a_throttle_resets_the_success_run(tmp_path):
    clock = AdaptiveClock()
    # 2 workers, throttled to 1. Request m=7 is throttled again, after which 10
    # more successes (m=8..17) are needed. Without a reset the limit would grow
    # earlier, once 10 successes in all had been seen.
    pair = threading.Barrier(2, timeout=10)
    transport = ScriptedTransport(
        clock, workers=2, throttle_at={7}, barriers={18: pair, 19: pair}
    )

    report = run_adaptive(tmp_path, transport, clock, workers=2)

    assert report.ok
    assert transport.peak(8, 17) == 1
    assert transport.peak(18, 19) == 2
