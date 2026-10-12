import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime

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

    def __init__(self, posts):
        # posts: {datemonth: [Response, ...]}; the last response repeats.
        self.posts = {k: list(v) for k, v in posts.items()}
        self.calls: list[tuple] = []

    def get(self, url, headers):
        self.calls.append(("GET", url, dict(headers)))
        return Response(200, PAGE.encode(), {})

    def post(self, url, form, headers):
        self.calls.append(("POST", url, dict(form), dict(headers)))
        queue = self.posts[form["datemonth"]]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def posts_for(self, month):
        return [c for c in self.calls if c[0] == "POST" and c[2]["datemonth"] == month]


def run(
    tmp_path, transport, *, instruments=("EURUSD",), start="2024-03", end="2024-03"
):
    sleeps: list[float] = []
    report = fetch_months(
        RawCache(tmp_path),
        transport,
        instruments,
        start,
        end,
        sleep=sleeps.append,
        now=lambda: NOW,
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


def test_requests_are_paced_politely(tmp_path):
    transport = StubTransport({"202403": [ok()], "202404": [ok(month="202404")]})

    report, sleeps = run(tmp_path, transport, end="2024-04")

    assert report.fetched == [("EURUSD", "202403"), ("EURUSD", "202404")]
    assert sleeps  # waits happen between requests
    assert all(0.2 <= s <= 0.5 for s in sleeps)


def test_range_spans_instruments_and_defaults_end_to_last_complete_month(tmp_path):
    months = {f"2026{m:02d}": [ok(month=f"2026{m:02d}")] for m in (8, 9)}
    transport = StubTransport(months)

    report, _ = run(
        tmp_path, transport, instruments=("EURUSD",), start="2026-08", end=None
    )

    # NOW is 2026-10-12, so the last complete month is 2026-09.
    assert [m for _, m in report.fetched] == ["202608", "202609"]
