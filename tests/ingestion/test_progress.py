"""The progress callback contract: what library code reports to a sink."""

import pytest

from shadow_trader.ingestion.backfill import (
    BuildFailure,
    build_backfill,
    infer_week_clock,
)
from shadow_trader.ingestion.dataset import RawCache
from shadow_trader.ingestion.sources import Response, fetch_months

from .test_backfill import fixture_csv, install_raw, make_config
from .test_histdata_fetch import NOW, StubTransport, ok


class Recorder:
    """A sink that records ``(event, *args)`` tuples."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def start(self, phase, total=None):
        self.events.append(("start", phase, total))

    def advance(self, detail="", outcome=None):
        self.events.append(("advance", detail, outcome))

    def warn(self, message):
        self.events.append(("warn", message))

    def status(self, text):
        self.events.append(("status", text))

    def done(self):
        self.events.append(("done",))

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]


def fetch(tmp_path, transport, sink, start="2024-03", end="2024-04"):
    return fetch_months(
        RawCache(tmp_path),
        transport,
        ("EURUSD",),
        start,
        end,
        sleep=lambda s: None,
        now=lambda: NOW,
        progress=sink,
    )


def test_fetch_reports_each_month_with_its_outcome_between_start_and_done(tmp_path):
    transport = StubTransport({"202403": [ok()], "202404": [Response(200, b"", {})]})
    sink = Recorder()

    fetch(tmp_path, transport, sink)

    assert sink.events == [
        ("start", "fetching months", 2),
        ("advance", "EURUSD 202403", "fetched"),
        ("advance", "EURUSD 202404", "absent"),
        ("done",),
    ]


def test_fetch_reports_cached_unsettled_and_rejected_months(tmp_path):
    fetch(tmp_path, StubTransport({"202403": [ok()]}), Recorder(), end="2024-03")
    transport = StubTransport({"202404": [Response(200, b"junk", {})]})
    sink = Recorder()

    fetch(tmp_path, transport, sink, start="2024-03", end="2024-04")
    fetch(tmp_path, transport, sink, start="2026-10", end="2026-10")

    outcomes = {e[1]: e[2] for e in sink.of("advance")}
    assert outcomes["EURUSD 202403"] == "cached"
    assert outcomes["EURUSD 202404"] == "rejected"
    assert outcomes["EURUSD 202610"] == "unsettled"
    assert any("202404" in w[1] for w in sink.of("warn"))  # why it was rejected


def test_clock_prepass_advances_per_month(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    sink = Recorder()

    infer_week_clock(
        RawCache(config.paths.data_root), "EURUSD", ["202401"], progress=sink
    )

    assert sink.events == [
        ("start", "scanning clocks", 1),
        ("advance", "EURUSD 202401", None),
        ("done",),
    ]


def test_build_reports_clock_scan_then_months_then_writing(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    sink = Recorder()

    build_backfill(config, 5, progress=sink)

    assert [e[:2] for e in sink.of("start")] == [
        ("start", "scanning clocks"),
        ("start", "building months"),
        ("start", "writing dataset"),
    ]
    assert [e[2] for e in sink.of("start")] == [1, 1, None]
    assert len(sink.of("done")) == 3
    assert [e[1] for e in sink.of("advance")] == ["EURUSD 202401", "EURUSD 202401"]


def test_build_surfaces_tolerated_warnings_as_they_happen(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2008-12")
    install_raw(config, "EURUSD", "200812", b"20081207 170000000,1.30000,1.30010,0\n")
    sink = Recorder()

    build_backfill(config, 5, progress=sink)

    assert any("thin_data" in w[1] for w in sink.of("warn"))


def test_build_failure_leaves_the_failing_phase_as_the_last_start(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    sink = Recorder()

    with pytest.raises(BuildFailure, match="202401"):
        build_backfill(config, 5, progress=sink)

    assert sink.events == [("start", "scanning clocks", 1)]


def test_build_surfaces_a_dropped_replay_hour_and_a_fallback_week(tmp_path):
    from .test_week_clock import build, replay_week

    config, _ = build(tmp_path, replay_week(), "2024-04")
    sink = Recorder()

    build_backfill(config, 5, progress=sink)

    assert any("replayed block" in w[1] for w in sink.of("warn"))


def test_live_ingester_reports_status_per_bar_not_per_tick(tmp_path):
    from datetime import timedelta

    from shadow_trader.ingestion.live import Heartbeat, LiveIngester
    from shadow_trader.ingestion.replay import ReplayAdapter

    from .test_dataset import make_config as dataset_config
    from .test_live import OPEN, q, ticking_clock

    sink = Recorder()
    quotes = [q(timedelta(seconds=s)) for s in range(0, 600, 5)]  # 120 ticks
    events = [*quotes, Heartbeat(OPEN + timedelta(minutes=30))]
    LiveIngester(
        dataset_config(tmp_path),
        ReplayAdapter(events, instrument="EURUSD"),
        clock=ticking_clock(),
        start=OPEN,
        progress=sink,
    ).run()

    statuses = [e[1] for e in sink.of("status")]
    assert 0 < len(statuses) < len(quotes) // 4
    assert "EURUSD" in statuses[-1] and "bar emissions" in statuses[-1]
    assert "last tick" in statuses[-1] and "session" in statuses[-1]


def test_build_live_and_diff_report_their_phases(tmp_path):
    from datetime import timedelta

    from shadow_trader.ingestion.livedataset import build_live
    from shadow_trader.ingestion.reconcile import reconcile

    from .test_live import q
    from .test_live_dataset import END, OPEN, record
    from .test_reconcile import bar, datasets

    config = record(tmp_path, [q(timedelta(minutes=1)), q(timedelta(hours=3))])
    sink = Recorder()

    build_live(config, 5, OPEN, END, progress=sink)

    assert [e[:2] for e in sink.of("start")] == [
        ("start", "replaying raw logs"),
        ("start", "writing dataset"),
    ]
    assert len(sink.of("done")) == 2

    live_id, back_id = datasets(config, [bar(0)], [bar(0)])
    sink = Recorder()
    reconcile(config, live_id, back_id, progress=sink)
    assert sink.events == [("start", "reconciling datasets", None), ("done",)]
