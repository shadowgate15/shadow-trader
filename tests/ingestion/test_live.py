"""Live ingester behaviour, driven through LiveIngester and the replay adapter."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow_trader.ingestion.anomaly import FlaggingBarBuilder
from shadow_trader.ingestion.bars import Quote
from shadow_trader.ingestion.config import AnomalyRuleConfig
from shadow_trader.ingestion.live import Heartbeat, LiveIngester, SourceDown, SourceUp
from shadow_trader.ingestion.rawlog import read_raw_log
from shadow_trader.ingestion.replay import ReplayAdapter
from shadow_trader.ingestion.session import session_of

from .test_dataset import make_config

OPEN = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)  # Session open


def q(offset: timedelta, bid: str = "1.10000", ask: str = "1.10010") -> Quote:
    return Quote(OPEN + offset, Decimal(bid), Decimal(ask))


def ticking_clock():
    """A deterministic clock: each call is one second after the previous."""
    base = datetime(2030, 1, 1, tzinfo=UTC)
    calls = iter(range(10**6))
    return lambda: base + timedelta(seconds=next(calls))


def live(tmp_path, events, **kwargs):
    """Run the events; return (emitted [(Bar, revision)], ingester)."""
    emitted = []
    ingester = LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(events, instrument="EURUSD"),
        sink=lambda bar, revision: emitted.append((bar, revision)),
        **kwargs,
    )
    ingester.run()
    return emitted, ingester


def test_bar_closes_when_the_watermark_passes_slot_end_plus_grace(tmp_path):
    emitted: list = []
    emitted_after_event: list[int] = []

    def events():
        for event in (
            q(timedelta(minutes=1)),
            q(timedelta(minutes=5, milliseconds=999)),  # inside grace: slot 0 open
            Heartbeat(OPEN + timedelta(minutes=5, seconds=1)),  # end + 1 s: closes
        ):
            yield event
            emitted_after_event.append(len(emitted))

    ingester = LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(events(), instrument="EURUSD"),
        sink=lambda bar, revision: emitted.append((bar, revision)),
    )
    ingester.run()

    # recorded when the generator resumes, i.e. after the ingester handled the event
    assert emitted_after_event == [0, 0, 1]
    bar, revision = emitted[0]
    assert (bar.open_time, bar.tick_count, revision) == (OPEN, 1, 0)


def test_empty_slots_close_as_gap_bars_once_a_heartbeat_passes(tmp_path):
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            Heartbeat(OPEN + timedelta(minutes=15, seconds=1)),
        ],
    )

    assert [(b.open_time, b.is_gap, r) for b, r in emitted[:3]] == [
        (OPEN, False, 0),
        (OPEN + timedelta(minutes=5), True, 0),
        (OPEN + timedelta(minutes=10), True, 0),
    ]


def test_end_of_stream_flushes_only_slots_the_watermark_has_closed(tmp_path):
    emitted, _ = live(
        tmp_path,
        [q(timedelta(minutes=1)), Heartbeat(OPEN + timedelta(minutes=10, seconds=1))],
    )

    # slots 0 and 1 are closed; the rest of the Session is not invented
    assert [(b.open_time, b.is_gap) for b, _ in emitted] == [
        (OPEN, False),
        (OPEN + timedelta(minutes=5), True),
    ]


def test_out_of_order_quotes_in_an_open_slot_order_by_timestamp_then_arrival(tmp_path):
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(seconds=30), bid="1.10030"),
            q(timedelta(seconds=10), bid="1.10010"),  # earlier ts, arrives later
            q(timedelta(seconds=30), bid="1.10040"),  # same ts as first, arrives after
            Heartbeat(OPEN + timedelta(minutes=6)),
        ],
    )

    bar = emitted[0][0]
    assert (bar.open, bar.close, bar.tick_count) == (1.1001, 1.1004, 3)


def test_late_quote_inside_the_repair_window_re_emits_the_slot_as_next_revision(
    tmp_path,
):
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(minutes=1), bid="1.10000"),
            Heartbeat(OPEN + timedelta(minutes=6)),  # slot 0 closes as revision 0
            q(timedelta(minutes=2), bid="1.10020"),  # late: slot 0 is closed
            q(timedelta(minutes=7), bid="1.10050"),
            Heartbeat(OPEN + timedelta(minutes=11)),
        ],
    )

    slot0 = [
        (bar.tick_count, bar.close, rev)
        for bar, rev in emitted
        if bar.open_time == OPEN
    ]
    assert slot0 == [(1, 1.1, 0), (2, 1.1002, 1)]  # revision 0 is kept, not replaced
    assert [(b.open_time, r) for b, r in emitted if b.open_time != OPEN][:1] == [
        (OPEN + timedelta(minutes=5), 0)
    ]


def test_quote_after_the_repair_window_is_dropped_counted_and_logged(tmp_path, caplog):
    sealed_at = timedelta(minutes=5 + 60)  # slot end + repair window
    with caplog.at_level("WARNING"):
        emitted, ingester = live(
            tmp_path,
            [
                q(timedelta(minutes=1)),
                Heartbeat(OPEN + sealed_at),  # slot 0 is sealed from here
                q(timedelta(minutes=2), bid="1.20000"),
            ],
        )

    slot0 = [(bar.tick_count, rev) for bar, rev in emitted if bar.open_time == OPEN]
    assert slot0 == [(1, 0)]  # no further Revision
    assert ingester.late_dropped == 1
    assert "sealed" in caplog.text


def test_quote_just_inside_the_repair_window_is_still_a_revision(tmp_path):
    inside = timedelta(minutes=5 + 60) - timedelta(milliseconds=1)
    emitted, ingester = live(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            Heartbeat(OPEN + inside),
            q(timedelta(minutes=2)),
        ],
    )

    assert [(b.tick_count, r) for b, r in emitted if b.open_time == OPEN] == [
        (1, 0),
        (2, 1),
    ]
    assert ingester.late_dropped == 0


def test_revisions_never_overwrite_and_the_as_seen_view_is_reconstructible(tmp_path):
    clock = ticking_clock()
    _, ingester = live(
        tmp_path,
        [
            q(timedelta(minutes=1), bid="1.10000"),
            Heartbeat(OPEN + timedelta(minutes=6)),  # revision 0 emitted
            q(timedelta(minutes=2), bid="1.10020"),  # revision 1 emitted
        ],
        clock=clock,
    )
    log = ingester.revisions

    history = log.history(OPEN)
    assert [(rev, bar.tick_count) for rev, bar, _ in history] == [(0, 1), (1, 2)]
    first_seen = history[0][2]
    # as seen right after revision 0, the agent saw one tick; the latest view has two
    assert [b.tick_count for b in log.as_seen(first_seen) if b.open_time == OPEN] == [1]
    assert [b.tick_count for b in log.latest() if b.open_time == OPEN] == [2]
    before_anything = first_seen - timedelta(seconds=1)
    assert log.as_seen(before_anything) == []


# --- stream equals batch --------------------------------------------------------

BAR_SIZES = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
SESSION_BASES = (
    OPEN,  # winter Session open
    datetime(2024, 3, 10, 21, 0, tzinfo=UTC) + timedelta(days=7),  # summer open
    OPEN + timedelta(hours=120, minutes=-90),  # 90 min before the weekly close
)
MINUTE_MS = 60_000


@st.composite
def arrival_sequences(draw):
    """In-Session quotes in arrival order: boundary ticks, ticks late by up to 40
    min (inside the 1 h repair window) and ticks out of order within the 1 s grace."""
    base = draw(st.sampled_from(SESSION_BASES))
    count = draw(st.integers(0, 40))
    offsets = st.one_of(
        st.integers(0, 180 * MINUTE_MS),
        st.integers(0, 180).map(lambda m: m * MINUTE_MS),  # slot boundaries
        st.integers(0, 12).map(lambda m: m * 5 * MINUTE_MS),
    )
    delays = st.one_of(st.just(0), st.integers(0, 999), st.integers(0, 40 * MINUTE_MS))
    keyed = []
    for _ in range(count):
        ts = base + timedelta(milliseconds=draw(offsets))
        bid = 110000 + draw(st.integers(-30, 30))
        spread = draw(st.integers(-2, 30))
        quote = Quote(ts, Decimal(bid) / 100000, Decimal(bid + spread) / 100000)
        keyed.append((ts + timedelta(milliseconds=draw(delays)), quote))
    keyed.sort(key=lambda pair: pair[0])  # stable: arrival
    return [quote for _, quote in keyed]


def batch_bars(quotes, minutes):
    """The backfill path: quotes sorted by timestamp (arrival breaks ties)."""
    builder = FlaggingBarBuilder("EURUSD", minutes, 5, AnomalyRuleConfig())
    bars = []
    for quote in sorted(quotes, key=lambda x: x.ts):
        bars.extend(builder.push(quote))
    bars.extend(builder.flush())
    return bars


@pytest.fixture(scope="module")
def shared_config(tmp_path_factory):
    return make_config(tmp_path_factory.mktemp("live"))


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(quotes=arrival_sequences(), minutes=st.sampled_from(BAR_SIZES))
def test_stream_equals_batch_for_sealed_bars(shared_config, quotes, minutes):
    config = shared_config
    # time passing to the Session close is what closes the trailing slots
    sessions = [s for x in quotes if (s := session_of(x.ts)) is not None]
    closing = [Heartbeat(sessions[0][1] + timedelta(minutes=1))] if sessions else []
    ingester = LiveIngester(
        config,
        ReplayAdapter([*quotes, *closing], instrument="EURUSD"),
        sink=lambda bar, revision: None,
        bar_size=minutes,
    )
    ingester.run()

    assert ingester.late_dropped == 0
    assert ingester.revisions.latest() == batch_bars(quotes, minutes)


def test_replay_of_raw_cache_ticks_gives_the_backfill_bars_without_network(tmp_path):
    from shadow_trader.ingestion.backfill import build_backfill
    from shadow_trader.ingestion.dataset import open_dataset

    from .test_backfill import fixture_csv, install_raw
    from .test_backfill import make_config as make_backfill_config

    config = make_backfill_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    backfill = build_backfill(config, 5)
    expected = open_dataset(config.paths.data_root, backfill.ref.dataset_id).read(
        "EURUSD"
    )

    ingester = LiveIngester(
        config,
        ReplayAdapter.from_raw_cache(config, "EURUSD"),
        sink=lambda bar, revision: None,
        bar_size=5,
    )
    ingester.run()

    # prices only: backfill applies its own Gap run-length rule and does not run
    # the tick detector, which the stream-equals-batch property covers instead
    def prices(bars):
        return [dataclasses.replace(b, quality_flags=()) for b in bars]

    assert prices(ingester.revisions.latest()) == prices(expected)
    assert all(
        rev == 0 for rev, _, _ in ingester.revisions.history(expected[0].open_time)
    )


class CatchUpAdapter(ReplayAdapter):
    """A replay adapter that can also supply the quotes missed while down."""

    def __init__(self, events, missed):
        super().__init__(events, instrument="EURUSD")
        self._missed = missed
        self.fetches = []

    def fetch_quotes(self, start, end):
        self.fetches.append((start, end))
        return self._missed


def test_reconnect_catch_up_quotes_repair_the_slot_as_a_late_revision(tmp_path):
    emitted = []
    adapter = CatchUpAdapter(
        [
            q(timedelta(minutes=1)),
            SourceDown(OPEN + timedelta(minutes=2)),
            SourceUp(OPEN + timedelta(minutes=6)),
        ],
        missed=[q(timedelta(minutes=3), bid="1.10030")],
    )
    ingester = LiveIngester(
        make_config(tmp_path),
        adapter,
        sink=lambda bar, revision: emitted.append((bar, revision)),
    )
    ingester.run()

    assert adapter.fetches == [
        (OPEN + timedelta(minutes=2), OPEN + timedelta(minutes=6))
    ]
    assert [(b.tick_count, r) for b, r in emitted if b.open_time == OPEN] == [
        (1, 0),
        (2, 1),
    ]


def test_without_fetch_quotes_a_reconnect_repairs_nothing(tmp_path):
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            SourceDown(OPEN + timedelta(minutes=2)),
            SourceUp(OPEN + timedelta(minutes=6)),
        ],
    )

    assert [(b.tick_count, r) for b, r in emitted if b.open_time == OPEN] == [(1, 0)]


def test_mid_session_start_emits_from_the_first_slot_at_or_after_the_start(tmp_path):
    start = OPEN + timedelta(hours=1, minutes=2)  # inside slot 12 (01:00-01:05)
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(hours=1, minutes=3)),  # in the partial slot: not emitted
            q(timedelta(hours=1, minutes=7)),
            Heartbeat(OPEN + timedelta(hours=1, minutes=16)),
        ],
        start=start,
    )

    first = OPEN + timedelta(hours=1, minutes=5)
    assert [(b.open_time, b.tick_count) for b, _ in emitted[:2]] == [
        (first, 1),
        (first + timedelta(minutes=5), None),
    ]
    assert emitted[0][0].open_time == first
    assert all(b.open_time >= first for b, _ in emitted)


def test_clean_shutdown_flushes_only_genuinely_closed_slots(tmp_path):
    emitted = []
    holder = []

    def events():
        yield q(timedelta(minutes=1))
        yield Heartbeat(OPEN + timedelta(minutes=11))  # slots 0 and 1 are closed
        holder[0].stop()
        yield q(timedelta(minutes=12))  # never handled: the run was stopped

    ingester = LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(events(), instrument="EURUSD"),
        sink=lambda bar, revision: emitted.append((bar, revision)),
    )
    holder.append(ingester)
    ingester.run()

    assert [b.open_time for b, _ in emitted] == [OPEN, OPEN + timedelta(minutes=5)]


def test_gap_bars_overlapping_a_source_down_interval_are_flagged_at_emission(tmp_path):
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            SourceDown(OPEN + timedelta(minutes=7)),  # inside slot 1
            SourceUp(OPEN + timedelta(minutes=11)),  # inside slot 2
            q(timedelta(minutes=12)),
            Heartbeat(OPEN + timedelta(minutes=21)),
        ],
    )

    by_slot = {b.open_time: b for b, _ in emitted}
    flags = [by_slot[OPEN + timedelta(minutes=5 * i)].quality_flags for i in range(4)]
    # slot 1 is a Gap overlapping the outage; slot 2 holds a quote so is priced
    # and unflagged; slot 3 is a plain Gap after the outage (10 min < 30 min rule)
    assert flags == [(), ("suspected_outage",), (), ()]
    assert by_slot[OPEN + timedelta(minutes=5)].is_gap
    assert not by_slot[OPEN + timedelta(minutes=10)].is_gap


ARRIVAL = datetime(2030, 1, 1, 9, 59, 58, tzinfo=UTC)


def arrival_clock(step: timedelta = timedelta(seconds=1), first: datetime = ARRIVAL):
    calls = iter(range(10**6))
    return lambda: first + step * next(calls)


def raw_dir(tmp_path):
    return tmp_path / "data" / "raw" / "live" / "replay" / "EURUSD"


def test_the_open_hour_file_has_a_sidecar_that_a_clean_close_brings_up_to_date(
    tmp_path,
):
    live(
        tmp_path,
        [q(timedelta(minutes=m)) for m in (1, 2)],
        clock=arrival_clock(timedelta(seconds=1)),
        start=OPEN,
    )

    plain = raw_dir(tmp_path) / "20300101T09.jsonl"
    sidecar = json.loads(
        (raw_dir(tmp_path) / "20300101T09.jsonl.meta.json").read_text()
    )
    assert sidecar["open"] is True  # provisional: the hour may still be appended to
    assert sidecar["sha256"] == hashlib.sha256(plain.read_bytes()).hexdigest()
    assert sidecar["size"] == len(plain.read_bytes())


def test_closed_hours_are_compressed_and_the_sidecar_hashes_the_compressed_bytes(
    tmp_path,
):
    live(
        tmp_path,
        [q(timedelta(minutes=m)) for m in (1, 2, 3)],
        clock=arrival_clock(timedelta(minutes=40)),  # arrivals 09:59, 10:39, 11:19
        start=OPEN,
    )

    directory = raw_dir(tmp_path)
    assert sorted(p.name for p in directory.iterdir()) == [
        "20300101T09.jsonl.gz",
        "20300101T09.jsonl.gz.meta.json",
        "20300101T10.jsonl.gz",
        "20300101T10.jsonl.gz.meta.json",
        "20300101T11.jsonl",  # the open hour is still plain, append-only
        "20300101T11.jsonl.meta.json",  # with a provisional sidecar
    ]
    packed = (directory / "20300101T09.jsonl.gz").read_bytes()
    sidecar = json.loads((directory / "20300101T09.jsonl.gz.meta.json").read_text())
    assert sidecar["sha256"] == hashlib.sha256(packed).hexdigest()
    first = json.loads(gzip.decompress(packed).decode())
    assert first["source_ts"] == "2024-01-07T22:01:00+00:00"


def test_the_raw_log_reproduces_the_emitted_bars_and_a_later_run_never_rewrites_it(
    tmp_path,
):
    events = [
        q(timedelta(minutes=1), bid="1.10000"),
        q(timedelta(minutes=12), bid="1.10040"),
        q(timedelta(minutes=2), bid="1.10020"),  # late: a revision of slot 0
        Heartbeat(OPEN + timedelta(minutes=30)),
    ]
    _, original = live(
        tmp_path, events, clock=arrival_clock(timedelta(minutes=30)), start=OPEN
    )
    closed = raw_dir(tmp_path) / "20300101T09.jsonl.gz"
    before = closed.read_bytes()

    replayed = []
    logged = [quote for quote, _ in read_raw_log(tmp_path / "data", "replay", "EURUSD")]
    assert len(logged) == 3
    LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(
            [*logged, Heartbeat(OPEN + timedelta(minutes=30))], instrument="EURUSD"
        ),
        sink=lambda bar, revision: replayed.append(bar),
        start=OPEN,
        clock=arrival_clock(first=ARRIVAL + timedelta(days=1)),  # new hours: new files
    ).run()

    cutoff = OPEN + timedelta(minutes=30)
    latest = {b.open_time: b for b in replayed}  # last Revision per slot wins
    assert [latest[t] for t in sorted(latest) if t < cutoff] == [
        b for b in original.revisions.latest() if b.open_time < cutoff
    ]
    assert closed.read_bytes() == before


def report_files(tmp_path):
    directory = tmp_path / "data" / "reports" / "live" / "replay" / "EURUSD"
    return sorted(directory.glob("*.json")) if directory.exists() else []


def read_report(tmp_path):
    (path,) = report_files(tmp_path)
    return json.loads(path.read_text())


def test_the_run_report_is_rewritten_at_each_closed_slot_and_finalised_on_shutdown(
    tmp_path,
):
    seen = []

    def events():
        yield q(timedelta(minutes=1))
        seen.append(report_files(tmp_path) and read_report(tmp_path))
        yield Heartbeat(OPEN + timedelta(minutes=11))  # closes slots 0 and 1
        seen.append(read_report(tmp_path))
        holder[0].stop()
        yield Heartbeat(OPEN + timedelta(minutes=21))

    holder = []
    ingester = LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(events(), instrument="EURUSD"),
        sink=lambda bar, revision: None,
        start=OPEN,
        clock=arrival_clock(),
    )
    holder.append(ingester)
    ingester.run()

    started, mid = seen
    assert (started["status"], started["counts"]["bars"]) == ("running", 0)
    assert (mid["status"], mid["counts"]["bars"], mid["counts"]["gap_bars"]) == (
        "running",
        2,
        1,
    )
    final = read_report(tmp_path)
    assert final["status"] == "stopped"
    assert final["report_version"] == 1
    assert (final["source"], final["instrument"], final["bar_size"]) == (
        "replay",
        "EURUSD",
        5,
    )
    assert final["run"]["ended_at"] is not None


def test_the_run_report_holds_counts_lag_distribution_and_first_20_sealed_drops(
    tmp_path,
):
    sealed = OPEN + timedelta(minutes=5 + 60)
    drops = [q(timedelta(minutes=1, seconds=s)) for s in range(25)]
    live(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            q(timedelta(minutes=7)),  # slot 0 closes at watermark 7 min
            q(timedelta(minutes=2)),  # late by 5 min: accepted as revision 1
            q(timedelta(minutes=3)),  # late by 4 min: accepted as revision 2
            Heartbeat(sealed),
            *drops,  # slot 0 is sealed: 25 drops, lag 3600 s - 1 min - s
            Quote(OPEN - timedelta(hours=3), Decimal("1.1"), Decimal("1.1001")),
        ],
        start=OPEN,
    )

    report = read_report(tmp_path)
    counts = report["counts"]
    assert counts["bars_by_revision"]["0"] == 12  # slots closed by the heartbeat
    assert counts["bars_by_revision"]["1"] == 1
    assert counts["bars_by_revision"]["2"] == 1
    assert counts["gap_bars"] >= 10
    assert counts["late_quotes_accepted"] == 2
    assert counts["quotes_dropped_after_seal"] == 25
    assert counts["quotes_dropped_out_of_session"] == 1
    assert report["late_quote_lag_seconds"] == {
        "count": 2,
        "p50": 240.0,
        "p95": 300.0,
        "max": 300.0,
    }
    examples = report["dropped_after_seal_examples"]
    assert len(examples) == 20
    assert examples[0]["source_ts"] == "2024-01-07T22:01:00+00:00"
    assert examples[19]["source_ts"] == "2024-01-07T22:01:19+00:00"
    assert [w["code"] for w in report["warnings"]] == ["live_quote_after_seal"]


def test_the_run_report_records_catch_up_calls_and_their_results(tmp_path):
    adapter = CatchUpAdapter(
        [
            q(timedelta(minutes=1)),
            SourceDown(OPEN + timedelta(minutes=2)),
            SourceUp(OPEN + timedelta(minutes=6)),
        ],
        missed=[q(timedelta(minutes=3)), q(timedelta(minutes=4))],
    )
    LiveIngester(
        make_config(tmp_path), adapter, sink=lambda bar, revision: None, start=OPEN
    ).run()

    report = read_report(tmp_path)
    assert report["catch_up"] == [
        {
            "start": "2024-01-07T22:02:00+00:00",
            "end": "2024-01-07T22:06:00+00:00",
            "quotes": 2,
            "accepted_as_revisions": 2,
            "dropped_after_seal": 0,
        }
    ]
    assert report["source_down"] == [
        {
            "start": "2024-01-07T22:02:00+00:00",
            "end": "2024-01-07T22:06:00+00:00",
            "reason": "adapter",
        }
    ]


def test_a_crash_and_restart_marks_the_old_report_crashed_and_records_downtime(
    tmp_path,
):
    def crashing():
        yield q(timedelta(minutes=1))
        yield Heartbeat(OPEN + timedelta(minutes=11))
        raise RuntimeError("process died")

    first = LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(crashing(), instrument="EURUSD"),
        sink=lambda bar, revision: None,
        start=OPEN,
    )
    with pytest.raises(RuntimeError):
        first.run()
    (crashed_path,) = report_files(tmp_path)
    assert json.loads(crashed_path.read_text())["status"] == "running"

    restart = OPEN + timedelta(minutes=47)
    emitted = []
    second = LiveIngester(
        make_config(tmp_path),
        ReplayAdapter(
            [q(timedelta(minutes=48)), Heartbeat(OPEN + timedelta(minutes=60))],
            instrument="EURUSD",
        ),
        sink=lambda bar, revision: emitted.append(bar),
        start=restart,
    )
    second.run()

    assert json.loads(crashed_path.read_text())["status"] == "crashed"
    (new_path,) = [p for p in report_files(tmp_path) if p != crashed_path]
    report = json.loads(new_path.read_text())
    assert report["source_down"] == [
        {
            "start": "2024-01-07T22:11:00+00:00",  # the dead run's last watermark
            "end": "2024-01-07T22:47:00+00:00",
            "reason": "between_runs",
        }
    ]
    # no Gaps are invented for slots before the restart
    assert min(b.open_time for b in emitted) == OPEN + timedelta(minutes=50)


def test_every_live_quote_is_appended_to_an_hourly_jsonl_with_both_timestamps(
    tmp_path,
):
    live(
        tmp_path,
        [
            q(timedelta(minutes=1), bid="1.10000"),
            q(timedelta(minutes=2), bid="1.10020"),
        ],
        clock=arrival_clock(),
        start=OPEN,
    )

    lines = (raw_dir(tmp_path) / "20300101T09.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "source_ts": "2024-01-07T22:01:00+00:00",
            "arrival_ts": "2030-01-01T09:59:58+00:00",
            "bid": "1.10000",
            "ask": "1.10010",
        },
        {
            "source_ts": "2024-01-07T22:02:00+00:00",
            "arrival_ts": "2030-01-01T09:59:59+00:00",
            "bid": "1.10020",
            "ask": "1.10010",
        },
    ]
