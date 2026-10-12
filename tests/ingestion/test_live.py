"""Live ingester behaviour, driven through LiveIngester and the replay adapter."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow_trader.ingestion.anomaly import FlaggingBarBuilder
from shadow_trader.ingestion.bars import Quote
from shadow_trader.ingestion.config import AnomalyRuleConfig
from shadow_trader.ingestion.live import Heartbeat, LiveIngester, SourceDown, SourceUp
from shadow_trader.ingestion.replay import ReplayAdapter

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


def test_end_of_stream_closes_the_rest_of_the_session_like_a_backfill_flush(tmp_path):
    emitted, _ = live(tmp_path, [q(timedelta(minutes=1))])

    assert len(emitted) == 120 * 60 // 5  # the whole Session grid, as in a backfill
    assert [b.tick_count for b, _ in emitted[:2]] == [1, None]


def test_out_of_order_quotes_in_an_open_slot_order_by_timestamp_then_arrival(tmp_path):
    emitted, _ = live(
        tmp_path,
        [
            q(timedelta(seconds=30), bid="1.10030"),
            q(timedelta(seconds=10), bid="1.10010"),  # earlier ts, arrives later
            q(timedelta(seconds=30), bid="1.10040"),  # same ts as first, arrives after
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
    ingester = LiveIngester(
        config,
        ReplayAdapter(quotes, instrument="EURUSD"),
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

    # build_backfill does not wire the anomaly detector yet, so compare the prices
    unflagged = [
        dataclasses.replace(b, quality_flags=()) for b in ingester.revisions.latest()
    ]
    assert unflagged == expected
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
