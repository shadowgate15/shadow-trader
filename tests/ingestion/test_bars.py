"""BarBuilder behaviour, checked against the independent reference builder."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadow_trader.ingestion.bars import BarBuilder, Quote, build_bars
from shadow_trader.ingestion.config import SUPPORTED_BAR_SIZES, ConfigError

from .reference_bars import d, reference_bars

# Winter Session: opens Sun 2024-01-07 22:00 UTC. Summer: Sun 2024-07-07 21:00 UTC.
WINTER = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)
SUMMER = datetime(2024, 7, 7, 21, 0, tzinfo=UTC)
ROW_FIELDS = (
    "open_time",
    "session_id",
    "is_rollover",
    "open",
    "high",
    "low",
    "close",
    "spread_close",
    "spread_mean",
    "tick_count",
    "is_gap",
)


def q(ts: datetime, bid: str, ask: str) -> Quote:
    return Quote(ts, d(bid), d(ask))


def build(quotes, minutes=5, scale=5):
    return build_bars("EURUSD", quotes, minutes, scale)


def as_rows(bars):
    return [{f: getattr(b, f) for f in ROW_FIELDS} for b in bars]


def assert_matches_reference(quotes, minutes, scale=5):
    bars, dropped = build(quotes, minutes, scale)
    ref, ref_dropped = reference_bars([(x.ts, x.bid, x.ask) for x in quotes], minutes)
    assert as_rows(bars) == ref
    assert dropped == ref_dropped


def test_ohlc_spread_and_tick_count_from_ticks():
    t = WINTER + timedelta(minutes=10)
    bars, _ = build(
        [
            q(t + timedelta(seconds=1), "1.10000", "1.10010"),
            q(t + timedelta(seconds=2), "1.10050", "1.10070"),
            q(t + timedelta(seconds=3), "1.09980", "1.10000"),
            q(t + timedelta(seconds=4), "1.10020", "1.10050"),
        ]
    )
    bar = next(b for b in bars if b.open_time == t)
    assert (bar.open, bar.high, bar.low, bar.close) == (1.1, 1.1005, 1.0998, 1.1002)
    assert bar.spread_close == pytest.approx(0.0003, abs=1e-12)
    assert bar.spread_mean == pytest.approx((10 + 20 + 20 + 30) / 4 * 1e-5, abs=1e-12)
    assert bar.tick_count == 4
    assert not bar.is_gap
    assert bar.instrument == "EURUSD"
    assert bar.quality_flags == ()


@pytest.mark.parametrize("minutes", SUPPORTED_BAR_SIZES)
def test_every_bar_size_matches_reference(minutes):
    base = WINTER + timedelta(hours=3, minutes=7)
    quotes = [
        q(base, "1.10000", "1.10012"),
        q(base + timedelta(seconds=30), "1.10020", "1.10031"),
        q(base + timedelta(minutes=minutes), "1.09990", "1.10004"),  # on a boundary
        q(base + timedelta(minutes=minutes, seconds=1), "1.10100", "1.10111"),
        q(base + timedelta(minutes=3 * minutes + 1), "1.10050", "1.10066"),
        q(WINTER + timedelta(hours=119, minutes=59, seconds=59), "1.1", "1.10002"),
    ]
    assert_matches_reference(quotes, minutes)


def test_bar_size_not_dividing_60_is_config_error():
    with pytest.raises(ConfigError):
        BarBuilder("EURUSD", 7, 5)


def test_gap_rows_have_null_fields_and_session_id():
    bars, _ = build([q(WINTER + timedelta(minutes=12), "1.1", "1.10002")])
    gaps = [b for b in bars if b.is_gap]
    assert len(gaps) == 120 * 12 - 1
    for g in gaps:
        assert (g.open, g.high, g.low, g.close) == (None,) * 4
        assert (g.spread_close, g.spread_mean, g.tick_count) == (None,) * 3
        assert g.quality_flags == ()
    assert {b.session_id for b in bars} == {WINTER}
    # empty slots are never carried forward from the previous close
    after = [b for b in bars if b.open_time > WINTER + timedelta(minutes=12)]
    assert all(b.is_gap and b.open is None for b in after)


@pytest.mark.parametrize(("open_", "hour"), [(WINTER, 22), (SUMMER, 21)])
@pytest.mark.parametrize("minutes", SUPPORTED_BAR_SIZES)
def test_session_grid_is_constant_and_opens_at_utc_hour(open_, hour, minutes):
    bars, _ = build([q(open_ + timedelta(seconds=1), "1.1", "1.10002")], minutes)
    assert bars[0].open_time == open_
    assert bars[0].open_time.hour == hour
    assert len(bars) == 120 * 60 // minutes
    assert bars[-1].open_time == open_ + timedelta(hours=120, minutes=-minutes)


def test_summer_and_winter_sessions_match_reference():
    assert_matches_reference(
        [
            q(SUMMER + timedelta(hours=1), "1.07000", "1.07010"),
            q(SUMMER + timedelta(hours=50), "1.07100", "1.07112"),
        ],
        15,
    )
    assert_matches_reference([q(WINTER + timedelta(hours=1), "1.07", "1.07010")], 30)


def test_ticks_outside_session_are_dropped_and_counted():
    quotes = [
        q(WINTER - timedelta(seconds=1), "1.1", "1.10002"),  # before open
        q(WINTER, "1.10000", "1.10002"),  # exactly open
        q(WINTER + timedelta(hours=120), "1.2", "1.20002"),  # exactly close
        q(WINTER + timedelta(hours=130), "1.3", "1.30002"),  # closure
        q(WINTER + timedelta(days=7), "1.4", "1.40002"),  # next Session open
    ]
    bars, dropped = build(quotes)
    assert dropped == 3
    assert sum(b.tick_count or 0 for b in bars) == 2
    assert_matches_reference(quotes, 5)


def test_multiple_sessions_include_empty_session_and_session_ids():
    quotes = [
        q(WINTER + timedelta(hours=1), "1.1", "1.10002"),
        q(WINTER + timedelta(days=14, hours=1), "1.2", "1.20002"),
    ]
    bars, _ = build(quotes, 60)
    assert [b.session_id for b in bars[::120]] == [
        WINTER,
        WINTER + timedelta(days=7),
        WINTER + timedelta(days=14),
    ]
    assert_matches_reference(quotes, 60)


def test_is_rollover_marks_bar_opening_1700_new_york_mon_to_thu():
    bars, _ = build([q(WINTER + timedelta(seconds=1), "1.1", "1.10002")], 5)
    rollover = [b.open_time for b in bars if b.is_rollover]
    # 17:00 NY winter = 22:00 UTC, Mon..Thu (not Sunday's open, not Friday's close)
    assert rollover == [WINTER + timedelta(days=i) for i in (1, 2, 3, 4)]


def test_ties_keep_arrival_order_and_late_arrival_orders_by_timestamp():
    t = WINTER + timedelta(minutes=5)
    quotes = [
        q(t + timedelta(seconds=10), "1.10010", "1.10020"),
        q(t + timedelta(seconds=10), "1.10030", "1.10040"),  # tie: arrives second
        q(t + timedelta(seconds=5), "1.10000", "1.10001"),  # earlier ts, arrives later
    ]
    bars, _ = build(quotes)
    bar = next(b for b in bars if b.open_time == t)
    assert bar.open == 1.1
    assert bar.close == 1.1003
    assert_matches_reference(quotes, 5)


def test_slot_boundary_tick_lands_in_the_later_half_open_slot():
    t = WINTER + timedelta(minutes=5)
    bars, _ = build(
        [q(t - timedelta(microseconds=1), "1.1", "1.10002"), q(t, "1.2", "1.20002")]
    )
    by_open = {b.open_time: b for b in bars}
    assert by_open[WINTER].close == 1.1
    assert by_open[t].open == 1.2


def test_jpy_scale_three_decimals():
    bars, _ = build_bars(
        "USDJPY",
        [Quote(WINTER + timedelta(seconds=1), d("150.123"), d("150.125"))],
        5,
        3,
    )
    assert bars[0].open == 150.123
    assert bars[0].spread_close == pytest.approx(0.002, abs=1e-12)


def test_excess_precision_fails_rather_than_rounding():
    with pytest.raises(ValueError, match="precision"):
        build([q(WINTER + timedelta(seconds=1), "1.100001", "1.10002")])


def test_incremental_push_emits_closed_bars_and_flush_finishes_session():
    b = BarBuilder("EURUSD", 5, 5)
    assert b.push(q(WINTER + timedelta(seconds=1), "1.1", "1.10002")) == []
    out = b.push(q(WINTER + timedelta(minutes=11), "1.2", "1.20002"))
    assert [x.open_time for x in out] == [WINTER, WINTER + timedelta(minutes=5)]
    assert out[0].tick_count == 1
    assert out[1].is_gap
    rest = b.flush()
    assert len(out) + len(rest) == 120 * 12
    assert rest[0].open_time == WINTER + timedelta(minutes=10)
    assert b.flush() == []


def test_push_into_a_closed_slot_is_rejected():
    b = BarBuilder("EURUSD", 5, 5)
    b.push(q(WINTER + timedelta(minutes=30), "1.1", "1.10002"))
    with pytest.raises(ValueError, match="closed"):
        b.push(q(WINTER + timedelta(minutes=1), "1.1", "1.10002"))


def test_bars_are_frozen():
    bars, _ = build([q(WINTER + timedelta(seconds=1), "1.1", "1.10002")])
    with pytest.raises(dataclasses.FrozenInstanceError):
        bars[0].open = 2.0  # type: ignore[misc]


@settings(max_examples=40, deadline=None)
@given(
    minutes=st.sampled_from(SUPPORTED_BAR_SIZES),
    offsets=st.lists(st.integers(-3600, 125 * 3600), min_size=1, max_size=40),
    moves=st.lists(
        st.tuples(st.integers(0, 400), st.integers(0, 40)), min_size=40, max_size=40
    ),
    summer=st.booleans(),
)
def test_random_ticks_match_reference(minutes, offsets, moves, summer):
    open_ = SUMMER if summer else WINTER
    quotes = [
        Quote(
            open_ + timedelta(seconds=off, milliseconds=7),
            Decimal(110000 + move) / 100000,
            Decimal(110000 + move + spr) / 100000,
        )
        for off, (move, spr) in zip(sorted(offsets), moves, strict=False)
    ]
    if all(not (open_ <= x.ts < open_ + timedelta(hours=120)) for x in quotes):
        return
    assert_matches_reference(quotes, minutes)
