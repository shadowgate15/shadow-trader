"""Anomaly detector behaviour, driven through the incremental FlaggingBarBuilder."""

from __future__ import annotations

import dataclasses
import itertools
import random
import statistics
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from shadow_trader.ingestion.anomaly import (
    VOCABULARY,
    FlaggingBarBuilder,
    summarize_flags,
)
from shadow_trader.ingestion.bars import Bar, Quote, build_bars
from shadow_trader.ingestion.config import AnomalyRuleConfig
from shadow_trader.ingestion.dataset import build_dataset

from .test_dataset import make_config

WINTER = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)  # Session open


def q(offset: timedelta, bid: str, ask: str) -> Quote:
    return Quote(WINTER + offset, Decimal(bid), Decimal(ask))


def run(quotes, rules=None, minutes=5, scale=5, instrument="EURUSD"):
    """Feed quotes through the flagging builder; return (bars, builder)."""
    builder = FlaggingBarBuilder(
        instrument, minutes, scale, rules or AnomalyRuleConfig()
    )
    bars = []
    for quote in quotes:
        bars.extend(builder.push(quote))
    bars.extend(builder.flush())
    return bars, builder


def flags_by_time(bars):
    return {b.open_time: b.quality_flags for b in bars if b.quality_flags}


def test_crossed_quote_flags_only_its_bar():
    bars, _ = run(
        [
            q(timedelta(minutes=1), "1.10000", "1.10010"),
            q(timedelta(minutes=6), "1.10010", "1.10000"),
            q(timedelta(minutes=11), "1.10000", "1.10010"),
        ]
    )
    assert flags_by_time(bars) == {WINTER + timedelta(minutes=5): ("crossed_quote",)}


def test_zero_spread_flags_its_bar():
    bars, _ = run(
        [
            q(timedelta(minutes=1), "1.10000", "1.10010"),
            q(timedelta(minutes=6), "1.10000", "1.10000"),
        ]
    )
    assert flags_by_time(bars) == {WINTER + timedelta(minutes=5): ("zero_spread",)}


def steady(n: int, start: timedelta = timedelta(seconds=1), step_s: int = 2):
    """n quiet ticks alternating 1.10000/1.10001 bid, one every step_s seconds."""
    return [
        q(start + timedelta(seconds=step_s * i), f"1.1000{i % 2}", f"1.1001{i % 2}")
        for i in range(n)
    ]


def test_price_spike_beyond_floor_flags_its_bar_once_warm():
    quotes = steady(60)  # 120 s, all in bar 0
    quotes.append(q(timedelta(minutes=6), "1.10150", "1.10160"))  # +15 pips
    quotes.append(q(timedelta(minutes=11), "1.10000", "1.10010"))
    bars, _ = run(quotes)
    assert flags_by_time(bars) == {WINTER + timedelta(minutes=5): ("price_spike",)}


def test_price_move_inside_floor_is_not_a_spike():
    quotes = steady(60)
    quotes.append(q(timedelta(minutes=6), "1.10050", "1.10060"))  # +5 pips
    bars, _ = run(quotes)
    assert flags_by_time(bars) == {}


def test_price_spike_suppressed_until_window_is_warm():
    quotes = steady(49)  # one tick short of N=50
    quotes.append(q(timedelta(minutes=6), "1.10150", "1.10160"))
    bars, _ = run(quotes)
    assert flags_by_time(bars) == {}


def test_price_spike_is_look_back_only():
    quotes = [*steady(60), q(timedelta(minutes=6), "1.10150", "1.10160")]
    prefix_bars, _ = run(quotes[:60])
    full_bars, _ = run(quotes)
    assert flags_by_time(prefix_bars) == {}
    assert flags_by_time(full_bars) == {WINTER + timedelta(minutes=5): ("price_spike",)}
    # Closed Bars of the shorter run keep the same flags in the longer one.
    assert [b for b in full_bars if b.open_time < WINTER + timedelta(minutes=5)] == [
        b for b in prefix_bars if b.open_time < WINTER + timedelta(minutes=5)
    ]


def test_price_spike_threshold_scales_with_volatility_above_floor():
    rng = random.Random(7)
    bid = 110000  # scaled integer, 5 dp
    bids = []
    for _ in range(60):
        bid += rng.randint(-400, 400)  # up to 40 pips per tick: very noisy
        bids.append(bid)
    window = bids[-50:]
    diffs = [b - a for a, b in itertools.pairwise(window)]
    med = statistics.median(diffs)
    scale = 1.4826 * statistics.median(abs(d - med) for d in diffs)
    threshold = max(10 * scale, 100)  # floor 10 pips = 100 units
    assert threshold > 100  # volatility term dominates the floor
    centre = statistics.median(window)

    def quote(i: int, b: int) -> Quote:
        return q(timedelta(seconds=i), f"{b / 1e5:.5f}", f"{(b + 10) / 1e5:.5f}")

    def with_final(final_bid: int):
        quotes = [quote(2 * i + 1, b) for i, b in enumerate(bids)]
        quotes.append(quote(360, final_bid))
        bars, _ = run(quotes)
        return flags_by_time(bars)

    assert with_final(int(centre + threshold - 2)) == {}
    assert with_final(int(centre + threshold + 2)) == {
        WINTER + timedelta(minutes=5): ("price_spike",)
    }


def frozen(start_s: int, count: int, step_s: int, bid="1.10000", ask="1.10010"):
    return [q(timedelta(seconds=start_s + step_s * i), bid, ask) for i in range(count)]


def minute(n: int) -> datetime:
    return WINTER + timedelta(minutes=n)


def test_stale_quote_flags_every_bar_with_a_run_tick_once_qualified():
    # 41 identical ticks, one per 30 s: the 20th (at 9.5 min) is the first to
    # reach both 20 ticks and 5 min; the quote changes at minute 21.
    quotes = [*frozen(0, 41, 30), q(timedelta(minutes=21), "1.10020", "1.10030")]
    bars, builder = run(quotes)
    assert flags_by_time(bars) == {
        minute(5): ("stale_quote",),
        minute(10): ("stale_quote",),
        minute(15): ("stale_quote",),
        minute(20): ("stale_quote",),
    }
    assert builder.tick_flag_counts["stale_quote"] == 22  # run ticks 20..41


def test_identical_ticks_short_of_the_tick_or_time_threshold_are_not_stale():
    many_but_brief = frozen(0, 40, 5)  # 40 ticks over 195 s
    change = q(timedelta(minutes=4), "1.10020", "1.10030")
    few_but_long = frozen(300, 19, 60)  # 19 ticks over 18 min
    bars, _ = run([*many_but_brief, change, *few_but_long])
    assert flags_by_time(bars) == {}


HOURLY = AnomalyRuleConfig(spread_spike_lookback_hours=1)


def spread_series(spreads_units: list[int], gaps: tuple[int, ...] = ()):
    """One quote per 5-min bar with the given spread in 1e-5 units; skip `gaps`."""
    return [
        q(timedelta(minutes=5 * i, seconds=1), "1.10000", f"{1.1 + s / 1e5:.5f}")
        for i, s in enumerate(spreads_units)
        if i not in gaps
    ]


def test_spread_spike_flags_bar_above_k_times_median_once_warm():
    # 14 Bars at 1 pip span 65 min (> the 1 h lookback); then 12 pips (> 10x).
    bars, _ = run(spread_series([10] * 14 + [120, 10]), HOURLY)
    assert flags_by_time(bars) == {minute(70): ("spread_spike",)}


def test_spread_spike_needs_more_than_k_times_median():
    bars, _ = run(spread_series([10] * 14 + [90, 10]), HOURLY)  # 9 pips < 10 pips
    assert flags_by_time(bars) == {}


def test_spread_spike_floor_applies_when_median_is_tiny():
    # Median 0.1 pip: 10x = 1 pip but median + 2 pips floor = 2.1 pips wins.
    low = run(spread_series([1] * 14 + [15, 1]), HOURLY)[0]  # 1.5 pips: inside
    high = run(spread_series([1] * 14 + [25, 1]), HOURLY)[0]  # 2.5 pips: beyond
    assert flags_by_time(low) == {}
    assert flags_by_time(high) == {minute(70): ("spread_spike",)}


def test_spread_spike_suppressed_until_baseline_is_warm():
    bars, _ = run(spread_series([10] * 6 + [120, 10]), HOURLY)  # only 30 min of history
    assert flags_by_time(bars) == {}


def test_spread_spike_median_ignores_gap_bars_and_is_look_back_only():
    series = spread_series([10] * 14 + [120, 10], gaps=(3, 4))
    full, _ = run(series, HOURLY)
    prefix, _ = run(series[:-2], HOURLY)
    assert flags_by_time(full) == {minute(70): ("spread_spike",)}
    assert flags_by_time(prefix) == {}


def test_flags_never_change_prices_spreads_or_tick_count():
    quotes = [
        *steady(60),
        q(timedelta(minutes=6), "1.10150", "1.10140"),  # spike and crossed
        q(timedelta(minutes=7), "1.10150", "1.10150"),  # zero spread
    ]
    flagged, _ = run(quotes)
    plain, _ = build_bars("EURUSD", quotes, 5, 5)
    assert any(b.quality_flags for b in flagged)
    assert [dataclasses.replace(b, quality_flags=()) for b in flagged] == plain


def test_disabled_rules_do_not_fire():
    rules = AnomalyRuleConfig(crossed_quote=False, zero_spread=False)
    bars, builder = run(
        [
            q(timedelta(minutes=1), "1.10010", "1.10000"),
            q(timedelta(minutes=2), "1.10000", "1.10000"),
        ],
        rules,
    )
    assert flags_by_time(bars) == {}
    assert not builder.tick_flag_counts


def test_jpy_pip_is_0_01_for_the_price_spike_floor():
    def jpy(final_bid: str):
        quotes = [
            q(timedelta(seconds=2 * i + 1), f"150.00{i % 2}", f"150.01{i % 2}")
            for i in range(60)
        ]
        quotes.append(q(timedelta(minutes=6), final_bid, "151.50"))
        bars, _ = run(quotes, scale=3, instrument="USDJPY")
        return flags_by_time(bars)

    assert jpy("150.050") == {}  # 5 pips
    assert jpy("151.100") == {minute(5): ("price_spike",)}  # 110 pips


def test_flags_are_emitted_in_closed_vocabulary_order():
    bars, _ = run([*steady(60), q(timedelta(minutes=6), "1.10150", "1.10140")])
    assert bars[1].quality_flags == ("crossed_quote", "price_spike")
    assert set(VOCABULARY) == {
        "crossed_quote",
        "zero_spread",
        "price_spike",
        "stale_quote",
        "spread_spike",
        "suspected_outage",
        "source_missing",
    }


def priced_bar(flags=(), i=0, gap=False):
    return Bar(
        instrument="EURUSD",
        open_time=minute(5 * i),
        open=None if gap else 1.1,
        high=None if gap else 1.1,
        low=None if gap else 1.1,
        close=None if gap else 1.1,
        spread_close=None if gap else 0.0001,
        spread_mean=None if gap else 0.0001,
        tick_count=None if gap else 1,
        is_gap=gap,
        session_id=WINTER,
        is_rollover=False,
        quality_flags=tuple(flags),
    )


def test_summary_counts_bars_and_flags_and_warns_above_alert_fraction():
    bars = [priced_bar(["price_spike"], 0), *(priced_bar(i=i) for i in range(1, 100))]
    quiet = summarize_flags(bars, {"price_spike": 3}, alert_fraction=0.01)
    assert quiet.bar_counts == {"price_spike": 1}
    assert quiet.tick_counts == {"price_spike": 3}
    assert quiet.warnings == []  # exactly 1 of 100 is not more than 1%

    loud = summarize_flags(
        [*bars, priced_bar(["stale_quote"], 100)], {}, alert_fraction=0.01
    )
    assert [w.code for w in loud.warnings] == ["anomaly_alert_fraction_exceeded"]


def test_summary_ignores_gap_bars_and_excludes_zero_spread_and_spread_spike():
    noisy = [
        priced_bar(["zero_spread", "spread_spike"], i) for i in range(10)
    ]  # excluded flags: no warning
    gaps = [priced_bar(i=20 + i, gap=True) for i in range(5)]
    assert summarize_flags([*noisy, *gaps], {}, alert_fraction=0.01).warnings == []
    # 2 flagged of 10 non-Gap = 20%; Gap Bars must not dilute it.
    mixed = [priced_bar(["crossed_quote"], 0), priced_bar(["price_spike"], 1)]
    mixed += [priced_bar(i=i) for i in range(2, 10)] + gaps
    assert summarize_flags(mixed, {}, alert_fraction=0.19).warnings
    assert not summarize_flags(mixed, {}, alert_fraction=0.2).warnings


def test_stale_quote_never_overlaps_a_gap():
    # Stale run, then silence, then a new quote.
    quotes = [*frozen(0, 41, 30), q(timedelta(minutes=31), "1.10020", "1.10030")]
    bars, _ = run(quotes)
    gaps = [b for b in bars if b.is_gap]
    assert gaps  # the silence is a Gap
    assert all(b.quality_flags == () for b in gaps)


def test_anomaly_thresholds_change_dataset_id_but_alert_fraction_does_not(tmp_path):
    bars, _ = run([*steady(60), q(timedelta(minutes=6), "1.10150", "1.10160")])
    series = {"EURUSD": bars}

    def dataset_id(rules: str) -> str:
        config = make_config(tmp_path, f"[rules.anomaly]\n{rules}\n")
        return build_dataset(config, 5, series).dataset_id

    base = dataset_id("")
    assert dataset_id("price_spike_k = 12.0") != base
    assert dataset_id("stale_quote_min_ticks = 30") != base
    assert dataset_id("alert_fraction = 0.5") == base


def test_manifest_records_per_flag_counts_for_bars_and_ticks(tmp_path):
    quotes = [
        *steady(60),
        q(timedelta(minutes=6), "1.10150", "1.10160"),
        q(timedelta(minutes=7), "1.10150", "1.10150"),
    ]
    bars, builder = run(quotes)
    ref = build_dataset(
        make_config(tmp_path),
        5,
        {"EURUSD": bars},
        tick_flag_counts={"EURUSD": builder.tick_flag_counts},
    )
    coverage = ref.manifest["coverage"]["EURUSD"]
    assert coverage["flag_counts"] == {"price_spike": 1, "zero_spread": 1}
    # Both off-level ticks are spikes, but they share one Bar.
    assert coverage["tick_flag_counts"] == {"price_spike": 2, "zero_spread": 1}
