import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from shadow_trader import bars, config
from shadow_trader.fetch import Week

HEADER = "DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose"


def m1(moment: datetime, bid=(1.1, 1.1, 1.1, 1.1), ask=(1.2, 1.2, 1.2, 1.2)) -> str:
    ts = moment.strftime("%m/%d/%Y %H:%M:%S.000")
    return ",".join([ts, *map(str, bid), *map(str, ask)])


def at(day: int, hour: int, minute: int = 0, month: int = 2, year: int = 2016):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def minutes(start: datetime, count: int, **kw) -> list[str]:
    return [m1(start + timedelta(minutes=i), **kw) for i in range(count)]


def make_week(tmp_path: Path, rows: list[str], name: str) -> Week:
    path = tmp_path / f"{name}.csv.gz"
    path.write_bytes(gzip.compress(("\n".join([HEADER, *rows]) + "\n").encode()))
    day = datetime(2016, 1, 1, tzinfo=UTC)
    return Week(2016, 0, path, day, day)


def ingest(tmp_path: Path, *week_rows: list[str]):
    weeks = [make_week(tmp_path, r, f"w{i}") for i, r in enumerate(week_rows)]
    out: list[str] = []
    summary = bars.ingest(weeks, tmp_path / "bars.parquet", progress=out.append)
    return summary, pq.read_table(tmp_path / "bars.parquet"), out


def test_complete_bucket_gives_per_side_ohlc_from_its_five_rows(tmp_path):
    rows = [
        m1(at(1, 10, 0), (1.0, 1.5, 0.9, 1.1), (2.0, 2.5, 1.9, 2.1)),
        m1(at(1, 10, 1), (1.1, 1.3, 1.0, 1.2), (2.1, 2.3, 2.0, 2.2)),
        m1(at(1, 10, 2), (1.2, 1.4, 0.8, 1.3), (2.2, 2.4, 2.0, 2.3)),
        m1(at(1, 10, 3), (1.3, 1.6, 1.2, 1.4), (2.3, 2.4, 2.2, 2.4)),
        m1(at(1, 10, 4), (1.4, 1.5, 1.3, 1.45), (2.4, 2.5, 2.3, 2.45)),
    ]
    _, table, _ = ingest(tmp_path, rows)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["time"] == at(1, 10, 0)
    bid = (row["bid_open"], row["bid_high"], row["bid_low"], row["bid_close"])
    ask = (row["ask_open"], row["ask_high"], row["ask_low"], row["ask_close"])
    assert bid == (1.0, 1.6, 0.8, 1.45)
    assert ask == (2.0, 2.5, 1.9, 2.45)
    assert table.column_names == [
        "time",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]


def test_bucket_missing_any_m1_row_makes_no_bar_and_widens_a_gap(tmp_path):
    rows = (
        minutes(at(1, 10, 0), 5)
        + minutes(at(1, 10, 5), 4)  # 10:05 bucket lacks 10:09
        + minutes(at(1, 10, 10), 5)
    )
    summary, table, _ = ingest(tmp_path, rows)
    assert table["time"].to_pylist() == [at(1, 10, 0), at(1, 10, 10)]
    assert len(summary.gaps) == 1
    assert summary.gaps[0].length == timedelta(minutes=5)


def test_mid_is_derived_from_the_stored_sides(tmp_path):
    rows = minutes(at(1, 10, 0), 5, bid=(1.0, 1.0, 1.0, 1.0), ask=(2.0, 3.0, 1.0, 2.0))
    _, table, _ = ingest(tmp_path, rows)
    derived = bars.mid(table)
    assert derived["mid_open"] == [1.5]
    assert derived["mid_high"] == [2.0]
    assert "mid_open" not in table.column_names


def test_only_real_bars_are_stored_across_a_weekend(tmp_path):
    friday = minutes(at(5, 21, 50), 10)  # Fri 21:50-21:59
    sunday = minutes(at(7, 22, 0), 5)
    _, table, _ = ingest(tmp_path, friday + sunday)
    assert table.num_rows == 3  # no bars synthesised for the closed hours


@pytest.mark.parametrize(
    "bad",
    [
        {"bid": (0.0, 1.1, 1.1, 1.1)},  # non-positive
        {"bid": (-1.1, 1.1, 1.1, 1.1)},
        {"ask": (1.2, float("nan"), 1.2, 1.2)},  # not finite
        {"ask": (1.2, float("inf"), 1.2, 1.2)},
    ],
)
def test_non_finite_or_non_positive_prices_abort_ingest(tmp_path, bad):
    rows = [*minutes(at(1, 10, 0), 4), m1(at(1, 10, 4), **bad)]
    with pytest.raises(bars.BarsError, match="finite and positive"):
        ingest(tmp_path, rows)


def one_bar(**overrides):
    columns = {c: [1.0] for c in bars.PRICE_COLUMNS}
    columns["time"] = [at(1, 10, 0)]
    for name, value in overrides.items():
        columns[name] = [value]
    return columns


@pytest.mark.parametrize(
    ("side", "broken"),
    [
        ("bid", {"bid_high": 0.9}),  # high below open/close
        ("bid", {"bid_low": 1.1}),  # low above open/close
        ("ask", {"ask_high": 0.9}),
        ("ask", {"ask_low": 1.1}),
        ("ask", {"ask_high": 0.8, "ask_low": 0.9, "ask_open": 0.85, "ask_close": 0.85}),
    ],
)
def test_inconsistent_ohlc_aborts_and_names_the_side(side, broken):
    bars.validate(one_bar())  # sanity: the baseline is valid
    with pytest.raises(bars.BarsError, match=f"{side} OHLC"):
        bars.validate(one_bar(**broken))


def test_high_below_low_aborts():
    with pytest.raises(bars.BarsError, match="bid OHLC"):
        bars.validate(one_bar(bid_high=1.0, bid_low=1.2, bid_open=1.1, bid_close=1.1))


def test_timestamps_out_of_order_across_weeks_abort(tmp_path):
    later = minutes(at(2, 10, 0), 5)
    earlier = minutes(at(1, 10, 0), 5)
    with pytest.raises(bars.BarsError, match="increasing"):
        ingest(tmp_path, later, earlier)


def test_duplicate_timestamps_across_weeks_abort(tmp_path):
    same = minutes(at(1, 10, 0), 5)
    with pytest.raises(bars.BarsError, match="increasing"):
        ingest(tmp_path, same, same)


def test_off_grid_bar_interval_aborts():
    columns = {c: [1.0, 1.0] for c in bars.PRICE_COLUMNS}
    columns["time"] = [at(1, 10, 0), at(1, 10, 7)]
    with pytest.raises(bars.BarsError, match="interval"):
        bars.validate(columns)


def test_gap_longer_than_a_weekend_aborts(tmp_path):
    rows = minutes(at(1, 10, 0), 5) + minutes(at(6, 10, 0), 5)  # 5 days
    with pytest.raises(bars.BarsError, match="Gap"):
        ingest(tmp_path, rows)


def test_a_weekend_sized_gap_is_allowed(tmp_path):
    rows = minutes(at(5, 21, 55), 5) + minutes(at(7, 22, 0), 5)
    summary, _, _ = ingest(tmp_path, rows)
    assert len(summary.gaps) == 1


@pytest.fixture
def known_break(monkeypatch):
    monkeypatch.setattr(
        config,
        "KNOWN_MISSING_WEEKS",
        ((at(7, 22, month=2), at(12, 21, 59, month=2)),),
    )


def test_known_missing_week_gap_is_allowed(tmp_path, known_break):
    before = minutes(at(5, 20, 55), 5)  # Fri
    after = minutes(at(14, 21, 0), 5)  # Sun after the missing week: ~9 days
    summary, _, out = ingest(tmp_path, before, after)
    assert len(summary.gaps) == 1
    assert summary.gaps[0].length == timedelta(days=9)
    assert "largest gap: 9 days, 0:00:00" in "\n".join(out)


def test_other_long_gaps_still_abort_despite_a_known_break(tmp_path, known_break):
    before = minutes(at(19, 20, 55), 5)
    after = minutes(at(28, 21, 0), 5)  # ~9 days, but not around the known week
    with pytest.raises(bars.BarsError, match="Gap"):
        ingest(tmp_path, before, after)


def test_crossed_and_zero_spread_rows_survive_unaltered(tmp_path):
    crossed = dict(bid=(1.3, 1.3, 1.3, 1.3), ask=(1.2, 1.2, 1.2, 1.2))
    flat = m1(at(1, 10, 1), bid=(1.1,) * 4, ask=(1.1,) * 4)
    rows = [m1(at(1, 10, 0), **crossed), flat, *minutes(at(1, 10, 2), 3)]
    summary, table, _ = ingest(tmp_path, rows)
    row = table.to_pylist()[0]
    assert row["bid_open"] == 1.3 and row["ask_open"] == 1.2  # crossed open kept
    assert row["bid_high"] == 1.3 and row["ask_low"] == 1.1
    assert summary.rows.crossed == 1
    assert summary.rows.zero_spread == 1


def test_crossed_uses_any_component_and_reports_deepest_crossing(tmp_path):
    # only the high component crosses, by 15 pips
    row = m1(at(1, 10, 0), bid=(1.1, 1.2015, 1.1, 1.1), ask=(1.2, 1.2, 1.2, 1.2))
    summary, _, out = ingest(tmp_path, [row, *minutes(at(1, 10, 1), 4)])
    text = "\n".join(out)
    assert summary.rows.crossed == 1
    assert "crossed rows (any component): 1 (20.00%)" in text
    assert "deepest crossing: 15.0 pips" in text


def test_summary_and_progress_are_printed(tmp_path):
    rows = minutes(at(1, 10, 0), 5) + minutes(at(1, 10, 20), 5)
    _, _, out = ingest(tmp_path, rows)
    text = "\n".join(out)
    for needle in (
        "aggregating 1/1",
        "bars: 2",
        "range: 2016-02-01 10:00 to 2016-02-01 10:20",
        "gaps: 1",
        "largest gap: 0:15:00",
        "gap histogram:",
        "gap min: 0:15:00  max: 0:15:00",
        "zero-spread rows: 0 (0.00%)",
    ):
        assert needle in text, needle


def test_bars_outside_the_pinned_range_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_START", at(1, 10, 5))
    monkeypatch.setattr(config, "HISTORY_END", at(1, 10, 15))
    _, table, _ = ingest(tmp_path, minutes(at(1, 10, 0), 25))
    assert table["time"].to_pylist() == [at(1, 10, 5), at(1, 10, 10)]
