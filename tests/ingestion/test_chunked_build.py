"""The chunked bar pass gives the same Dataset however the weeks are split."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from shadow_trader.ingestion.backfill import build_backfill
from shadow_trader.ingestion.dataset import open_dataset

from .test_backfill import install_raw, make_config

FIRST_SUNDAY = date(2024, 1, 7)
WEEKS = 8  # Jan 7 .. Feb 25: seams fall inside and across months
STEP = timedelta(minutes=4)
TICKS_PER_WEEK = 5 * 24 * 15  # Sunday 17:00 to Friday 17:00, every 4 minutes
STALE = ("1.10006", "1.10016")
LATE_TICK = 15 * 91  # Thursday 12:00 New York


def _line(stamp: datetime, bid: str, ask: str) -> str:
    return f"{stamp:%Y%m%d %H%M%S}{stamp.microsecond // 1000:03d},{bid},{ask},0"


def week_lines(week: int) -> list[str]:
    sunday = FIRST_SUNDAY + timedelta(weeks=week)
    opened = datetime(sunday.year, sunday.month, sunday.day, 17)
    lines = []
    if week not in (2, 3):  # a lead tick would count as the week's first tick
        lines.append(_line(opened - timedelta(minutes=30), "1.10000", "1.10010"))
    for i in range(TICKS_PER_WEEK):
        if week in (2, 3) and i < 23:
            continue  # first tick ~90 min after the open: a misaligned week
        bid = f"{1.1 + ((i * 7919) % 13) * 0.00001:.5f}"
        spread = 0.0030 if 255 <= i <= 258 and week else 0.0001  # Monday 10:00
        ask = None
        if i < 5 or i >= TICKS_PER_WEEK - 25:
            bid, ask = STALE  # an identical-quote run that spans the weekend
        elif i == 6 and week:
            bid = f"{float(bid) + 0.02:.5f}"  # price spike right after the open
        ask = ask or f"{float(bid) + spread:.5f}"
        stamp = opened + i * STEP
        lines.append(_line(stamp, bid, ask))
        if i == LATE_TICK:  # 0.1 s behind a closed slot
            lines.append(_line(stamp - timedelta(milliseconds=100), bid, ask))
    saturday = opened + timedelta(days=6, hours=-5)
    return [*lines, _line(saturday, "1.10000", "1.10010")]


def install_history(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-02", start="2024-01")
    by_month: dict[str, list[str]] = {"202401": [], "202402": []}
    for week in range(WEEKS):
        for line in week_lines(week):
            if line[:6] in by_month:
                by_month[line[:6]].append(line)
    for period, lines in by_month.items():
        install_raw(config, "EURUSD", period, ("\n".join(lines) + "\n").encode())
    return config


def observed(config, chunk_weeks):
    result = build_backfill(config, 5, chunk_weeks=chunk_weeks)
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text())
    report.pop("timings")
    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    return result, report, bars


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    return install_history(tmp_path_factory.mktemp("chunked"))


@pytest.fixture(scope="module")
def serial(history):
    return observed(history, 1000)


def test_history_exercises_the_seam_sensitive_rules(serial):
    result, _, bars = serial
    counts = result.ref.manifest["coverage"]["EURUSD"]["tick_flag_counts"]
    assert counts["price_spike"] >= 5
    assert counts["stale_quote"] > 0
    assert any("spread_spike" in bar.quality_flags for bar in bars)
    codes = [w.code for w in result.warnings]
    assert "late_ticks_dropped" in codes
    assert "out_of_session_ticks" in codes
    assert codes.count("session_alignment") == 2


@pytest.mark.parametrize("chunk_weeks", [1, 2, 3, 5])
def test_any_chunk_size_builds_the_same_dataset(history, serial, chunk_weeks):
    result, report, bars = observed(history, chunk_weeks)
    base_result, base_report, base_bars = serial

    assert result.ref.dataset_id == base_result.ref.dataset_id
    assert result.warnings == base_result.warnings
    assert result.dropped_out_of_session == base_result.dropped_out_of_session
    assert report == base_report
    assert bars == base_bars


def test_weeks_without_ticks_between_chunks_are_still_gaps(tmp_path):
    from .test_gaps import three_month_config

    config = three_month_config(tmp_path)

    ids = {
        build_backfill(config, 5, chunk_weeks=n).ref.dataset_id for n in (1, 2, 1000)
    }

    assert len(ids) == 1
