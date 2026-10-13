"""Per-week raw-clock inference: each Session week is NY-local or fixed UTC-5.

HistData's raw clock follows New York local time in most weeks, but from 2019 it
behaves as fixed UTC-5 in the weeks where US and EU daylight saving differ. The
backfill decides per week, from the first and last tick against the 17:00 New
York open and close, and fails when a week fits neither.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from shadow_trader.ingestion.backfill import BuildFailure, build_backfill
from shadow_trader.ingestion.dataset import open_dataset
from tests.ingestion.test_backfill import install_raw, make_config

TICK = "1.10000,1.10010,0"
EDT, EST = -240, -300  # raw clock UTC offsets, minutes


def stamp(day: date, hour: int, minute: int = 0, second: int = 0) -> str:
    return f"{day:%Y%m%d} {hour:02d}{minute:02d}{second:02d}000,{TICK}"


def session_lines(
    sunday: date,
    open_hour: int,
    close_hour: int,
    *,
    first_minute: int = 0,
) -> list[str]:
    """One Session week on a raw clock: first tick at ``open_hour`` Sunday, last tick
    just before ``close_hour`` Friday, and a tick at noon on Monday to Thursday."""
    friday = sunday + timedelta(days=5)
    return [
        stamp(sunday, open_hour, first_minute),
        *(stamp(sunday + timedelta(days=d), 12) for d in range(1, 5)),
        f"{friday:%Y%m%d} {close_hour - 1:02d}5959000,{TICK}",
    ]


NY_WEEK = (17, 17)  # NY-local clock: open and close both read 17:00
FIXED_DST_WEEK = (16, 16)  # fixed UTC-5 clock while New York is on EDT: 17:00 EDT


def build(tmp_path, lines, month="2024-03"):
    config = make_config(tmp_path, "EURUSD", month)
    install_raw(
        config, "EURUSD", month.replace("-", ""), ("\n".join(lines) + "\n").encode()
    )
    return config, lambda: build_backfill(config, 5)


def first_open(config, result):
    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    return bars[0].open_time


def offsets(result):
    return result.ref.manifest["session_clock"]["EURUSD"]["offsets"]


def test_ny_local_week_converts_with_the_new_york_offset_and_records_it(tmp_path):
    config, run = build(tmp_path, session_lines(date(2024, 3, 17), *NY_WEEK))

    result = run()

    assert first_open(config, result) == datetime(2024, 3, 17, 21, 0, tzinfo=UTC)
    assert offsets(result) == {"2024-03-17": EDT}
    assert not [w for w in result.warnings if w.code == "session_alignment"]


def test_gap_week_on_a_fixed_utc_minus_5_clock_is_inferred_and_does_not_warn(tmp_path):
    # US DST began 2024-03-10, EU's has not: the raw clock reads 16:00 at the open
    config, run = build(tmp_path, session_lines(date(2024, 3, 10), *FIXED_DST_WEEK))

    result = run()

    assert first_open(config, result) == datetime(2024, 3, 10, 21, 0, tzinfo=UTC)
    assert offsets(result) == {"2024-03-10": EST}
    assert not [w for w in result.warnings if w.code == "session_alignment"]


def test_each_week_gets_its_own_offset_across_a_gap_and_a_normal_week(tmp_path):
    lines = [
        *session_lines(date(2024, 3, 10), *FIXED_DST_WEEK),
        *session_lines(date(2024, 3, 17), *NY_WEEK),
    ]
    _, run = build(tmp_path, lines)

    result = run()

    assert offsets(result) == {"2024-03-10": EST, "2024-03-17": EDT}


def test_winter_week_is_the_same_under_both_rules(tmp_path):
    _, run = build(tmp_path, session_lines(date(2024, 1, 7), *NY_WEEK), "2024-01")

    result = run()

    assert offsets(result) == {"2024-01-07": EST}
    assert result.ref.manifest["session_clock"]["EURUSD"]["inherited"] == []


def test_week_fitting_neither_rule_fails_the_build_naming_the_week(tmp_path):
    # first tick 2.5 h before the open on either clock
    lines = [
        stamp(date(2024, 3, 10), 14, 30),
        *session_lines(date(2024, 3, 10), 16, 16)[1:],
    ]
    config, run = build(tmp_path, lines)

    with pytest.raises(BuildFailure, match=r"EURUSD.*2024-03-10") as caught:
        run()

    assert caught.value.locator == "EURUSD/week-2024-03-10"
    assert not (config.paths.data_root / "datasets").exists()
    assert caught.value.report_path is not None
    failed = json.loads(caught.value.report_path.read_text())
    assert failed["status"] == "failed"
    assert "2024-03-10" in failed["condition"]


def thin_week(sunday: date) -> list[str]:
    """Late first tick and early last tick: fits both rules, so neither is chosen."""
    friday = sunday + timedelta(days=5)
    return [
        stamp(sunday, 19, 30),
        stamp(sunday + timedelta(days=2), 12),
        stamp(friday, 13),
    ]


def test_thin_week_inherits_the_rule_when_both_neighbours_agree(tmp_path):
    lines = [
        *session_lines(date(2024, 3, 17), *NY_WEEK),
        *thin_week(date(2024, 3, 24)),
        *session_lines(date(2024, 3, 31), *NY_WEEK),
    ]
    config, run = build(tmp_path, lines)

    result = run()

    assert offsets(result)["2024-03-24"] == EDT
    assert result.ref.manifest["session_clock"]["EURUSD"]["inherited"] == ["2024-03-24"]
    report = json.loads(result.report_path.read_text())
    assert report["instruments"]["EURUSD"]["session_clock"]["inherited"] == [
        "2024-03-24"
    ]
    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    assert datetime(2024, 3, 24, 23, 30, tzinfo=UTC) in {b.open_time for b in bars}


def calendar_weeks(result):
    return result.ref.manifest["session_clock"]["EURUSD"]["calendar"]


def test_thin_week_between_disagreeing_neighbours_falls_back_to_the_calendar(
    tmp_path,
):
    # the neighbours cannot be inherited, but 2024-03-17 sits in the US-only DST
    # window, where the calendar says the raw clock is fixed UTC-5
    lines = [
        *session_lines(date(2024, 3, 10), *FIXED_DST_WEEK),
        *thin_week(date(2024, 3, 17)),
        *session_lines(date(2024, 3, 24), *NY_WEEK),
    ]
    _, run = build(tmp_path, lines)

    result = run()

    assert offsets(result)["2024-03-17"] == EST
    assert calendar_weeks(result) == ["2024-03-17"]
    assert result.ref.manifest["session_clock"]["EURUSD"]["inherited"] == []


def test_thin_week_next_to_another_thin_week_falls_back_to_the_calendar(tmp_path):
    lines = [
        *session_lines(date(2024, 3, 17), *NY_WEEK),
        *thin_week(date(2024, 3, 24)),
        *thin_week(date(2024, 3, 31)),
    ]
    _, run = build(tmp_path, lines)

    result = run()

    # US-only window, then both on summer time from the EU change on 03-31
    assert offsets(result)["2024-03-24"] == EST
    assert offsets(result)["2024-03-31"] == EDT
    assert calendar_weeks(result) == ["2024-03-24", "2024-03-31"]


def test_inherited_and_decided_weeks_are_not_recorded_as_calendar(tmp_path):
    lines = [
        *session_lines(date(2024, 3, 17), *NY_WEEK),
        *thin_week(date(2024, 3, 24)),
        *session_lines(date(2024, 3, 31), *NY_WEEK),
    ]
    _, run = build(tmp_path, lines)

    assert calendar_weeks(run()) == []


def test_calendar_does_not_rescue_a_week_whose_ticks_contradict_every_rule(tmp_path):
    # a thin week is undecided, but a stray tick 3 h before the open kills both
    # candidates, so the calendar has nothing to confirm
    lines = [
        *session_lines(date(2024, 3, 17), *NY_WEEK),
        stamp(date(2024, 3, 24), 13, 50),
        stamp(date(2024, 3, 26), 12),
        stamp(date(2024, 3, 29), 13),
    ]
    _, run = build(tmp_path, lines)

    with pytest.raises(BuildFailure, match=r"2024-03-24.*fits no raw-clock rule"):
        run()


def test_thin_week_at_the_edge_of_the_range_inherits_from_its_only_neighbour(
    tmp_path,
):
    lines = [*session_lines(date(2024, 3, 17), *NY_WEEK), *thin_week(date(2024, 3, 24))]
    _, run = build(tmp_path, lines)

    result = run()

    assert offsets(result)["2024-03-24"] == EDT


def test_tick_in_the_dst_change_hour_is_dropped_not_fatal(tmp_path):
    # Sun 2024-03-10 02:30 does not exist on a New York wall clock; the week's
    # offset is fixed, so it is simply a tick outside any Session
    lines = [
        stamp(date(2024, 3, 8), 16, 59, 59),
        stamp(date(2024, 3, 10), 2, 30),
        *session_lines(date(2024, 3, 10), *FIXED_DST_WEEK),
    ]
    _, run = build(tmp_path, lines)

    result = run()

    assert any(w.code == "out_of_session_ticks" for w in result.warnings)


def test_thin_week_next_to_a_winter_week_falls_back_to_the_calendar(tmp_path):
    # a winter neighbour says nothing about the rule, so the calendar decides
    lines = [
        *session_lines(date(2024, 3, 3), *NY_WEEK),
        *thin_week(date(2024, 3, 10)),
    ]
    _, run = build(tmp_path, lines)

    result = run()

    assert offsets(result)["2024-03-10"] == EST
    assert calendar_weeks(result) == ["2024-03-10"]


def single_end_week(open_hour: int, close_tick: str) -> list[str]:
    sunday = date(2024, 4, 7)
    return [
        stamp(sunday, open_hour),
        stamp(sunday + timedelta(days=2), 12),
        f"{sunday + timedelta(days=5):%Y%m%d} {close_tick}000,{TICK}",
    ]


def test_single_end_choice_against_the_calendar_defers_to_the_calendar(tmp_path):
    # both clocks on summer time: the calendar says New York local. Only the last
    # tick (1 s before the close on a fixed UTC-5 reading) backs the other rule,
    # and both ends sit inside the session under New York local.
    _, run = build(tmp_path, single_end_week(18, "155959"), "2024-04")

    result = run()

    assert offsets(result) == {"2024-04-07": EDT}
    assert calendar_weeks(result) == ["2024-04-07"]
    assert result.ref.manifest["session_clock"]["EURUSD"]["single_end"] == []


def test_single_end_choice_that_matches_the_calendar_stays_single_end(tmp_path):
    _, run = build(tmp_path, single_end_week(17, "140000"), "2024-04")

    result = run()

    assert offsets(result) == {"2024-04-07": EDT}
    assert calendar_weeks(result) == []
    assert result.ref.manifest["session_clock"]["EURUSD"]["single_end"] == [
        "2024-04-07"
    ]
