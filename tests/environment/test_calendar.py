"""Calendar helper reference cases: Session-final Bars, rollover days, fillable."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from shadow_trader.environment import Market, market_from_row
from shadow_trader.environment.session_calendar import (
    bars_to_session_final,
    is_fillable,
    is_session_final,
    rollover_days,
)


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


# Winter: Session opens Sun 2024-01-07 17:00 EST = 22:00 UTC, closes Fri 22:00 UTC.
def test_only_the_last_bar_before_the_weekly_close_is_session_final() -> None:
    assert is_session_final(utc(2024, 1, 12, 21, 55), 5)
    assert not is_session_final(utc(2024, 1, 12, 21, 50), 5)
    assert not is_session_final(utc(2024, 1, 7, 22, 0), 5)


def test_session_final_follows_the_daylight_saving_offset() -> None:
    # Summer: close is Fri 2024-07-19 17:00 EDT = 21:00 UTC.
    assert is_session_final(utc(2024, 7, 19, 20, 55), 5)
    assert not is_session_final(utc(2024, 7, 19, 21, 55), 5)  # outside any Session


def test_bars_to_session_final_counts_down_from_the_open() -> None:
    assert bars_to_session_final(utc(2024, 1, 7, 22, 0), 5) == 1439
    assert bars_to_session_final(utc(2024, 1, 12, 21, 55), 5) == 0
    assert bars_to_session_final(utc(2024, 1, 12, 21, 50), 5) == 1


def test_rollover_days_are_1_on_mon_tue_thu_and_3_on_wednesday() -> None:
    assert rollover_days(utc(2024, 1, 8, 22, 0), 5) == 1  # Monday 17:00 NY
    assert rollover_days(utc(2024, 1, 9, 22, 0), 5) == 1  # Tuesday
    assert rollover_days(utc(2024, 1, 10, 22, 0), 5) == 3  # Wednesday
    assert rollover_days(utc(2024, 1, 11, 22, 0), 5) == 1  # Thursday


def test_rollover_days_are_0_on_every_other_bar() -> None:
    assert rollover_days(utc(2024, 1, 10, 21, 55), 5) == 0
    assert rollover_days(utc(2024, 1, 10, 22, 5), 5) == 0
    assert rollover_days(utc(2024, 1, 7, 22, 0), 5) == 0  # Session open Sunday
    assert rollover_days(utc(2024, 1, 12, 21, 55), 5) == 0  # Friday final Bar


def test_rollover_days_use_new_york_time_in_summer() -> None:
    assert rollover_days(utc(2024, 7, 17, 21, 0), 5) == 3
    assert rollover_days(utc(2024, 7, 17, 22, 0), 5) == 0


def test_fillable_is_priced_and_not_distrusted() -> None:
    assert is_fillable(True, ())
    assert is_fillable(True, ("zero_spread", "spread_spike"))  # informational
    assert not is_fillable(False, ())
    for flag in ("crossed_quote", "price_spike", "stale_quote"):
        assert not is_fillable(True, ("zero_spread", flag))
    assert not is_fillable(True, ("odd",), distrusted=("odd",))


ROW = {
    "open_time": utc(2024, 1, 10, 22, 0),
    "open": 1.1,
    "close": 1.1005,
    "spread_mean": 0.0002,
    "spread_close": 0.0003,
    "is_gap": False,
    "quality_flags": [],
}


def test_market_from_row_maps_prices_and_calendar_facts() -> None:
    market = market_from_row(ROW, 5)
    assert market == Market(
        open_bid=1.1,
        open_spread=0.0002,
        close_bid=1.1005,
        close_spread=0.0003,
        fillable=True,
        is_session_final=False,
        rollover_days=3,
        bars_to_session_final=1439 - 3 * 288,
    )


def test_market_from_row_marks_gap_and_distrusted_bars_unfillable() -> None:
    gap_row = {
        **ROW,
        "is_gap": True,
        "open": None,
        "close": None,
        "spread_mean": None,
        "spread_close": None,
    }
    gap = market_from_row(gap_row, 5)
    assert not gap.fillable
    assert math.isnan(gap.open_bid)
    spiked = market_from_row({**ROW, "quality_flags": ["price_spike"]}, 5)
    assert not spiked.fillable
    assert spiked.open_bid == 1.1  # priced values are kept
