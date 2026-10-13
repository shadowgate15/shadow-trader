"""The calendar oracle: which raw-clock rule each Session week should follow.

Before 2018-12-16 HistData stamps are New York local time (US daylight saving);
from then they are Europe/Zurich local time minus six hours (EU daylight saving).
Both are UTC-5 in winter, so the rules only differ in the weeks between the US
and EU clock changes. The real-cache test checks the per-week inference, which
reads only ticks, against this independent calendar.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from shadow_trader.ingestion.backfill import read_raw_month
from shadow_trader.ingestion.clock import (
    FIXED_UTC_MINUS_5,
    NEW_YORK_LOCAL,
    WeekEvidence,
    calendar_offset,
    calendar_rule,
)
from shadow_trader.ingestion.dataset import RawCache

EDT, EST = timedelta(hours=-4), timedelta(hours=-5)

CASES = [
    # US era: New York local, so the US change dates decide
    (date(2018, 3, 11), EDT, NEW_YORK_LOCAL),
    (date(2018, 10, 28), EDT, NEW_YORK_LOCAL),  # EU already on winter time: unused
    (date(2018, 11, 4), EST, None),
    # EU era, week of the era switch itself
    (date(2018, 12, 16), EST, None),
    # EU era, US-only DST window: US on EDT, EU still winter -> UTC-5
    (date(2019, 3, 10), EST, FIXED_UTC_MINUS_5),
    (date(2024, 3, 10), EST, FIXED_UTC_MINUS_5),
    (date(2024, 3, 17), EST, FIXED_UTC_MINUS_5),
    # both on summer time
    (date(2024, 3, 31), EDT, NEW_YORK_LOCAL),
    (date(2024, 7, 14), EDT, NEW_YORK_LOCAL),
    # autumn window: EU back on winter time while the US is still on EDT
    (date(2019, 10, 27), EST, FIXED_UTC_MINUS_5),
    (date(2024, 10, 20), EDT, NEW_YORK_LOCAL),
    (date(2024, 10, 27), EST, FIXED_UTC_MINUS_5),
    # both on winter time
    (date(2024, 11, 3), EST, None),
    (date(2024, 1, 7), EST, None),
]


@pytest.mark.parametrize(("sunday", "offset", "rule"), CASES)
def test_calendar_predicts_the_raw_clock_of_a_session_week(sunday, offset, rule):
    assert calendar_offset(sunday) == offset
    assert calendar_rule(sunday) == rule


# --- real-cache cross-check (opt in with -m slow) -------------------------------


@pytest.mark.slow
def test_real_eurusd_weeks_decided_on_evidence_agree_with_the_calendar():
    root = os.environ.get("SHADOW_TRADER_RAW_ROOT")
    if not root or not (Path(root) / "raw/histdata/EURUSD").is_dir():
        pytest.skip("set SHADOW_TRADER_RAW_ROOT to a data_root with EURUSD raw months")
    cache = RawCache(Path(root).resolve())
    periods = sorted(
        p.name.removesuffix(".zip")
        for p in (Path(root) / "raw/histdata/EURUSD").glob("??????.zip")
    )
    evidence = WeekEvidence()
    for period in periods:
        evidence.add_text(read_raw_month(cache, "EURUSD", period))

    clock, failures = evidence.analyse()

    soft = clock.inherited | clock.single_end
    decided = {w: o for w, o in clock.offsets.items() if w not in soft}
    disagreements = {
        week.isoformat(): f"inferred {offset}, calendar {calendar_offset(week)}"
        for week, offset in sorted(decided.items())
        if offset != calendar_offset(week)
    }
    weak = {
        week.isoformat()
        for week in soft
        if clock.offsets[week] != calendar_offset(week)
    }
    print(
        f"{len(decided)} evidence-decided weeks, disagreements: {disagreements}; "
        f"failing weeks: {sorted(map(str, failures))}; "
        f"inherited/single_end weeks off the calendar: {sorted(weak)}"
    )
    assert decided
    assert not disagreements
