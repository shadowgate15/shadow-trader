"""Clock evidence is gathered per month and merged by the parent in any order."""

from __future__ import annotations

import random
from datetime import date, timedelta

from shadow_trader.ingestion.clock import WeekEvidence, month_evidence


def _month_text(year: int, month: int) -> str:
    """Ticks stamped on New York local time: Sunday open and Friday close of each
    week of the month, plus a mid-week tick."""
    lines: list[str] = []
    day = date(year, month, 1)
    while day.month == month:
        if day.weekday() == 6:
            lines.append(f"{day:%Y%m%d} 170100,1.1,1.1,0")
        if day.weekday() == 2:
            lines.append(f"{day:%Y%m%d} 120000,1.1,1.1,0")
        if day.weekday() == 4:
            lines.append(f"{day:%Y%m%d} 165500,1.1,1.1,0")
        day += timedelta(days=1)
    return "\n".join(lines) + "\n"


def _summary(evidence: WeekEvidence):
    clock = evidence.infer()
    return clock.summary()


def test_merging_months_in_any_order_infers_the_same_clock():
    months = [(2024, m) for m in range(1, 9)]
    months = [(f"EURUSD/2024{m:02d}", _month_text(y, m)) for y, m in months]

    serial = WeekEvidence()
    for label, text in months:
        serial.add_text(text, label)
    expected = _summary(serial)

    parts = [month_evidence(text, label) for label, text in months]
    for seed in range(5):
        shuffled = parts[:]
        random.Random(seed).shuffle(shuffled)
        merged = WeekEvidence()
        for part in shuffled:
            merged.merge(part)
        assert _summary(merged) == expected
