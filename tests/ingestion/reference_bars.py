"""Independent reference Bar builder: plain Python, Decimal, tick-based.

Deliberately shares no code with the production builder. It enumerates every
Session between the first and last tick, then every slot of each Session, and
selects ticks by brute-force scan.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
PRICE_KEYS = (
    "open",
    "high",
    "low",
    "close",
    "spread_close",
    "spread_mean",
    "tick_count",
)


def _sunday_opens(first: datetime, last: datetime) -> list[datetime]:
    day = (first - timedelta(days=8)).astimezone(NY).date()
    opens = []
    for i in range(400 * 7):
        d = day + timedelta(days=i)
        if d.weekday() != 6:
            continue
        opn = datetime.combine(d, time(17, 0), tzinfo=NY).astimezone(UTC)
        if opn > last:
            break
        opens.append(opn)
    return opens


def reference_bars(ticks, minutes: int):
    """ticks: iterable of (datetime UTC, Decimal bid, Decimal ask).

    Returns (rows, dropped_count) for the dense grid from the first Session
    that holds a tick to the last.
    """
    ticks = sorted(ticks, key=lambda t: t[0])  # sorted() is stable
    first, last = ticks[0][0], ticks[-1][0]
    sessions = []
    for opn in _sunday_opens(first, last):
        close = datetime.combine(
            opn.astimezone(NY).date() + timedelta(days=5), time(17, 0), tzinfo=NY
        ).astimezone(UTC)
        sessions.append((opn, close))
    held = [(o, c) for o, c in sessions if any(o <= t < c for t, _, _ in ticks)]
    sessions = [(o, c) for o, c in sessions if held[0][0] <= o <= held[-1][0]]
    dropped = sum(1 for t, _, _ in ticks if not any(o <= t < c for o, c in sessions))
    rows = []
    for opn, close in sessions:
        slot = opn
        while slot < close:
            end = slot + timedelta(minutes=minutes)
            inside = [(b, a) for t, b, a in ticks if slot <= t < end]
            local = slot.astimezone(NY)
            rollover = local.time() == time(17, 0) and local.weekday() in (0, 1, 2, 3)
            row: dict[str, Any] = {
                "open_time": slot,
                "session_id": opn,
                "is_rollover": rollover,
            }
            if inside:
                spreads = [a - b for b, a in inside]
                row |= {
                    "open": float(inside[0][0]),
                    "high": float(max(b for b, _ in inside)),
                    "low": float(min(b for b, _ in inside)),
                    "close": float(inside[-1][0]),
                    "spread_close": float(spreads[-1]),
                    "spread_mean": float(Fraction(sum(spreads)) / len(spreads)),
                    "tick_count": len(inside),
                    "is_gap": False,
                }
            else:
                row |= dict.fromkeys(PRICE_KEYS) | {"is_gap": True}
            rows.append(row)
            slot = end
    return rows, dropped


def d(x: str) -> Decimal:
    return Decimal(x)
