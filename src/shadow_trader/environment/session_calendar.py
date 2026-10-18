"""Calendar helper: Session-final, rollover days and fillability from timestamps.

Pure and shared by training and live. The engine never reads a clock or timezone;
it receives these facts through ``Market``. Session geometry comes from
``shadow_trader.ingestion.session`` (weekly open Sunday 17:00 New York, close Friday
17:00); a Bar is identified by its OPEN time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from shadow_trader.ingestion.session import NY, ROLLOVER, session_of
from shadow_trader.preparation.anomaly import DEFAULT_DISTRUSTED

from .engine import Market

__all__ = [
    "bars_to_session_final",
    "is_fillable",
    "is_session_final",
    "market_from_row",
    "rollover_days",
]

WEDNESDAY = 2
LAST_ROLLOVER_WEEKDAY = 3  # Thursday; the Friday 17:00 marker is the weekly close


def bars_to_session_final(open_time: datetime, bar_minutes: int) -> int:
    """Bars after this one until the Session's final Bar (0 on the final Bar).

    Raises ``ValueError`` for a Bar outside any Session (the weekly closure).
    """
    session = session_of(open_time)
    if session is None:
        raise ValueError(f"{open_time} is outside any Session")
    remaining = (session[1] - open_time) // timedelta(minutes=bar_minutes)
    return int(remaining) - 1


def is_session_final(open_time: datetime, bar_minutes: int) -> bool:
    """Whether the Bar opening at ``open_time`` ends at the weekly close."""
    return (
        session_of(open_time) is not None
        and bars_to_session_final(open_time, bar_minutes) == 0
    )


def rollover_days(open_time: datetime, bar_minutes: int) -> int:
    """Financing days charged on this Bar: 1 on the Mon/Tue/Thu 17:00 New York
    Bar, 3 on Wednesday's (triple), 0 on every other Bar. ``bar_minutes`` is
    accepted for symmetry with the other helpers; the rollover Bar is the one that
    opens at 17:00."""
    local = open_time.astimezone(NY)
    if local.time() != ROLLOVER or local.weekday() > LAST_ROLLOVER_WEEKDAY:
        return 0
    return 3 if local.weekday() == WEDNESDAY else 1


def is_fillable(
    priced: bool,
    quality_flags: Iterable[str],
    distrusted: Iterable[str] = DEFAULT_DISTRUSTED,
) -> bool:
    """Priced and carrying no distrusted quality flag."""
    return priced and not set(quality_flags) & set(distrusted)


def _price(value: Any) -> float:
    return math.nan if value is None else float(value)


def market_from_row(
    row: Mapping[str, Any],
    bar_minutes: int,
    distrusted: Iterable[str] = DEFAULT_DISTRUSTED,
) -> Market:
    """Build a ``Market`` from one prepared-Dataset (or live Bar) row.

    ``open_spread`` is the Bar's ``spread_mean`` (the Dataset has no open-spread
    column); ``close_spread`` is ``spread_close``. Null prices (Gaps) become NaN
    and the Bar is not fillable, so the engine must not trade on it.
    """
    open_time: datetime = row["open_time"]
    priced = not row["is_gap"] and row["open"] is not None
    return Market(
        open_bid=_price(row["open"]),
        open_spread=_price(row["spread_mean"]),
        close_bid=_price(row["close"]),
        close_spread=_price(row["spread_close"]),
        fillable=is_fillable(priced, row["quality_flags"] or (), distrusted),
        is_session_final=is_session_final(open_time, bar_minutes),
        rollover_days=rollover_days(open_time, bar_minutes),
        bars_to_session_final=bars_to_session_final(open_time, bar_minutes),
    )
