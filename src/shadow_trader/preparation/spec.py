"""Feature spec registry: named polars expressions under an integer version.

A spec is bound to one bar size and lists the ``feat_*`` columns a prepared
Dataset carries. Horizons count Bars and are not configurable; changing a
definition, the column set or numerical behaviour means a new spec version in
code. The content hash covers each column's declared definition, so editing a
definition without bumping the version is caught by the golden tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

BAR_SIZE_5 = 5
RETURN_HORIZONS = (1, 3, 6, 12, 24, 48, 288)  # Bars: short, medium, long
# A sample std needs two returns, so the 1-Bar horizon has no volatility column.
VOLATILITY_HORIZONS = (3, 6, 12, 24, 48, 288)
ATR_WINDOW = 14  # Bars of true range averaged to scale intrabar shape
MEDIAN_WINDOW = 48  # Bars in the rolling median behind spread and activity


@dataclass(frozen=True)
class FeatureColumn:
    """One ``feat_<family>_<param>`` column and how it is computed."""

    name: str
    family: str
    definition: str  # canonical description, hashed into the spec hash
    lookback: int  # Bars of history the value needs
    expression: Callable[[], pl.Expr]
    always_valid: bool = False  # time/session columns: never nulled by the policy


@dataclass(frozen=True)
class FeatureSpec:
    version: int
    bar_size: int
    columns: tuple[FeatureColumn, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def hash(self) -> str:
        description = {
            "version": self.version,
            "bar_size": self.bar_size,
            "columns": [[c.name, c.definition] for c in self.columns],
        }
        payload = json.dumps(description, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _log_return(horizon: int) -> Callable[[], pl.Expr]:
    def expression() -> pl.Expr:
        # Per Session so a window never crosses a Session start; null (not NaN)
        # wherever either end is unpriced or the ratio is not finite.
        ratio = pl.col("close") / pl.col("close").shift(horizon).over("session_id")
        value = ratio.log()
        return pl.when(value.is_finite()).then(value).otherwise(None)

    return expression


def _volatility_family() -> tuple[FeatureColumn, ...]:
    def make(horizon: int) -> Callable[[], pl.Expr]:
        def expression() -> pl.Expr:
            # Every return in the window must exist (min_samples), so a single
            # null makes the std null; the policy adds the Gap/flag rule.
            value = (
                _one_bar_return()
                .rolling_std(horizon, min_samples=horizon)
                .over("session_id")
            )
            return pl.when(value.is_finite()).then(value).otherwise(None)

        return expression

    return tuple(
        FeatureColumn(
            name=f"feat_vol_{h}",
            family="volatility",
            definition=(
                f"sample std (ddof=1) of the last {h} 1-Bar log returns "
                "within session_id"
            ),
            lookback=h,
            expression=make(h),
        )
        for h in VOLATILITY_HORIZONS
    )


def _finite(value: pl.Expr) -> pl.Expr:
    return pl.when(value.is_finite()).then(value).otherwise(None)


def _atr() -> pl.Expr:
    """Simple 14-Bar mean of true range within a Session (finite memory).

    True range needs the previous close, so it is null at a Session's first Bar
    and after an unpriced Bar; the mean needs all 14 true ranges.
    """
    previous = pl.col("close").shift(1).over("session_id")
    true_range = (
        pl.when(previous.is_null())
        .then(None)
        .otherwise(
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - previous).abs(),
                (pl.col("low") - previous).abs(),
            )
        )
    )
    return true_range.rolling_mean(ATR_WINDOW, min_samples=ATR_WINDOW).over(
        "session_id"
    )


def _shape_family() -> tuple[FeatureColumn, ...]:
    body_top = pl.max_horizontal("open", "close")
    body_bottom = pl.min_horizontal("open", "close")
    parts = {
        "body": ("close - open", lambda: pl.col("close") - pl.col("open")),
        "range": ("high - low", lambda: pl.col("high") - pl.col("low")),
        "wick_up": ("high - max(open, close)", lambda: pl.col("high") - body_top),
        "wick_dn": ("min(open, close) - low", lambda: body_bottom - pl.col("low")),
    }

    def make(numerator: Callable[[], pl.Expr]) -> Callable[[], pl.Expr]:
        return lambda: _finite(numerator() / _atr())

    return tuple(
        FeatureColumn(
            name=f"feat_shape_{param}",
            family="shape",
            definition=(
                f"({text}) / atr, atr = mean of the last {ATR_WINDOW} true "
                "ranges within session_id (true range null without a previous close)"
            ),
            lookback=ATR_WINDOW,
            expression=make(numerator),
        )
        for param, (text, numerator) in parts.items()
    )


def _relative_to_median(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64)
    median = value.rolling_median(MEDIAN_WINDOW, min_samples=MEDIAN_WINDOW)
    return _finite(value / median.over("session_id"))


def _cost_and_activity_family() -> tuple[FeatureColumn, ...]:
    lookback = MEDIAN_WINDOW - 1
    return (
        FeatureColumn(
            name="feat_spr_close",
            family="spread",
            definition="spread_close / close",
            lookback=0,
            expression=lambda: _finite(pl.col("spread_close") / pl.col("close")),
        ),
        FeatureColumn(
            name="feat_spr_mean",
            family="spread",
            definition=(
                f"spread_mean / median of the last {MEDIAN_WINDOW} spread_mean "
                "within session_id"
            ),
            lookback=lookback,
            expression=lambda: _relative_to_median("spread_mean"),
        ),
        FeatureColumn(
            name="feat_act_tick",
            family="activity",
            definition=(
                f"tick_count / median of the last {MEDIAN_WINDOW} tick_count "
                "within session_id; null whenever tick_count is null"
            ),
            lookback=lookback,
            expression=lambda: _relative_to_median("tick_count"),
        ),
    )


def _time_family() -> tuple[FeatureColumn, ...]:
    hour = pl.col("open_time").dt.hour() + pl.col("open_time").dt.minute() / 60
    hour_angle = hour * (2 * math.pi / 24)
    day_angle = (pl.col("open_time").dt.weekday() - 1) * (2 * math.pi / 7)
    since_open = (pl.col("open_time") - pl.col("session_id")).dt.total_minutes()
    definitions = {
        "hour_sin": (
            "sin(2*pi*hour/24), hour = fractional UTC hour of open_time",
            lambda: hour_angle.sin(),
        ),
        "hour_cos": (
            "cos(2*pi*hour/24), hour = fractional UTC hour of open_time",
            lambda: hour_angle.cos(),
        ),
        "dow_sin": (
            "sin(2*pi*weekday/7), weekday Monday=0 of open_time",
            lambda: day_angle.sin(),
        ),
        "dow_cos": (
            "cos(2*pi*weekday/7), weekday Monday=0 of open_time",
            lambda: day_angle.cos(),
        ),
        "since_open": (
            "minutes from session_id (Session open) to open_time",
            lambda: since_open.cast(pl.Float64),
        ),
        "rollover": (
            "is_rollover as 0.0 / 1.0",
            lambda: pl.col("is_rollover").cast(pl.Float64),
        ),
    }
    return tuple(
        FeatureColumn(
            name=f"feat_time_{param}",
            family="time",
            definition=text,
            lookback=0,
            expression=expression,
            always_valid=True,
        )
        for param, (text, expression) in definitions.items()
    )


def _one_bar_return() -> pl.Expr:
    return (pl.col("close") / pl.col("close").shift(1).over("session_id")).log()


def _returns_family() -> tuple[FeatureColumn, ...]:
    return tuple(
        FeatureColumn(
            name=f"feat_ret_{h}",
            family="returns",
            definition=f"ln(close / close.shift({h})) within session_id",
            lookback=h,
            expression=_log_return(h),
        )
        for h in RETURN_HORIZONS
    )


SPECS: dict[int, FeatureSpec] = {
    1: FeatureSpec(1, BAR_SIZE_5, _returns_family()),
    2: FeatureSpec(
        2,
        BAR_SIZE_5,
        _returns_family()
        + _volatility_family()
        + _shape_family()
        + _cost_and_activity_family()
        + _time_family(),
    ),
}


def compute_features(
    history: pl.LazyFrame | pl.DataFrame, spec: FeatureSpec
) -> pl.LazyFrame:
    """Append the spec's ``feat_*`` columns to one Instrument's Bar history.

    ``history`` holds the ingestion Bar columns for a single Instrument. Rows
    are ordered by ``open_time`` and none is dropped; each value depends only
    on its own Bar and earlier Bars of the same Session.
    """
    ordered = history.lazy().sort("open_time")
    return ordered.with_columns(*(c.expression().alias(c.name) for c in spec.columns))
