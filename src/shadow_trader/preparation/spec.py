"""Feature spec registry: named polars expressions under an integer version.

A spec is bound to one bar size and lists the ``feat_*`` columns a prepared
Dataset carries. Horizons count Bars and are not configurable; changing a
definition, the column set or numerical behaviour means a new spec version in
code. The content hash covers each column's declared definition, so editing a
definition without bumping the version is caught by the golden tests.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import polars as pl

from ._io import canonical_json

BAR_SIZE_5 = 5
RETURN_HORIZONS = (1, 3, 6, 12, 24, 48, 288)  # Bars: short, medium, long
# A sample std needs two returns, so the 1-Bar horizon has no volatility column.
VOLATILITY_HORIZONS = (3, 6, 12, 24, 48, 288)
ATR_WINDOW = 14  # Bars of true range averaged to scale intrabar shape
MEDIAN_WINDOW = 48  # Bars in the rolling median behind spread and activity
EMA_SPANS = (12, 24)  # Bars; trend distance is close minus the EMA, in ATR units
RSI_WINDOW = 14
WILDER_WINDOW = 14  # ATR (and RSI) smoothing: alpha = 1 / 14
FULL_HASH_FROM_VERSION = 4  # first spec whose hash covers lookback and scaling
LONG_MEMORY_TOLERANCE = 1e-9  # default absolute parity tolerance for EMA/RSI/ATR


def _decay_bars(alpha: float) -> int:
    """Bars after which an input's weight in a recursive average is below 1e-9."""
    return math.ceil(math.log(1e-9) / math.log(1 - alpha))


ATR_LOOKBACK = _decay_bars(1 / WILDER_WINDOW)  # 280


@dataclass(frozen=True)
class FeatureColumn:
    """One ``feat_<family>_<param>`` column and how it is computed."""

    name: str
    family: str
    definition: str  # canonical description, hashed into the spec hash
    lookback: int  # Bars of history the value needs
    expression: Callable[[], pl.Expr]
    always_valid: bool = False  # time/session columns: never nulled by the policy
    # Absolute parity tolerance between batch and tail evaluation. 0.0 means
    # bitwise (finite memory); recursive EMA/RSI/ATR columns declare 1e-9.
    tolerance: float = 0.0
    # Scaler method: ``robust_clip`` or ``identity`` (bounded / flag columns).
    scaling: str = "robust_clip"


@dataclass(frozen=True)
class FeatureSpec:
    version: int
    bar_size: int
    columns: tuple[FeatureColumn, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def longest_lookback(self) -> int:
        """Bars of history the deepest column needs; sizes the Embargo."""
        return max((c.lookback for c in self.columns), default=0)

    @property
    def hash(self) -> str:
        """SHA-256 of the canonical description of the spec.

        From version 4 every column contributes its name, definition, lookback,
        parity tolerance and scaling method. Versions 1-3 keep their original
        payload (definition, plus lookback and tolerance only for recursive
        columns) so their hashes, and every Dataset built from them, stay valid.
        """
        if self.version >= FULL_HASH_FROM_VERSION:
            columns = [
                [c.name, c.definition, c.lookback, c.tolerance, c.scaling]
                for c in self.columns
            ]
        else:
            columns = [
                [c.name, c.definition]
                + ([c.tolerance, c.lookback] if c.tolerance else [])
                for c in self.columns
            ]
        description = {
            "version": self.version,
            "bar_size": self.bar_size,
            "columns": columns,
        }
        return hashlib.sha256(canonical_json(description)).hexdigest()


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
            # Every return in the window must exist, so a single null makes the
            # std null; the policy adds the Gap/flag rule. The first return of a
            # Session is null, so a window never crosses a Session start. The std
            # is a per-window group aggregate rather than the sliding-sum
            # rolling_std, whose rounding depends on where the series starts and
            # would break bitwise tail/batch parity (ADR 0004).
            same_session = pl.col("session_id") == pl.col("session_id").shift(1)
            step = pl.when(same_session).then(
                (pl.col("close") / pl.col("close").shift(1)).log()
            )
            index = pl.int_range(pl.len(), dtype=pl.Int64)
            # a window holds horizon + 1 Bars: the first only seeds its neighbour
            period = f"{horizon + 1}i"
            std = step.std(ddof=1).rolling(index_column=index, period=period)
            seen = step.count().rolling(index_column=index, period=period)
            value = pl.when(seen == horizon).then(std)
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


def _wilder(value: pl.Expr, first: int) -> pl.Expr:
    """Wilder-smoothed average (alpha = 1/14) seeded with a plain 14-value mean.

    ``value`` first exists at Session position ``first``; the seed is the mean of
    its first 14 values and sits at position ``first + 13``. A null input is
    skipped (the average carries forward unchanged). Per Session.
    """
    return _seeded_average(value, first, WILDER_WINDOW, 1 / WILDER_WINDOW)


def _seeded_average(value: pl.Expr, first: int, n: int, alpha: float) -> pl.Expr:
    position = pl.int_range(pl.len()).over("session_id")
    seed_at = first + n - 1
    start = (
        pl.when(position < seed_at)
        .then(None)
        .when(position == seed_at)
        .then(value.rolling_mean(n, min_samples=n).over("session_id"))
        .otherwise(value)
    )
    return start.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True).over(
        "session_id"
    )


def _true_range() -> pl.Expr:
    previous = pl.col("close").shift(1).over("session_id")
    return (
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


def _wilder_atr() -> pl.Expr:
    """Wilder ATR: seeded at Session position 14 with the mean of true ranges 1..14."""
    return _wilder(_true_range(), first=1)


def _atr_volatility() -> FeatureColumn:
    return FeatureColumn(
        name="feat_vol_atr",
        family="volatility",
        definition=(
            f"atr / close; atr is Wilder ATR (alpha = 1/{WILDER_WINDOW}) seeded "
            f"with the mean of true ranges 1..{WILDER_WINDOW} within session_id; "
            "lookback = Bars until an older input weighs under 1e-9"
        ),
        lookback=ATR_LOOKBACK,
        expression=lambda: _finite(_wilder_atr() / pl.col("close")),
        tolerance=LONG_MEMORY_TOLERANCE,
    )


def _trend_family() -> tuple[FeatureColumn, ...]:
    def ema_distance(span: int) -> Callable[[], pl.Expr]:
        def expression() -> pl.Expr:
            ema = _seeded_average(pl.col("close"), 0, span, 2 / (span + 1))
            return _finite((pl.col("close") - ema) / _wilder_atr())

        return expression

    def rsi() -> pl.Expr:
        change = pl.col("close") - pl.col("close").shift(1).over("session_id")
        gain = _wilder(change.clip(lower_bound=0), first=1)
        loss = _wilder((-change).clip(lower_bound=0), first=1)
        value = (
            pl.when((gain == 0) & (loss == 0))
            .then(50.0)
            .otherwise(100 * gain / (gain + loss))
        )
        return _finite(value)

    columns = tuple(
        FeatureColumn(
            name=f"feat_trend_ema_{span}",
            family="trend",
            definition=(
                f"(close - ema) / atr; ema span {span} seeded with the SMA of the "
                f"first {span} closes in session_id, alpha = 2/({span}+1); atr is "
                f"Wilder ATR (alpha = 1/{WILDER_WINDOW}) seeded with the mean of "
                f"true ranges 1..{WILDER_WINDOW}; a null input is skipped; "
                "lookback = Bars until an older input weighs under 1e-9"
            ),
            lookback=ATR_LOOKBACK,
            expression=ema_distance(span),
            tolerance=LONG_MEMORY_TOLERANCE,
        )
        for span in EMA_SPANS
    )
    rsi_column = FeatureColumn(
        name=f"feat_trend_rsi_{RSI_WINDOW}",
        family="trend",
        definition=(
            f"100 * avg_gain / (avg_gain + avg_loss) of 1-Bar close changes, "
            f"Wilder smoothed (alpha = 1/{RSI_WINDOW}) and seeded with the mean "
            f"of the first {RSI_WINDOW} changes in session_id; 50 when both "
            "are zero; lookback = Bars until an older input weighs under 1e-9"
        ),
        lookback=ATR_LOOKBACK,
        expression=rsi,
        tolerance=LONG_MEMORY_TOLERANCE,
    )
    return (*columns, rsi_column)


def _wilder_shape_family() -> tuple[FeatureColumn, ...]:
    return tuple(
        replace(
            column,
            definition=column.definition.replace(
                f"atr = mean of the last {ATR_WINDOW} true ranges within session_id "
                "(true range null without a previous close)",
                f"atr = Wilder ATR (alpha = 1/{WILDER_WINDOW}) seeded with the mean "
                f"of true ranges 1..{WILDER_WINDOW} within session_id; "
                "lookback = Bars until an older input weighs under 1e-9",
            ),
            lookback=ATR_LOOKBACK,
            tolerance=LONG_MEMORY_TOLERANCE,
        )
        for column in _shape_family(_wilder_atr)
    )


def _shape_family(
    atr: Callable[[], pl.Expr] | None = None,
) -> tuple[FeatureColumn, ...]:
    atr = atr or _atr
    body_top = pl.max_horizontal("open", "close")
    body_bottom = pl.min_horizontal("open", "close")
    parts = {
        "body": ("close - open", lambda: pl.col("close") - pl.col("open")),
        "range": ("high - low", lambda: pl.col("high") - pl.col("low")),
        "wick_up": ("high - max(open, close)", lambda: pl.col("high") - body_top),
        "wick_dn": ("min(open, close) - low", lambda: body_bottom - pl.col("low")),
    }

    def make(numerator: Callable[[], pl.Expr]) -> Callable[[], pl.Expr]:
        return lambda: _finite(numerator() / atr())

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
            scaling="robust_clip" if param == "since_open" else "identity",
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
    3: FeatureSpec(
        3,
        BAR_SIZE_5,
        _returns_family()
        + _volatility_family()
        + _wilder_shape_family()
        + _cost_and_activity_family()
        + _trend_family()
        + _time_family(),
    ),
    # v3 plus ATR / close in the volatility family; the hash now covers every
    # column's lookback, tolerance and scaling method.
    4: FeatureSpec(
        4,
        BAR_SIZE_5,
        _returns_family()
        + _volatility_family()
        + (_atr_volatility(),)
        + _wilder_shape_family()
        + _cost_and_activity_family()
        + _trend_family()
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
