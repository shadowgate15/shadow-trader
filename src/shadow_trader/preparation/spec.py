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
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

BAR_SIZE_5 = 5
RETURN_HORIZONS = (1, 3, 6, 12, 24, 48, 288)  # Bars: short, medium, long


@dataclass(frozen=True)
class FeatureColumn:
    """One ``feat_<family>_<param>`` column and how it is computed."""

    name: str
    family: str
    definition: str  # canonical description, hashed into the spec hash
    lookback: int  # Bars of history the value needs
    expression: Callable[[], pl.Expr]


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
