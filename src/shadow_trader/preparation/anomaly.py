"""Anomaly policy: which Bars are trustworthy, and why the others are not.

A ``feat_*`` value is null whenever its own lookback window (the Bar and the
``lookback`` Bars before it, within one Session) holds a Gap Bar or a Bar
carrying a distrusted quality flag. Windows never cross a Session start or the
Dataset start. Each Bar gets ``usable`` and a single ``unusable_reason``.
A Bar is usable only when every history-dependent ``feat_*`` (lookback > 0) is
non-null; a null that survives the window rules (a missing ``tick_count``)
is reported as ``window_contains_gap``, the last-resort reason.
Raw rows are never altered or dropped; the policy only nulls ``feat_*`` values
and adds the two derived columns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import polars as pl

from .spec import FeatureSpec

POLICY_VERSION = 1
DEFAULT_DISTRUSTED = ("crossed_quote", "price_spike", "stale_quote")
# Informational only (never distrusted by default): zero_spread, spread_spike.

# Fixed priority, highest first.
REASONS = (
    "gap",
    "distrusted_flag",
    "window_contains_gap",
    "window_contains_distrusted_flag",
    "session_warm_up",
    "dataset_start",
)


@dataclass(frozen=True)
class AnomalyPolicy:
    distrusted_flags: tuple[str, ...] = DEFAULT_DISTRUSTED
    version: int = POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "distrusted_flags", tuple(sorted(set(self.distrusted_flags)))
        )

    @property
    def hash(self) -> str:
        payload = json.dumps(
            {"version": self.version, "distrusted_flags": list(self.distrusted_flags)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


DEFAULT_POLICY = AnomalyPolicy()


def _window_has(flag: pl.Expr, horizon: int) -> pl.Expr:
    """True when ``flag`` holds anywhere in the Bar and ``horizon`` Bars before it."""
    return (
        flag.cast(pl.UInt32).rolling_sum(horizon + 1, min_samples=1).over("session_id")
        > 0
    )


def apply_anomaly_policy(
    frame: pl.DataFrame, spec: FeatureSpec, policy: AnomalyPolicy
) -> pl.DataFrame:
    """Null window-bad ``feat_*`` values and add ``usable`` / ``unusable_reason``.

    ``frame`` is one Instrument's Bars with features, ordered by ``open_time``.
    """
    if frame.height == 0:
        return frame.with_columns(
            pl.lit(None, pl.Boolean).alias("usable"),
            pl.lit(None, pl.String).alias("unusable_reason"),
        )
    # Every column but the always-valid time family is nulled by its own window
    # (lookback 0 means the Bar itself); only lookback > 0 sets the warm-up.
    windowed = [c for c in spec.columns if not c.always_valid]
    horizon = max((c.lookback for c in windowed), default=0)
    history = [c.name for c in windowed if c.lookback > 0]
    distrusted = pl.Series(list(policy.distrusted_flags), dtype=pl.String)

    gap = pl.col("is_gap")
    flagged = (
        pl.col("quality_flags")
        .list.eval(pl.element().is_in(distrusted.implode()))
        .list.any()
        .fill_null(False)
    )
    first_session = frame["session_id"][0]
    position = pl.int_range(pl.len()).over("session_id")

    nulled = [
        pl.when(~_window_has(gap | flagged, c.lookback))
        .then(pl.col(c.name))
        .otherwise(None)
        .alias(c.name)
        for c in windowed
    ]
    reason = (
        pl.when(gap | pl.col("close").is_null())
        .then(pl.lit("gap"))
        .when(flagged)
        .then(pl.lit("distrusted_flag"))
        .when(_window_has(gap, horizon))
        .then(pl.lit("window_contains_gap"))
        .when(_window_has(flagged, horizon))
        .then(pl.lit("window_contains_distrusted_flag"))
        .when(position < horizon)
        .then(
            pl.when(pl.col("session_id") == first_session)
            .then(pl.lit("dataset_start"))
            .otherwise(pl.lit("session_warm_up"))
        )
        # Story 24: usable needs every history-dependent feat_* non-null. A null
        # left over after the policy came from a missing input (e.g. tick_count
        # null nulls feat_act_tick); the closest fixed reason is a window that
        # holds a bad Bar, so it is reported as window_contains_gap.
        .when(pl.any_horizontal(pl.lit(False), *(pl.col(n).is_null() for n in history)))
        .then(pl.lit("window_contains_gap"))
        .otherwise(None)
    )
    return (
        frame.with_columns(*nulled)
        .with_columns(reason.alias("unusable_reason"))
        .with_columns(pl.col("unusable_reason").is_null().alias("usable"))
        .select(pl.exclude("unusable_reason"), "unusable_reason")
    )


def policy_summary(
    policy: AnomalyPolicy, frames: Iterable[pl.DataFrame]
) -> Mapping[str, object]:
    """The manifest's anomaly-policy block: the policy, counts and usable fraction."""
    counts = dict.fromkeys(REASONS, 0)
    usable = rows = 0
    for frame in frames:
        rows += frame.height
        usable += int(frame["usable"].sum())
        for value, n in (
            frame["unusable_reason"].drop_nulls().value_counts().iter_rows()
        ):
            counts[value] += n
    return {
        "distrusted_flags": list(policy.distrusted_flags),
        "version": policy.version,
        "hash": policy.hash,
        "counts_by_reason": counts,
        "usable_count": usable,
        "usable_fraction": usable / rows if rows else 0.0,
    }
