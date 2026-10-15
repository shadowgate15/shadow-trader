"""The live seam: what live orchestration calls to compute features on a tail.

ADR 0004: there is one feature implementation (``compute_features``). The live
run re-evaluates it over a fixed-length tail of recent Bars and keeps the last
row; there is no per-Bar update, cache or second code path. The tail start is
treated like a Dataset start, so a tail shorter than ``min_tail_bars`` yields
nulls rather than errors, and parity with the batch build is claimed only at or
above ``min_tail_bars``. Bar ``revision`` plays no part here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from shadow_trader.ingestion.bars import Bar

from .anomaly import DEFAULT_POLICY, AnomalyPolicy
from .build import featured_frame
from .scaler import Scaler, ScalerError
from .spec import SPECS, FeatureSpec

LIVE_SPEC_VERSION = 3
# A recursive column's declared lookback is where an older input weighs under
# 1e-9; the warm-up goes on top so truncation error stays well under that.
WARM_UP_WEIGHT = 1e-12
_RECURSIVE_ALPHA = 1 / 14  # the slowest decay: Wilder (alpha = 1/14)


def _extra_warm_up(lookback: int) -> int:
    total = math.ceil(math.log(WARM_UP_WEIGHT) / math.log(1 - _RECURSIVE_ALPHA))
    return max(0, total - lookback)


@dataclass(frozen=True)
class LiveFeatureSpec:
    """A feature spec as seen by live orchestration."""

    spec: FeatureSpec
    policy: AnomalyPolicy
    warm_up: dict[str, int]  # per column: Bars beyond the lookback a tail needs

    @property
    def version(self) -> int:
        return self.spec.version

    @property
    def hash(self) -> str:
        return self.spec.hash

    @property
    def policy_hash(self) -> str:
        return self.policy.hash

    @property
    def min_tail_bars(self) -> int:
        """Shortest tail with a parity guarantee: deepest lookback + warm-up."""
        return max(
            (c.lookback + 1 + self.warm_up[c.name] for c in self.spec.columns),
            default=1,
        )


def live_spec(
    spec: FeatureSpec, policy: AnomalyPolicy = DEFAULT_POLICY
) -> LiveFeatureSpec:
    warm_up = {
        c.name: _extra_warm_up(c.lookback) if c.tolerance else 0 for c in spec.columns
    }
    return LiveFeatureSpec(spec, policy, warm_up)


FEATURE_SPEC = live_spec(SPECS[LIVE_SPEC_VERSION])


def evaluate_tail(
    tail_bars: Sequence[Bar], spec: LiveFeatureSpec = FEATURE_SPEC
) -> pl.DataFrame:
    """The last Bar's ``feat_*``, ``usable`` and ``unusable_reason`` as one row.

    Stateless and pure: ``tail_bars`` is one Instrument's recent Bars in
    ``open_time`` order. The tail start is treated like a Dataset start, so a
    short tail gives nulls, never an error; an empty tail gives an unusable
    all-null row.
    """
    if not tail_bars:
        return pl.DataFrame(
            {
                **{n: pl.Series([None], dtype=pl.Float64) for n in spec.spec.names},
                "usable": [False],
                "unusable_reason": ["dataset_start"],
            }
        )
    frame = featured_frame(tail_bars, spec.spec, spec.policy)
    return frame.select(*spec.spec.names, "usable", "unusable_reason").tail(1)


def check_compat(
    scaler: Scaler,
    spec: LiveFeatureSpec = FEATURE_SPEC,
    *,
    manifest_polars_version: str | None = None,
) -> list[str]:
    """Refuse a scaler fit under another spec or policy; return soft warnings.

    A spec or anomaly-policy hash mismatch raises ``ScalerError``. A polars
    version differing from ``manifest_polars_version`` (the Dataset manifest's)
    is only a warning: numerical behaviour may drift, so parity should be re-run.
    """
    artifact = scaler.artifact
    if artifact["spec_hash"] != spec.hash:
        raise ScalerError(
            f"spec hash mismatch: scaler was fit on spec {artifact['spec_hash']}, "
            f"live spec v{spec.version} is {spec.hash}"
        )
    if artifact["anomaly_policy_hash"] != spec.policy_hash:
        raise ScalerError(
            "anomaly policy hash mismatch: scaler was fit under policy "
            f"{artifact['anomaly_policy_hash']}, live policy is {spec.policy_hash}"
        )
    warnings: list[str] = []
    if manifest_polars_version not in (None, pl.__version__):
        warnings.append(
            f"polars version {pl.__version__} differs from the manifest's "
            f"{manifest_polars_version}; re-run the tail parity check"
        )
    return warnings


@dataclass(frozen=True)
class ParityReport:
    """Tail evaluation against the batch build over the compared rows."""

    rows_compared: int
    max_deviation: dict[str, float]  # per column, over rows non-null in both
    null_mismatches: dict[str, int]  # per column, rows null in exactly one side
    usable_mismatches: int
    reason_mismatches: int

    @property
    def ok(self) -> bool:
        return (
            not self.null_mismatches
            and self.usable_mismatches == 0
            and self.reason_mismatches == 0
        )


def check_tail_parity(
    bars: Sequence[Bar],
    tail_len: int,
    spec: LiveFeatureSpec = FEATURE_SPEC,
    *,
    every: int = 1,
) -> ParityReport:
    """Replay ``bars`` through ``evaluate_tail`` and compare with the batch build.

    Row ``i`` is evaluated on ``bars[i - tail_len + 1 : i + 1]``; only rows with
    a full ``tail_len`` Bars are compared, every ``every``-th plus the last.
    """
    batch = featured_frame(bars, spec.spec, spec.policy)
    first = tail_len - 1
    indexes = sorted(
        {*range(first, len(bars), every), len(bars) - 1} - set(range(first))
    )
    names = spec.spec.names
    deviation = dict.fromkeys(names, 0.0)
    nulls = dict.fromkeys(names, 0)
    usable = reasons = 0
    for i in indexes:
        live = evaluate_tail(bars[i - first : i + 1], spec)
        expected = batch.row(i, named=True)
        got = live.row(0, named=True)
        for name in names:
            a, b = got[name], expected[name]
            if (a is None) != (b is None):
                nulls[name] += 1
            elif a is not None:
                deviation[name] = max(deviation[name], abs(a - b))
        usable += got["usable"] != expected["usable"]
        reasons += got["unusable_reason"] != expected["unusable_reason"]
    return ParityReport(
        rows_compared=len(indexes),
        max_deviation=deviation,
        null_mismatches={n: c for n, c in nulls.items() if c},
        usable_mismatches=usable,
        reason_mismatches=reasons,
    )
