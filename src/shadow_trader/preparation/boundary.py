"""``boundary_check``: leakage safeguards run on every build.

Checks run in a fixed order and the first failure raises ``BoundaryCheckFailure``
naming it; the build then aborts before anything is written. There is no
override. A passing run returns the block recorded in all three manifests and the
build record.

A check is a ``(name, function)`` pair; the function takes a ``BoundaryContext``
and returns ``None`` to pass or a detail string to fail. ``boundary_check`` takes
extra checks after the built-in ones, which is how the scaler ticket adds its
"fit range inside train" check without touching this module.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from .spec import FeatureSpec
from .split import ROLES, RoleSlice, SplitBounds

# Closed failure codes that are not ``boundary_check.<name>``.
SPECIAL_CODES = {
    "dataset_length": "embargo_exceeds_dataset",
    "usable_bars": "no_usable_bars",
}


class BoundaryCheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        self.name = name
        self.detail = detail
        self.code = SPECIAL_CODES.get(name, f"boundary_check.{name}")
        super().__init__(f"{self.code}: {detail}")


@dataclass(frozen=True)
class BoundaryContext:
    bounds: SplitBounds
    roles: Mapping[str, RoleSlice]
    parent_open_times: pl.Series  # every parent Bar of the Instrument, ascending
    embargo_bars: int
    spec: FeatureSpec


Check = Callable[[BoundaryContext], str | None]


def _ordering(ctx: BoundaryContext) -> str | None:
    for label, bounds in (
        ("nominal", ctx.bounds.nominal),
        ("snapped", ctx.bounds.snapped),
    ):
        given = [(k, v) for k, v in bounds.items() if v is not None]
        for (a, early), (b, late) in itertools.pairwise(given):
            if not early < late:
                return (
                    f"{label} bounds must be strictly increasing: "
                    f"{a} {early.isoformat()} is not before {b} {late.isoformat()}"
                )
    return None


def _session_open(ctx: BoundaryContext) -> str | None:
    for role, part in ctx.roles.items():
        frame = part.frame
        if frame.height == 0:
            continue  # reported by dataset_length
        start = ctx.bounds.start(role)
        first = frame.row(0, named=True)
        if first["open_time"] != start or first["session_id"] != start:
            return (
                f"{role} must start at a Session open ({start.isoformat()}); "
                f"its first Bar opens {first['open_time'].isoformat()}"
            )
    return None


def _dataset_length(ctx: BoundaryContext) -> str | None:
    for role, part in ctx.roles.items():
        if part.frame.height == 0:
            return (
                f"{role} has no Bars after the Embargo trim "
                f"(E={ctx.embargo_bars}, {part.raw_height} Bars in range)"
            )
    return None


def _usable_bars(ctx: BoundaryContext) -> str | None:
    for role, part in ctx.roles.items():
        if not part.frame["usable"].any():
            return f"{role} has no usable Bars after the Embargo trim"
    return None


def _gap_between_splits(ctx: BoundaryContext) -> str | None:
    times = ctx.parent_open_times
    for earlier, later in itertools.pairwise(ROLES):
        last = ctx.roles[earlier].frame["open_time"][-1]
        first = ctx.roles[later].frame["open_time"][0]
        between = int(((times > last) & (times < first)).sum())
        if between < ctx.embargo_bars:
            return (
                f"{between} Bars between {earlier} and {later}; "
                f"at least E={ctx.embargo_bars} are required"
            )
    return None


def _no_window_before_dataset_start(ctx: BoundaryContext) -> str | None:
    """Each feature is null for its first ``lookback`` Bars of the Dataset."""
    for role, part in ctx.roles.items():
        for column in ctx.spec.columns:
            head = part.frame[column.name].head(column.lookback)
            if head.null_count() != head.len():
                return (
                    f"{role}: {column.name} has a non-null value within its first "
                    f"{column.lookback} Bars, so its window reaches before the "
                    "Dataset start"
                )
    return None


BUILT_IN: tuple[tuple[str, Check], ...] = (
    ("ordering", _ordering),
    ("session_open", _session_open),
    ("dataset_length", _dataset_length),
    ("usable_bars", _usable_bars),
    ("gap_between_splits", _gap_between_splits),
    ("window_before_dataset_start", _no_window_before_dataset_start),
)


def boundary_check(
    context: BoundaryContext, extra: Sequence[tuple[str, Check]] = ()
) -> dict[str, Any]:
    """Run every check; raise ``BoundaryCheckFailure`` on the first that fails."""
    names: list[str] = []
    for name, check in (*BUILT_IN, *extra):
        detail = check(context)
        if detail is not None:
            raise BoundaryCheckFailure(name, detail)
        names.append(name)
    return {
        "passed": True,
        "embargo_bars": context.embargo_bars,
        "checks": names,
    }
