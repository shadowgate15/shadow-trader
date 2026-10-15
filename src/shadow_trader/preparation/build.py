"""The preparation build: read one parent ingestion Dataset, add features, split, write.

Reads the parent only through ingestion's public ``open_dataset``. Build order:
open parent, check bar size and Instrument, compute features, split + Embargo,
``boundary_check``, then write the three Datasets and the build record. Every
check runs before the first write, so a failed build leaves nothing behind.

The scaler (fit on train, written before the Datasets) is not built yet: its slot
sits between ``boundary_check`` and the writes, and ``boundary_check`` accepts
extra checks, which is where the scaler's fit-range check plugs in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from shadow_trader.ingestion.dataset import (
    BAR_SCHEMA,
    DatasetError,
    open_dataset,
)
from shadow_trader.ingestion.progress import NULL_PROGRESS, ProgressSink

from .anomaly import apply_anomaly_policy
from .boundary import BoundaryCheckFailure, BoundaryContext, Check, boundary_check
from .config import PreparedConfig
from .dataset import PreparedRef, build_id_of, plan_prepared, write_build_record
from .spec import compute_features
from .split import (
    ROLES,
    SplitFailure,
    dataset_identity,
    embargo_bars,
    snap_bounds,
    split_frame,
)

# Checks run after the built-in boundary checks. STUB: the scaler ticket adds its
# "scaler fit range inside train" check here.
EXTRA_CHECKS: tuple[tuple[str, Check], ...] = ()


class BuildFailure(Exception):
    """The build cannot proceed (CLI exit code 1); nothing is written.

    ``code`` is the closed failure code (``boundary_check.*``,
    ``embargo_exceeds_dataset``, ``no_usable_bars``) when there is one.
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BuildRef:
    """The result of one build: its id, record and the three Datasets by role."""

    build_id: str
    path: Path
    record: Mapping[str, Any]
    datasets: Mapping[str, PreparedRef]


def _bars_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    columns = {name: [row[name] for row in rows] for name in BAR_SCHEMA.names}
    table = pa.table(columns, schema=BAR_SCHEMA)
    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)
    return frame


def build_prepared(
    config: PreparedConfig, progress: ProgressSink = NULL_PROGRESS
) -> BuildRef:
    """Build the train, validation and test Datasets and return the build's ref."""
    spec = config.feature_spec
    parent_id = config.input.dataset_id
    try:
        parent = open_dataset(config.paths.data_root, parent_id)
    except (DatasetError, OSError) as exc:
        raise BuildFailure(f"cannot open parent Dataset {parent_id}: {exc}") from exc

    bar_size = parent.manifest["bar_size"]
    if bar_size != spec.bar_size:
        raise BuildFailure(
            f"bar size mismatch: parent Dataset {parent_id} has {bar_size}-minute "
            f"Bars but feature spec v{spec.version} is bound to "
            f"{spec.bar_size}-minute Bars"
        )
    instruments = list(parent.manifest["instruments"])
    if len(instruments) != 1:
        raise BuildFailure(
            f"parent Dataset {parent_id} has {len(instruments)} Instruments; "
            "this build prepares exactly one"
        )

    progress.start("features", len(instruments))
    frames: dict[str, pl.DataFrame] = {}
    for instrument in instruments:
        bars = parent.read(instrument)
        rows = [{**vars(bar), "quality_flags": list(bar.quality_flags)} for bar in bars]
        featured = compute_features(_bars_frame(rows), spec).collect()
        frames[instrument] = apply_anomaly_policy(featured, spec, config.anomaly)
        progress.advance(instrument)
    progress.done()

    (instrument,) = instruments
    featured = frames[instrument]
    embargo = embargo_bars(spec, config.embargo)
    try:
        bounds = snap_bounds(
            config.split, sorted(featured["session_id"].unique().to_list())
        )
        roles = split_frame(featured, bounds, embargo)
        # Scaler slot (not built yet): fit on roles["train"] here and add its
        # fit-range check to EXTRA_CHECKS.
        checked = boundary_check(
            BoundaryContext(bounds, roles, featured["open_time"], embargo, spec),
            extra=EXTRA_CHECKS,
        )
    except (SplitFailure, BoundaryCheckFailure) as exc:
        raise BuildFailure(f"{exc.code}: {exc.detail}", exc.code) from exc

    plans = {
        role: plan_prepared(
            {instrument: roles[role].frame},
            parent_ingestion=parent_id,
            spec=spec,
            policy=config.anomaly,
            identity=dataset_identity(config.split, embargo, role),
        )
        for role in ROLES
    }
    ids = {role: plan.dataset_id for role, plan in plans.items()}
    split_block = {"scheme": config.split.scheme, **bounds.block()}
    shared: dict[str, Any] = {
        "parent_ingestion": parent_id,
        "spec_hash": spec.hash,
        "policy_hash": config.anomaly.hash,
        "split": split_block,
        "embargo_bars": embargo,
        "boundary_check": checked,
    }
    build_id = build_id_of(shared, ids)

    progress.start("write", len(ROLES))
    datasets: dict[str, PreparedRef] = {}
    for role in ROLES:
        datasets[role] = plans[role].write(
            config.paths.prepared_root,
            {
                "role": role,
                "build_id": build_id,
                "siblings": {r: i for r, i in ids.items() if r != role},
                "split": split_block,
                "embargo": roles[role].embargo_block(
                    embargo, spec, config.embargo.episode_bars
                ),
                "boundary_check": checked,
            },
        )
        progress.advance(role)
    record = write_build_record(config.paths.prepared_root, shared, ids)
    progress.done()
    return BuildRef(record.build_id, record.path, record.record, datasets)
