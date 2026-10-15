"""The preparation build: read one parent ingestion Dataset, add features, write.

Reads the parent only through ingestion's public ``open_dataset``. Build order
(this tracer bullet): open parent, check bar size and Instrument, compute
features per Instrument, write the prepared Dataset.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from shadow_trader.ingestion.dataset import (
    BAR_SCHEMA,
    DatasetError,
    open_dataset,
)
from shadow_trader.ingestion.progress import NULL_PROGRESS, ProgressSink

from .anomaly import apply_anomaly_policy
from .config import PreparedConfig
from .dataset import PreparedRef, write_prepared
from .spec import compute_features

BuildRef = PreparedRef


class BuildFailure(Exception):
    """The build cannot proceed (CLI exit code 1); nothing is written."""


def _bars_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    columns = {name: [row[name] for row in rows] for name in BAR_SCHEMA.names}
    table = pa.table(columns, schema=BAR_SCHEMA)
    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)
    return frame


def build_prepared(
    config: PreparedConfig, progress: ProgressSink = NULL_PROGRESS
) -> BuildRef:
    """Build the prepared Dataset described by ``config`` and return its ref."""
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

    progress.start("write", 1)
    ref = write_prepared(
        config.paths.prepared_root,
        frames,
        parent_ingestion=parent_id,
        spec=spec,
        policy=config.anomaly,
    )
    progress.advance(ref.dataset_id)
    progress.done()
    return ref
