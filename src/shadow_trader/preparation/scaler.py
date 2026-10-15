"""The scaler: a train-only fit, an immutable hash-addressed artifact and ``apply``.

Robust scaling ``(x - median) / IQR`` (IQR = p75 - p25) then a clip at +-5, fit on
the train Dataset only, per Instrument and column over non-null values. Columns
whose spec marks them ``identity`` (time/session sin/cos, ``is_rollover``) are
left alone. The fit is static: there is no refit. Prepared Datasets keep
unscaled ``feat_*``; scaled values are only ever produced by ``Scaler.apply``.

The artifact is JSON stored at ``<prepared_root>/scalers/<scaler_id>.json``;
``scaler_id`` is the SHA-256 of its canonical content, so it is immutable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, overload

import polars as pl

from .boundary import BoundaryContext, Check
from .spec import FeatureSpec

SCALER_VERSION = 1
MIN_N_FIT = 30  # fewest non-null train values a robust_clip column may be fit on
CLIP = 5.0
ROBUST = "robust_clip"
IDENTITY = "identity"


class ScalerError(Exception):
    """A scaler artifact is missing, corrupt or does not fit the frame."""


class ScalerFitError(ScalerError):
    """The fit is degenerate (CLI exit code 1); ``code`` is the closed failure code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class Scaler:
    """A loaded scaler artifact."""

    artifact: Mapping[str, Any]

    @property
    def scaler_id(self) -> str:
        return hashlib.sha256(_canonical(self.artifact)).hexdigest()

    def _columns(self) -> set[str]:
        return {
            name
            for entry in self.artifact["instruments"].values()
            for name in entry["columns"]
        }

    def _expression(self, name: str) -> pl.Expr | None:
        branches = {
            pair: entry["columns"][name]
            for pair, entry in self.artifact["instruments"].items()
            if entry["columns"][name]["method"] == ROBUST
        }
        if not branches:
            return None
        value = pl.col(name)
        chain: Any = pl
        for pair, stats in branches.items():
            scaled = ((value - stats["median"]) / stats["iqr"]).clip(-CLIP, CLIP)
            chain = chain.when(pl.col("instrument") == pair).then(scaled)
        return chain.otherwise(None).alias(name)

    @overload
    def apply(
        self, frame: pl.DataFrame, *, spec_hash: str | None = ...
    ) -> pl.DataFrame: ...
    @overload
    def apply(
        self, frame: pl.LazyFrame, *, spec_hash: str | None = ...
    ) -> pl.LazyFrame: ...
    def apply(
        self, frame: pl.DataFrame | pl.LazyFrame, *, spec_hash: str | None = None
    ) -> pl.DataFrame | pl.LazyFrame:
        """Scale ``feat_*`` columns; same kind, same names, nulls preserved.

        ``spec_hash`` is the feature spec hash of the frame's Dataset (from its
        manifest); a mismatch with the artifact is refused. The frame's ``feat_*``
        column set must always equal the artifact's.
        """
        if spec_hash is not None and spec_hash != self.artifact["spec_hash"]:
            raise ScalerError(
                f"spec hash mismatch: scaler was fit on spec "
                f"{self.artifact['spec_hash']}, the frame is spec {spec_hash}"
            )
        names = (
            frame.columns
            if isinstance(frame, pl.DataFrame)
            else frame.collect_schema().names()
        )
        present = {n for n in names if n.startswith("feat_")}
        if present != self._columns():
            raise ScalerError(
                "feature columns do not match the scaler: "
                f"missing {sorted(self._columns() - present)}, "
                f"unexpected {sorted(present - self._columns())}"
            )
        expressions = [
            e for n in sorted(present) if (e := self._expression(n)) is not None
        ]
        return frame.with_columns(*expressions)


def fit_scaler(
    train: pl.DataFrame,
    spec: FeatureSpec,
    *,
    policy_hash: str,
    fit_dataset: str,
) -> Scaler:
    """Fit on the train frame (all Instruments, with an ``instrument`` column)."""
    if train.height == 0:
        raise ScalerFitError("scaler_insufficient_fit", "the train Dataset is empty")
    instruments: dict[str, Any] = {}
    for (pair,), part in train.group_by("instrument", maintain_order=True):
        columns: dict[str, Any] = {}
        for column in spec.columns:
            if column.scaling == IDENTITY:
                columns[column.name] = {"method": IDENTITY}
                continue
            values = part[column.name].drop_nulls()
            n_fit = len(values)
            if n_fit < MIN_N_FIT:
                raise ScalerFitError(
                    "scaler_insufficient_fit",
                    f"{pair} {column.name} has {n_fit} non-null train values; "
                    f"at least {MIN_N_FIT} are required",
                )
            median, p25, p75 = (
                values.to_frame("v")
                .select(
                    pl.col("v").median().alias("median"),
                    pl.col("v").quantile(0.25, "linear").alias("p25"),
                    pl.col("v").quantile(0.75, "linear").alias("p75"),
                )
                .row(0)
            )
            if p75 - p25 == 0:
                raise ScalerFitError(
                    "scaler_zero_iqr",
                    f"{pair} {column.name} has a zero IQR over its {n_fit} train "
                    "values and cannot be robust-scaled",
                )
            columns[column.name] = {
                "method": ROBUST,
                "median": median,
                "iqr": p75 - p25,
                "n_fit": n_fit,
            }
        instruments[str(pair)] = {"columns": columns}
    first, last = train.select(
        pl.col("open_time").min().alias("first"),
        pl.col("open_time").max().alias("last"),
    ).row(0)
    artifact = {
        "kind": "scaler",
        "scaler_version": SCALER_VERSION,
        "spec_version": spec.version,
        "spec_hash": spec.hash,
        "anomaly_policy_hash": policy_hash,
        "fit_dataset": fit_dataset,
        "fit_range": [first.isoformat(), last.isoformat()],
        "clip": CLIP,
        "instruments": instruments,
    }
    return Scaler(artifact)


def scaler_fit_range_check(scaler: Scaler) -> Check:
    """The ``boundary_check`` check: the fit range lies inside the train range."""

    def check(ctx: BoundaryContext) -> str | None:
        frame = ctx.roles["train"].frame
        if frame.height == 0:
            return None  # reported by dataset_length
        low, high = frame.select(
            pl.col("open_time").min().alias("low"),
            pl.col("open_time").max().alias("high"),
        ).row(0)
        start, end = (datetime.fromisoformat(v) for v in scaler.artifact["fit_range"])
        if start < low or end > high:
            return (
                f"scaler fit range {start.isoformat()}..{end.isoformat()} is not "
                f"inside the train range {low.isoformat()}..{high.isoformat()}"
            )
        return None

    return check


def scaler_path(prepared_root: Path, scaler_id: str) -> Path:
    return Path(prepared_root) / "scalers" / f"{scaler_id}.json"


def write_scaler(prepared_root: Path, scaler: Scaler) -> Path:
    """Write the immutable artifact by atomic rename; an existing one is kept."""
    path = scaler_path(prepared_root, scaler.scaler_id)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scaler.artifact, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_scaler(data_root: Path, scaler_id: str) -> Scaler:
    """Load a scaler from the prepared root; its content must hash to the id."""
    path = scaler_path(data_root, scaler_id)
    if not path.is_file():
        raise ScalerError(f"scaler {scaler_id} not found under {data_root}")
    scaler = Scaler(json.loads(path.read_text()))
    if scaler.scaler_id != scaler_id:
        raise ScalerError(
            f"scaler artifact hash mismatch: {path} hashes to {scaler.scaler_id}"
        )
    return scaler
