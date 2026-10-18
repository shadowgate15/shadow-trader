"""A prepared Dataset converted once to contiguous numpy arrays (Gymnasium-free).

``MarketData`` is what the stepping loop reads: scaled features, ``usable``, the
price columns and the calendar facts (``fillable``, Session-final, rollover days,
Bars to Session final) precomputed per Bar, plus Session boundaries. The stepping
loop never touches Polars. Rows are one Instrument ordered by ``open_time``;
Session ids are contiguous runs because the weekly closure has no rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import polars as pl

from shadow_trader.preparation.dataset import PreparedDataset
from shadow_trader.preparation.scaler import Scaler, ScalerError

from .config import ConfigError
from .engine import Market
from .observation import ObservationConfig, window_inputs
from .session_calendar import market_from_row

__all__ = ["MarketData", "eligible_starts", "from_prepared"]

log = logging.getLogger(__name__)

_ROW_COLUMNS = (
    "open_time",
    "open",
    "close",
    "spread_close",
    "spread_mean",
    "is_gap",
    "quality_flags",
)


@dataclass(frozen=True)
class MarketData:
    instrument: str
    bar_minutes: int
    open_time: tuple[datetime, ...]
    features: npt.NDArray[np.float64]  # [n, F], scaled; NaN where null
    usable: npt.NDArray[np.bool_]
    open_bid: npt.NDArray[np.float64]
    open_spread: npt.NDArray[np.float64]
    close_bid: npt.NDArray[np.float64]
    close_spread: npt.NDArray[np.float64]
    fillable: npt.NDArray[np.bool_]
    is_session_final: npt.NDArray[np.bool_]
    rollover_days: npt.NDArray[np.int64]
    bars_to_session_final: npt.NDArray[np.int64]
    session_index: npt.NDArray[np.int64]  # [n] ordinal of the Bar's Session
    session_start: npt.NDArray[np.int64]  # [S] first row of each Session
    session_end: npt.NDArray[np.int64]  # [S] one past the last row
    session_open: tuple[datetime, ...]  # [S] the Session ids (UTC open)

    def __len__(self) -> int:
        return len(self.open_time)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def market(self, row: int) -> Market:
        return Market(
            open_bid=float(self.open_bid[row]),
            open_spread=float(self.open_spread[row]),
            close_bid=float(self.close_bid[row]),
            close_spread=float(self.close_spread[row]),
            fillable=bool(self.fillable[row]),
            is_session_final=bool(self.is_session_final[row]),
            rollover_days=int(self.rollover_days[row]),
            bars_to_session_final=int(self.bars_to_session_final[row]),
        )

    def window(
        self, row: int, bars: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """``[1, bars, F]`` features and ``[1, bars]`` in-Session mask ending at
        ``row``. Rows before the Session start are zero and not valid."""
        start = int(self.session_start[self.session_index[row]])
        rows = self.features[max(start, row - bars + 1) : row + 1]
        return window_inputs(rows[None], ObservationConfig(window_bars=bars))


def from_prepared(
    dataset: PreparedDataset, scaler: Scaler, instrument: str, bar_minutes: int
) -> MarketData:
    """Read ``dataset`` once, scale it once and convert it to arrays."""
    if int(dataset.manifest["bar_size"]) != bar_minutes:
        raise ConfigError(
            f"Dataset Bars are {dataset.manifest['bar_size']} minutes but the "
            f"config says bar_minutes={bar_minutes}"
        )
    declared = dataset.manifest.get("scaler", {}).get("id")
    if declared is not None and declared != scaler.scaler_id:
        raise ScalerError(
            f"Dataset {dataset.dataset_id} names scaler {declared}, got "
            f"{scaler.scaler_id}"
        )
    frame = dataset.read(instrument)
    scaled = scaler.apply(frame, spec_hash=dataset.manifest["feature_spec"]["hash"])
    names = [c for c in scaled.columns if c.startswith("feat_")]
    features = (
        scaled.select(names)
        .with_columns(pl.all().fill_null(float("nan")))
        .to_numpy()
        .astype(np.float64)
    )
    markets = [
        market_from_row(row, bar_minutes)
        for row in scaled.select(_ROW_COLUMNS).iter_rows(named=True)
    ]

    def column(values: list[float] | list[bool] | list[int], dtype: type):
        return np.array(values, dtype=dtype)

    session_ids = scaled["session_id"].to_list()
    boundaries = [0] + [
        i for i in range(1, len(session_ids)) if session_ids[i] != session_ids[i - 1]
    ]
    starts = np.array(boundaries, dtype=np.int64)
    ends = np.append(starts[1:], len(session_ids)).astype(np.int64)
    index = np.repeat(np.arange(len(starts)), ends - starts).astype(np.int64)
    return MarketData(
        instrument=instrument,
        bar_minutes=bar_minutes,
        open_time=tuple(scaled["open_time"].to_list()),
        features=features,
        usable=scaled["usable"].to_numpy().astype(np.bool_),
        open_bid=column([m.open_bid for m in markets], np.float64),
        open_spread=column([m.open_spread for m in markets], np.float64),
        close_bid=column([m.close_bid for m in markets], np.float64),
        close_spread=column([m.close_spread for m in markets], np.float64),
        fillable=column([m.fillable for m in markets], np.bool_),
        is_session_final=column([m.is_session_final for m in markets], np.bool_),
        rollover_days=column([m.rollover_days for m in markets], np.int64),
        bars_to_session_final=column(
            [m.bars_to_session_final for m in markets], np.int64
        ),
        session_index=index,
        session_start=starts,
        session_end=ends,
        session_open=tuple(session_ids[i] for i in boundaries),
    )


def eligible_starts(data: MarketData, window_bars: int) -> npt.NDArray[np.int64]:
    """Rows a training window of ``window_bars`` stepped Bars may start from, in
    ascending order.

    The start Bar and every Bar stepped after it must be usable and not the
    Session's final Bar, and the last stepped Bar must lie in the start's Session.
    (Gap stepping, which lifts the all-usable requirement, is a later ticket.)
    Sessions with no usable Bar, or none that fit a window, are logged.
    """
    good = data.usable & ~data.is_session_final
    csum = np.concatenate([[0], np.cumsum(good)])
    n = len(data)
    rows = np.arange(n)
    last = rows + window_bars
    in_range = last < n
    clipped = np.minimum(last, n - 1)
    all_good = in_range & (csum[clipped + 1] - csum[rows] == window_bars + 1)
    same_session = in_range & (data.session_end[data.session_index] > last)
    starts = rows[all_good & same_session].astype(np.int64)
    covered = set(data.session_index[starts].tolist())
    for k, opened in enumerate(data.session_open):
        lo, hi = data.session_start[k], data.session_end[k]
        if not data.usable[lo:hi].any():
            log.warning("skipping Session %s: it has no usable Bars", opened)
        elif k not in covered:
            log.warning(
                "skipping Session %s: no window of %d Bars fits", opened, window_bars
            )
    return starts
