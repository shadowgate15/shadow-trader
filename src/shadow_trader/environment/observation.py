"""Pure observation assembly shared by training and live (Gymnasium-free).

``build_observation`` turns an already scaled feature window, the engine state and
the current ``Market`` into a structured ``Observation``: the features and their
validity mask, a position block (``position_frac``, ``pending_target``,
``unrealized_return``) per Instrument and an account block (``used_margin_frac``,
``session_progress``). Everything is float32 and finite; absolute equity is never
exposed, so a policy cannot memorise account size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from shadow_trader.ingestion.session import slots_per_session

from .config import ConfigError, EngineConfig
from .engine import EngineState, Market, level_to_units

__all__ = ["Observation", "ObservationConfig", "build_observation"]

POSITION_FIELDS = ("position_frac", "pending_target", "unrealized_return")
ACCOUNT_FIELDS = ("used_margin_frac", "session_progress")


@dataclass(frozen=True)
class ObservationConfig:
    """``window_bars`` is W, ``position_clip`` the safety bound on
    ``position_frac`` and ``bar_minutes`` sets the nominal Session length."""

    window_bars: int = 1
    position_clip: float = 5.0
    bar_minutes: int = 5

    def __post_init__(self) -> None:
        if self.window_bars < 1:
            raise ConfigError("window_bars must be at least 1")
        if not 0 < self.position_clip < math.inf:
            raise ConfigError("position_clip must be positive and finite")
        if self.bar_minutes < 1:
            raise ConfigError("bar_minutes must be at least 1")


@dataclass(frozen=True, slots=True)
class Observation:
    """``features`` [I, W, F] oldest to newest, ``valid`` [I, W], ``position``
    [I, 3] in ``POSITION_FIELDS`` order and ``account`` [2] in ``ACCOUNT_FIELDS``
    order."""

    features: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    position: npt.NDArray[np.float32]
    account: npt.NDArray[np.float32]


def build_observation(
    scaled_window: npt.ArrayLike,
    valid: npt.ArrayLike,
    engine_state: EngineState,
    market: Market | Sequence[Market],
    config: EngineConfig,
    observation_config: ObservationConfig = ObservationConfig(),  # noqa: B008
) -> Observation:
    """Assemble the observation for the Bar just closed.

    ``scaled_window`` is [I, W, F] already scaled by the shared Scaler step; an
    older row with a non-finite value is zeroed entirely and marked invalid, while a
    non-finite value in the current (last) row raises ``ValueError``. ``market``
    is the Bar being observed; its close prices mark the position.
    """
    markets = (market,) if isinstance(market, Market) else tuple(market)
    n = len(config.instruments)
    window = np.asarray(scaled_window, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    w = observation_config.window_bars
    if window.ndim != 3 or window.shape[:2] != (n, w) or mask.shape != (n, w):
        raise ValueError(
            f"window must be [{n}, {w}, F] with valid [{n}, {w}]; got "
            f"{window.shape} and {mask.shape}"
        )
    if len(markets) != n or len(engine_state.positions) != n:
        raise ValueError(f"expected {n} market(s) for {config.symbols}")

    finite = np.isfinite(window)
    if not finite[:, -1, :].all():
        raise ValueError("null or non-finite value in the current row")
    row_ok = finite.all(axis=2)
    features = np.where(row_ok[:, :, None], window, 0.0).astype(np.float32)
    out_valid = mask & row_ok

    entries = engine_state.entry_prices or (0.0,) * n
    position = np.zeros((n, len(POSITION_FIELDS)), dtype=np.float64)
    used_margin = 0.0
    for i, spec in enumerate(config.instruments):
        bar = markets[i]
        units = engine_state.positions[i]
        mark = bar.close_bid + bar.close_spread if units < 0 else bar.close_bid
        max_units = abs(
            level_to_units(
                1.0, engine_state.equity, mark, config.effective_leverage_cap
            )
        )
        clip = observation_config.position_clip
        position[i, 0] = max(-clip, min(clip, units / max_units))
        position[i, 1] = engine_state.target_levels[i]
        if units != 0 and entries[i] > 0:
            direction = 1.0 if units > 0 else -1.0
            position[i, 2] = direction * (mark - entries[i]) / entries[i]
        used_margin += abs(units) * mark * spec.margin_rate

    nominal = slots_per_session(observation_config.bar_minutes) - 1
    progress = max(0.0, min(1.0, markets[0].bars_to_session_final / nominal))
    account = np.array([used_margin / engine_state.equity, progress])

    result = Observation(
        features=features,
        valid=out_valid,
        position=position.astype(np.float32),
        account=account.astype(np.float32),
    )
    for name in ("position", "account"):
        if not np.isfinite(getattr(result, name)).all():
            raise ValueError(f"non-finite {name} block (unpriced or bust account)")
    return result
