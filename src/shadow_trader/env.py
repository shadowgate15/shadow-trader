"""The Gymnasium environment the Agent trades in, for training and evaluation.

Timing. At Bar ``t`` the Agent sees the Observation built from Bars up to and
including ``t`` and picks a target Position. That Position is held into Bar
``t+1``: the Reward is ``position * (Mid[t+1] - Mid[t])`` minus the Spread cost
of the change just made, both in pips (unscaled, commensurate with the
Spread cost), so an Action cannot earn the move that prompted it.
Everything is marked to Mid; no fill is simulated at bid or ask.

Warm-up. N log-return vectors need N+1 Bars, so the first Action of a Segment
falls on its Bar N+1 (index ``first + N``) and no Reward accrues before it.

Observation, float32, ``3 + 3N + F*N`` values (N = window size, F = Feature
count): ``[Position, Spread cost, Time in position, high log returns (N), low
log returns (N), close log returns (N), then F Feature blocks of N]``. The Spread
cost is the decision Bar's effective cost in plain pips (``max(close Spread,
floor)``): no clipping, transform or scale, and no Window of past costs. Time in
position is the plain count of Bars since the last Position change as of the
decision Bar (flat included; a direct reversal resets it; 0 at ``reset``):
uncapped and unscaled, so it can far exceed an Episode's 288 Bars. Every block
runs oldest Bar to newest. Feature ``k`` at Window position ``j`` is that
Feature evaluated at Bar ``t - N + 1 + j``, so raw and Feature blocks line up
Bar for Bar. Feature blocks follow ``FAMILIES`` order, ascending horizon within
a family.

Features. Each is derived only from Mid up to and including its Bar, precomputed
once over the whole History's Mid (global Bar index) and sliced per step. They
look back by Bar count across Gaps and Segment boundaries; the raw Window does
not. A Segment whose oldest Window Bar lacks the longest Feature lookback
behind it is not usable and is dropped from ``segments``; its Bars still serve
as lookback for later Segments.

Adding a Feature family: add a horizon tuple to ``config``, a precompute
function here, and one ``FeatureFamily`` entry in ``FAMILIES``.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shadow_trader import config

RETURN_SCALE = 1e4  # brings log returns onto a network-friendly scale; no clipping
PIP = 1e-4
POSITIONS = (-1.0, 0.0, 1.0)  # Action 0 = short, 1 = flat, 2 = long

# Multi-horizon return Features: RETURN_SCALE / sqrt(h) lands every horizon at
# comparable variance. Fixed constants, no clipping, no running statistics.
RETURN_FEATURE_SCALES = tuple(
    RETURN_SCALE / math.sqrt(h) for h in config.RETURN_HORIZONS
)

SMA_FEATURE_SCALES = tuple(RETURN_SCALE / math.sqrt(n) for n in config.SMA_SPANS)

# Volatility Features: RETURN_SCALE * stdev only, no horizon adjustment.
VOLATILITY_FEATURE_SCALES = tuple(RETURN_SCALE for _ in config.VOLATILITY_HORIZONS)

# Range position is raw 0 to 1: no scaling, no recentring.
RANGE_FEATURE_SCALES = (1.0,) * len(config.RANGE_HORIZONS)


def _return_series(mid: np.ndarray, horizon: int, scale: float) -> np.ndarray:
    """``scale * log(c_t / c_{t-h})`` per Bar; NaN where fewer than h Bars precede."""
    close = mid[:, 3]
    out = np.full(len(close), np.nan)
    if len(close) > horizon:
        out[horizon:] = scale * np.log(close[horizon:] / close[:-horizon])
    return out


def _sma_series(mid: np.ndarray, span: int, scale: float) -> np.ndarray:
    """``scale * log(c_t / SMA_n)`` per Bar; NaN where fewer than n-1 Bars precede."""
    close = mid[:, 3]
    out = np.full(len(close), np.nan)
    if len(close) >= span:
        csum = np.concatenate([[0.0], np.cumsum(close)])
        sma = (csum[span:] - csum[:-span]) / span
        out[span - 1 :] = scale * np.log(close[span - 1 :] / sma)
    return out


def _volatility_series(mid: np.ndarray, horizon: int, scale: float) -> np.ndarray:
    """``scale * stdev`` (ddof=1) of the ``horizon`` one-Bar close log returns
    ending at each Bar; NaN where fewer than ``horizon + 1`` closes are available.

    A one-Bar return across a Gap includes the Gap's jump; nothing neutralises it.
    """
    close = mid[:, 3]
    out = np.full(len(close), np.nan)
    if len(close) > horizon:
        windows = np.lib.stride_tricks.sliding_window_view(
            np.diff(np.log(close)), horizon
        )
        # Chunked so the temporaries stay small on a long History.
        for i in range(0, len(windows), 8192):
            out[horizon + i : horizon + i + 8192] = scale * np.std(
                windows[i : i + 8192], axis=1, ddof=1
            )
    return out


def _range_series(mid: np.ndarray, horizon: int, scale: float) -> np.ndarray:
    """``(c_t - min l) / (max h - min l)`` over the ``horizon`` Bars to ``t``.

    A flat range reads exactly 0.5. NaN where fewer than ``horizon`` Bars exist.
    """
    n = len(mid)
    out = np.full(n, np.nan)
    if n >= horizon:
        windows = np.lib.stride_tricks.sliding_window_view
        high = windows(mid[:, 1], horizon).max(axis=1)
        low = windows(mid[:, 2], horizon).min(axis=1)
        span = high - low
        flat = span == 0
        pos = (mid[horizon - 1 :, 3] - low) / np.where(flat, 1.0, span)
        out[horizon - 1 :] = scale * np.where(flat, 0.5, pos)
    return out


@dataclass(frozen=True)
class FeatureFamily:
    """One family of Feature series, one per horizon, in ascending horizon order.

    ``reach(h)``: how many Bars before Bar ``t`` a series of horizon ``h`` reads.
    ``precompute(mid, h, scale)``: the whole-History series, NaN where lacking
    history. ``scales`` has one entry per horizon.
    """

    horizons: tuple[int, ...]
    scales: tuple[float, ...]
    reach: Callable[[int], int]
    precompute: Callable[[np.ndarray, int, float], np.ndarray]


FAMILIES: tuple[FeatureFamily, ...] = (
    FeatureFamily(
        config.RETURN_HORIZONS, RETURN_FEATURE_SCALES, lambda h: h, _return_series
    ),
    FeatureFamily(config.SMA_SPANS, SMA_FEATURE_SCALES, lambda n: n - 1, _sma_series),
    FeatureFamily(
        config.VOLATILITY_HORIZONS,
        VOLATILITY_FEATURE_SCALES,
        lambda n: n,
        _volatility_series,
    ),
    FeatureFamily(
        config.RANGE_HORIZONS, RANGE_FEATURE_SCALES, lambda h: h - 1, _range_series
    ),
)
FEATURE_COUNT = sum(len(f.horizons) for f in FAMILIES)
LOOKBACK = max(f.reach(h) for f in FAMILIES for h in f.horizons)  # Bars


class TradingEnv(gym.Env):
    """Trade one fixed-size Position over Mid OHLC Bars.

    ``mid`` is an ``(n, 4)`` array of Mid open, high, low, close, indexed by
    Bar. ``segments`` are half-open ``[first, stop)`` Bar-index ranges for one
    Period, as ``(first, stop)`` pairs or objects with ``.first``/``.stop``
    (``segments.Segment``). Segments should come from ``segments_within`` so
    they are clipped to the Period.

    ``reset()`` starts a random Episode of ``episode_length`` rewarded Bars
    inside a single Segment. ``reset(options={"segment": i})`` instead runs
    the i-th Segment from its start through its last Bar, starting flat, for
    deterministic evaluation. ``info["bar"]`` is the Bar the next Action falls on;
    after ``step``, ``info["cost"]`` is the Spread cost in pips just charged.

    Spread cost. ``close_spread_pips`` is the close Spread (ask close minus bid
    close, in pips) of every Bar, or None for none. A Position change decided at
    Bar ``t`` pays half of ``cost_pips[t] = max(close Spread at t, spread_pips)``
    per unit changed, so ``spread_pips`` is the floor and Bar ``t+1``'s Spread is
    never used.
    """

    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        mid: np.ndarray,
        segments: Sequence[Any],
        *,
        window_size: int = config.WINDOW_SIZE,
        episode_length: int = config.EPISODE_LENGTH,
        spread_pips: float = config.SPREAD_PIPS,
        close_spread_pips: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.mid = np.asarray(mid, dtype=np.float64)
        self.window_size = window_size
        self.segments = [r for r in map(_range, segments) if self._usable(r[0])]
        self.episode_length = episode_length
        self.spread_pips = spread_pips  # the floor on the per-Bar Spread cost
        # Spread cost in pips of a Position change decided at each Bar:
        # max(close Spread, floor). Zero, crossed and narrow Spreads all floor.
        spread = (
            np.zeros(len(self.mid))
            if close_spread_pips is None
            else np.asarray(close_spread_pips, dtype=np.float64)
        )
        if spread.shape != (len(self.mid),):
            raise ValueError("close_spread_pips needs one value per Bar")
        self.cost_pips = np.maximum(spread, spread_pips)
        n = window_size
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (3 + (3 + FEATURE_COUNT) * n,), np.float32
        )
        self.action_space = spaces.Discrete(3)

        # High, Low, Close as log returns against the PREVIOUS Bar's Mid Close.
        # One shared base, so the vector reads as where this Bar went relative
        # to where the last one closed. Row 0 has no previous Bar; never used.
        base = self.mid[:-1, 3:4]
        returns = np.zeros((len(self.mid), 3))
        returns[1:] = RETURN_SCALE * np.log(self.mid[1:, 1:4] / base)
        self._returns = returns
        self._features = np.column_stack(
            [
                family.precompute(self.mid, h, scale)
                for family in FAMILIES
                for h, scale in zip(family.horizons, family.scales, strict=True)
            ]
        )

        self._bar = 0
        self._last = 0  # last Bar of the current run
        self._position = 0.0
        self._time_in_position = 0  # Bars since the last Position change

    def _usable(self, first: int) -> bool:
        # The oldest Window Bar is first + 1; the Feature needing the most
        # history there reaches LOOKBACK Bars back (and the raw return one).
        return first + 1 - max(LOOKBACK, 1) >= 0

    def _observation(self) -> np.ndarray:
        # Deliberately NOT in the vector:
        # - No Open channel. Open == previous Close holds bit-exact on this
        #   data (0 of 135,770 contiguous Bars disagree), so a log return of
        #   Open against the previous Close is identically 0.0. It would feed
        #   the network a constant and waste a quarter of the width.
        # - No clock or calendar feature (hour, day of week, session marker).
        #   The MVP asks whether price movement alone carries signal; a clock
        #   answers a different question. Features are Mid-derived price
        #   context only; no clock, volume or other-pair Feature either.
        # - No raw Spread and no Window of past Spread costs. The one Spread
        #   quantity in the vector is the decision Bar's floored Spread cost
        #   (slot 1), so indicative-quote zeros and crossings never reach the
        #   Agent.
        # - Time in position (slot 2) is the one Position-history quantity: raw
        #   Bar count, deliberately uncapped and unscaled.
        # Layout: Position, Spread cost, Time in position, the high, low, close blocks
        # (channel-major), then the Feature blocks in FAMILIES order.
        t, n = self._bar, self.window_size
        window = self._returns[t - n + 1 : t + 1]  # (n, 3): high, low, close
        features = self._features[t - n + 1 : t + 1]  # (n, F)
        return np.concatenate(
            [
                [self._position, self.cost_pips[t], self._time_in_position],
                window.T.ravel(),
                features.T.ravel(),
            ]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        n = self.window_size
        if options and "segment" in options:
            first, stop = self.segments[options["segment"]]
            if stop - first < n + 2:
                raise ValueError("segment too short for warm-up plus one step")
            self._bar, self._last = first + n, stop - 1
        else:
            need = n + self.episode_length + 1  # warm-up, steps, final move
            fits = [(f, s - need - f + 1) for f, s in self.segments if s - f >= need]
            if not fits:
                raise ValueError(f"no Segment holds {need} Bars for an Episode")
            counts = np.array([c for _, c in fits])
            pick = int(self.np_random.integers(counts.sum()))
            offsets = np.cumsum(counts) - counts  # index of each Segment's first start
            i = int(np.searchsorted(offsets, pick, side="right")) - 1
            start = fits[i][0] + pick - int(offsets[i])
            self._bar = start + n
            self._last = self._bar + self.episode_length
        self._position = 0.0
        self._time_in_position = 0
        return self._observation(), {"bar": self._bar}

    def flatten_cost(self) -> float:
        """Pips to close the current Position at the current Bar's Spread cost."""
        return float(self.cost_pips[self._bar] / 2 * abs(self._position))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = POSITIONS[int(action)]
        t = self._bar
        cost = self.cost_pips[t] / 2 * abs(target - self._position)
        move = (self.mid[t + 1, 3] - self.mid[t, 3]) / PIP
        reward = target * move - float(cost)
        self._time_in_position = (
            self._time_in_position + 1 if target == self._position else 0
        )
        self._position = target
        self._bar = t + 1
        truncated = self._bar >= self._last
        info = {"bar": self._bar, "cost": float(cost)}
        return self._observation(), float(reward), False, truncated, info


def _range(segment: Any) -> tuple[int, int]:
    if hasattr(segment, "first"):
        return int(segment.first), int(segment.stop)
    first, stop = segment
    return int(first), int(stop)
