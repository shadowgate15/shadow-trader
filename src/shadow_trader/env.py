"""The Gymnasium environment the Agent trades in, for training and evaluation.

Timing. At Bar ``t`` the Agent sees the Observation built from Bars up to and
including ``t`` and picks a target Position. That Position is held into Bar
``t+1``: the Reward is ``position * (Mid[t+1] - Mid[t])`` minus the Spread cost
of the change just made, so an Action cannot earn the move that prompted it.
Everything is marked to Mid; no fill is simulated at bid or ask.

Warm-up. N log-return vectors need N+1 Bars, so the first Action of a Segment
falls on its Bar N+1 (index ``first + N``) and no Reward accrues before it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WINDOW_SIZE = 24
EPISODE_LENGTH = 288
SPREAD_PIPS = 1.0
RETURN_SCALE = 1e4  # brings log returns onto a network-friendly scale; no clipping
PIP = 1e-4
POSITIONS = (-1.0, 0.0, 1.0)  # Action 0 = short, 1 = flat, 2 = long


class TradingEnv(gym.Env):
    """Trade one fixed-size Position over Mid OHLC Bars.

    ``prices`` is an ``(n, 4)`` array of Mid open, high, low, close, indexed by
    Bar. ``segments`` are half-open ``[first, stop)`` Bar-index ranges for one
    Period, as ``(first, stop)`` pairs or objects with ``.first``/``.stop``
    (``segments.Segment``). Segments should come from ``segments_within`` so
    they are clipped to the Period.

    ``reset()`` starts a random Episode of ``episode_length`` rewarded Bars
    inside a single Segment. ``reset(options={"segment": i})`` instead runs
    the i-th Segment from its start through its last Bar, starting flat, for
    deterministic evaluation. ``info["bar"]`` is the Bar the next Action falls on.
    """

    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        prices: np.ndarray,
        segments: Sequence[Any],
        *,
        window_size: int = WINDOW_SIZE,
        episode_length: int = EPISODE_LENGTH,
        spread_pips: float = SPREAD_PIPS,
    ) -> None:
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float64)
        self.segments = [_range(s) for s in segments]
        self.window_size = window_size
        self.episode_length = episode_length
        self.spread = spread_pips * PIP
        n = window_size
        self.observation_space = spaces.Box(-np.inf, np.inf, (3 * n + 1,), np.float32)
        self.action_space = spaces.Discrete(3)

        # High, Low, Close as log returns against the PREVIOUS Bar's Mid Close.
        # One shared base, so the vector reads as where this Bar went relative
        # to where the last one closed. Row 0 has no previous Bar; never used.
        base = self.prices[:-1, 3:4]
        returns = np.zeros((len(self.prices), 3))
        returns[1:] = RETURN_SCALE * np.log(self.prices[1:, 1:4] / base)
        self._returns = returns

        self._bar = 0
        self._last = 0  # last Bar of the current run
        self._position = 0.0

    def _observation(self) -> np.ndarray:
        # Deliberately NOT in the vector:
        # - No Open channel. Open == previous Close holds bit-exact on this
        #   data (0 of 135,770 contiguous Bars disagree), so a log return of
        #   Open against the previous Close is identically 0.0. It would feed
        #   the network a constant and waste a quarter of the width.
        # - No clock or calendar feature (hour, day of week, session marker).
        #   The MVP asks whether price movement alone carries signal; a clock
        #   answers a different question.
        t, n = self._bar, self.window_size
        window = self._returns[t - n + 1 : t + 1]  # (n, 3): high, low, close
        return np.concatenate([window.T.ravel(), [self._position]]).astype(np.float32)

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
        return self._observation(), {"bar": self._bar}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = POSITIONS[int(action)]
        cost = self.spread / 2 * abs(target - self._position)
        t = self._bar
        reward = target * (self.prices[t + 1, 3] - self.prices[t, 3]) - cost
        self._position = target
        self._bar = t + 1
        truncated = self._bar >= self._last
        return self._observation(), float(reward), False, truncated, {"bar": self._bar}


def _range(segment: Any) -> tuple[int, int]:
    if hasattr(segment, "first"):
        return int(segment.first), int(segment.stop)
    first, stop = segment
    return int(first), int(stop)
