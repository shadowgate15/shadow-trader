"""The Gymnasium environment layer: ``TradingEnv`` and ``make_env``.

The env adds actions, episodes, reward and ``info`` on top of the pure engine,
observation assembler and ledger. This is the skeleton: one agent step is exactly
one Bar, and a training episode is a random window of ``EpisodeConfig.window_bars``
Bars inside one Session whose Bars are all usable. A margin closeout ends the
episode as ``terminated``; window end is ``truncated``. Session-end
force-flat, Gap stepping and evaluation sweeps arrive with later tickets.

Gymnasium is imported here (and in nothing else of the package); the package
``__init__`` exposes this module lazily.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from shadow_trader.preparation.dataset import PreparedDataset
from shadow_trader.preparation.scaler import Scaler

from . import market_data
from .config import ConfigError
from .engine import EngineState, initial_state, step
from .env_config import EnvConfig
from .ledger import Ledger, LedgerBuilder
from .market_data import MarketData
from .observation import POSITION_FIELDS, Observation, build_observation
from .reward import log_return_reward

__all__ = ["TradingEnv", "make_env"]

WINDOW_END = "window_end"
CLOSEOUT = "closeout"


class TradingEnv(gym.Env):
    """Single-Instrument trading environment over a prepared Dataset."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        data: MarketData,
        config: EnvConfig,
        *,
        seed: int | None = None,
        record_ledger: bool = False,
    ) -> None:
        super().__init__()
        if len(config.engine.instruments) != 1:
            raise ConfigError("the env supports a single Instrument for now")
        self._data = data
        self._config = config
        self._starts = market_data.eligible_starts(data, config.episode.window_bars)
        if len(self._starts) == 0:
            raise ValueError(
                f"no window of {config.episode.window_bars} usable Bars fits in a "
                "Session of the Dataset"
            )
        self._record = record_ledger
        self._levels = config.engine.levels
        self._np_random, _ = seeding.np_random(seed)

        w = config.observation.window_bars
        f = data.n_features
        unbounded = np.inf
        self.action_space = spaces.Discrete(len(self._levels))
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(-unbounded, unbounded, (1, w, f), np.float32),
                "valid": spaces.MultiBinary((1, w)),
                "position": spaces.Box(
                    -unbounded, unbounded, (1, len(POSITION_FIELDS)), np.float32
                ),
                "account": spaces.Box(-unbounded, unbounded, (2,), np.float32),
            }
        )
        self._state: EngineState | None = None
        self._row = 0
        self._end = 0
        self._episode = -1
        self._done = True
        self._builder: LedgerBuilder | None = None

    # -- Gymnasium API -------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        data = self._data
        start = int(self._starts[self.np_random.integers(len(self._starts))])
        self._row = start
        self._end = start + self._config.episode.window_bars
        self._state = initial_state(self._config.engine)
        self._done = False
        self._episode += 1
        self._builder = LedgerBuilder(self._config.engine) if self._record else None
        session = int(data.session_index[start])
        info = {
            "session": session,
            "session_open": data.session_open[session],
            "start_time": data.open_time[start],
            "start_bar_in_session": start - int(data.session_start[session]),
        }
        return self._observation(), info

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._state is None or self._done:
            raise RuntimeError("call reset() before step(): the episode is over")
        data = self._data
        previous = self._state.equity
        row = self._row + 1
        self._state, events = step(
            self._state,
            self._config.engine,
            data.market(row),
            self._levels[int(action)],
        )
        self._row = row
        if self._builder is not None:
            self._builder.add_step(
                data.open_time[row],
                events,
                episode_id=self._episode,
                session_id=int(data.session_index[row]),
            )
        terminated = events.closeout
        truncated = row >= self._end and not terminated
        self._done = terminated or truncated
        ending = CLOSEOUT if terminated else WINDOW_END if truncated else None
        info = {
            "ending": ending,
            "equity_delta": events.equity - previous,
            "bars_advanced": 1,
            "clipped": events.clipped,
            "closeout": events.closeout,
        }
        reward = log_return_reward(previous, events.equity, self._config.reward)
        return self._observation(), reward, terminated, truncated, info

    # -- Ledger ----------------------------------------------------------------

    def ledger(self) -> Ledger:
        """The Ledger of the current (or just finished) episode.

        Only available when the env was made with ``record_ledger=True``. Rewards
        are never stored in it.
        """
        if self._builder is None:
            raise RuntimeError("the env was not made with record_ledger=True")
        return self._builder.build()

    # -- internals ---------------------------------------------------------------

    def _observation(self) -> dict[str, np.ndarray]:
        assert self._state is not None
        data = self._data
        window, valid = data.window(self._row, self._config.observation.window_bars)
        obs: Observation = build_observation(
            window,
            valid,
            self._state,
            data.market(self._row),
            self._config.engine,
            self._config.observation,
        )
        return {
            "features": obs.features,
            "valid": obs.valid.astype(np.int8),
            "position": obs.position,
            "account": obs.account,
        }


def make_env(
    dataset: PreparedDataset,
    scaler: Scaler,
    config: EnvConfig,
    *,
    seed: int | None = None,
    record_ledger: bool = False,
) -> TradingEnv:
    """Build a ``TradingEnv`` from a prepared Dataset, its Scaler and a config.

    The Dataset is read and scaled once and converted to numpy. The Scaler must
    match the Dataset's feature spec (``ScalerError`` otherwise) and, when the
    manifest names one, be that Scaler.
    """
    data = market_data.from_prepared(
        dataset,
        scaler,
        config.engine.instruments[0].symbol,
        config.observation.bar_minutes,
    )
    return TradingEnv(data, config, seed=seed, record_ledger=record_ledger)
