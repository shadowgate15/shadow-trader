"""The Gymnasium environment layer: ``TradingEnv`` and ``make_env``.

The env adds actions, episodes, reward and ``info`` on top of the pure engine,
observation assembler and ledger. The agent acts only on Decision Bars (usable
Bars): one env step advances at least one Bar and spans every following Gap or
non-usable Bar with the target frozen, up to and including the next usable Bar, the
window end or the Session end. The engine still runs on each spanned Bar, so
pending fills, rollover and the Session-final force-flat happen; unfillable Bars
are marked at the last trusted close. The observation is that of the last Decision
Bar. A training episode is a random window of ``EpisodeConfig.window_bars``
Bars inside one Session that starts on a usable Bar. A margin closeout ends the
episode as ``terminated``; window end, Session end and dataset end are
``truncated``. On the Session's final Bar the engine forces any position flat at
the open (the cost lands in that last step).

``reset(options={"session": k, ...})`` instead starts an evaluation episode: the
Session with ordinal ``k`` from ``start_bar`` (default: its first usable Bar) with
``equity`` (default: the starting equity) through to the Session's end. Options
make the episode fully deterministic; the RNG is not touched. ``chain=True`` keeps
recording into the same Ledger (an evaluation sweep, see ``sweep.run_sweep``).

``info`` reports ``bars_advanced`` (Bars stepped), ``gap_bars`` (Gap rows among
them), ``unfillable_bars`` (Bars that could not fill: Gaps plus distrusted ones) and
``blind_run`` (stepped Bars that were not Decision Bars, i.e. the agent was blind).
There is no blind-run cap.

Gymnasium is imported here (and in nothing else of the package); the package
``__init__`` exposes this module lazily.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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

log = logging.getLogger(__name__)

WINDOW_END = "window_end"
SESSION_END = "session_end"
DATASET_END = "dataset_end"
CLOSEOUT = "closeout"
_OPTION_KEYS = frozenset({"session", "start_bar", "equity", "chain"})


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
        self._obs_row = 0  # the last Decision Bar: the observation is built there
        self._end = 0
        self._episode = -1
        self._done = True
        self._explicit = False
        self._builder: LedgerBuilder | None = None
        #: Dataset id and Scaler hash for manifests; ``make_env`` fills them in.
        self.provenance: dict[str, str] = {}

    @property
    def config(self) -> EnvConfig:
        return self._config

    @property
    def seed_entropy(self) -> int:
        """Entropy of the env's current ``SeedSequence`` (recorded in manifests)."""
        seed_seq = getattr(self.np_random.bit_generator, "seed_seq")  # noqa: B009
        return int(seed_seq.entropy)

    @property
    def records_ledger(self) -> bool:
        return self._record

    @property
    def session_count(self) -> int:
        return len(self._data.session_open)

    @property
    def equity(self) -> float:
        """Account equity now (after the last step, or the carried-in equity)."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state.equity

    def sweep_sessions(self) -> list[int]:
        """Ordinals of the Sessions an evaluation sweep can run, chronologically.

        A Session with no usable Bar is skipped and logged.
        """
        data = self._data
        runnable = []
        for k, opened in enumerate(data.session_open):
            lo, hi = int(data.session_start[k]), int(data.session_end[k])
            if data.usable[lo:hi].any():
                runnable.append(k)
            else:
                log.warning("skipping Session %s: it has no usable Bars", opened)
        return runnable

    # -- Gymnasium API -------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        data = self._data
        equity = self._config.engine.starting_equity
        chain = False
        if options:
            start, self._end, equity, chain = self._explicit_start(options)
            self._explicit = True
        else:
            start = int(self._starts[self.np_random.integers(len(self._starts))])
            self._end = start + self._config.episode.window_bars
            self._explicit = False
        self._row = self._obs_row = start
        self._state = initial_state(
            self._config.engine,
            equity,
            ((float(data.close_bid[start]), float(data.close_spread[start])),),
        )
        self._done = False
        self._episode += 1
        session = int(data.session_index[start])
        if self._record:
            if not (chain and self._builder is not None):
                self._builder = LedgerBuilder(self._config.engine)
            self._builder.begin_episode(
                self._episode,
                session,
                data.open_time[start],
                equity,
                chained=chain and self._episode > 0,
            )
        else:
            self._builder = None
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
        level = self._levels[int(action)]
        gap_bars = unfillable = blind = skipped = advanced = 0
        flags = {"clipped": False, "closeout": False, "forced_flat": False}
        unfilled_forced_flat = False
        events = None
        row = self._row
        while True:
            row += 1
            self._state, events = step(
                self._state, self._config.engine, data.market(row), level
            )
            advanced += 1
            decision = bool(data.usable[row])
            blind = 0 if decision else blind + 1  # the run so far, 0 on a decision
            skipped += int(not decision)
            gap_bars += int(data.is_gap[row])
            unfillable += int(not data.fillable[row])
            flags["clipped"] |= events.clipped
            flags["closeout"] |= events.closeout
            flags["forced_flat"] |= events.forced_flat
            unfilled_forced_flat |= events.unfilled_forced_flat
            if self._builder is not None:
                self._builder.add_step(
                    data.open_time[row],
                    events,
                    episode_id=self._episode,
                    session_id=int(data.session_index[row]),
                    is_decision=decision,
                    blind_run=blind,
                )
            if events.closeout or row >= self._end or decision:
                break
        self._row = row
        if data.usable[row]:
            self._obs_row = row
        terminated = events.closeout
        truncated = row >= self._end and not terminated
        self._done = terminated or truncated
        ending = None
        if terminated:
            ending = CLOSEOUT
        elif truncated:
            if data.is_session_final[row]:
                ending = SESSION_END
            else:
                ending = DATASET_END if self._explicit else WINDOW_END
        if self._builder is not None and ending is not None:
            self._builder.end_episode(ending)
        info = {
            "ending": ending,
            "equity_delta": events.equity - previous,
            "bars_advanced": advanced,
            "gap_bars": gap_bars,
            "unfillable_bars": unfillable,
            "blind_run": skipped,
            "clipped": flags["clipped"],
            "closeout": flags["closeout"],
            "forced_flat": flags["forced_flat"],
            "unfilled_forced_flat": unfilled_forced_flat,
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

    def _explicit_start(self, options: dict[str, Any]) -> tuple[int, int, float, bool]:
        """Validate ``reset`` options: (start row, last row, equity, chain)."""
        data = self._data
        unknown = sorted(set(options) - _OPTION_KEYS)
        if unknown:
            raise ValueError(f"unknown reset option(s): {', '.join(unknown)}")
        if "session" not in options:
            raise ValueError("reset options need a 'session'")
        session = options["session"]
        if (
            isinstance(session, bool)
            or not isinstance(session, int | np.integer)
            or not 0 <= session < len(data.session_open)
        ):
            raise ValueError(f"session {session!r} is not a Session ordinal")
        lo, hi = int(data.session_start[session]), int(data.session_end[session])
        usable = np.flatnonzero(data.usable[lo:hi])
        if len(usable) == 0:
            raise ValueError(f"Session {session} has no usable Bars")
        offset = int(options.get("start_bar", usable[0]))
        start = lo + offset
        if not 0 <= offset < hi - lo - 1 or not data.usable[start]:
            raise ValueError(
                f"start_bar {offset} is not a usable Bar before the end of Session "
                f"{session}"
            )
        equity = options.get("equity", self._config.engine.starting_equity)
        if (
            isinstance(equity, bool)
            or not isinstance(equity, int | float)
            or not np.isfinite(equity)
            or equity <= 0
        ):
            raise ValueError(f"equity {equity!r} must be a positive number")
        return start, hi - 1, float(equity), bool(options.get("chain", False))

    def _observation(self) -> dict[str, np.ndarray]:
        assert self._state is not None
        data = self._data
        row = self._obs_row  # the last Decision Bar; later Bars are blind
        window, valid = data.window(row, self._config.observation.window_bars)
        market = replace(
            data.market(row),
            bars_to_session_final=int(data.bars_to_session_final[self._row]),
        )
        obs: Observation = build_observation(
            window,
            valid,
            self._state,
            market,
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
    env = TradingEnv(data, config, seed=seed, record_ledger=record_ledger)
    env.provenance = {
        "dataset_id": dataset.dataset_id,
        "scaler_hash": scaler.scaler_id,
    }
    return env
