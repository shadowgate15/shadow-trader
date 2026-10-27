"""Fit the PPO policy on the train Period and save it.

:func:`load_market` is the one shared "stored Bars -> Mid OHLC + Structure"
step; ``evaluate`` calls it too and builds its environment with
:meth:`Market.make_env`, so both commands see identical data and settings.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from stable_baselines3 import PPO

from shadow_trader import bars, config, segments
from shadow_trader.env import TradingEnv


@dataclass(frozen=True)
class Market:
    mid: np.ndarray  # (n, 4) Mid open/high/low/close, indexed by global Bar
    structure: segments.Structure

    def make_env(self, period: segments.Period) -> TradingEnv:
        """The environment for one Period, with the configured settings."""
        return TradingEnv(
            self.mid,
            self.structure.segments_within(period),
            window_size=config.WINDOW_SIZE,
            episode_length=config.EPISODE_LENGTH,
            spread_pips=config.SPREAD_PIPS,
        )


def load_market() -> Market:
    mid = bars.mid(pq.read_table(bars.bars_path()))
    mid_ohlc = np.column_stack([mid[f"mid_{f}"] for f in bars.FIELDS])
    return Market(mid_ohlc, segments.load())


def model_path() -> Path:
    return config.DATA_DIR / "model" / "ppo_eurusd.zip"


def fit(
    market: Market,
    *,
    total_timesteps: int | None = None,
    ppo: Mapping[str, Any] | None = None,
) -> PPO:
    """Train on the train Period; defaults come from config, never from SB3."""
    params = dict(config.PPO if ppo is None else ppo)
    steps = config.TOTAL_TIMESTEPS if total_timesteps is None else total_timesteps
    env = market.make_env(market.structure.split.train)
    # Hyperparameters are an untyped config mapping; SB3 validates them itself.
    model = PPO(params.pop("policy"), env, verbose=1, **params)  # ty: ignore[invalid-argument-type]
    # verbose=1 prints a progress table after every rollout.
    model.learn(total_timesteps=steps)
    return model


def run() -> None:
    model = fit(load_market())
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"saved Agent to {path}")
