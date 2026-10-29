"""Fit the PPO policy on the train Period and save it.

:func:`load_market` is the one shared "stored Bars -> Mid OHLC + Structure"
step; ``evaluate`` calls it too and builds its environment with
:meth:`Market.make_env`, so both commands see identical data and settings.
"""

import csv
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from shadow_trader import bars, config, plotting, segments
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


def best_model_path() -> Path:
    return config.DATA_DIR / "model" / "ppo_eurusd_best.zip"


def learning_curve_path() -> Path:
    return config.DATA_DIR / "model" / "learning_curve.csv"


def learning_curve_plot_path() -> Path:
    return learning_curve_path().with_suffix(".png")


@dataclass(frozen=True)
class Checkpoint:
    """The deterministic Agent scored on the validation Period mid-training."""

    timestep: int
    metrics: Any  # evaluation.Metrics
    position_changes: int
    selected: bool = False


def _say(text: str) -> None:
    print(text, flush=True)


class _CheckpointCallback(BaseCallback):
    """Score validation every ``every`` rollouts and once when training ends.

    Rollout-end checkpoints score the policy that collected that rollout; the
    final one runs after the last update, so it scores the model that is saved.
    Validation has its own environment and prediction is deterministic, so
    training's RNG stream, and hence the weights, are untouched.
    """

    def __init__(
        self,
        env: TradingEnv,
        every: int,
        total_rollouts: int,
        best_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._env = env
        self._best_path = best_path
        self.best_index: int | None = None
        self._every = every
        self._rollouts = 0
        self._last = total_rollouts  # scored after its update, at training end
        self._total = len({r for r in range(1, total_rollouts) if r % every == 0}) + 1
        self.checkpoints: list[Checkpoint] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts % self._every == 0 and self._rollouts < self._last:
            self._score()

    def _on_training_end(self) -> None:
        self._score()

    def _score(self) -> None:
        from shadow_trader import evaluation

        timestep = int(self.model.num_timesteps)

        def policy(obs: np.ndarray) -> int:
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        result = evaluation.score(self._env, policy)
        m = result.metrics
        changes = sum(result.position_changes)
        self.checkpoints.append(Checkpoint(timestep, m, changes))
        # Never-trading Checkpoints are ineligible; strictly greater replaces,
        # so ties keep the earliest.
        is_best = changes > 0 and (
            self.best_index is None
            or m.net_pips > self.checkpoints[self.best_index].metrics.net_pips
        )
        if is_best:
            self.best_index = len(self.checkpoints) - 1
            if self._best_path is not None:
                self._best_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self._best_path)
        _say(
            f"checkpoint {len(self.checkpoints)}/{self._total} @ {timestep:,}: "
            f"net {m.net_pips:.1f} pips, sharpe {m.sharpe:.4f}, changes {changes:,}"
            + (" *" if is_best else "")
        )


@dataclass(frozen=True)
class Fit:
    model: PPO
    checkpoints: list[Checkpoint]


_FROM_CONFIG: Any = object()


def fit(
    market: Market,
    *,
    total_timesteps: int | None = None,
    ppo: Mapping[str, Any] | None = None,
    checkpoint_every: int | None = _FROM_CONFIG,
    best_path: Path | None = None,
) -> Fit:
    """Train on the train Period; defaults come from config, never from SB3.

    ``checkpoint_every`` is a rollout count; ``None`` turns scoring off.
    The best Checkpoint's model is saved to ``best_path`` (if given) as it
    appears, and its row comes back with ``selected`` set.
    """
    params = dict(config.PPO if ppo is None else ppo)
    steps = config.TOTAL_TIMESTEPS if total_timesteps is None else total_timesteps
    every = (
        config.CHECKPOINT_EVERY_ROLLOUTS
        if checkpoint_every is _FROM_CONFIG
        else checkpoint_every
    )
    env = market.make_env(market.structure.split.train)
    # Hyperparameters are an untyped config mapping; SB3 validates them itself.
    model = PPO(params.pop("policy"), env, verbose=1, **params)  # ty: ignore[invalid-argument-type]
    callback = None
    if every is not None:
        rollouts = math.ceil(steps / model.n_steps)
        val_env = market.make_env(market.structure.split.validation)
        callback = _CheckpointCallback(val_env, every, rollouts, best_path)
    # verbose=1 prints a progress table after every rollout.
    model.learn(total_timesteps=steps, callback=callback)
    if callback is None:
        return Fit(model, [])
    checkpoints = [
        replace(c, selected=i == callback.best_index)
        for i, c in enumerate(callback.checkpoints)
    ]
    return Fit(model, checkpoints)


def write_learning_curve(checkpoints: list[Checkpoint], path: Path) -> None:
    from shadow_trader import evaluation

    metric_cols = [f.name for f in fields(evaluation.Metrics)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", *metric_cols, "position_changes", "selected"])
        for c in checkpoints:
            m = asdict(c.metrics)
            writer.writerow(
                [
                    c.timestep,
                    *(m[k] for k in metric_cols),
                    c.position_changes,
                    c.selected,
                ]
            )


def run() -> None:
    best_path = best_model_path()
    best_path.unlink(missing_ok=True)
    result = fit(load_market(), best_path=best_path)
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(path)
    print(f"saved Agent to {path}")
    write_learning_curve(result.checkpoints, learning_curve_path())
    print(f"saved learning curve to {learning_curve_path()}")
    png = learning_curve_plot_path()
    try:
        plotting.plot_learning_curve(result.checkpoints, png)
        plotted = True
    except Exception as e:  # cosmetic: never lose the run over a plot
        plotted = False
        print(f"warning: learning-curve plot failed: {type(e).__name__}: {e}")
    selected = next((c for c in result.checkpoints if c.selected), None)
    if selected is None:
        print("no checkpoint had a Position change: no best model saved")
    else:
        print(f"selected checkpoint @ {selected.timestep:,}")
        print(f"saved best Agent to {best_path}")
    if plotted:
        print(f"saved learning-curve plot to {png}")
