"""Score a policy on the test Period, Segment by Segment, ending flat.

:func:`score` rolls any policy (a callable ``observation -> Action``) through
every scoreable Segment of an environment. ``evaluate`` and the Baselines both
use it, so every policy is treated identically.

Ending flat. The environment does not force the Position flat, so after the
last Bar's move the scorer closes whatever Position is open at that Bar's Mid
and charges the usual Spread cost for it (half a spread per unit of Position
crossed). That cost is subtracted from the Segment's final Reward and counts as
one Position change. A Segment is entered flat, so no Position crosses a Gap.

Segments with fewer than ``window_size + 2`` Bars are skipped entirely: 25 Bars
give one Observation but no Reward, so there is nothing to score.

Equity is one continuous additive-pip curve across Segments (Segment two starts
where one ended). Sharpe is computed over the per-Bar Rewards of each Segment
joined together; each Segment's Rewards are its own, so no cross-Gap return is
ever in the series.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from shadow_trader import training
from shadow_trader.env import PIP, POSITIONS, TradingEnv

Policy = Callable[[np.ndarray], int]


@dataclass(frozen=True)
class Metrics:
    net_pips: float
    net_pct: float  # percentage of notional
    sharpe: float  # per-Bar, un-annualised, ddof=1; 0.0 on zero variance
    max_drawdown_pips: float  # from a zero starting peak
    bars: int  # rewarded Bars
    segments: int

    def render(self) -> str:
        return "\n".join(
            [
                f"net return: {self.net_pips:.2f} pips",
                f"net return: {self.net_pct:.4f} % of notional",
                f"sharpe (per-Bar, un-annualised): {self.sharpe:.4f}",
                f"max drawdown: {self.max_drawdown_pips:.2f} pips",
                f"bars: {self.bars}",
                f"segments: {self.segments}",
            ]
        )


@dataclass(frozen=True)
class Score:
    metrics: Metrics
    position_changes: tuple[int, ...]  # per scored Segment, final flatten included


def compute_metrics(rewards: Sequence[np.ndarray], notional: float) -> Metrics:
    """Metrics from per-Segment Rewards in price units, one array per Segment.

    ``notional`` is the price level net return is expressed against.
    """
    if not rewards:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0, 0)
    pips = np.concatenate([np.asarray(r, dtype=np.float64) for r in rewards]) / PIP
    if len(pips) == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0, len(rewards))
    equity = np.cumsum(pips)
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)
    drawdown = float(np.max(peak - equity))
    net = float(equity[-1])
    sd = float(np.std(pips, ddof=1)) if len(pips) > 1 else 0.0
    sharpe = float(pips.mean()) / sd if sd > 1e-12 else 0.0
    pct = net * PIP / notional * 100
    return Metrics(net, pct, sharpe, drawdown, len(pips), len(rewards))


def score(env: TradingEnv, policy: Policy) -> Score:
    """Roll ``policy`` through each scoreable Segment of ``env`` and measure it."""
    rewards, changes, closes = [], [], []
    for i, (first, stop) in enumerate(env.segments):
        if stop - first < env.window_size + 2:
            continue
        obs, _ = env.reset(options={"segment": i})
        rs, held, truncated = [], [0.0], False
        while not truncated:
            action = int(policy(obs))
            obs, reward, _, truncated, _ = env.step(action)
            rs.append(reward)
            held.append(POSITIONS[action])
        rs[-1] -= env.spread / 2 * abs(held[-1])  # the final flatten
        held.append(0.0)
        rewards.append(np.array(rs))
        changes.append(int(np.count_nonzero(np.diff(held))))
        closes.append(env.prices[first + env.window_size : stop, 3])
    notional = float(np.concatenate(closes).mean()) if closes else 1.0
    return Score(compute_metrics(rewards, notional), tuple(changes))


def run() -> None:
    market = training.load_market()
    model = PPO.load(training.model_path(), device="cpu")

    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    result = score(market.make_env(market.structure.split.test), policy)
    print(result.metrics.render())
