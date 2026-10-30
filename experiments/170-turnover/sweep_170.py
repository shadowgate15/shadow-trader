"""Throwaway (#170): does lowering turnover pressure fix the collapse?

Run from the repo root:  uv run python experiments/170-turnover/sweep_170.py <config> <seed>
Writes JSON to out/<config>_s<seed>.json. Same yardstick as #167 (validation,
Spread 1.0 pip, final Checkpoint judged on the multi-seed mean).

Action persistence k: the Agent decides every k Bars and the Position is held
in between (training: env repeats the action; validation: the policy is held).
Timesteps are decisions, so k>1 sees k x more Bars for the same 1M steps.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shadow_trader import config, evaluation, training
from shadow_trader.env import POSITIONS, TradingEnv

torch.set_num_threads(1)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

CONFIGS = {
    "ent0": dict(ppo=dict(ent_coef=0.0)),
    "ent001": dict(ppo=dict(ent_coef=0.001)),
    "g95": dict(ppo=dict(gamma=0.95)),
    "g90": dict(ppo=dict(gamma=0.90)),
    "g95_ent001": dict(ppo=dict(gamma=0.95, ent_coef=0.001)),
    "k3": dict(k=3),
    "k6": dict(k=6),
    "k12": dict(k=12),
    "k6_g95_ent001": dict(k=6, ppo=dict(gamma=0.95, ent_coef=0.001)),
}


class PersistentEnv(TradingEnv):
    k = 1  # set by the runner; scoring drops it to 1 and holds in the policy

    def step(self, action):
        total = 0.0
        for _ in range(self.k):
            obs, r, term, trunc, info = super().step(action)
            total += r
            if trunc:
                break
        return obs, total, term, trunc, info


def score_gross(env, policy, k):
    saved, env.k = env.k, 1
    rewards, changes, crossed, notional = [], [], 0.0, None
    for i, (first, stop) in enumerate(env.segments):
        if stop - first < env.window_size + 2:
            continue
        obs, _ = env.reset(options={"segment": i})
        rs, held, trunc, n, a = [], [0.0], False, 0, 1
        while not trunc:
            if n % k == 0:
                a = int(policy(obs))
            n += 1
            obs, r, _, trunc, _ = env.step(a)
            rs.append(r)
            held.append(POSITIONS[a])
        rs[-1] -= env.spread_pips / 2 * abs(held[-1])
        held.append(0.0)
        rewards.append(np.array(rs))
        d = np.abs(np.diff(held))
        changes.append(int(np.count_nonzero(d)))
        crossed += float(d.sum())
        if notional is None:
            notional = float(env.mid[first + env.window_size, 3])
    env.k = saved
    m = evaluation.compute_metrics(rewards, notional or 1.0)
    cost = env.spread_pips / 2 * crossed
    return m, changes, dict(net=m.net_pips, gross=m.net_pips + cost, cost=cost,
                            changes=sum(changes), bars=m.bars, sharpe=m.sharpe)


name, seed = sys.argv[1], int(sys.argv[2])
cfg = CONFIGS[name]
k = cfg.get("k", 1)
LOG = []


def score(env, policy):
    m, changes, row = score_gross(env, policy, k)
    LOG.append(row)
    return evaluation.Score(m, tuple(changes))


evaluation.score = score


def make_env(self, period):
    env = PersistentEnv(self.mid, self.structure.segments_within(period),
                        window_size=config.WINDOW_SIZE,
                        episode_length=config.EPISODE_LENGTH,
                        spread_pips=config.SPREAD_PIPS)
    env.k = k
    return env


training.Market.make_env = make_env
market = training.load_market()
t0 = time.time()
smoke = {"total_timesteps": int(sys.argv[3]), "checkpoint_every": 1} if len(sys.argv) > 3 else {}
res = training.fit(market, **smoke, ppo={**config.PPO, **cfg.get("ppo", {}), "seed": seed}, best_path=None)
for r, c in zip(LOG, res.checkpoints):
    r["timestep"] = c.timestep
json.dump(dict(config=name, seed=seed, seconds=time.time() - t0, checkpoints=LOG),
          open(OUT / (f"{name}_s{seed}.json" if not smoke else "smoke.json"), "w"))
print("done", name, seed, time.time() - t0)
