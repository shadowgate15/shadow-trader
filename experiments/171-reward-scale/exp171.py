"""Throwaway (#171): does Reward scaling / normalisation / vf settings help?

Run from the repo root:  uv run python experiments/171-reward-scale/exp171.py <variant> <seed>
Variants: unit (reward / per-Bar close-move std), vecnorm (VecNormalize, reward only),
vf2 (vf_coef 2.0, raw reward), vecnorm_vf (VecNormalize + vf_coef 2.0).
Training Reward is transformed; validation scoring stays in raw pips (separate env).
Same logging as baseline_167.py (gross = net + Spread cost).
"""
import json, sys, time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from shadow_trader import config, evaluation, training
from shadow_trader.env import POSITIONS

torch.set_num_threads(1)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LOG = []


def score_gross(env, policy):
    rewards, changes, crossed, notional = [], [], 0.0, None
    for i, (first, stop) in enumerate(env.segments):
        if stop - first < env.window_size + 2:
            continue
        obs, _ = env.reset(options={"segment": i})
        rs, held, trunc = [], [0.0], False
        while not trunc:
            a = int(policy(obs))
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
    m = evaluation.compute_metrics(rewards, notional or 1.0)
    cost = env.spread_pips / 2 * crossed
    LOG.append(dict(net=m.net_pips, gross=m.net_pips + cost, cost=cost,
                    changes=sum(changes), bars=m.bars, sharpe=m.sharpe))
    return evaluation.Score(m, tuple(changes))


class Scale(gym.RewardWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k

    def reward(self, r):
        return r * self.k


variant, seed = sys.argv[1], int(sys.argv[2])
evaluation.score = score_gross
market = training.load_market()

ppo = {**config.PPO, "seed": seed}
real_ppo = training.PPO
if variant == "unit":
    std = float(np.std(np.diff(market.mid[:, 3]) * 1e4))
    print("per-Bar close move std (pips):", std)
    k = 1.0 / std

    def wrap(env):
        return Scale(env, k)
elif variant in ("vecnorm", "vecnorm_vf"):
    def wrap(env):
        return VecNormalize(DummyVecEnv([lambda: env]), norm_obs=False, norm_reward=True, gamma=ppo["gamma"])
else:
    def wrap(env):
        return env
if variant in ("vf2", "vecnorm_vf"):
    ppo["vf_coef"] = 2.0


def patched(policy, env, **kw):
    return real_ppo(policy, wrap(env), **kw)


training.PPO = patched
t0 = time.time()
steps = int(sys.argv[3]) if len(sys.argv) > 3 else None  # smoke tests only
res = training.fit(market, ppo=ppo, best_path=None, total_timesteps=steps, **({"checkpoint_every": 1} if steps else {}))
for r, c in zip(LOG, res.checkpoints):
    r["timestep"] = c.timestep
json.dump(dict(variant=variant, seed=seed, seconds=time.time() - t0, checkpoints=LOG),
          open(OUT / f"{variant}_seed{seed}.json", "w"))
print("done", variant, seed, time.time() - t0)
