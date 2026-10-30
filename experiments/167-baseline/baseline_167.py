"""Throwaway (#167): multi-seed validation baseline + reference policies.

Run from the repo root:  uv run python baseline_167.py seed <s>   |   refs
Writes JSON to out/.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from shadow_trader import config, evaluation, training
from shadow_trader.env import POSITIONS

torch.set_num_threads(1)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LOG = []  # one row per evaluation.score call


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
    row = dict(net=m.net_pips, gross=m.net_pips + cost, cost=cost,
               changes=sum(changes), bars=m.bars, sharpe=m.sharpe)
    LOG.append(row)
    return evaluation.Score(m, tuple(changes))


evaluation.score = score_gross
market = training.load_market()
val = market.make_env(market.structure.split.validation)

if sys.argv[1] == "seed":
    s = int(sys.argv[2])
    t0 = time.time()
    res = training.fit(market, ppo={**config.PPO, "seed": s}, best_path=None)
    LOG[:] = LOG  # checkpoint rows in order
    for r, c in zip(LOG, res.checkpoints):
        r["timestep"] = c.timestep
    secs = time.time() - t0
    json.dump(dict(seed=s, seconds=secs, checkpoints=LOG), open(OUT / f"seed{s}.json", "w"))
    print("done", s, secs)
else:
    refs = {}
    refs["flat"] = score_gross(val, lambda o: 1) and LOG[-1]
    refs["buy_and_hold"] = score_gross(val, lambda o: 2) and LOG[-1]
    lengths = evaluation._scored_lengths(val)
    for frac in (0.35,):
        draws = []
        for d in range(20):
            rng = np.random.default_rng([config.SEED, d])
            plan = np.concatenate([evaluation.turnover_matched_plan(rng, n, max(2, round(frac * n))) for n in lengths])
            it = iter(plan)
            score_gross(val, lambda o, it=it: int(next(it)))
            draws.append(LOG[-1])
        refs[f"random_{frac}"] = dict(
            mean={k: float(np.mean([x[k] for x in draws])) for k in draws[0]},
            std_net=float(np.std([x["net"] for x in draws], ddof=1)), draws=draws)
    json.dump(refs, open(OUT / "refs.json", "w"), indent=1)
    print(json.dumps({k: (v if "mean" not in v else v["mean"]) for k, v in refs.items()}, indent=1))
