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

Notional. Percentages are of one unit at the first evaluated Mid: the Mid close
of the first scored Bar (index ``first + window_size``, where the first Action
falls) of the first scored Segment. Buy-and-hold's % thus reads close to the
pair's price change over the test Period.

Rewards come from the environment already in pips. Equity is one continuous
 additive-pip curve across Segments (Segment two starts
where one ended). Sharpe is computed over the per-Bar Rewards of each Segment
joined together; each Segment's Rewards are its own, so no cross-Gap return is
ever in the series.

Baselines. Both go through :func:`score` on the same env, so they pay the same
Spread cost and are flattened the same way.

* Buy-and-hold is long from each Segment's first Action to its end, then the
  shared final flatten closes it: two Position changes per Segment, never held
  across a Gap.
* The random Baseline is turnover-matched per Segment. A Segment's count of
  Position changes includes the final flatten (the Agent's does too, see
  above), so a plan with ``c`` changes picks ``c`` of the ``K + 1`` transitions
  in ``flat, a_1 .. a_K, flat`` uniformly, and gives each a random new Position
  different from the one before it (no no-op changes). The last change must land
  on flat, so ``c = 1`` cannot occur (an Agent's count is 0 or at least 2).
  Draw ``d`` is seeded with ``[config.SEED, d]``, so the verdict reproduces.

Verdict. See :func:`verdict`: strictly greater on net return and Sharpe against
both Baselines, plus at least one Position change.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from shadow_trader import config, training
from shadow_trader.env import PIP, POSITIONS, TradingEnv

Policy = Callable[[np.ndarray], int]


@dataclass(frozen=True)
class Metrics:
    net_pips: float
    net_pct: float  # percentage of notional
    sharpe: float  # per-Bar, un-annualised, ddof=1; 0.0 on zero variance
    max_drawdown_pips: float  # from a zero starting peak
    max_drawdown_pct: float  # percentage of notional
    bars: int  # rewarded Bars
    segments: int


@dataclass(frozen=True)
class Score:
    metrics: Metrics
    position_changes: tuple[int, ...]  # per scored Segment, final flatten included


def compute_metrics(rewards: Sequence[np.ndarray], notional: float) -> Metrics:
    """Metrics from per-Segment Rewards in pips, one array per Segment.

    ``notional`` is the Mid level net return and drawdown are expressed against.
    """
    pips = np.concatenate([np.asarray(r, dtype=np.float64) for r in rewards] or [[]])
    if len(pips) == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, len(rewards))
    equity = np.cumsum(pips)
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)
    drawdown = float(np.max(peak - equity))
    net = float(equity[-1])
    sd = float(np.std(pips, ddof=1)) if len(pips) > 1 else 0.0
    sharpe = float(pips.mean()) / sd if sd > 1e-12 else 0.0
    to_pct = PIP / notional * 100
    return Metrics(
        net, net * to_pct, sharpe, drawdown, drawdown * to_pct, len(pips), len(rewards)
    )


def score(env: TradingEnv, policy: Policy) -> Score:
    """Roll ``policy`` through each scoreable Segment of ``env`` and measure it."""
    rewards, changes, notional = [], [], None
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
        rs[-1] -= env.spread_pips / 2 * abs(held[-1])  # the final flatten
        held.append(0.0)
        rewards.append(np.array(rs))
        changes.append(int(np.count_nonzero(np.diff(held))))
        if notional is None:
            notional = float(env.mid[first + env.window_size, 3])
    return Score(compute_metrics(rewards, notional or 1.0), tuple(changes))


def _scored_lengths(env: TradingEnv) -> list[int]:
    """Action count of each Segment :func:`score` would score, in order."""
    return [
        stop - first - env.window_size - 1
        for first, stop in env.segments
        if stop - first >= env.window_size + 2
    ]


def buy_and_hold(env: TradingEnv) -> Score:
    """Long from each scored Segment's start; the final flatten exits it."""
    return score(env, lambda obs: 2)


def turnover_matched_plan(
    rng: np.random.Generator, n_actions: int, changes: int
) -> np.ndarray:
    """Random Actions for one Segment with exactly ``changes`` Position changes.

    The count includes the final flatten, as :func:`score` counts it.
    """
    plan = np.ones(n_actions, dtype=np.int64)
    if changes == 0 or n_actions == 0:
        return plan
    changes = min(max(changes, 2), n_actions + 1)
    # transition j (0..K) is the change into Action j; j == K is the final flatten
    times = np.sort(rng.choice(n_actions + 1, size=changes, replace=False))
    held = [0]  # held[k] is the Position after change k-1; it starts and ends flat
    for k in range(1, changes):
        options = [x for x in (-1, 0, 1) if x != held[-1]]
        if k == changes - 1:  # the move before the closing flat must be a real one
            options = [x for x in options if x != 0]
        held.append(int(rng.choice(options)))
    held.append(0)
    for k in range(changes):
        end = n_actions if k == changes - 1 else int(times[k + 1])
        plan[int(times[k]) : end] = held[k + 1] + 1
    return plan


def random_baseline(
    env: TradingEnv,
    agent_changes: Sequence[int],
    draws: int = config.N_BASELINE_DRAWS,
    seed: int = config.SEED,
    progress: Callable[[int, int], None] | None = None,
) -> list[Score]:
    """Score ``draws`` seeded random policies matching ``agent_changes`` per Segment.

    ``agent_changes`` is the Agent's :attr:`Score.position_changes` on ``env``.
    ``progress(done, total)`` is called after each draw.
    """
    lengths = _scored_lengths(env)
    scores = []
    for d in range(draws):
        rng = np.random.default_rng([seed, d])
        plan = np.concatenate(
            [
                turnover_matched_plan(rng, n, c)
                for n, c in zip(lengths, agent_changes, strict=True)
            ]
        )
        actions = iter(plan)
        scores.append(score(env, lambda obs, a=actions: int(next(a))))
        if progress:
            progress(d + 1, draws)
    return scores


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: tuple[str, ...]  # why it failed; empty on a pass

    def render(self) -> str:
        if self.passed:
            return "VERDICT: PASS"
        return "\n".join(["VERDICT: FAIL", *(f"  - {r}" for r in self.reasons)])


def verdict(
    *,
    agent_net: float,
    agent_sharpe: float,
    hold_net: float,
    hold_sharpe: float,
    random_net: float,
    random_sharpe: float,
    agent_changes: int,
) -> Verdict:
    """Pass only if the Agent is strictly greater on net return and Sharpe against
    both Baselines (random's means over its draws) and made a Position change."""
    reasons = []
    for name, net, sharpe in (
        ("buy-and-hold", hold_net, hold_sharpe),
        ("random", random_net, random_sharpe),
    ):
        if not agent_net > net:
            reasons.append(f"net return not above {name}: {agent_net:.2f} vs {net:.2f}")
        if not agent_sharpe > sharpe:
            reasons.append(
                f"Sharpe not above {name}: {agent_sharpe:.4f} vs {sharpe:.4f}"
            )
    if agent_changes < 1:
        reasons.append("no Position change over the test period")
    return Verdict(not reasons, tuple(reasons))


def _mean_metrics(scores: Sequence[Score]) -> Metrics:
    ms = [s.metrics for s in scores]

    def mean(field: str) -> float:
        return float(np.mean([getattr(m, field) for m in ms]))

    return Metrics(
        mean("net_pips"),
        mean("net_pct"),
        mean("sharpe"),
        mean("max_drawdown_pips"),
        mean("max_drawdown_pct"),
        ms[0].bars,
        ms[0].segments,
    )


def _say(text: str = "") -> None:
    print(text, flush=True)


def run() -> None:
    _say("loading Bars and the trained Agent")
    market = training.load_market()
    best_path = training.best_model_path()
    if best_path.exists():
        model = PPO.load(best_path, device=config.DEVICE)
        _say(f"checkpoint: best @ {model.num_timesteps:,} steps ({best_path.name})")
    else:
        model = PPO.load(training.model_path(), device=config.DEVICE)
        _say(
            f"checkpoint: final @ {model.num_timesteps:,} steps "
            f"({best_path.name} not found: no checkpoint had a Position change)"
        )

    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    env = market.make_env(market.structure.split.test)
    _say("scoring the Agent on the test Period")
    agent = score(env, policy)
    _say("scoring buy-and-hold")
    hold = buy_and_hold(env)

    def progress(done: int, total: int) -> None:
        if done % 5 == 0 or done == total:
            _say(f"scoring random Baseline: draw {done}/{total}")

    draws = random_baseline(env, agent.position_changes, progress=progress)
    rand = _mean_metrics(draws)
    am, hm = agent.metrics, hold.metrics
    beat_net = float(np.mean([am.net_pips > d.metrics.net_pips for d in draws]))
    beat_sharpe = float(np.mean([am.sharpe > d.metrics.sharpe for d in draws]))
    changes = sum(agent.position_changes)

    _say()
    _say(f"bars: {am.bars}")
    _say(f"segments: {am.segments}")
    _say(f"Agent position changes: {changes}")
    _say()
    header = ("net (pips)", "net (% notional)", "sharpe", "max DD (pips)", "max DD (%)")
    _say(f"{'':14}" + "".join(f"{h:>18}" for h in header))
    for name, m in (("Agent", am), ("buy-and-hold", hm), ("random (mean)", rand)):
        cells = (
            m.net_pips,
            m.net_pct,
            m.sharpe,
            m.max_drawdown_pips,
            m.max_drawdown_pct,
        )
        _say(f"{name:14}" + "".join(f"{c:18.4f}" for c in cells))
    _say("sharpe is per-Bar, un-annualised; max drawdown is from a zero starting peak")
    _say()
    _say(
        f"random Baseline: {len(draws)} seeded draws, turnover matched per Segment; "
        f"Agent beats {beat_net:.0%} of draws on net return, "
        f"{beat_sharpe:.0%} on Sharpe"
    )
    _say()
    _say(
        verdict(
            agent_net=am.net_pips,
            agent_sharpe=am.sharpe,
            hold_net=hm.net_pips,
            hold_sharpe=hm.sharpe,
            random_net=rand.net_pips,
            random_sharpe=rand.sharpe,
            agent_changes=changes,
        ).render()
    )
