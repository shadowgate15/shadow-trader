from typing import Any

import numpy as np
import pytest
from stable_baselines3 import PPO
from test_env import make_prices
from test_training import TINY_PPO, stored_bars  # noqa: F401

from shadow_trader import cli, config, evaluation, training
from shadow_trader.env import PIP, POSITIONS, TradingEnv

N = 24


def test_metrics_net_return_in_pips_and_percent_of_notional():
    m = evaluation.compute_metrics([np.array([1.0, -3.0]), np.array([5.0])], 1.25)
    assert m.net_pips == pytest.approx(3.0)
    assert m.net_pct == pytest.approx(3 * PIP / 1.25 * 100)
    assert m.bars == 3
    assert m.segments == 2


def test_sharpe_is_per_bar_with_ddof_one_and_ignores_segment_seams():
    a, b = np.array([1.0, 2.0]), np.array([3.0, 6.0])
    m = evaluation.compute_metrics([a, b], 1.0)
    pips = np.array([1.0, 2.0, 3.0, 6.0])
    assert m.sharpe == pytest.approx(pips.mean() / pips.std(ddof=1))


@pytest.mark.parametrize("rewards", [[np.zeros(5)], [np.full(5, 0.3)], [np.zeros(1)]])
def test_zero_variance_sharpe_is_zero_not_nan(rewards):
    assert evaluation.compute_metrics(rewards, 1.0).sharpe == 0.0


def test_no_segments_gives_zeroed_metrics():
    m = evaluation.compute_metrics([], 1.0)
    assert (m.net_pips, m.sharpe, m.max_drawdown_pips, m.bars, m.segments) == (
        0.0,
        0.0,
        0.0,
        0,
        0,
    )


def test_drawdown_runs_from_a_zero_peak():
    losing = evaluation.compute_metrics([np.array([-2.0, -1.0])], 1.0)
    assert losing.max_drawdown_pips == pytest.approx(3.0)
    assert losing.max_drawdown_pct == pytest.approx(3 * PIP / 1.0 * 100)


def test_notional_is_the_first_scored_bar_mid_close_of_the_first_scored_segment():
    env = env_for([(0, 10), (50, 150), (160, 260)])  # first Segment is skipped
    s = evaluation.buy_and_hold(env)
    notional = env.mid[50 + N, 3]
    assert s.metrics.net_pct == pytest.approx(s.metrics.net_pips * PIP / notional * 100)


def test_equity_is_continuous_across_segments_for_drawdown():
    # +5 then segment two starts where one ended: peak 5, trough 5-4-3 = -2
    m = evaluation.compute_metrics([np.array([5.0]), np.array([-4.0, -3.0, 1.0])], 1.0)
    assert m.max_drawdown_pips == pytest.approx(7.0)


def always(action):
    return lambda obs: action


def env_for(segs, count=400):
    return TradingEnv(make_prices(count), segs)


def test_segments_shorter_than_the_window_plus_a_step_are_skipped_entirely():
    env = env_for([(0, 10), (10, 10 + N), (50, 150), (160, 160 + N + 1)])
    s = evaluation.score(env, always(1))
    assert s.metrics.segments == 1
    assert s.metrics.bars == 100 - N - 1
    assert len(s.position_changes) == 1


def test_flat_policy_earns_nothing_and_never_changes_position():
    s = evaluation.score(env_for([(0, 100), (120, 220)]), always(1))
    assert s.metrics.net_pips == 0.0
    assert s.metrics.sharpe == 0.0
    assert s.position_changes == (0, 0)


def test_long_policy_is_flattened_at_each_segment_end_and_pays_spread_both_ways():
    env = env_for([(0, 100), (120, 220)])
    s = evaluation.score(env, always(2))
    p = env.mid[:, 3]
    gross = (p[99] - p[N]) + (p[219] - p[120 + N])
    assert s.metrics.net_pips == pytest.approx(gross / PIP - 4 * 0.5)
    assert s.position_changes == (2, 2)
    assert s.metrics.segments == 2


def test_bar_count_is_rewarded_bars_and_equity_has_no_cross_gap_return():
    env = env_for([(0, 100), (120, 220)])
    s = evaluation.score(env, always(2))
    assert s.metrics.bars == (99 - N) + (219 - 120 - N)


def test_reversal_counts_as_one_position_change():
    flip = iter([2, 0, 0, 1] + [1] * 500)
    s = evaluation.score(env_for([(0, 100)]), lambda obs: next(flip))
    assert s.position_changes == (3,)  # flat->long, long->short, short->flat


def test_cli_evaluate_prints_the_metrics_for_the_test_period(
    stored_bars,  # noqa: F811
    capsys,
):
    market = training.load_market()
    model = training.fit(market, total_timesteps=64, ppo=TINY_PPO)
    path = training.model_path()
    path.parent.mkdir(parents=True)
    model.save(path)
    assert isinstance(PPO.load(path, device="cpu"), PPO)
    capsys.readouterr()
    cli.main(["evaluate"])
    out = capsys.readouterr().out
    for label in (
        "net (pips)",
        "% notional",
        "sharpe",
        "max DD (pips)",
        "max DD (%)",
        "Agent",
        "loading",
        "random Baseline: draw",
        "bars:",
        "segments:",
    ):
        assert label in out


# --- Baselines and verdict (#150) ---------------------------------------------


def held_changes(plan):
    return int(np.count_nonzero(np.diff([0.0, *(POSITIONS[a] for a in plan), 0.0])))


@pytest.mark.parametrize("changes", [0, 2, 3, 4, 7, 20])
def test_turnover_plan_reproduces_the_position_change_count_with_final_flatten(
    changes,
):
    for seed in range(25):
        rng = np.random.default_rng(seed)
        plan = evaluation.turnover_matched_plan(rng, 60, changes)
        assert len(plan) == 60
        assert held_changes(plan) == changes


def test_turnover_plan_for_zero_changes_stays_flat():
    plan = evaluation.turnover_matched_plan(np.random.default_rng(0), 30, 0)
    assert set(plan) == {1}


def test_buy_and_hold_is_long_from_each_segment_start_and_flat_at_the_end():
    env = env_for([(0, 100), (120, 220)])
    s = evaluation.buy_and_hold(env)
    p = env.mid[:, 3]
    gross = (p[99] - p[N]) + (p[219] - p[120 + N])
    assert s.position_changes == (2, 2)  # enter once, flatten once, per Segment
    assert s.metrics.net_pips == pytest.approx(gross / PIP - 4 * 0.5)


def test_random_baseline_matches_the_agent_turnover_per_segment_not_pooled():
    env = env_for([(0, 100), (120, 220), (240, 340)], count=400)
    agent = (0, 6, 2)
    draws = evaluation.random_baseline(env, agent)
    assert len(draws) == config.N_BASELINE_DRAWS == 20
    assert all(d.position_changes == agent for d in draws)


def test_random_baseline_is_reproducible_and_draws_differ():
    env = env_for([(0, 200)])
    a = evaluation.random_baseline(env, (8,))
    b = evaluation.random_baseline(env, (8,))
    assert [d.metrics for d in a] == [d.metrics for d in b]
    assert len({d.metrics.net_pips for d in a}) > 1


def test_random_baseline_skips_short_segments_like_score_does():
    env = env_for([(0, 10), (50, 150)])
    draws = evaluation.random_baseline(env, (4,))
    assert all(d.position_changes == (4,) for d in draws)


GOOD: dict[str, Any] = {
    "agent_net": 10.0,
    "agent_sharpe": 0.2,
    "hold_net": -5.0,
    "hold_sharpe": -0.1,
    "random_net": -8.0,
    "random_sharpe": -0.2,
    "agent_changes": 12,
}


def test_verdict_passes_when_strictly_greater_everywhere_and_trading():
    v = evaluation.verdict(**GOOD)
    assert v.passed
    assert v.reasons == ()


@pytest.mark.parametrize(
    "field", ["hold_net", "hold_sharpe", "random_net", "random_sharpe"]
)
def test_verdict_fails_on_a_tie_with_any_single_bar(field):
    args = {**GOOD, field: GOOD["agent_net" if "net" in field else "agent_sharpe"]}
    v = evaluation.verdict(**args)
    assert not v.passed
    assert len(v.reasons) == 1


@pytest.mark.parametrize(
    "field,value",
    [("hold_net", 11.0), ("hold_sharpe", 0.5), ("random_net", 99.0)],
)
def test_verdict_fails_when_a_baseline_is_ahead(field, value):
    assert not evaluation.verdict(**{**GOOD, field: value}).passed


def test_permanently_flat_agent_fails_even_when_it_outscores_both_baselines():
    flat = {**GOOD, "agent_net": 0.0, "agent_sharpe": 0.0, "agent_changes": 0}
    v = evaluation.verdict(**flat)
    assert not v.passed
    assert any("Position change" in r for r in v.reasons)


def test_cli_evaluate_prints_baselines_and_an_explicit_verdict(
    stored_bars,  # noqa: F811
    capsys,
):
    market = training.load_market()
    model = training.fit(market, total_timesteps=64, ppo=TINY_PPO)
    path = training.model_path()
    path.parent.mkdir(parents=True)
    model.save(path)
    capsys.readouterr()
    cli.main(["evaluate"])
    out = capsys.readouterr().out
    assert "buy-and-hold" in out
    assert "random" in out
    assert "draws" in out
    assert "VERDICT: PASS" in out or "VERDICT: FAIL" in out
