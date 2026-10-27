import numpy as np
import pytest
from stable_baselines3 import PPO
from test_env import make_prices
from test_training import TINY_PPO, stored_bars  # noqa: F401

from shadow_trader import cli, evaluation, training
from shadow_trader.env import PIP, TradingEnv

N = 24


def test_metrics_net_return_in_pips_and_percent_of_notional():
    m = evaluation.compute_metrics([np.array([1e-4, -3e-4]), np.array([5e-4])], 1.25)
    assert m.net_pips == pytest.approx(3.0)
    assert m.net_pct == pytest.approx(3e-4 / 1.25 * 100)
    assert m.bars == 3
    assert m.segments == 2


def test_sharpe_is_per_bar_with_ddof_one_and_ignores_segment_seams():
    a, b = np.array([1.0, 2.0]) * PIP, np.array([3.0, 6.0]) * PIP
    m = evaluation.compute_metrics([a, b], 1.0)
    pips = np.array([1.0, 2.0, 3.0, 6.0])
    assert m.sharpe == pytest.approx(pips.mean() / pips.std(ddof=1))


@pytest.mark.parametrize(
    "rewards", [[np.zeros(5)], [np.full(5, 0.3e-4)], [np.zeros(1)]]
)
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
    losing = evaluation.compute_metrics([np.array([-2.0, -1.0]) * PIP], 1.0)
    assert losing.max_drawdown_pips == pytest.approx(3.0)


def test_equity_is_continuous_across_segments_for_drawdown():
    # +5 then segment two starts where one ended: peak 5, trough 5-4-3 = -2
    m = evaluation.compute_metrics(
        [np.array([5.0]) * PIP, np.array([-4.0, -3.0, 1.0]) * PIP], 1.0
    )
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
    p = env.prices[:, 3]
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
        "pips",
        "% of notional",
        "sharpe",
        "max drawdown",
        "bars:",
        "segments:",
    ):
        assert label in out
