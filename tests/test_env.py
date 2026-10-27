import math

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from shadow_trader.env import TradingEnv
from shadow_trader.segments import Segment

N = 24
SCALE = 1e4


def make_prices(count: int, seed: int = 0) -> np.ndarray:
    """Mid OHLC (open, high, low, close) with Open == previous Close."""
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, count))
    open_ = np.concatenate([[1.1], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 1e-4, count)
    low = np.minimum(open_, close) - rng.uniform(0, 1e-4, count)
    return np.column_stack([open_, high, low, close])


def env_for(count=400, segs=((0, 400),), **kw) -> TradingEnv:
    return TradingEnv(make_prices(count), list(segs), **kw)


def run_to_end(env: TradingEnv, action: int = 1):
    steps, done, info = 0, False, {}
    terminated = truncated = False
    while not done:
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    return steps, terminated, truncated, info


def test_spaces():
    env = env_for()
    assert env.observation_space == spaces.Box(-np.inf, np.inf, (73,), np.float32)
    assert env.action_space == spaces.Discrete(3)


def test_check_env_passes():
    check_env(env_for(count=1000, segs=((0, 1000),)), skip_render_check=True)


def test_observation_holds_scaled_log_returns_against_previous_close():
    prices = make_prices(400)
    env = TradingEnv(prices, [(0, 400)])
    obs, info = env.reset(seed=1, options={"segment": 0})
    t = N  # first Action falls on the 25th Bar: index 24
    assert info["bar"] == t
    bars = range(t - N + 1, t + 1)
    for ch, col in enumerate((1, 2, 3)):  # high, low, close
        expected = [SCALE * math.log(prices[b, col] / prices[b - 1, 3]) for b in bars]
        np.testing.assert_allclose(obs[ch * N : (ch + 1) * N], expected, rtol=1e-4)
    assert obs[-1] == 0.0
    assert obs.dtype == np.float32


def test_observation_is_not_clipped():
    prices = make_prices(400)
    prices[30] *= 1.5  # a genuine shock
    env = TradingEnv(prices, [(0, 400)])
    env.reset(options={"segment": 0})
    for _ in range(6):
        obs, *_ = env.step(1)
    assert obs[:N].max() > 3000


def test_position_is_last_observation_element_and_target_not_increment():
    env = env_for()
    env.reset(seed=0, options={"segment": 0})
    for action, pos in [(2, 1.0), (2, 1.0), (0, -1.0), (1, 0.0)]:
        obs, *_ = env.step(action)
        assert obs[-1] == pos


def test_reward_is_pnl_on_position_held_minus_half_spread_per_unit():
    prices = make_prices(400)
    env = TradingEnv(prices, [(0, 400)])
    env.reset(options={"segment": 0})
    t = N
    half = 1.0 * 1e-4 / 2
    # Go long at t: held into t+1, pays half a spread to open.
    _, r, *_ = env.step(2)
    assert r == pytest.approx((prices[t + 1, 3] - prices[t, 3]) - half)
    # Hold: no cost.
    _, r, *_ = env.step(2)
    assert r == pytest.approx(prices[t + 2, 3] - prices[t + 1, 3])
    # Reverse long -> short: pays a full spread.
    _, r, *_ = env.step(0)
    assert r == pytest.approx(-(prices[t + 3, 3] - prices[t + 2, 3]) - 2 * half)
    # Flat: closing costs half, earns nothing.
    _, r, *_ = env.step(1)
    assert r == pytest.approx(-half)
    # Flat and staying flat is free.
    _, r, *_ = env.step(1)
    assert r == 0.0


def test_spread_pips_is_configurable():
    prices = make_prices(400)
    env = TradingEnv(prices, [(0, 400)], spread_pips=2.0)
    env.reset(options={"segment": 0})
    _, r, *_ = env.step(2)
    assert r == pytest.approx(prices[N + 1, 3] - prices[N, 3] - 1e-4)


def test_episode_is_288_steps_and_truncates():
    env = env_for(count=1000, segs=((0, 1000),))
    env.reset(seed=3)
    steps, terminated, truncated, _ = run_to_end(env)
    assert steps == 288 and truncated and not terminated


def test_episodes_never_span_a_segment_boundary_and_start_randomly():
    segs = [(0, 400), (400, 450), (450, 900)]  # middle one is too short
    env = env_for(count=900, segs=segs)
    starts = set()
    for seed in range(60):
        _, info = env.reset(seed=seed)
        first_action = info["bar"]
        last_bar = first_action + 288  # last Bar whose move is rewarded
        seg = next(s for s in segs if s[0] <= first_action < s[1])
        assert seg[0] + N <= first_action and last_bar < seg[1]
        starts.add(first_action)
    assert len(starts) > 10


def test_same_seed_gives_same_episode_start():
    env = env_for(count=1000, segs=((0, 1000),))
    _, a = env.reset(seed=7)
    _, b = env.reset(seed=7)
    assert a["bar"] == b["bar"]


def test_segment_option_runs_one_segment_from_its_start_to_its_end():
    segs = [Segment(0, 100, None, None, None), Segment(100, 300, None, None, None)]  # ty: ignore[invalid-argument-type]
    env = TradingEnv(make_prices(300), segs)
    _, info = env.reset(options={"segment": 1})
    assert info["bar"] == 100 + N  # warm-up of 24 Bars, first Action on Bar 25
    steps, _, truncated, info = run_to_end(env)
    assert truncated
    assert steps == 200 - N - 1  # last rewarded Bar is the Segment's last
    assert info["bar"] == 299


def test_no_segment_long_enough_is_an_error():
    with pytest.raises(ValueError):
        env_for(count=100, segs=((0, 100),)).reset()
