import math

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from shadow_trader.env import TradingEnv
from shadow_trader.segments import Segment

N = 24
SCALE = 1e4
LEAD = 300  # leading history: more than the 288-Bar lookback a Segment needs
RETURN_HORIZONS = (48, 144, 288)


def make_prices(count: int, seed: int = 0) -> np.ndarray:
    """Mid OHLC (open, high, low, close) with Open == previous Close."""
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, count))
    open_ = np.concatenate([[1.1], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 1e-4, count)
    low = np.minimum(open_, close) - rng.uniform(0, 1e-4, count)
    return np.column_stack([open_, high, low, close])


def env_for(count=LEAD + 400, segs=((LEAD, LEAD + 400),), **kw) -> TradingEnv:
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
    assert env.observation_space == spaces.Box(
        -np.inf, np.inf, (1 + 3 * N + 3 * N,), np.float32
    )
    assert env.action_space == spaces.Discrete(3)


def test_check_env_passes():
    check_env(env_for(count=1300, segs=((LEAD, 1300),)), skip_render_check=True)


def test_observation_holds_scaled_log_returns_against_previous_close():
    prices = make_prices(LEAD + 400)
    env = TradingEnv(prices, [(LEAD, LEAD + 400)])
    obs, info = env.reset(seed=1, options={"segment": 0})
    t = LEAD + N  # first Action falls on the 25th Bar of the Segment
    assert info["bar"] == t
    bars = range(t - N + 1, t + 1)
    for ch, col in enumerate((1, 2, 3)):  # high, low, close
        expected = [SCALE * math.log(prices[b, col] / prices[b - 1, 3]) for b in bars]
        np.testing.assert_allclose(
            obs[1 + ch * N : 1 + (ch + 1) * N], expected, rtol=1e-4
        )
    assert obs[0] == 0.0
    assert obs.dtype == np.float32


def test_observation_is_not_clipped():
    prices = make_prices(LEAD + 400)
    prices[LEAD + 30] *= 1.5  # a genuine shock
    env = TradingEnv(prices, [(LEAD, LEAD + 400)])
    env.reset(options={"segment": 0})
    for _ in range(6):
        obs, *_ = env.step(1)
    assert obs[1 : 1 + N].max() > 3000


def test_position_is_first_observation_element_and_target_not_increment():
    env = env_for()
    env.reset(seed=0, options={"segment": 0})
    for action, pos in [(2, 1.0), (2, 1.0), (0, -1.0), (1, 0.0)]:
        obs, *_ = env.step(action)
        assert obs[0] == pos


def test_reward_is_pnl_on_position_held_minus_half_spread_per_unit():
    prices = make_prices(LEAD + 400)
    env = TradingEnv(prices, [(LEAD, LEAD + 400)])
    env.reset(options={"segment": 0})
    t = LEAD + N
    half = 1.0 / 2  # pips
    # Go long at t: held into t+1, pays half a spread to open.
    _, r, *_ = env.step(2)
    assert r == pytest.approx((prices[t + 1, 3] - prices[t, 3]) / 1e-4 - half)
    # Hold: no cost.
    _, r, *_ = env.step(2)
    assert r == pytest.approx((prices[t + 2, 3] - prices[t + 1, 3]) / 1e-4)
    # Reverse long -> short: pays a full spread.
    _, r, *_ = env.step(0)
    assert r == pytest.approx(-(prices[t + 3, 3] - prices[t + 2, 3]) / 1e-4 - 2 * half)
    # Flat: closing costs half, earns nothing.
    _, r, *_ = env.step(1)
    assert r == pytest.approx(-half)
    # Flat and staying flat is free.
    _, r, *_ = env.step(1)
    assert r == 0.0


def test_spread_pips_is_configurable():
    prices = make_prices(LEAD + 400)
    env = TradingEnv(prices, [(LEAD, LEAD + 400)], spread_pips=2.0)
    env.reset(options={"segment": 0})
    _, r, *_ = env.step(2)
    assert r == pytest.approx(
        (prices[LEAD + N + 1, 3] - prices[LEAD + N, 3]) / 1e-4 - 1.0
    )


def test_episode_is_288_steps_and_truncates():
    env = env_for(count=1300, segs=((LEAD, 1300),))
    env.reset(seed=3)
    steps, terminated, truncated, _ = run_to_end(env)
    assert steps == 288 and truncated and not terminated


def test_episodes_never_span_a_segment_boundary_and_start_randomly():
    segs = [(LEAD, 700), (700, 750), (750, 1200)]  # middle one is too short
    env = env_for(count=1200, segs=segs)
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
    env = env_for(count=1300, segs=((LEAD, 1300),))
    _, a = env.reset(seed=7)
    _, b = env.reset(seed=7)
    assert a["bar"] == b["bar"]


def test_segment_option_runs_one_segment_from_its_start_to_its_end():
    segs = [
        Segment(LEAD, LEAD + 100, None, None, None),  # ty: ignore[invalid-argument-type]
        Segment(LEAD + 100, LEAD + 300, None, None, None),  # ty: ignore[invalid-argument-type]
    ]
    env = TradingEnv(make_prices(LEAD + 300), segs)
    _, info = env.reset(options={"segment": 1})
    assert info["bar"] == LEAD + 100 + N  # warm-up of 24 Bars, first Action on Bar 25
    steps, _, truncated, info = run_to_end(env)
    assert truncated
    assert steps == 200 - N - 1  # last rewarded Bar is the Segment's last
    assert info["bar"] == LEAD + 299


def test_no_segment_long_enough_is_an_error():
    with pytest.raises(ValueError):
        env_for(count=LEAD + 100, segs=((LEAD, LEAD + 100),)).reset()


# --- Multi-horizon return Features (#183) -------------------------------------


def ref_return_block(prices, t, h):
    """Independent reference: scaled log(c_b / c_{b-h}) for the 24 Bars to t."""
    c = prices[:, 3]
    return [
        SCALE / math.sqrt(h) * math.log(c[b] / c[b - h])
        for b in range(t - N + 1, t + 1)
    ]


def return_blocks(obs):
    base = 1 + 3 * N
    return [obs[base + k * N : base + (k + 1) * N] for k in range(len(RETURN_HORIZONS))]


def obs_after(prices, segs, steps):
    env = TradingEnv(prices, segs)
    obs, _ = env.reset(options={"segment": 0})
    for _ in range(steps):
        obs, *_ = env.step(1)
    return obs, env._bar


def test_return_blocks_follow_the_raw_blocks_and_match_the_reference():
    prices = make_prices(LEAD + 400)
    obs, t = obs_after(prices, [(LEAD, LEAD + 400)], 5)
    for block, h in zip(return_blocks(obs), RETURN_HORIZONS, strict=True):
        expected = ref_return_block(prices, t, h)
        np.testing.assert_allclose(block, expected, rtol=1e-4, atol=1e-6)


def test_observation_at_t_ignores_later_bars_but_not_earlier_ones():
    prices = make_prices(LEAD + 400)
    segs = [(LEAD, LEAD + 400)]
    obs, t = obs_after(prices, segs, 3)
    later = prices.copy()
    later[t + 1 :] *= 1.3
    obs2, _ = obs_after(later, segs, 3)
    assert obs.tobytes() == obs2.tobytes()
    earlier = prices.copy()
    earlier[t - 48] *= 1.01
    obs3, _ = obs_after(earlier, segs, 3)
    assert not np.array_equal(obs, obs3)


def test_features_look_back_across_a_gap_while_the_raw_window_does_not():
    prices = make_prices(900)
    env = TradingEnv(prices, [(0, 500), (600, 900)])  # first Segment is dropped
    assert env.segments == [(600, 900)]
    obs, info = env.reset(options={"segment": 0})
    t = info["bar"]
    assert t == 600 + N
    for block, h in zip(return_blocks(obs), RETURN_HORIZONS, strict=True):
        expected = ref_return_block(prices, t, h)
        np.testing.assert_allclose(block, expected, rtol=1e-4, atol=1e-6)
    # the oldest Window Bar (601) at the longest horizon reaches Bars before the Gap
    assert 601 - 288 < 500
    # raw close block: only Bars of this Segment, first against Bar 600's close
    raw_close = [
        SCALE * math.log(prices[b, 3] / prices[b - 1, 3])
        for b in range(t - N + 1, t + 1)
    ]
    np.testing.assert_allclose(obs[1 + 2 * N : 1 + 3 * N], raw_close, rtol=1e-4)


def test_segment_without_enough_lookback_is_dropped_and_serves_as_lookback():
    prices = make_prices(700)
    env = TradingEnv(prices, [(0, 320), (330, 700)])
    assert env.segments == [(330, 700)]
    obs, info = env.reset(options={"segment": 0})
    assert info["bar"] == 330 + N
    expected = ref_return_block(prices, 330 + N, 288)
    np.testing.assert_allclose(return_blocks(obs)[2], expected, rtol=1e-4, atol=1e-6)


def test_lookback_boundary_is_the_longest_horizon_behind_the_oldest_window_bar():
    # oldest Window Bar is first + 1; it needs 288 Bars behind it: first >= 287
    assert TradingEnv(make_prices(800), [(286, 800)]).segments == []
    assert TradingEnv(make_prices(800), [(287, 800)]).segments == [(287, 800)]


def test_no_usable_segment_raises_the_existing_error():
    with pytest.raises(ValueError):
        TradingEnv(make_prices(400), [(0, 400)]).reset()


def test_a_shock_gives_large_unclipped_return_features():
    prices = make_prices(LEAD + 400)
    prices[LEAD + 6 :] *= 1.5  # permanent 50% jump at Bar LEAD + 6
    obs, _ = obs_after(prices, [(LEAD, LEAD + 400)], 3)
    assert return_blocks(obs)[0].max() > 500  # 1e4 * ln(1.5) / sqrt(48) = 585
