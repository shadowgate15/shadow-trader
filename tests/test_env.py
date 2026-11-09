import math

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from shadow_trader import config
from shadow_trader.env import (
    FAMILIES,
    FEATURE_COUNT,
    LOOKBACK,
    TradingEnv,
    _volatility_series,
)
from shadow_trader.segments import Segment

N = 24
SCALE = 1e4
LEAD = LOOKBACK + 12  # leading history: more than the lookback a Segment needs
RETURN_HORIZONS = (48, 144, 288)
SMA_SPANS = (48, 288)
RANGE_HORIZONS = (48, 288)


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
        -np.inf, np.inf, ((2 + 3 * N + FEATURE_COUNT * N,)), np.float32
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
            obs[2 + ch * N : 2 + (ch + 1) * N], expected, rtol=1e-4
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
    assert obs[2 : 2 + N].max() > 3000


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


def spread_env(spread_by_bar: dict[int, float], **kw):
    """Env whose close Spread (pips) is ``spread_by_bar`` and zero elsewhere."""
    prices = make_prices(LEAD + 400)
    spread = np.zeros(len(prices))
    for bar, pips in spread_by_bar.items():
        spread[bar] = pips
    env = TradingEnv(prices, [(LEAD, LEAD + 400)], close_spread_pips=spread, **kw)
    env.reset(options={"segment": 0})
    return prices, env


@pytest.mark.parametrize(
    ("spread", "cost"),
    [(0.0, 1.0), (-3.0, 1.0), (0.4, 1.0), (1.0, 1.0), (4.0, 4.0), (15.0, 15.0)],
)
def test_cost_is_the_floor_unless_the_spread_is_wider(spread, cost):
    t = LEAD + N
    prices, env = spread_env({t: spread})
    _, r, *_ = env.step(2)  # open: half the cost
    assert r == pytest.approx((prices[t + 1, 3] - prices[t, 3]) / 1e-4 - cost / 2)


def test_the_decision_bars_spread_is_charged_not_the_next_bars():
    t = LEAD + N
    prices, env = spread_env({t: 6.0, t + 1: 20.0})
    _, r, *_ = env.step(2)
    assert r == pytest.approx((prices[t + 1, 3] - prices[t, 3]) / 1e-4 - 3.0)


def test_open_pays_half_and_reversal_pays_the_full_decision_bar_cost():
    t = LEAD + N
    prices, env = spread_env({t: 4.0, t + 1: 6.0, t + 2: 8.0})
    env.step(2)  # open at t: pays 2.0
    _, r, *_ = env.step(2)  # hold at t+1: free
    assert r == pytest.approx((prices[t + 2, 3] - prices[t + 1, 3]) / 1e-4)
    _, r, *_ = env.step(0)  # reverse at t+2: two units of half of 8
    assert r == pytest.approx(-(prices[t + 3, 3] - prices[t + 2, 3]) / 1e-4 - 8.0)


def test_spread_pips_is_the_floor_for_the_spread_data():
    t = LEAD + N
    prices, env = spread_env({t: 1.5}, spread_pips=2.0)
    _, r, *_ = env.step(2)
    assert r == pytest.approx((prices[t + 1, 3] - prices[t, 3]) / 1e-4 - 1.0)


@pytest.mark.parametrize(
    ("spread", "cost"),
    [(0.0, 1.0), (-3.0, 1.0), (0.4, 1.0), (1.0, 1.0), (4.0, 4.0), (15.0, 15.0)],
)
def test_observation_second_value_is_the_decision_bars_floored_cost_in_pips(
    spread, cost
):
    t = LEAD + N
    _, env = spread_env({t: spread, t + 1: 7.5})
    obs, _ = env.reset(options={"segment": 0})
    assert obs[0] == 0.0
    assert obs[1] == np.float32(cost)
    # After a step the Observation is of Bar t+1: its cost, not Bar t's.
    obs, *_ = env.step(2)
    assert obs[0] == 1.0
    assert obs[1] == np.float32(7.5)


def test_observation_cost_floors_at_spread_pips_and_without_spread_data():
    t = LEAD + N
    _, env = spread_env({t: 1.5}, spread_pips=2.0)
    obs, _ = env.reset(options={"segment": 0})
    assert obs[1] == np.float32(2.0)
    obs, _ = env_for(spread_pips=3.0).reset(options={"segment": 0})
    assert obs[1] == np.float32(3.0)


def test_observation_cost_is_unclipped_and_adds_no_window_of_costs():
    t = LEAD + N
    _, env = spread_env({t: 5000.0})
    obs, _ = env.reset(options={"segment": 0})
    assert obs[1] == np.float32(5000.0)
    assert obs.shape == (2 + (3 + FEATURE_COUNT) * N,)


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
    base = 2 + 3 * N
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
    np.testing.assert_allclose(obs[2 + 2 * N : 2 + 3 * N], raw_close, rtol=1e-4)


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


# --- SMA distance Features (#184) ---------------------------------------------


def ref_sma_block(prices, t, n):
    """Independent reference: scaled log(c_b / mean(c[b-n+1..b])) for 24 Bars to t."""
    c = prices[:, 3]
    return [
        SCALE / math.sqrt(n) * math.log(c[b] / (sum(c[b - n + 1 : b + 1]) / n))
        for b in range(t - N + 1, t + 1)
    ]


def sma_blocks(obs):
    base = 2 + 3 * N + len(RETURN_HORIZONS) * N
    return [obs[base + k * N : base + (k + 1) * N] for k in range(len(SMA_SPANS))]


def test_sma_blocks_follow_the_return_blocks_and_match_the_reference():
    prices = make_prices(LEAD + 400)
    obs, t = obs_after(prices, [(LEAD, LEAD + 400)], 5)
    for block, n in zip(sma_blocks(obs), SMA_SPANS, strict=True):
        np.testing.assert_allclose(
            block, ref_sma_block(prices, t, n), rtol=1e-3, atol=1e-5
        )


def test_sma_features_look_back_across_a_gap_and_ignore_the_future():
    prices = make_prices(900)
    env = TradingEnv(prices, [(0, 500), (600, 900)])
    obs, info = env.reset(options={"segment": 0})
    t = info["bar"]
    for block, n in zip(sma_blocks(obs), SMA_SPANS, strict=True):
        np.testing.assert_allclose(
            block, ref_sma_block(prices, t, n), rtol=1e-3, atol=1e-5
        )
    later = prices.copy()
    later[t + 1 :] *= 1.3
    obs2, _ = TradingEnv(later, [(0, 500), (600, 900)]).reset(options={"segment": 0})
    assert obs.tobytes() == obs2.tobytes()


def test_observation_width_counts_the_sma_blocks():
    env = env_for()
    expected = 2 + 3 * N + FEATURE_COUNT * N
    assert env.observation_space.shape == (expected,)
    check_env(env, skip_render_check=True)


# --- Volatility Features (#185) -----------------------------------------------


def volatility_blocks(obs):
    """The volatility Feature blocks, located via FAMILIES so other families
    may sit before or after them."""
    k = 0
    for family in FAMILIES:
        if family.precompute is _volatility_series:
            base = 2 + 3 * N + k * N
            return [
                obs[base + i * N : base + (i + 1) * N]
                for i in range(len(family.horizons))
            ]
        k += len(family.horizons)
    raise AssertionError("no volatility family")


def ref_volatility_block(prices, t, n):
    """Independent reference: scaled sample stdev (ddof=1) of the n one-Bar
    close log returns ending at each of the 24 Bars to t."""
    logc = np.log(prices[:, 3])
    return [
        SCALE * np.std(np.diff(logc[b - n : b + 1]), ddof=1)
        for b in range(t - N + 1, t + 1)
    ]


def test_volatility_blocks_match_the_reference_at_all_positions():
    prices = make_prices(LEAD + 400)
    obs, t = obs_after(prices, [(LEAD, LEAD + 400)], 5)
    for block, n in zip(
        volatility_blocks(obs), config.VOLATILITY_HORIZONS, strict=True
    ):
        np.testing.assert_allclose(
            block, ref_volatility_block(prices, t, n), rtol=1e-4, atol=1e-6
        )


def test_constant_price_gives_zero_finite_volatility():
    prices = np.full((LEAD + 400, 4), 1.1)
    obs, _ = obs_after(prices, [(LEAD, LEAD + 400)], 3)
    for block in volatility_blocks(obs):
        assert np.all(np.isfinite(block))
        assert np.all(block == 0)


def test_volatility_looks_back_across_a_gap_and_includes_its_jump():
    prices = make_prices(900)
    prices[560:] *= 1.02  # the Gap's price jump, inside every lookback window
    env = TradingEnv(prices, [(0, 500), (600, 900)])
    obs, info = env.reset(options={"segment": 0})
    t = info["bar"]
    for block, n in zip(
        volatility_blocks(obs), config.VOLATILITY_HORIZONS, strict=True
    ):
        np.testing.assert_allclose(
            block, ref_volatility_block(prices, t, n), rtol=1e-4, atol=1e-6
        )
    # 1e4 * ln(1.02) / sqrt(288) = 11.7, against ~1 without the jump
    assert volatility_blocks(obs)[1].min() > 5


def test_volatility_at_t_ignores_later_bars():
    prices = make_prices(LEAD + 400)
    segs = [(LEAD, LEAD + 400)]
    obs, t = obs_after(prices, segs, 3)
    later = prices.copy()
    later[t + 1 :] *= 1.3
    obs2, _ = obs_after(later, segs, 3)
    assert obs.tobytes() == obs2.tobytes()


# --- Range position Features (#186) -------------------------------------------


def ref_range_block(prices, t, n):
    """Independent reference: (c - min l) / (max h - min l) over the n Bars to b."""
    out = []
    for b in range(t - N + 1, t + 1):
        lo = min(prices[i, 2] for i in range(b - n + 1, b + 1))
        hi = max(prices[i, 1] for i in range(b - n + 1, b + 1))
        out.append(0.5 if hi == lo else (prices[b, 3] - lo) / (hi - lo))
    return out


def range_blocks(obs):
    """The range blocks are last in FAMILIES order: 48 then 288."""
    return [obs[len(obs) - 2 * N : len(obs) - N], obs[len(obs) - N :]]


def test_range_blocks_match_the_reference_and_stay_within_zero_to_one():
    prices = make_prices(LEAD + 400)
    for steps in (0, 5, 100):
        obs, t = obs_after(prices, [(LEAD, LEAD + 400)], steps)
        for block, n in zip(range_blocks(obs), RANGE_HORIZONS, strict=True):
            np.testing.assert_allclose(
                block, ref_range_block(prices, t, n), rtol=1e-4, atol=1e-6
            )
            assert block.min() >= 0.0 and block.max() <= 1.0


def test_constant_price_stretch_reads_exactly_half_and_is_finite():
    prices = make_prices(LEAD + 400)
    prices[LEAD - 300 :] = 1.2  # every Bar flat, so every range is flat
    obs, _ = obs_after(prices, [(LEAD, LEAD + 400)], 3)
    for block in range_blocks(obs):
        assert np.all(np.isfinite(block))
        assert np.all(block == np.float32(0.5))


def test_range_blocks_look_back_across_a_gap():
    prices = make_prices(900)
    env = TradingEnv(prices, [(0, 500), (600, 900)])
    obs, info = env.reset(options={"segment": 0})
    for block, n in zip(range_blocks(obs), RANGE_HORIZONS, strict=True):
        expected = ref_range_block(prices, info["bar"], n)
        np.testing.assert_allclose(block, expected, rtol=1e-4, atol=1e-6)


def test_range_blocks_ignore_later_bars_but_not_earlier_ones():
    prices = make_prices(LEAD + 400)
    segs = [(LEAD, LEAD + 400)]
    obs, t = obs_after(prices, segs, 3)
    later = prices.copy()
    later[t + 1 :] *= 1.3
    obs2, _ = obs_after(later, segs, 3)
    assert obs[-2 * N :].tobytes() == obs2[-2 * N :].tobytes()
    earlier = prices.copy()
    earlier[t - 100, 1] += 0.05  # a high 100 Bars back lies inside the 288 range only
    obs3, _ = obs_after(earlier, segs, 3)
    assert not np.array_equal(range_blocks(obs)[1], range_blocks(obs3)[1])
    assert np.array_equal(range_blocks(obs)[0], range_blocks(obs3)[0])


def test_lookback_continues_through_a_previous_segment_shorter_than_the_horizon():
    prices = make_prices(900)
    segs = [(0, 200), (210, 310), (320, 900)]  # middle one is far shorter than 288
    env = TradingEnv(prices, segs)
    assert env.segments == [(320, 900)]
    obs, info = env.reset(options={"segment": 0})
    t = info["bar"]
    assert t - 288 < 200  # reaches past the short Segment into the earlier one
    for block, h in zip(return_blocks(obs), RETURN_HORIZONS, strict=True):
        np.testing.assert_allclose(
            block, ref_return_block(prices, t, h), rtol=1e-4, atol=1e-6
        )


def test_observation_width_is_pinned_and_float32():
    env = env_for()
    assert env.observation_space.shape == (290,)
    assert env.observation_space.dtype == np.float32
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(options={"segment": 0})
    assert obs.shape == (290,)
    assert obs.dtype == np.float32


def test_every_feature_value_is_finite_at_every_usable_segment_start():
    prices = make_prices(2000)
    segs = [(LEAD, 900), (1000, 2000)]
    env = TradingEnv(prices, segs)
    assert len(env.segments) == 2
    for i in range(len(env.segments)):
        obs, _ = env.reset(options={"segment": i})
        assert np.isfinite(obs).all()
        for _ in range(5):
            obs, *_ = env.step(1)
            assert np.isfinite(obs).all()
