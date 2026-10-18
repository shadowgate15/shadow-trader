"""Observation assembly reference cases (I=1), hand-computed.

Default config: EURUSD, 10,000 USD, leverage cap 10, slippage 1e-5. At 1.1000 a
+1 level is floor(10_000 * 10 / 1.1) = 90_909 units.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shadow_trader.environment import (
    EngineConfig,
    Lot,
    Market,
    Observation,
    ObservationConfig,
    build_observation,
    initial_state,
    step,
)
from shadow_trader.environment import observation as observation_module

CONFIG = EngineConfig()
MARKET = Market(
    open_bid=1.1000,
    open_spread=0.0001,
    close_bid=1.1000,
    close_spread=0.0001,
    bars_to_session_final=1439,
)


def window(rows: list[list[float]]) -> np.ndarray:
    return np.array([rows], dtype=np.float64)  # [I=1, W, F]


def observe(state, market=MARKET, rows=((0.5, -0.5),), valid=((True,),), config=CONFIG):
    obs_config = ObservationConfig(window_bars=len(rows))
    return build_observation(
        window([list(r) for r in rows]),
        np.array(valid, dtype=bool),
        state,
        market,
        config,
        obs_config,
    )


def test_flat_account_at_session_open() -> None:
    obs = observe(initial_state(CONFIG))

    assert isinstance(obs, Observation)
    assert obs.features.shape == (1, 1, 2)
    assert obs.features[0, 0].tolist() == [0.5, -0.5]
    assert obs.valid.tolist() == [[True]]
    assert obs.position.tolist() == [[0.0, 0.0, 0.0]]
    assert obs.account.tolist() == [0.0, 1.0]  # no margin used; Session just opened


def test_long_position_pending_target_and_unrealized_return() -> None:
    market = replace(MARKET, close_bid=1.1011)
    state, _ = step(initial_state(CONFIG), CONFIG, market, 1.0)
    # filled at 1.1000 + 0.0001 + 0.00001 = 1.10011; marked at close bid 1.1011
    obs = observe(state, market)

    position_frac, pending_target, unrealized = obs.position[0].tolist()
    # max units at equity ~ 10_000 + 90_909 * (1.1011 - 1.10011) = 10_008.9999
    max_units = math.floor(state.equity * 10 / 1.1011)
    assert position_frac == pytest.approx(np.float32(90_909 / max_units), rel=1e-6)
    assert pending_target == 1.0  # the target when nothing is pending
    assert unrealized == pytest.approx((1.1011 - 1.10011) / 1.10011, rel=1e-5)
    margin_frac = 90_909 * 1.1011 * 0.02 / state.equity
    assert obs.account[0] == pytest.approx(margin_frac, rel=1e-6)


def test_short_position_unrealized_return_uses_the_ask_mark() -> None:
    market = replace(MARKET, close_bid=1.0990)
    state, _ = step(initial_state(CONFIG), CONFIG, market, -1.0)
    # sold at 1.1000 - 0.00001 = 1.09999; marked at close ask 1.0990 + 0.0001
    obs = observe(state, market)

    position_frac, _, unrealized = obs.position[0].tolist()
    assert position_frac < 0
    assert unrealized == pytest.approx((1.09999 - 1.0991) / 1.09999, rel=1e-5)


def test_pending_target_differs_from_position_before_the_fill() -> None:
    state = replace(initial_state(CONFIG), target_levels=(1.0,), target_units=(90_909,))
    obs = observe(state)
    assert obs.position[0, 0] == 0.0
    assert obs.position[0, 1] == 1.0


def test_position_frac_is_safety_clipped_at_five() -> None:
    state = replace(
        initial_state(CONFIG),
        positions=(10_000_000,),
        lots=((Lot(10_000_000, 1.1),),),
    )
    obs = observe(state)
    assert obs.position[0, 0] == 5.0
    negative = replace(state, positions=(-10_000_000,))
    assert observe(negative).position[0, 0] == -5.0


@pytest.mark.parametrize(
    ("bars_left", "expected"),
    [(1439, 1.0), (0, 0.0), (1000, 1000 / 1439), (5000, 1.0)],
)
def test_session_progress_runs_from_one_at_open_to_zero_on_final_bar(
    bars_left: int, expected: float
) -> None:
    market = replace(MARKET, bars_to_session_final=bars_left)
    obs = observe(initial_state(CONFIG), market)
    assert obs.account[1] == pytest.approx(expected)


def test_output_is_float32_and_finite() -> None:
    state, _ = step(initial_state(CONFIG), CONFIG, MARKET, 1.0)
    obs = observe(state)
    assert obs.features.dtype == np.float32
    assert obs.position.dtype == np.float32
    assert obs.account.dtype == np.float32
    assert obs.valid.dtype == np.bool_
    for array in (obs.features, obs.position, obs.account):
        assert np.isfinite(array).all()


def test_null_in_the_current_row_raises() -> None:
    with pytest.raises(ValueError, match="current row"):
        observe(initial_state(CONFIG), rows=((0.5, math.nan),))


def test_infinite_feature_raises() -> None:
    with pytest.raises(ValueError, match="current row"):
        observe(initial_state(CONFIG), rows=((0.5, math.inf),))


def test_older_nulls_are_zero_filled_and_masked() -> None:
    obs = observe(
        initial_state(CONFIG),
        rows=((math.nan, 0.25), (1.0, 2.0)),
        valid=((True, True),),
    )
    assert obs.features[0, 0].tolist() == [0.0, 0.0]
    assert obs.features[0, 1].tolist() == [1.0, 2.0]
    assert obs.valid.tolist() == [[False, True]]


def test_input_valid_mask_is_preserved() -> None:
    obs = observe(
        initial_state(CONFIG), rows=((1.0, 1.0), (2.0, 2.0)), valid=((False, True),)
    )
    assert obs.valid.tolist() == [[False, True]]


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        build_observation(
            window([[0.0, 0.0]]),
            np.array([[True]]),
            initial_state(CONFIG),
            MARKET,
            CONFIG,
            ObservationConfig(window_bars=3),
        )


def test_observation_does_not_depend_on_absolute_equity() -> None:
    rich = replace(CONFIG, starting_equity=1_000_000.0)
    a = observe(step(initial_state(CONFIG), CONFIG, MARKET, 1.0)[0])
    b = observe(step(initial_state(rich), rich, MARKET, 1.0)[0], config=rich)
    assert a.position[0, 0] == pytest.approx(b.position[0, 0], rel=1e-3)
    assert a.account[0] == pytest.approx(b.account[0], rel=1e-3)


def test_observation_module_never_imports_gymnasium() -> None:
    tree = ast.parse(Path(observation_module.__file__).read_text())
    names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "gymnasium" not in names
