"""Observation windows: window_inputs, the validity mask and W>1 in the env."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from shadow_trader.environment import make_env
from shadow_trader.environment.env_config import EnvConfig, EpisodeConfig
from shadow_trader.environment.market_data import from_prepared
from shadow_trader.environment.observation import ObservationConfig, window_inputs
from tests.environment.fixture import load_fixture

NAN = float("nan")


def test_full_window_is_returned_oldest_to_newest_all_valid() -> None:
    rows = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    features, valid = window_inputs(rows, ObservationConfig(window_bars=3))
    assert features.shape == (1, 3, 2) and features.dtype == np.float32
    assert features[0, 0].tolist() == [1.0, 2.0]
    assert features[0, 2].tolist() == [5.0, 6.0]
    assert valid.tolist() == [[True, True, True]]


def test_older_nulls_are_zero_filled_and_masked() -> None:
    rows = [[[1.0, 2.0], [NAN, 4.0], [5.0, 6.0]]]
    features, valid = window_inputs(rows, ObservationConfig(window_bars=3))
    assert features[0, 1].tolist() == [0.0, 0.0]
    assert valid.tolist() == [[True, False, True]]


def test_null_current_row_raises() -> None:
    rows = [[[1.0, 2.0], [5.0, NAN]]]
    with pytest.raises(ValueError, match="current row"):
        window_inputs(rows, ObservationConfig(window_bars=2))


def test_session_start_boundary_pads_before_the_first_row() -> None:
    rows = [[[7.0, 8.0], [9.0, 10.0]]]  # only 2 rows exist in the Session
    features, valid = window_inputs(rows, ObservationConfig(window_bars=4))
    assert features.shape == (1, 4, 2)
    assert features[0, :2].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert features[0, 2:].tolist() == [[7.0, 8.0], [9.0, 10.0]]
    assert valid.tolist() == [[False, False, True, True]]


def test_w1_is_the_degenerate_case_with_the_same_rank() -> None:
    features, valid = window_inputs([[[1.0, 2.0]]], ObservationConfig(window_bars=1))
    assert features.shape == (1, 1, 2) and valid.tolist() == [[True]]


def test_more_rows_than_window_keeps_the_newest() -> None:
    rows = [[[1.0], [2.0], [3.0]]]
    features, _ = window_inputs(rows, ObservationConfig(window_bars=2))
    assert features[0, :, 0].tolist() == [2.0, 3.0]


def test_env_observation_matches_declared_space_for_w_gt_1() -> None:
    dataset, scaler = load_fixture()
    config = EnvConfig(
        episode=EpisodeConfig(window_bars=20),
        observation=ObservationConfig(window_bars=6),
    )
    env = make_env(dataset, scaler, config, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs["features"].shape[1] == 6
    assert obs["valid"].shape == (1, 6)
    assert env.observation_space.contains(obs)
    check_env(env.unwrapped, skip_render_check=True)


def test_market_data_window_never_reaches_before_the_session_start() -> None:
    dataset, scaler = load_fixture()
    data = from_prepared(dataset, scaler, "EURUSD", 5)
    start = int(data.session_start[1])
    first = start + int(data.usable[start : data.session_end[1]].argmax())
    offset = first - start  # warm-up Bars with null features, still in-Session
    features, valid = data.window(first, offset + 3)
    assert valid.tolist() == [[False] * (offset + 2) + [True]]
    assert not features[0, :-1].any()
    assert features[0, -1].tolist() == pytest.approx(data.features[first].tolist())
