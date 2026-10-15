"""Finite-memory feature families, checked against hand-worked values.

Expected values are worked by hand from the closed forms in each test, never
recomputed with the implementation's own formula.
"""

import math
from dataclasses import replace
from datetime import timedelta

import pytest

from shadow_trader.ingestion.bars import gap_bar
from shadow_trader.preparation import SPECS

from .synthetic import (
    SESSION_1,
    prepared_frame,
    priced,
    write_ingestion,
    write_prepared_toml,
)
from .test_cli import build_in_subprocess

UP = 0.01  # alternating log returns of +-UP: closes flip between 1.0 and e**UP
N = 300


def alternating(count=N, **overrides):
    """One Session whose 1-Bar log returns alternate +UP, -UP, +UP, ..."""
    bars = []
    for i in range(count):
        close = 1.0 if i % 2 == 0 else math.exp(UP)
        bar = priced("EURUSD", SESSION_1 + timedelta(minutes=5 * i), SESSION_1, i)
        bars.append(
            replace(bar, open=close, high=close, low=close, close=close, **overrides)
        )
    return bars


def test_volatility_is_the_rolling_std_of_one_bar_log_returns(tmp_path):
    frame = prepared_frame(tmp_path, alternating())

    # n alternating +-UP returns have mean 0 (n even) so sample std is
    # UP * sqrt(n / (n - 1)); n = 3 starting with + is UP * sqrt(4 / 3).
    for horizon in (6, 12, 24, 48, 288):
        expected = UP * math.sqrt(horizon / (horizon - 1))
        column = frame[f"feat_vol_{horizon}"]
        assert column[horizon - 1] is None  # needs `horizon` returns, first is at 1
        assert math.isclose(column[horizon], expected, rel_tol=1e-9)
        assert math.isclose(column[N - 1], expected, rel_tol=1e-9)
    assert math.isclose(frame["feat_vol_3"][3], UP * math.sqrt(4 / 3), rel_tol=1e-9)
    assert "feat_vol_1" not in frame.columns  # a std over one return is undefined


def test_spec_v2_declares_lookback_for_each_new_column():
    spec = SPECS[2]
    lookback = {c.name: c.lookback for c in spec.columns}

    assert lookback["feat_vol_3"] == 3
    assert lookback["feat_vol_288"] == 288
    assert spec.version == 2 and spec.hash != SPECS[1].hash
    assert SPECS[1].names == tuple(f"feat_ret_{h}" for h in (1, 3, 6, 12, 24, 48, 288))
    with pytest.raises(KeyError):
        SPECS[4]


def test_spec_hashes_are_pinned_so_a_column_change_needs_a_version_bump():
    # Update these only together with a new spec version (and its goldens).
    assert SPECS[1].hash == (
        "e9680f6f3cf48f772ea0889e4a5d58737eb481209c8da1cfc377050860ce981b"
    )
    assert SPECS[2].hash == (
        "16d17ab3cd78190297595e44cfc8b1354a81f7bd52fa49ebc697254b520b1bba"
    )
    assert len(SPECS[2].names) == 26 and len(set(SPECS[2].names)) == 26


def test_each_column_recovers_after_a_gap_by_its_own_lookback(tmp_path):
    bars = alternating()
    bars[100] = gap_bar("EURUSD", bars[100].open_time, SESSION_1)
    frame = prepared_frame(tmp_path, bars)

    def first_valid(name):
        return next(i for i in range(101, N) if frame[name][i] is not None)

    assert frame["feat_vol_3"][103] is None and first_valid("feat_vol_3") == 104
    assert first_valid("feat_vol_48") == 149
    assert first_valid("feat_spr_mean") == 148  # 47 Bars of lookback
    assert first_valid("feat_spr_close") == 101  # own Bar only
    assert all(frame[c][100] is not None for c in frame.columns if "time" in c)
    assert frame["feat_vol_288"].slice(101).null_count() == N - 101  # never recovers


def test_finite_memory_dataset_hash_is_identical_at_one_and_many_threads(tmp_path):
    parent = write_ingestion(tmp_path, gaps=(10,), flags={20: ("price_spike",)})
    ids = []
    for name, threads in (("one", 1), ("many", 4)):
        path = write_prepared_toml(tmp_path, parent.dataset_id)
        text = path.read_text().replace("version = 1", "version = 2")
        path.write_text(text.replace('/prepared"', f'/prepared_{name}"'))
        ids.append(build_in_subprocess(path, threads))

    assert ids[0] == ids[1]


def shaped(count, ohlc):
    """One Session whose Bar ``i`` has (open, high, low, close) = ``ohlc(i)``."""
    bars = []
    for i in range(count):
        o, h, low, c = ohlc(i)
        bar = priced("EURUSD", SESSION_1 + timedelta(minutes=5 * i), SESSION_1, i)
        bars.append(replace(bar, open=o, high=h, low=low, close=c))
    return bars


def test_intrabar_shape_is_scaled_by_the_average_true_range(tmp_path):
    # Every Bar: open 1.00, high 1.03, low 0.99, close 1.01. Its previous close
    # (1.01) is inside the range, so true range = 0.04 and the 14-Bar ATR is 0.04.
    frame = prepared_frame(tmp_path, shaped(20, lambda i: (1.0, 1.03, 0.99, 1.01)))

    expected = {
        "feat_shape_body": 0.01 / 0.04,
        "feat_shape_range": 0.04 / 0.04,
        "feat_shape_wick_up": 0.02 / 0.04,
        "feat_shape_wick_dn": 0.01 / 0.04,
    }
    for name, value in expected.items():
        assert frame[name][13] is None  # 14 true ranges first exist at Bar 14
        assert math.isclose(frame[name][14], value, rel_tol=1e-9), name
        assert math.isclose(frame[name][19], value, rel_tol=1e-9), name


def test_true_range_includes_the_jump_from_the_previous_close(tmp_path):
    # Each Bar closes 0.03 above the last: range 0.01, |high - prev close| 0.04,
    # |low - prev close| 0.03, so true range (and ATR) is 0.04, not 0.01.
    frame = prepared_frame(
        tmp_path,
        shaped(
            20,
            lambda i: (
                1 + 0.03 * i,
                1 + 0.03 * i + 0.01,
                1 + 0.03 * i,
                1 + 0.03 * i,
            ),
        ),
    )

    assert math.isclose(frame["feat_shape_range"][19], 0.01 / 0.04, rel_tol=1e-9)
    assert math.isclose(frame["feat_shape_wick_up"][19], 0.01 / 0.04, rel_tol=1e-9)
    assert frame["feat_shape_wick_dn"][19] == 0.0
    assert frame["feat_shape_body"][19] == 0.0


def with_fields(count, **fields):
    """``count`` Bars at close 2.0; each field is a callable of the Bar index."""
    bars = []
    for i in range(count):
        bar = priced("EURUSD", SESSION_1 + timedelta(minutes=5 * i), SESSION_1, i)
        values = {k: f(i) for k, f in fields.items()}
        bars.append(replace(bar, open=2.0, high=2.0, low=2.0, close=2.0, **values))
    return bars


def test_spread_is_relative_to_price_and_to_its_rolling_median(tmp_path):
    # spread_mean is 0.0002 except 0.0004 at Bar 55, so against the median of
    # the last 48 Bars (0.0002) it reads 2.0 there and 1.0 elsewhere.
    frame = prepared_frame(
        tmp_path,
        with_fields(
            60,
            spread_close=lambda i: 0.0004,
            spread_mean=lambda i: 0.0004 if i == 55 else 0.0002,
        ),
    )

    assert math.isclose(frame["feat_spr_close"][10], 0.0004 / 2.0, rel_tol=1e-9)
    assert frame["feat_spr_mean"][46] is None  # 48 values first exist at Bar 47
    assert frame["feat_spr_mean"][47] == 1.0
    assert frame["feat_spr_mean"][54] == 1.0
    assert frame["feat_spr_mean"][55] == 2.0


def test_zero_median_spread_is_null_not_infinite(tmp_path):
    frame = prepared_frame(
        tmp_path, with_fields(60, spread_close=lambda i: 0.0, spread_mean=lambda i: 0.0)
    )

    assert frame["feat_spr_mean"].null_count() == 60
    assert frame["feat_spr_close"].null_count() == 0  # 0 / 2.0 is a real zero


def test_activity_is_tick_count_relative_to_its_rolling_median(tmp_path):
    frame = prepared_frame(
        tmp_path,
        with_fields(60, tick_count=lambda i: 300 if i == 55 else 100),
    )

    assert frame["feat_act_tick"][46] is None
    assert frame["feat_act_tick"][47] == 1.0
    assert frame["feat_act_tick"][55] == 3.0


def test_activity_is_null_wherever_tick_count_is_null(tmp_path):
    frame = prepared_frame(
        tmp_path,
        with_fields(60, tick_count=lambda i: None if i == 56 else 100),
    )

    assert frame["feat_act_tick"][55] == 1.0
    assert frame["feat_act_tick"][56] is None
    assert frame["feat_act_tick"].is_nan().sum() == 0


def test_lookback_zero_columns_still_go_null_on_a_distrusted_bar(tmp_path):
    bars = with_fields(60, spread_close=lambda i: 0.0004)
    bars[30] = replace(bars[30], quality_flags=("price_spike",))
    frame = prepared_frame(tmp_path, bars)

    assert frame["feat_spr_close"][30] is None
    assert frame["feat_spr_close"][29] is not None
    assert frame["feat_spr_close"][31] is not None  # lookback 0: own Bar only


def test_time_features_come_from_open_time_and_session_open(tmp_path):
    # SESSION_1 opens Sunday 2024-01-07 22:00 UTC. Hour-of-day is fractional
    # (22:30 = 22.5 h) and the angles are 2*pi*hour/24 and 2*pi*weekday/7 with
    # Monday = 0, so Sunday is 6.
    bars = with_fields(30, is_rollover=lambda i: i == 2)
    frame = prepared_frame(tmp_path, bars)
    close = {"rel_tol": 1e-12, "abs_tol": 1e-12}

    first, half = frame.row(0, named=True), frame.row(6, named=True)
    assert math.isclose(first["feat_time_hour_sin"], -0.5, **close)  # 22 h = 11pi/6
    assert math.isclose(first["feat_time_hour_cos"], math.sqrt(3) / 2, **close)
    assert math.isclose(first["feat_time_dow_sin"], -0.7818314824680298, **close)
    assert math.isclose(first["feat_time_dow_cos"], 0.6234898018587336, **close)
    assert first["feat_time_since_open"] == 0.0
    assert math.isclose(half["feat_time_hour_sin"], -0.3826834323650898, **close)
    assert math.isclose(half["feat_time_hour_cos"], 0.9238795325112867, **close)
    assert half["feat_time_since_open"] == 30.0
    assert [frame["feat_time_rollover"][i] for i in (1, 2, 3)] == [0.0, 1.0, 0.0]

    monday = frame.row(24, named=True)  # 2024-01-08 00:00 UTC
    assert math.isclose(monday["feat_time_hour_sin"], 0.0, **close)
    assert math.isclose(monday["feat_time_hour_cos"], 1.0, **close)
    assert math.isclose(monday["feat_time_dow_sin"], 0.0, **close)
    assert math.isclose(monday["feat_time_dow_cos"], 1.0, **close)
    assert monday["feat_time_since_open"] == 120.0


def test_time_features_are_valid_on_gap_and_flagged_bars(tmp_path):
    bars = with_fields(30)
    bars[3] = gap_bar("EURUSD", bars[3].open_time, SESSION_1)
    bars[4] = replace(bars[4], quality_flags=("crossed_quote",))
    frame = prepared_frame(tmp_path, bars)

    time_columns = [c for c in frame.columns if c.startswith("feat_time_")]
    assert len(time_columns) == 6
    assert all(frame[c].null_count() == 0 for c in time_columns)
    assert frame["feat_time_since_open"][3] == 15.0
    assert frame["feat_spr_close"][3] is None  # the Gap's other features are null


def test_flat_bars_have_no_atr_scale_so_shape_is_null_not_nan(tmp_path):
    frame = prepared_frame(tmp_path, shaped(20, lambda i: (1.0, 1.0, 1.0, 1.0)))

    assert frame["feat_shape_range"].null_count() == 20
    assert frame["feat_shape_range"].is_nan().sum() == 0
