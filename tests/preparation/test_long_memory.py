"""Long-memory families (spec v3), checked against hand-worked values.

Conventions under test: EMA seeded with the SMA of the first ``span`` closes in
a Session; Wilder smoothing (alpha = 1/n) for ATR and RSI, seeded with the plain
mean of the first ``n`` true ranges / price changes. Expected values are worked
by hand in each test, never recomputed with the implementation's formula.
"""

import math
from dataclasses import replace
from datetime import timedelta

import pytest

from shadow_trader.ingestion.bars import gap_bar
from shadow_trader.preparation import SPECS

from .synthetic import SESSION_1, prepared_frame, priced

V3 = 3


def shaped(count, ohlc):
    """One Session whose Bar ``i`` has (open, high, low, close) = ``ohlc(i)``."""
    bars = []
    for i in range(count):
        o, h, low, c = ohlc(i)
        bar = priced("EURUSD", SESSION_1 + timedelta(minutes=5 * i), SESSION_1, i)
        bars.append(replace(bar, open=o, high=h, low=low, close=c))
    return bars


def ramp(i):
    c = 100.0 + i
    return (c, c + 1, c - 1, c)


def test_spec_v3_declares_warm_up_and_tolerance_for_each_long_memory_column():
    spec = SPECS[V3]
    by_name = {c.name: c for c in spec.columns}

    long_memory = [
        "feat_trend_ema_12",
        "feat_trend_ema_24",
        "feat_trend_rsi_14",
        "feat_shape_body",
        "feat_shape_range",
        "feat_shape_wick_up",
        "feat_shape_wick_dn",
    ]
    for name in long_memory:
        assert by_name[name].tolerance == 1e-9, name
    # weight of an input older than the lookback is below 1e-9
    # EMA distance is in ATR units, so it inherits the ATR's longer memory
    assert by_name["feat_trend_ema_12"].lookback == 280
    assert by_name["feat_trend_ema_24"].lookback == 280
    assert by_name["feat_trend_rsi_14"].lookback == 280
    assert by_name["feat_shape_range"].lookback == 280
    # finite-memory columns are bitwise
    assert by_name["feat_ret_288"].tolerance == 0.0
    assert by_name["feat_vol_12"].tolerance == 0.0
    assert len(spec.names) == 29 and len(set(spec.names)) == 29


def test_longest_lookback_is_readable_from_the_spec_for_embargo_sizing():
    assert SPECS[V3].longest_lookback == 288
    assert SPECS[2].longest_lookback == 288


def test_spec_v3_hash_is_pinned_and_prior_versions_are_untouched():
    assert SPECS[V3].version == 3
    assert SPECS[V3].hash not in (SPECS[1].hash, SPECS[2].hash)
    assert SPECS[V3].hash == (
        "50e353e62ba39315b3954fda87cb576c200231cac6cfe15d09411f625a32ead5"
    )
    assert SPECS[2].hash == (
        "16d17ab3cd78190297595e44cfc8b1354a81f7bd52fa49ebc697254b520b1bba"
    )


def test_ema_distance_is_in_wilder_atr_units(tmp_path):
    # Closes 100, 101, ... with high/low one either side: true range is 2 from
    # Bar 1, so the ATR is 2 once Bar 14 seeds it (mean of true ranges 1..14).
    # An EMA of a unit ramp trails by (span - 1) / 2 Bars from its SMA seed on:
    # 5.5 for span 12 (seed at Bar 11), 11.5 for span 24 (seed at Bar 23).
    frame = prepared_frame(tmp_path, shaped(40, ramp), version=V3)

    ema12 = frame["feat_trend_ema_12"]
    assert ema12[13] is None  # the ATR scale does not exist yet
    for i in (14, 15, 39):
        assert ema12[i] == pytest.approx(5.5 / 2, abs=1e-9)
    ema24 = frame["feat_trend_ema_24"]
    assert ema24[22] is None
    for i in (23, 24, 39):
        assert ema24[i] == pytest.approx(11.5 / 2, abs=1e-9)


def test_rsi_is_wilder_smoothed_and_seeded_with_the_first_fourteen_changes(tmp_path):
    # Changes +1, -1, ... for Bars 1..14 give mean gain = mean loss = 0.5, so the
    # seed RSI at Bar 14 is 50. Bar 15 adds +3: gain (0.5*13 + 3)/14 = 9.5/14,
    # loss 0.5*13/14 = 6.5/14, so RSI = 100 * 9.5 / (9.5 + 6.5) = 59.375.
    closes = [100.0 + (i % 2) * 1.0 for i in range(15)] + [103.0]
    frame = prepared_frame(tmp_path, shaped(16, lambda i: (closes[i],) * 4), version=V3)

    rsi = frame["feat_trend_rsi_14"]
    assert rsi[13] is None
    assert rsi[14] == pytest.approx(50.0, abs=1e-9)
    assert rsi[15] == pytest.approx(59.375, abs=1e-9)


def test_rsi_is_100_when_nothing_fell_and_50_when_nothing_moved(tmp_path):
    (tmp_path / "up").mkdir()
    (tmp_path / "flat").mkdir()
    up = prepared_frame(tmp_path / "up", shaped(20, ramp), version=V3)
    flat = prepared_frame(
        tmp_path / "flat", shaped(20, lambda i: (5.0,) * 4), version=V3
    )

    assert up["feat_trend_rsi_14"][19] == 100.0
    assert flat["feat_trend_rsi_14"][19] == 50.0
    assert flat["feat_trend_rsi_14"].is_nan().sum() == 0


def test_shape_uses_wilder_smoothed_atr(tmp_path):
    # Flat close 100, high 101, low 99: true range 2 from Bar 1, ATR 2 at Bar 14.
    # Bar 15 spikes to a high of 115: true range 16, ATR = (2*13 + 16)/14 = 3, so
    # range / ATR = 16 / 3. (A simple 14-Bar mean would drop Bar 1's true range
    # and give (2*13 + 16)/14 as well, but at Bar 16 it differs: see below.)
    def ohlc(i):
        return (100.0, 115.0 if i == 15 else 101.0, 99.0, 100.0)

    frame = prepared_frame(tmp_path, shaped(17, ohlc), version=V3)

    rng = frame["feat_shape_range"]
    assert rng[13] is None
    assert rng[14] == pytest.approx(1.0, abs=1e-9)
    assert rng[15] == pytest.approx(16 / 3, abs=1e-9)
    # Bar 16: true range 2 again. Wilder ATR = (3*13 + 2)/14 = 41/14; a simple
    # mean would be (2*12 + 16 + 2)/14 = 3.0.
    assert rng[16] == pytest.approx(2 / (41 / 14), abs=1e-9)


def test_a_gap_nulls_long_memory_columns_for_their_whole_lookback(tmp_path):
    bars = shaped(400, ramp)
    bars[10] = gap_bar("EURUSD", bars[10].open_time, SESSION_1)
    frame = prepared_frame(tmp_path, bars, version=V3)

    def first_valid(name):
        return next(i for i in range(11, 400) if frame[name][i] is not None)

    assert first_valid("feat_trend_ema_12") == 10 + 280 + 1
    assert first_valid("feat_trend_ema_24") == 10 + 280 + 1
    assert first_valid("feat_trend_rsi_14") == 10 + 280 + 1


@pytest.mark.oracle
def test_ta_lib_agrees_on_ema_rsi_and_atr_conventions_past_warm_up(tmp_path):
    talib = pytest.importorskip("talib")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    close = 1.1 + np.cumsum(rng.normal(0, 0.001, 500))
    high = close + rng.uniform(0.0001, 0.002, 500)
    low = close - rng.uniform(0.0001, 0.002, 500)
    open_ = np.concatenate([[close[0]], close[:-1]])
    frame = prepared_frame(
        tmp_path,
        shaped(500, lambda i: (open_[i], high[i], low[i], close[i])),
        version=V3,
    )

    atr = talib.ATR(high, low, close, 14)
    # A mismatch here is a spec finding: TA-Lib never defines the features.
    for span in (12, 24):
        expected = (close - talib.EMA(close, span)) / atr
        got = frame[f"feat_trend_ema_{span}"].to_numpy()
        past = slice(300, None)
        assert got[past] == pytest.approx(expected[past], abs=1e-6)
    got_rsi = frame["feat_trend_rsi_14"].to_numpy()
    assert got_rsi[300:] == pytest.approx(talib.RSI(close, 14)[300:], abs=1e-6)
    got_range = frame["feat_shape_range"].to_numpy()
    assert got_range[300:] == pytest.approx(((high - low) / atr)[300:], abs=1e-6)
    assert not math.isnan(got_range[300])


V4 = 4


def test_spec_v4_adds_atr_over_close_to_the_volatility_family(tmp_path):
    # Closes 100, 101, ... with high/low one either side: the Wilder ATR is 2
    # from Bar 14 (mean of true ranges 1..14), and close at Bar 14 is 114.
    spec = SPECS[V4]
    by_name = {c.name: c for c in spec.columns}
    assert by_name["feat_vol_atr"].family == "volatility"
    assert by_name["feat_vol_atr"].lookback == 280
    assert by_name["feat_vol_atr"].tolerance == 1e-9
    assert set(SPECS[V3].names) < set(spec.names)
    assert len(spec.names) == 30 and "feat_vol_1" not in spec.names

    frame = prepared_frame(tmp_path, shaped(40, ramp), version=V4)

    assert frame["feat_vol_atr"][13] is None
    assert frame["feat_vol_atr"][14] == pytest.approx(2 / 114, abs=1e-12)
    assert frame["feat_vol_atr"][39] == pytest.approx(2 / 139, abs=1e-12)


def test_v4_hash_covers_scaling_and_lookback_while_old_pins_stay():
    v4 = SPECS[V4]
    first = v4.columns[0]
    rescaled = replace(v4, columns=(replace(first, scaling="identity"), *v4.columns[1:]))
    deeper = replace(v4, columns=(replace(first, lookback=first.lookback + 1), *v4.columns[1:]))

    assert rescaled.hash != v4.hash and deeper.hash != v4.hash
    # versions 1-3 keep their original hash payload, so their pins are unchanged
    old = SPECS[3]
    again = replace(old, columns=(replace(old.columns[0], scaling="identity"), *old.columns[1:]))
    assert again.hash == old.hash
