"""The live seam: ``FEATURE_SPEC``, ``evaluate_tail``, ``check_compat`` and
``check_tail_parity`` (ADR 0004)."""

import math
import random
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow_trader.ingestion.bars import Bar, gap_bar
from shadow_trader.ingestion.session import is_rollover
from shadow_trader.preparation import SPECS, Scaler, ScalerError, compute_features
from shadow_trader.preparation.anomaly import DEFAULT_POLICY
from shadow_trader.preparation.build import featured_frame
from shadow_trader.preparation.live import (
    FEATURE_SPEC,
    check_compat,
    check_tail_parity,
    evaluate_tail,
)

from .synthetic import SESSION_1, session_bars

TAIL = FEATURE_SPEC.min_tail_bars


def series(count, gaps=()):
    return session_bars("EURUSD", SESSION_1, count, gaps=gaps)


# --- FEATURE_SPEC -----------------------------------------------------------


def test_feature_spec_exposes_identity_and_warm_up():
    spec = SPECS[max(SPECS)]

    assert FEATURE_SPEC.version == max(SPECS)
    assert FEATURE_SPEC.hash == spec.hash
    assert FEATURE_SPEC.policy_hash == DEFAULT_POLICY.hash
    assert set(FEATURE_SPEC.warm_up) == set(spec.names)
    # finite-memory columns need no warm-up; recursive ones do
    assert FEATURE_SPEC.warm_up["feat_ret_288"] == 0
    assert FEATURE_SPEC.warm_up["feat_trend_rsi_14"] > 0
    # a tail holds the deepest window (lookback + the Bar itself) plus its warm-up
    assert FEATURE_SPEC.min_tail_bars == max(
        c.lookback + 1 + FEATURE_SPEC.warm_up[c.name] for c in spec.columns
    )
    # Wilder alpha 1/14: an input min_tail_bars - 1 Bars back weighs under 1e-12
    assert (1 - 1 / 14) ** (FEATURE_SPEC.min_tail_bars - 1) < 1e-12


# --- evaluate_tail ----------------------------------------------------------


def test_evaluate_tail_returns_one_row_with_features_and_usability():
    bars = series(TAIL + 30)

    row = evaluate_tail(bars[-TAIL:])

    assert row.height == 1
    assert row.columns == [*FEATURE_SPEC.spec.names, "usable", "unusable_reason"]
    assert row["usable"][0] is True
    assert row["unusable_reason"][0] is None
    assert all(row[name][0] is not None for name in FEATURE_SPEC.spec.names)


def test_evaluate_tail_matches_the_batch_last_row():
    bars = series(900)
    batch = featured_frame(bars, FEATURE_SPEC.spec).tail(1)

    row = evaluate_tail(bars[-TAIL:])

    assert row["usable"][0] == batch["usable"][0]
    assert row["feat_ret_288"][0] == batch["feat_ret_288"][0]
    assert row["feat_trend_rsi_14"][0] == pytest.approx(
        batch["feat_trend_rsi_14"][0], abs=1e-9
    )


def test_a_short_tail_gives_nulls_not_errors():
    row = evaluate_tail(series(10))

    assert row.height == 1
    assert row["usable"][0] is False
    assert row["unusable_reason"][0] == "dataset_start"
    assert row["feat_ret_288"][0] is None
    assert row["feat_trend_rsi_14"][0] is None
    assert row["feat_time_hour_sin"][0] is not None  # always valid


def test_an_empty_tail_gives_a_null_row():
    row = evaluate_tail([])

    assert row.height == 1
    assert row["usable"][0] is False
    assert row["feat_ret_1"][0] is None


def test_a_gap_as_the_last_bar_makes_the_row_unusable():
    row = evaluate_tail(series(500, gaps=(499,)))

    assert row["unusable_reason"][0] == "gap"
    assert row["feat_ret_1"][0] is None


def test_the_revision_of_a_bar_is_ignored():
    # Bar has no revision field: a replayed tail is identified by content alone
    bars = series(TAIL)

    assert evaluate_tail(bars).equals(evaluate_tail(list(bars)))


# --- check_compat -----------------------------------------------------------


def scaler_for(spec_hash, policy_hash):
    return Scaler({"spec_hash": spec_hash, "anomaly_policy_hash": policy_hash})


def test_check_compat_accepts_a_matching_scaler_without_warnings():
    scaler = scaler_for(FEATURE_SPEC.hash, FEATURE_SPEC.policy_hash)

    assert check_compat(scaler, FEATURE_SPEC) == []


def test_check_compat_refuses_a_spec_hash_mismatch():
    scaler = scaler_for("0" * 64, FEATURE_SPEC.policy_hash)

    with pytest.raises(ScalerError, match="spec hash"):
        check_compat(scaler, FEATURE_SPEC)


def test_check_compat_refuses_a_policy_hash_mismatch():
    scaler = scaler_for(FEATURE_SPEC.hash, "0" * 64)

    with pytest.raises(ScalerError, match="policy hash"):
        check_compat(scaler, FEATURE_SPEC)


def prepared_with_polars(version):
    from pathlib import Path

    from shadow_trader.preparation import PreparedDataset

    return PreparedDataset("d", Path("."), {"polars_version": version})


def test_check_compat_warns_on_a_polars_version_difference():
    scaler = scaler_for(FEATURE_SPEC.hash, FEATURE_SPEC.policy_hash)

    warnings = check_compat(scaler, FEATURE_SPEC, dataset=prepared_with_polars("0.0.1"))

    assert len(warnings) == 1
    assert "polars" in warnings[0]
    assert "0.0.1" in warnings[0]
    same = check_compat(
        scaler, FEATURE_SPEC, dataset=prepared_with_polars(pl.__version__)
    )
    assert same == []


# --- check_tail_parity ------------------------------------------------------


def test_check_tail_parity_reports_agreement_on_a_clean_series():
    bars = series(TAIL + 60)

    report = check_tail_parity(bars, TAIL, every=7)

    assert report.rows_compared > 0
    assert report.ok
    assert report.null_mismatches == {}
    assert report.usable_mismatches == 0
    assert report.reason_mismatches == 0
    assert report.max_deviation["feat_ret_288"] == 0.0
    assert report.max_deviation["feat_trend_ema_12"] <= 1e-9


def test_check_tail_parity_reports_a_deviation_below_the_stated_tail_length():
    bars = series(TAIL + 60)

    report = check_tail_parity(bars, 40, every=7)

    # a 40-Bar tail has no 288-Bar return where the batch does
    assert report.null_mismatches["feat_ret_288"] > 0
    assert not report.ok


# --- Hypothesis parity ------------------------------------------------------

START = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)


@st.composite
def bar_series(draw, min_size=TAIL + 1, max_size=TAIL + 100):
    """Bars with Gaps, Session boundaries, flagged Bars and null/NaN mixes."""
    count = draw(st.integers(min_size, max_size))
    starts = {0, *draw(st.sets(st.integers(1, count - 1), max_size=3))}
    rate = draw(st.sampled_from([0.0, 0.002, 0.01, 0.05]))
    rng = random.Random(draw(st.integers(0, 2**32)))
    kinds = [
        rng.choice(("gap", "flag", "null_tick", "nan"))
        if rng.random() < rate
        else "normal"
        for _ in range(count)
    ]
    steps = [rng.uniform(-0.01, 0.01) for _ in range(count)]
    shapes = [
        (
            rng.uniform(0, 0.004),
            rng.uniform(0, 0.004),
            rng.uniform(0, 0.004),
            rng.uniform(0.00005, 0.0005),
            rng.randint(1, 200),
        )
        for _ in range(count)
    ]
    bars: list[Bar] = []
    session = START
    price = 1.1
    for i in range(count):
        opened = START + timedelta(minutes=5 * i)
        if i in starts:
            session = opened
        if kinds[i] == "gap":
            bars.append(gap_bar("EURUSD", opened, session))
            continue
        price *= math.exp(steps[i])
        up, down, body, spread, ticks = shapes[i]
        open_ = price * (1 + body - 0.002)
        high = max(open_, price) * (1 + up)
        low = min(open_, price) * (1 - down)
        bars.append(
            Bar(
                instrument="EURUSD",
                open_time=opened,
                open=open_,
                high=high,
                low=low,
                close=price,
                spread_close=float("nan") if kinds[i] == "nan" else spread,
                spread_mean=spread,
                tick_count=None if kinds[i] == "null_tick" else ticks,
                is_gap=False,
                session_id=session,
                is_rollover=is_rollover(opened),
                quality_flags=("price_spike",) if kinds[i] == "flag" else (),
            )
        )
    return bars


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
)
@given(data=st.data())
def test_tail_evaluation_matches_the_batch_build_at_the_production_tail_length(data):
    bars = data.draw(bar_series())
    report = check_tail_parity(bars, TAIL, every=19)

    assert report.rows_compared > 0
    assert report.usable_mismatches == 0
    assert report.reason_mismatches == 0
    assert report.null_mismatches == {}
    tolerance = {c.name: c.tolerance for c in FEATURE_SPEC.spec.columns}
    for name, deviation in report.max_deviation.items():
        assert deviation <= tolerance[name], name  # 0.0 means bitwise


# --- Hypothesis causality ---------------------------------------------------


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
)
@given(data=st.data(), cut=st.integers(0, 59))
def test_replacing_every_bar_after_a_cut_leaves_earlier_rows_bitwise_identical(
    data, cut
):
    original = data.draw(bar_series(min_size=60, max_size=400))
    other = data.draw(bar_series(min_size=400, max_size=400))
    perturbed = original[: cut + 1] + [b for b in other[cut + 1 : len(original)]]
    assert len(perturbed) == len(original)

    before = compute_features(_frame(original), FEATURE_SPEC.spec).collect()
    after = compute_features(_frame(perturbed), FEATURE_SPEC.spec).collect()

    columns = ["open_time", *FEATURE_SPEC.spec.names]  # incl. time / session
    assert (
        before.select(columns).head(cut + 1).equals(after.select(columns).head(cut + 1))
    )


def _frame(bars):
    from shadow_trader.preparation.build import _bars_frame

    return _bars_frame(
        [{**vars(b), "quality_flags": list(b.quality_flags)} for b in bars]
    )
