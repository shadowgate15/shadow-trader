from datetime import timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow_trader.preparation import BuildFailure, build_prepared, open_prepared
from shadow_trader.preparation.align import AlignmentFailure, check_alignment

from .synthetic import (
    BARS_PER_SESSION,
    EMBARGO,
    SESSION_1,
    SESSION_2,
    SESSION_3,
    SESSIONS,
    prepared_config,
    session_bars,
    write_bars,
)

ROLES = ("train", "validation", "test")


def series(pair, sessions=SESSIONS, count=BARS_PER_SESSION, gaps=(5,), flags=None):
    return [
        bar
        for s in sessions
        for bar in session_bars(pair, s, count, gaps=gaps, flags=flags)
    ]


def build(root, bars):
    parent = write_bars(root, bars)
    config = prepared_config(root, parent.dataset_id)
    ref = build_prepared(config)
    return config, ref


def read(config, ref, role, pair):
    dataset = open_prepared(config.paths.prepared_root, ref.datasets[role].dataset_id)
    return dataset, dataset.read(pair)


def test_index_is_one_row_per_instrument_per_slot_with_no_invented_rows(tmp_path):
    bars = {"EURUSD": series("EURUSD"), "GBPUSD": series("GBPUSD", gaps=(9, 40))}
    config, ref = build(tmp_path, bars)

    for role in ROLES:
        dataset, eur = read(config, ref, role, "EURUSD")
        gbp = dataset.read("GBPUSD")
        expected = BARS_PER_SESSION - (EMBARGO if role != "test" else 0)
        assert eur.height == gbp.height == expected
        assert eur["open_time"].to_list() == gbp["open_time"].to_list()
        assert eur["open_time"].is_unique().all()
        assert sorted(dataset.manifest["instruments"]) == ["EURUSD", "GBPUSD"]


def test_unusable_is_per_instrument_with_no_joint_column(tmp_path):
    bars = {"EURUSD": series("EURUSD", gaps=(5,)), "GBPUSD": series("GBPUSD", gaps=())}
    config, ref = build(tmp_path, bars)

    dataset, eur = read(config, ref, "train", "EURUSD")
    gbp = dataset.read("GBPUSD")

    assert eur["unusable_reason"][5] == "gap"
    assert gbp["is_gap"][5] is False
    assert gbp["unusable_reason"][5] != "gap"
    assert [c for c in gbp.columns if "usable" in c] == ["usable", "unusable_reason"]


def test_manifest_records_the_actual_span_per_instrument(tmp_path):
    bars = {"EURUSD": series("EURUSD"), "GBPUSD": series("GBPUSD")}
    config, ref = build(tmp_path, bars)

    dataset, eur = read(config, ref, "validation", "EURUSD")

    span = dataset.manifest["instrument_spans"]["GBPUSD"]
    assert span["start"] == eur["open_time"][0].isoformat()
    assert span["end"] == (eur["open_time"][-1] + timedelta(minutes=5)).isoformat()


def test_an_instrument_short_of_the_requested_span_fails_naming_it(tmp_path):
    bars = {
        "EURUSD": series("EURUSD"),
        "GBPUSD": series("GBPUSD", sessions=(SESSION_1, SESSION_2)),
    }
    parent = write_bars(tmp_path, bars)
    config = prepared_config(tmp_path, parent.dataset_id)

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    assert raised.value.code == "coverage"
    assert "GBPUSD" in str(raised.value)
    gbp_end = SESSION_2 + timedelta(minutes=5 * BARS_PER_SESSION)
    assert gbp_end.isoformat() in str(raised.value)  # where the uncovered span starts
    assert not (tmp_path / "prepared").exists()


def test_an_instrument_starting_late_fails_naming_the_uncovered_span(tmp_path):
    bars = {
        "EURUSD": series("EURUSD"),
        "GBPUSD": series("GBPUSD", sessions=(SESSION_2, SESSION_3)),
    }
    parent = write_bars(tmp_path, bars)
    config = prepared_config(tmp_path, parent.dataset_id)

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    assert raised.value.code == "coverage"
    assert "GBPUSD" in str(raised.value)
    assert SESSION_1.isoformat() in str(raised.value)


def test_differing_session_geometry_fails_and_nothing_is_resampled(tmp_path):
    short_middle = (
        series("GBPUSD", sessions=(SESSION_1,))
        + series("GBPUSD", sessions=(SESSION_2,), count=BARS_PER_SESSION - 1)
        + series("GBPUSD", sessions=(SESSION_3,))
    )
    bars = {"EURUSD": series("EURUSD"), "GBPUSD": short_middle}
    parent = write_bars(tmp_path, bars)
    config = prepared_config(tmp_path, parent.dataset_id)

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    assert raised.value.code == "session_geometry"
    assert "GBPUSD" in str(raised.value)
    assert not (tmp_path / "prepared").exists()


def frame(minutes, count=10):
    times = [SESSION_1 + timedelta(minutes=minutes * i) for i in range(count)]
    return pl.DataFrame({"open_time": times, "session_id": [SESSION_1] * count})


def test_instruments_differing_in_bar_size_fail():
    frames = {"EURUSD": frame(5), "GBPUSD": frame(15)}

    with pytest.raises(AlignmentFailure) as raised:
        check_alignment(frames, bar_size=5, start=SESSION_1, end=None)

    assert raised.value.code == "bar_size_mismatch"
    assert "GBPUSD" in str(raised.value)


@settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    gaps_a=st.sets(st.integers(0, 50), max_size=4),
    gaps_b=st.sets(st.integers(0, 50), max_size=4),
    flagged=st.sets(st.integers(0, 50), max_size=3),
)
def test_building_a_and_b_together_equals_building_them_apart(
    tmp_path_factory, gaps_a, gaps_b, flagged
):
    flags = {i: ("price_spike",) for i in flagged}
    a = series("EURUSD", gaps=tuple(gaps_a))
    b = series("GBPUSD", gaps=tuple(gaps_b), flags=flags)
    both = build(tmp_path_factory.mktemp("both"), {"EURUSD": a, "GBPUSD": b})
    runs = {
        "EURUSD": build(tmp_path_factory.mktemp("a"), {"EURUSD": a}),
        "GBPUSD": build(tmp_path_factory.mktemp("b"), {"GBPUSD": b}),
    }

    for pair, (config, ref) in runs.items():
        for role in ROLES:
            _, together = read(both[0], both[1], role, pair)
            _, apart = read(config, ref, role, pair)
            assert together.sort("open_time").equals(apart.sort("open_time"))
