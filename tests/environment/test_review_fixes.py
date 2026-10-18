"""Behaviour pinned by the code-review pass over the Trading environment spec."""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import shadow_trader.environment as environment
from shadow_trader.environment import persist, seam
from shadow_trader.environment.config import EngineConfig, InstrumentSpec
from shadow_trader.environment.determinism import make_env_fns
from shadow_trader.environment.engine import Market, initial_state, step
from shadow_trader.environment.env import make_env
from shadow_trader.environment.env_config import EnvConfig, EpisodeConfig
from shadow_trader.environment.ledger import (
    EPISODES_SCHEMA,
    LedgerError,
    verify_ledger,
)
from shadow_trader.environment.observation import ObservationConfig, build_observation
from shadow_trader.environment.sweep import run_sweep

from .fixture import SESSION_OPENS, load_fixture

LONG, FLAT = 2, 1
SEAM = 2


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


def build(prepared, *, seed: int = 3, **kwargs):
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    return make_env(dataset, scaler, config, seed=seed, record_ledger=True, **kwargs)


def play(env, action: int = LONG) -> list[dict]:
    infos, done = [], False
    while not done:
        _, _, term, trunc, info = env.step(action)
        infos.append(info)
        done = term or trunc
    return infos


# --- Spec "missing" 1: make_env signature ----------------------------------------


def test_make_env_needs_a_seed(prepared) -> None:
    dataset, scaler = prepared
    with pytest.raises(TypeError):
        make_env(dataset, scaler, EnvConfig())  # ty: ignore[missing-argument]
    assert "split_sessions" in inspect.signature(make_env).parameters


def test_split_sessions_restrict_windows_and_sweeps(prepared) -> None:
    split = [SESSION_OPENS[1], SESSION_OPENS[2]]
    env = build(prepared, split_sessions=split)

    for seed in range(15):
        _, info = env.reset(seed=seed)
        assert info["session_open"] in split

    result = run_sweep(env, lambda _obs: FLAT)
    assert result.sessions_run == [1, 2]
    with pytest.raises(ValueError, match="split"):
        env.reset(options={"session": 0})


def test_split_sessions_must_name_known_sessions(prepared) -> None:
    with pytest.raises(ValueError, match="Session"):
        build(prepared, split_sessions=[datetime(2000, 1, 1, tzinfo=UTC)])


# --- Spec "missing" 2: fill count in info -----------------------------------------


def test_info_reports_the_steps_fill_count(prepared) -> None:
    env = build(prepared)
    env.reset(options={"session": 0})
    infos = play(env)
    ledger = env.ledger()
    assert sum(i["fills"] for i in infos) == ledger.fills.height > 0


# --- Spec "missing" 3: episode counters and schema versions -------------------------


COUNTERS = (
    "decisions",
    "fills",
    "clips",
    "forced_flats",
    "synthetic_liquidations",
    "gap_spans",
    "closed_lots",
    "longest_blind_run",
)


def test_episodes_table_carries_the_per_episode_counters(prepared) -> None:
    for name in COUNTERS:
        assert name in EPISODES_SCHEMA
    env = build(prepared)
    env.reset(options={"session": 0})
    play(env)
    ledger = env.ledger()
    row = ledger.episodes.row(0, named=True)
    assert row["fills"] == ledger.fills.height
    assert row["decisions"] == ledger.steps["is_decision"].sum()
    assert row["closed_lots"] == ledger.closed_lots.height


def test_verify_rejects_a_counter_that_disagrees_with_the_steps(prepared) -> None:
    env = build(prepared)
    env.reset(options={"session": 0})
    play(env)
    ledger = env.ledger()
    bad = replace(
        ledger, episodes=ledger.episodes.with_columns(fills=pl.col("fills") + 1)
    )
    with pytest.raises(LedgerError, match="fills"):
        verify_ledger(bad)


def test_schema_and_seam_versions_were_bumped() -> None:
    assert persist.SCHEMA_VERSION == 3
    assert seam.SEAM_VERSION == SEAM


# --- Spec "missing" 5: a ledger must be checkable against what it should be ---------


def written(prepared, tmp_path):
    env = build(prepared)
    env.reset(options={"session": 0})
    play(env)
    path = tmp_path / "ledger"
    persist.write_ledger(env, path, episode_mode="eval", seam_version=SEAM)
    return path, env


def test_read_ledger_checks_the_expected_dataset_scaler_and_config(
    prepared, tmp_path
) -> None:
    path, env = written(prepared, tmp_path)
    prov = env.provenance
    persist.read_ledger(
        path,
        dataset_id=prov["dataset_id"],
        scaler_hash=prov["scaler_hash"],
        config=env.config,
    )
    with pytest.raises(persist.ManifestError, match="dataset"):
        persist.read_ledger(path, dataset_id="other")
    with pytest.raises(persist.ManifestError, match="scaler"):
        persist.read_ledger(path, scaler_hash="other")
    other = replace(env.config, episode=EpisodeConfig(window_bars=21))
    with pytest.raises(persist.ManifestError, match="config"):
        persist.read_ledger(path, config=other)


# --- Spec "missing" 6: per-worker directories ---------------------------------------


def test_workers_write_to_their_own_spawn_index_directory(prepared, tmp_path) -> None:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    factories = make_env_fns(dataset, scaler, config, 2, seed=5, record_ledger=True)
    assert [f.spawn_index for f in factories] == [0, 1]
    for factory in factories:
        env = factory()
        env.reset(options={"session": 0})
        play(env)
        persist.write_worker_ledger(
            env,
            tmp_path,
            factory.spawn_index,
            episode_mode="train",
            seam_version=SEAM,
        )
    for index in (0, 1):
        assert persist.read_ledger(tmp_path / f"worker-{index}").steps.height > 0


# --- Spec "missing" 7: read_ledger / verify_ledger public in the ledger module ------


def test_ledger_module_exposes_read_and_verify(prepared, tmp_path) -> None:
    from shadow_trader.environment import ledger as ledger_module

    path, _ = written(prepared, tmp_path)
    assert ledger_module.read_ledger(path).steps.height > 0
    assert "read_ledger" in ledger_module.__all__
    assert "verify_ledger" in ledger_module.__all__


# --- Spec "missing" 8: P&L scales linearly -------------------------------------------

price = st.floats(min_value=0.5, max_value=2.0, allow_nan=False)
spread = st.floats(min_value=0.0, max_value=0.001, allow_nan=False)
markets_st = st.builds(
    lambda o, c, s1, s2: Market(
        open_bid=o, open_spread=s1, close_bid=c, close_spread=s2
    ),
    price,
    price,
    spread,
    spread,
)
series_st = st.lists(
    st.tuples(markets_st, st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0])),
    min_size=1,
    max_size=40,
)


def pnl_and_closeout(equity: float, series) -> tuple[float, bool]:
    config = EngineConfig(starting_equity=equity)
    state = initial_state(config)
    closeout = False
    for market, target in series:
        state, events = step(state, config, market, target)
        if events.closeout:
            closeout = True
            break
    return state.equity - equity, closeout


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(series=series_st)
def test_doubling_equity_doubles_pnl_within_tolerance(series) -> None:
    base = 1_000_000.0  # large, so the whole-unit rounding is negligible
    one, closed_one = pnl_and_closeout(base, series)
    two, closed_two = pnl_and_closeout(2 * base, series)
    if closed_one != closed_two:  # a borderline closeout flips on unit rounding
        return
    assert two == pytest.approx(2 * one, abs=1e-3 * base)


# --- Scope creep -----------------------------------------------------------------


def test_package_root_keeps_clip_order_and_the_sizing_helpers_internal() -> None:
    for name in ("clip_order", "level_to_units", "Fill", "Lot", "ClosedLot"):
        assert not hasattr(environment, name), name
        assert name not in environment.__all__, name


def test_step_info_has_no_running_or_reward_facts(prepared) -> None:
    env = build(prepared)
    env.reset(options={"session": 0})
    _, _, _, _, info = env.step(LONG)
    assert "equity_delta" not in info


def test_the_environment_logs_nothing_when_it_skips_sessions(prepared, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        env = build(prepared)
        result = run_sweep(env, lambda _obs: FLAT)
    assert caplog.records == []
    assert 3 in result.skipped  # the Session with no usable Bar is reported instead


def test_the_effective_leverage_cap_is_just_the_leverage_cap() -> None:
    assert not hasattr(EngineConfig(), "effective_leverage_cap")
    assert InstrumentSpec().name == "EURUSD"
    assert EngineConfig().names == ("EURUSD",)


# --- "Wrong" 4: info blind_run agrees with the ledger -------------------------------


def test_info_blind_run_is_the_longest_ledger_blind_run_of_the_step(prepared) -> None:
    from .fixture import OUTAGE_SESSION

    env = build(prepared)
    env.reset(options={"session": OUTAGE_SESSION})
    infos = play(env, FLAT)
    steps = env.ledger().steps
    assert max(i["blind_run"] for i in infos) == steps["blind_run"].max() > 0
    # per step: the Bars skipped are the blind run the step ended
    cursor = 0
    for info in infos:
        rows = steps.slice(cursor, info["bars_advanced"])
        assert info["blind_run"] == rows["blind_run"].max()
        cursor += info["bars_advanced"]


# --- "Wrong" 5: an empty account must not crash the observation -----------


def test_observation_survives_zero_and_negative_equity() -> None:
    config = EngineConfig()
    window = [[[0.1, 0.2, 0.3]]]
    valid = [[True]]
    market = Market(1.1, 0.0001, 1.1, 0.0001, bars_to_session_final=10)
    flat_bust = replace(initial_state(config), equity=0.0, cash=0.0)
    obs = build_observation(window, valid, flat_bust, market, config)
    assert obs.account[0] == 0.0

    held, _ = step(initial_state(config), config, market, 1.0)
    bust = replace(held, equity=-5.0)
    obs = build_observation(window, valid, bust, market, config)
    clip = ObservationConfig().position_clip
    assert obs.account[0] == pytest.approx(clip)

    deep = replace(held, equity=1e-9)
    obs = build_observation(window, valid, deep, market, config)
    assert obs.account[0] == pytest.approx(clip)


# --- Standards: shared helpers ------------------------------------------------------


def test_ending_names_are_defined_once() -> None:
    from shadow_trader.environment import endings, ledger, report

    assert ledger.CLOSEOUT is endings.CLOSEOUT
    assert report.ENDINGS is endings.ENDINGS
    assert set(endings.ENDINGS) == {
        "closeout",
        "session_end",
        "dataset_end",
        "window_end",
        "halt",
        "stopped",
    }
