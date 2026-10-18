"""The Run summary: one accumulator for the live env and for ``summarize``."""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path

import pytest

from shadow_trader.environment import EnvConfig, EpisodeConfig
from shadow_trader.environment.env import TradingEnv, make_env
from shadow_trader.environment.persist import (
    ManifestError,
    read_ledger,
    write_ledger,
)
from shadow_trader.environment.report import (
    ENDINGS,
    EpisodeSummary,
    RunSummary,
    summarize,
)
from shadow_trader.environment.sweep import run_sweep

from .fixture import OUTAGE_SESSION, TAIL_SESSION, load_fixture

LONG, FLAT = 2, 1


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


def build(prepared) -> TradingEnv:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    return make_env(dataset, scaler, config, seed=3, record_ledger=True)


def play(env: TradingEnv, session: int, action: int = LONG, **options):
    env.reset(options={"session": session, **options})
    done = False
    info: dict = {}
    while not done:
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


@pytest.mark.parametrize("session", [0, 4, OUTAGE_SESSION, TAIL_SESSION])
def test_env_episode_summary_equals_summarize_of_the_ledger(
    prepared, session: int
) -> None:
    env = build(prepared)
    info = play(env, session)
    summary = summarize(env.ledger())
    assert isinstance(info["episode"], EpisodeSummary)
    assert summary.episodes == (info["episode"],)


def test_a_chained_sweep_summarises_every_episode_like_the_env_did(prepared) -> None:
    result = run_sweep(build(prepared), lambda _o: LONG)
    # replay the same sweep, collecting what the env reported at each episode end
    env = build(prepared)
    seen: list[EpisodeSummary] = []
    equity = None
    for k in result.sessions_run:
        options = {"session": k}
        if equity is not None:
            options.update(equity=equity, chain=True)
        env.reset(options=options)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(LONG)
            done = term or trunc
        seen.append(info["episode"])
        equity = env.equity
    assert summarize(result.ledger).episodes == tuple(seen)


def test_info_has_no_episode_summary_before_the_episode_ends(prepared) -> None:
    env = build(prepared)
    env.reset(options={"session": 0})
    _, _, terminated, truncated, info = env.step(LONG)
    assert not (terminated or truncated)
    assert "episode" not in info


def test_the_env_summarises_even_without_recording_a_ledger(prepared) -> None:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=20))
    plain = make_env(dataset, scaler, config, seed=3)
    info = play(plain, 0)
    recorded = build(prepared)
    assert info["episode"] == play(recorded, 0)["episode"]


def test_counts_and_totals_read_straight_off_the_tables(prepared) -> None:
    env = build(prepared)
    play(env, 0)
    ledger = env.ledger()
    (episode,) = summarize(ledger).episodes
    assert episode.bars == ledger.steps.height
    assert episode.decisions == int(ledger.steps["is_decision"].sum())
    assert episode.fills == ledger.fills.height > 0
    assert episode.closed_lots == ledger.closed_lots.height
    assert episode.clips == int(ledger.steps["clipped"].sum())
    assert episode.spread_cost == pytest.approx(ledger.fills["spread_cost"].sum())
    assert episode.slippage_cost == pytest.approx(ledger.fills["slippage_cost"].sum())
    assert episode.rollover_cost == pytest.approx(ledger.steps["rollover_charge"].sum())
    assert episode.start_equity == ledger.episodes["start_equity"][0]
    assert episode.final_equity == ledger.steps["equity"][-1]
    assert episode.net_pnl == pytest.approx(episode.final_equity - episode.start_equity)
    assert episode.longest_blind_run == ledger.steps["blind_run"].max()


def test_gap_spans_blind_runs_and_synthetic_liquidations(prepared) -> None:
    env = build(prepared)
    play(env, OUTAGE_SESSION)
    (outage,) = summarize(env.ledger()).episodes
    assert outage.gap_spans >= 1
    assert outage.longest_blind_run >= 5  # the five Gap rows in a row
    assert outage.decisions < outage.bars
    env = build(prepared)
    play(env, TAIL_SESSION)
    (tail,) = summarize(env.ledger()).episodes
    assert tail.synthetic_liquidations == 1  # the trailing Gaps strand the long
    assert tail.forced_flats == 0


def test_forced_flat_and_ending_are_counted(prepared) -> None:
    env = build(prepared)
    play(env, 4)
    (episode,) = summarize(env.ledger()).episodes
    assert episode.forced_flats == 1
    assert episode.ending == "session_end"


def test_run_summary_totals_episodes_and_counts_endings(prepared) -> None:
    env = build(prepared)
    result = run_sweep(env, lambda _o: LONG, sessions=[0, 4])
    run = summarize(result.ledger)
    assert isinstance(run, RunSummary)
    assert run.episode_count == 2
    assert run.bars == sum(e.bars for e in run.episodes)
    assert run.fills == sum(e.fills for e in run.episodes)
    assert run.longest_blind_run == max(e.longest_blind_run for e in run.episodes)
    assert run.start_equity == run.episodes[0].start_equity
    assert run.final_equity == run.episodes[-1].final_equity
    assert run.net_pnl == pytest.approx(run.final_equity - run.start_equity)
    assert run.endings["dataset_end"] == 1
    assert run.endings["session_end"] == 1


def test_halt_and_stopped_are_always_present_and_zero_outside_live(prepared) -> None:
    env = build(prepared)
    play(env, 0)
    run = summarize(env.ledger())
    assert set(run.endings) == set(ENDINGS)
    assert run.endings["halt"] == 0
    assert run.endings["stopped"] == 0


def test_summary_has_no_ratios_or_drawdown(prepared) -> None:
    env = build(prepared)
    play(env, 0)
    names = " ".join(f.name for f in fields(summarize(env.ledger())))
    for banned in ("ratio", "sharpe", "drawdown", "win", "return"):
        assert banned not in names


def test_the_environment_emits_no_logging_or_output(prepared, caplog, capsys) -> None:
    env = build(prepared)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        for session in (0, OUTAGE_SESSION, TAIL_SESSION):
            play(env, session)
    assert caplog.records == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


# --- summary.json -------------------------------------------------------------


@pytest.fixture
def persisted(prepared, tmp_path) -> Path:
    env = build(prepared)
    play(env, 0)
    path = tmp_path / "ledger"
    write_ledger(env, path, episode_mode="eval", seam_version=1)
    return path


def test_persisting_writes_a_canonical_summary_beside_the_tables(persisted) -> None:
    raw = (persisted / "summary.json").read_bytes()
    doc = json.loads(raw)
    assert doc["summary_version"] == 1
    assert raw == json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    assert doc["summary"]["episode_count"] == 1
    assert "summary.json" not in (persisted / "manifest.json").read_text()
    assert read_ledger(persisted).steps.height == doc["summary"]["bars"]


def test_a_tampered_summary_is_rejected_on_load(persisted) -> None:
    file = persisted / "summary.json"
    doc = json.loads(file.read_text())
    doc["summary"]["fills"] += 1
    file.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ManifestError, match="summary"):
        read_ledger(persisted)


def test_a_non_canonical_or_missing_summary_is_rejected(persisted) -> None:
    file = persisted / "summary.json"
    file.write_text(json.dumps(json.loads(file.read_text()), indent=2))
    with pytest.raises(ManifestError, match="summary"):
        read_ledger(persisted)
    file.unlink()
    with pytest.raises(ManifestError, match="summary"):
        read_ledger(persisted)
