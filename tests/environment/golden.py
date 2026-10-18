"""The golden scenario: scripted runs over the fixture Dataset, hashed and stored.

``run_golden(seed)`` plays three scripted runs through the real env:

- ``sweep``: an evaluation sweep over every runnable Session with a policy that
  is long for 40 steps and short for 30, repeating. The fixture gives it Gaps and
  a Gap outage, a rollover Bar (nonzero financing rates), Session-final forced
  flats and an unfilled forced flat on the Gap tail;
- ``closeout``: a hair-trigger-margin sweep of Session 0 that ends in a closeout;
- ``random-<i>``: ``RANDOM_EPISODES`` random-start episodes of ``WINDOW`` Bars, each
  in an env of its own. The seed decides only where those windows start.

The golden hash (``determinism.hash_parts``) and the four Ledger tables of each
part are checked in under ``tests/environment/golden``, with the platform the hash
was recorded on. Refresh them only with ``shadow-trader env regen-golden``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from shadow_trader.environment import (
    EngineConfig,
    EnvConfig,
    EpisodeConfig,
    InstrumentSpec,
    verify_ledger,
)
from shadow_trader.environment.determinism import (
    LEDGER_TABLES,
    OBSERVATION_KEYS,
    GoldenPart,
    hash_parts,
    seed_int,
)
from shadow_trader.environment.env import TradingEnv, make_env
from shadow_trader.environment.sweep import run_sweep

from .fixture import load_fixture

GOLDEN_DIR = Path(__file__).parent / "golden"
RECORD_FILE = "golden.json"
GOLDEN_ENTROPY = 125
WINDOW = 20
RANDOM_EPISODES = 4
FINANCING = InstrumentSpec(rollover_long=0.0002, rollover_short=-0.0001)
CARRY = EngineConfig(instruments=(FINANCING,))
HAIR_TRIGGER = EngineConfig(leverage_cap=50.0, closeout_fraction=1.0)
LONG, SHORT = 2, 0  # indices into the default levels (-1, 0, 1)


@dataclass(frozen=True)
class GoldenRun:
    parts: list[GoldenPart]


class Recorder:
    """Collects the observations a run saw, in order."""

    def __init__(self) -> None:
        self._seen: dict[str, list[np.ndarray]] = {k: [] for k in OBSERVATION_KEYS}

    def add(self, obs: dict[str, np.ndarray]) -> None:
        for key in OBSERVATION_KEYS:
            self._seen[key].append(np.array(obs[key]))

    def stacked(self) -> dict[str, np.ndarray]:
        return {key: np.stack(values) for key, values in self._seen.items()}


def cycling_policy(recorder: Recorder) -> Callable[[Any], int]:
    count = 0

    def policy(obs: Any) -> int:
        nonlocal count
        recorder.add(obs)
        action = LONG if count % 70 < 40 else SHORT
        count += 1
        return action

    return policy


def _env(prepared, engine: EngineConfig, seq: np.random.SeedSequence) -> TradingEnv:
    dataset, scaler = prepared
    config = EnvConfig(engine=engine, episode=EpisodeConfig(window_bars=WINDOW))
    return make_env(dataset, scaler, config, seed=seed_int(seq), record_ledger=True)


def _sweep(prepared, name: str, engine, seq, sessions=None) -> GoldenPart:
    env = _env(prepared, engine, seq)
    recorder = Recorder()
    result = run_sweep(env, cycling_policy(recorder), sessions=sessions)
    verify_ledger(result.ledger)
    return GoldenPart(name, result.ledger, recorder.stacked())


def _random(prepared, index: int, seq: np.random.SeedSequence) -> GoldenPart:
    """One random-start episode in an env of its own (a Ledger holds one)."""
    env = _env(prepared, CARRY, seq)
    recorder = Recorder()
    policy = cycling_policy(recorder)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        done = terminated or truncated
    recorder.add(obs)
    ledger = env.ledger()
    verify_ledger(ledger)
    return GoldenPart(f"random-{index}", ledger, recorder.stacked())


def run_golden(seed: np.random.SeedSequence) -> GoldenRun:
    """The golden scenario's parts for ``seed`` (a fresh ``SeedSequence``)."""
    prepared = load_fixture()
    sweep_seq, closeout_seq, *random_seqs = seed.spawn(2 + RANDOM_EPISODES)
    return GoldenRun(
        [
            _sweep(prepared, "sweep", CARRY, sweep_seq),
            _sweep(prepared, "closeout", HAIR_TRIGGER, closeout_seq, sessions=[0]),
            *(_random(prepared, i, seq) for i, seq in enumerate(random_seqs)),
        ]
    )


def replace_steps(part: GoldenPart, steps: pl.DataFrame) -> GoldenPart:
    return replace(part, ledger=replace(part.ledger, steps=steps))


def replace_observations(part: GoldenPart, observations: Any) -> GoldenPart:
    return replace(part, observations=observations)


def load_record() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / RECORD_FILE).read_text())


def write_golden(platform: str) -> str:
    """Replay the scenario and store its Ledger tables, hash and platform."""
    run = run_golden(np.random.SeedSequence(GOLDEN_ENTROPY))
    digest = hash_parts(run.parts)
    for part in run.parts:
        folder = GOLDEN_DIR / part.name
        folder.mkdir(parents=True, exist_ok=True)
        for table in LEDGER_TABLES:
            getattr(part.ledger, table).write_parquet(folder / f"{table}.parquet")
    record = {
        "hash": digest,
        "platform": platform,
        "entropy": GOLDEN_ENTROPY,
        "parts": [part.name for part in run.parts],
    }
    (GOLDEN_DIR / RECORD_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return digest
