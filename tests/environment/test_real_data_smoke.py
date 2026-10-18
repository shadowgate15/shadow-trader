"""Opt-in smoke: replay a real prepared Dataset month. Opt in with ``-m slow``.

Set ``SHADOW_TRADER_RAW_ROOT`` (the ``data_root`` the other slow tests use) to a
root holding a built 5-minute EURUSD ingestion Dataset spanning at least 40 days.
The newest such Dataset is prepared into a temporary root with the first 31 days
as the train role; that train Dataset (about a month) is replayed with a scripted
and a random policy. Each must run without an exception and pass ``verify_ledger``,
and the engine on its own must manage 50,000 steps a second (the floor is for an
M-series Mac; set ``SHADOW_TRADER_SKIP_THROUGHPUT_FLOOR=1`` on slower hardware to
keep the replay checks and only report the rate).

Skipped, with the reason, when the variable is unset or no such Dataset exists.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shadow_trader.environment import (
    EngineConfig,
    EnvConfig,
    market_data,
    verify_ledger,
)
from shadow_trader.environment.env import make_env
from shadow_trader.environment.sweep import run_sweep
from shadow_trader.preparation import (
    build_prepared,
    load_prepared_config,
    load_scaler,
    open_prepared,
)

from .throughput import measure_engine_throughput

pytestmark = pytest.mark.slow

FLOOR_STEPS_PER_SECOND = 50_000
MIN_SPAN_DAYS = 40
TRAIN_DAYS = 31
LEVELS = 3  # default levels (-1, 0, 1)


def _latest_eurusd_manifest() -> tuple[Path, dict[str, Any]]:
    root = os.environ.get("SHADOW_TRADER_RAW_ROOT")
    if not root:
        pytest.skip("set SHADOW_TRADER_RAW_ROOT to a data_root with an EURUSD Dataset")
    found: list[dict[str, Any]] = []
    for path in (Path(root) / "datasets").glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest["bar_size"] != 5 or manifest["instruments"] != ["EURUSD"]:
            continue
        span = datetime.fromisoformat(manifest["time_range"]["end"]) - (
            datetime.fromisoformat(manifest["time_range"]["start"])
        )
        if span.days >= MIN_SPAN_DAYS:
            found.append(manifest)
    if not found:
        pytest.skip(
            f"data_root holds no 5-minute EURUSD Dataset of {MIN_SPAN_DAYS}+ days"
        )
    return Path(root).resolve(), max(found, key=lambda m: m["created_at"])


@pytest.fixture(scope="module")
def real_month(tmp_path_factory: pytest.TempPathFactory):
    data_root, manifest = _latest_eurusd_manifest()
    start = datetime.fromisoformat(manifest["time_range"]["start"])
    end = datetime.fromisoformat(manifest["time_range"]["end"])
    validation_start = start + timedelta(days=TRAIN_DAYS)
    test_start = validation_start + (end - validation_start) / 2
    tmp = tmp_path_factory.mktemp("real_month")
    config_path = tmp / "prepare.toml"
    config_path.write_text(
        f'[paths]\ndata_root = "{data_root}"\nprepared_root = "{tmp / "prepared"}"\n'
        f'[input]\ndataset_id = "{manifest["dataset_id"]}"\n'
        "[spec]\nversion = 4\n"
        '[split]\nscheme = "chronological"\n'
        f"train_start = {start:%Y-%m-%dT%H:%M:%SZ}\n"
        f"validation_start = {validation_start:%Y-%m-%dT%H:%M:%SZ}\n"
        f"test_start = {test_start:%Y-%m-%dT%H:%M:%SZ}\n"
        "[embargo]\nepisode_bars = 288\n"
    )
    config = load_prepared_config(config_path)
    ref = build_prepared(config)
    root = config.paths.prepared_root
    dataset = open_prepared(root, ref.datasets["train"].dataset_id)
    scaler = load_scaler(root, dataset.manifest["scaler"]["id"])
    return dataset, scaler


def _scripted():
    """A policy holding each of long, flat and short for 40 steps in turn."""
    count = -1

    def policy(obs: Any) -> int:
        nonlocal count
        count += 1
        return (count // 40) % LEVELS

    return policy


def _replay(real_month, policy) -> None:
    dataset, scaler = real_month
    env = make_env(dataset, scaler, EnvConfig(), seed=126, record_ledger=True)
    result = run_sweep(env, policy)
    assert result.sessions_run
    assert result.ledger.steps.height > 0
    verify_ledger(result.ledger)


def test_scripted_policy_replays_a_real_month(real_month):
    _replay(real_month, _scripted())


def test_random_policy_replays_a_real_month(real_month):
    rng = np.random.default_rng(126)
    _replay(real_month, lambda obs: int(rng.integers(LEVELS)))


def test_engine_meets_the_throughput_floor(real_month):
    dataset, scaler = real_month
    data = market_data.from_prepared(dataset, scaler, "EURUSD", 5)
    measured = measure_engine_throughput(data, EngineConfig())
    print(f"engine: {measured.steps} steps, {measured.steps_per_second:,.0f} steps/s")
    assert measured.steps > 4_000, "a month of Bars is ~5,000+; the data looks wrong"
    if os.environ.get("SHADOW_TRADER_SKIP_THROUGHPUT_FLOOR"):
        pytest.skip(f"floor not enforced; measured {measured.steps_per_second:,.0f}/s")
    assert measured.steps_per_second >= FLOOR_STEPS_PER_SECOND
