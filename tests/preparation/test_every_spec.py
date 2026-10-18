"""A full build and verify at every registered spec version (issue #91).

Uses its own random-walk Bars: the shared synthetic Bars double every Bar, so
RSI is a constant 100 and the scaler rightly rejects it (zero IQR).
"""

import math
import random
from dataclasses import replace
from datetime import timedelta

import pytest

from shadow_trader import main
from shadow_trader.ingestion.bars import Bar, gap_bar
from shadow_trader.preparation import SPECS, build_prepared, open_prepared
from shadow_trader.preparation.config import load_prepared_config

from .synthetic import (
    BARS_PER_SESSION,
    BASE,
    GAP_INDEX,
    SESSIONS,
    priced,
    write_bars,
    write_prepared_toml,
)

SEED = 91


def random_walk_bars(instrument: str = "EURUSD") -> list[Bar]:
    """Deterministic log-return steps of both signs; valid OHLC and spreads."""
    rng = random.Random(SEED)
    bars: list[Bar] = []
    for s, session in enumerate(SESSIONS):
        close = BASE
        for i in range(BARS_PER_SESSION):
            opened = session + timedelta(minutes=5 * i)
            if s == 0 and i == GAP_INDEX:
                bars.append(gap_bar(instrument, opened, session))
                continue
            before = close
            close = before * math.exp(rng.gauss(0.0, 0.0005))
            high = max(before, close) * (1 + abs(rng.gauss(0.0, 0.0002)))
            low = min(before, close) * (1 - abs(rng.gauss(0.0, 0.0002)))
            spread = 0.0001 * (1 + rng.random())
            bars.append(
                replace(
                    priced(instrument, opened, session, i),
                    open=before,
                    high=high,
                    low=low,
                    close=close,
                    spread_close=spread,
                    spread_mean=spread,
                    tick_count=10 + rng.randrange(50),
                )
            )
    return bars


@pytest.mark.parametrize("version", sorted(SPECS))
def test_full_build_and_verify_succeed_at_every_spec_version(tmp_path, version):
    parent = write_bars(tmp_path, {"EURUSD": random_walk_bars()})
    path = write_prepared_toml(tmp_path, parent.dataset_id)
    path.write_text(path.read_text().replace("version = 1", f"version = {version}"))
    config = load_prepared_config(path)

    ref = build_prepared(config)

    opened = {}
    for role, dataset in ref.datasets.items():
        opened[role] = open_prepared(config.paths.prepared_root, dataset.dataset_id)
        assert opened[role].manifest["parent_ingestion"] == parent.dataset_id
        assert opened[role].manifest["build_id"] == ref.build_id
    assert main(["prepare", "verify", ref.build_id, "--config", str(path)]) == 0
    assert opened["train"].read("EURUSD")["usable"].sum() > 0
