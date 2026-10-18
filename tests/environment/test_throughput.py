"""The engine-only throughput helper runs on the fixture (rate is informational)."""

from __future__ import annotations

from shadow_trader.environment import EngineConfig, market_data

from .fixture import load_fixture
from .throughput import measure_engine_throughput


def test_throughput_helper_steps_every_fillable_bar_of_the_fixture():
    dataset, scaler = load_fixture()
    data = market_data.from_prepared(dataset, scaler, "EURUSD", 5)
    measured = measure_engine_throughput(data, EngineConfig())
    assert measured.steps == int(data.fillable.sum())
    assert measured.steps_per_second > 0
