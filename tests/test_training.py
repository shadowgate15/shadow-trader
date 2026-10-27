from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from stable_baselines3 import PPO

from shadow_trader import bars, cli, config, training

T0 = datetime(2016, 1, 4, tzinfo=UTC)
M5 = timedelta(minutes=5)
COUNT = 3000
TINY_PPO = {**config.PPO, "n_steps": 64, "batch_size": 32, "n_epochs": 2}


@pytest.fixture
def stored_bars(tmp_path, monkeypatch):
    """Synthetic Bars on disk: one contiguous run, enough for a train Episode."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    rng = np.random.default_rng(1)
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, COUNT))
    open_ = np.concatenate([[1.1], close[:-1]])
    high = np.maximum(open_, close) + 1e-4
    low = np.minimum(open_, close) - 1e-4
    cols = {}
    for field, values in zip(bars.FIELDS, (open_, high, low, close), strict=True):
        cols[f"bid_{field}"] = list(values - 5e-5)
        cols[f"ask_{field}"] = list(values + 5e-5)
    times = [T0 + i * M5 for i in range(COUNT)]
    path = bars.bars_path()
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"time": times, **cols}, schema=bars.SCHEMA), path)
    return close


def weights(model: PPO) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.policy.parameters()]


def test_load_market_gives_mid_prices_and_structure(stored_bars):
    market = training.load_market()
    assert market.prices.shape == (COUNT, 4)
    assert market.structure.bar_count == COUNT
    np.testing.assert_allclose(market.prices[:, 3], stored_bars)


def test_same_config_and_seed_gives_the_same_model(stored_bars):
    market = training.load_market()
    a = training.fit(market, total_timesteps=256, ppo=TINY_PPO)
    b = training.fit(market, total_timesteps=256, ppo=TINY_PPO)
    assert all(torch.equal(x, y) for x, y in zip(weights(a), weights(b), strict=True))


def test_a_different_seed_gives_a_different_model(stored_bars):
    market = training.load_market()
    a = training.fit(market, total_timesteps=256, ppo=TINY_PPO)
    b = training.fit(market, total_timesteps=256, ppo={**TINY_PPO, "seed": 7})
    assert not all(
        torch.equal(x, y) for x, y in zip(weights(a), weights(b), strict=True)
    )


def test_cli_train_saves_a_loadable_model_and_prints_progress(
    stored_bars, monkeypatch, capsys
):
    monkeypatch.setattr(config, "TOTAL_TIMESTEPS", 256)
    monkeypatch.setattr(config, "PPO", TINY_PPO)
    cli.main(["train"])
    assert "total_timesteps" in capsys.readouterr().out
    path = training.model_path()
    assert path == config.DATA_DIR / "model" / "ppo_eurusd.zip"
    model = PPO.load(path, device="cpu")
    assert model.seed == 42
    assert model.policy.observation_space.shape == (73,)


def test_config_pins_the_requested_run():
    assert config.TOTAL_TIMESTEPS == 1_000_000
    assert config.PPO["device"] == "cpu"
    assert config.PPO["seed"] == 42
