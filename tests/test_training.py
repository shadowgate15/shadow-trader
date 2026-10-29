import csv
from datetime import UTC, datetime, timedelta
from typing import Any

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
    assert market.mid.shape == (COUNT, 4)
    assert market.structure.bar_count == COUNT
    np.testing.assert_allclose(market.mid[:, 3], stored_bars)


def test_same_config_and_seed_gives_the_same_model(stored_bars):
    market = training.load_market()
    a = training.fit(market, total_timesteps=256, ppo=TINY_PPO).model
    b = training.fit(market, total_timesteps=256, ppo=TINY_PPO).model
    assert all(torch.equal(x, y) for x, y in zip(weights(a), weights(b), strict=True))


def test_a_different_seed_gives_a_different_model(stored_bars):
    market = training.load_market()
    a = training.fit(market, total_timesteps=256, ppo=TINY_PPO).model
    b = training.fit(market, total_timesteps=256, ppo={**TINY_PPO, "seed": 7}).model
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
    assert config.CHECKPOINT_EVERY_ROLLOUTS == 25


COLUMNS = [
    "timestep",
    "net_pips",
    "net_pct",
    "sharpe",
    "max_drawdown_pips",
    "max_drawdown_pct",
    "bars",
    "segments",
    "position_changes",
    "selected",
]


def test_cli_train_writes_the_learning_curve_and_prints_a_line_per_checkpoint(
    stored_bars, monkeypatch, capsys
):
    # 320 steps at n_steps 64 = 5 rollouts; every 2nd plus the last = 2, 4, 5.
    monkeypatch.setattr(config, "TOTAL_TIMESTEPS", 320)
    monkeypatch.setattr(config, "PPO", TINY_PPO)
    monkeypatch.setattr(config, "CHECKPOINT_EVERY_ROLLOUTS", 2)
    cli.main(["train"])
    out = capsys.readouterr().out
    path = config.DATA_DIR / "model" / "learning_curve.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert [r["timestep"] for r in rows] == ["128", "256", "320"]
    assert sum(r["selected"] == "True" for r in rows) <= 1
    lines = [ln for ln in out.splitlines() if ln.startswith("checkpoint ")]
    assert len(lines) == 3
    assert lines[0].startswith("checkpoint 1/3 @ 128: net ")
    assert lines[2].startswith("checkpoint 3/3 @ 320: net ")
    assert "sharpe " in lines[0] and ", changes " in lines[0]


def test_the_callback_leaves_the_final_weights_byte_identical(stored_bars):
    market = training.load_market()
    kwargs: dict[str, Any] = {"total_timesteps": 320, "ppo": TINY_PPO}
    plain = training.fit(market, checkpoint_every=None, **kwargs)
    scored = training.fit(market, checkpoint_every=2, **kwargs)
    assert plain.checkpoints == []
    assert len(scored.checkpoints) == 3
    assert all(
        torch.equal(x, y)
        for x, y in zip(weights(plain.model), weights(scored.model), strict=True)
    )


def test_the_final_rollout_is_scored_once_when_it_falls_on_the_cadence(stored_bars):
    # 4 rollouts, every 2nd: Checkpoints at rollouts 2 and 4, not a third.
    market = training.load_market()
    result = training.fit(market, total_timesteps=256, ppo=TINY_PPO, checkpoint_every=2)
    assert [c.timestep for c in result.checkpoints] == [128, 256]


def fake_scores(monkeypatch, plan):
    """Make successive Checkpoints score (net_pips, position_changes) from plan."""
    from shadow_trader import evaluation

    it = iter(plan)

    def score(env, policy):
        net, changes = next(it)
        return evaluation.Score(
            evaluation.Metrics(net, 0.0, 0.0, 0.0, 0.0, 10, 1), (changes,)
        )

    monkeypatch.setattr(evaluation, "score", score)


def run_train(monkeypatch, capsys):
    monkeypatch.setattr(config, "TOTAL_TIMESTEPS", 320)
    monkeypatch.setattr(config, "PPO", TINY_PPO)
    monkeypatch.setattr(config, "CHECKPOINT_EVERY_ROLLOUTS", 2)
    cli.main(["train"])
    out = capsys.readouterr().out
    with training.learning_curve_path().open() as f:
        return out, list(csv.DictReader(f))


def test_cli_train_saves_the_best_checkpoint_and_marks_it(
    stored_bars, monkeypatch, capsys
):
    # Checkpoints at 128, 256, 320; the 50-pip one never trades so cannot win.
    fake_scores(monkeypatch, [(5.0, 3), (50.0, 0), (9.0, 4)])
    out, rows = run_train(monkeypatch, capsys)
    assert [r["selected"] for r in rows] == ["False", "False", "True"]
    best = training.best_model_path()
    assert best == config.DATA_DIR / "model" / "ppo_eurusd_best.zip"
    assert PPO.load(best, device="cpu").num_timesteps == 320
    lines = [ln for ln in out.splitlines() if ln.startswith("checkpoint ")]
    assert lines[0].endswith(" *") and not lines[1].endswith(" *")
    assert lines[2].endswith(" *")
    assert "selected checkpoint @ 320" in out
    assert str(best) in out and str(training.model_path()) in out


def test_a_never_trading_checkpoint_cannot_be_selected(
    stored_bars, monkeypatch, capsys
):
    fake_scores(monkeypatch, [(5.0, 0), (50.0, 0), (9.0, 0)])
    out, rows = run_train(monkeypatch, capsys)
    assert {r["selected"] for r in rows} == {"False"}
    assert not training.best_model_path().exists()
    assert "no checkpoint" in out.lower()
    assert training.model_path().exists()


def test_ties_keep_the_earliest_checkpoint(stored_bars, monkeypatch, capsys):
    fake_scores(monkeypatch, [(7.0, 1), (7.0, 2), (3.0, 1)])
    _, rows = run_train(monkeypatch, capsys)
    assert [r["selected"] for r in rows] == ["True", "False", "False"]
    assert PPO.load(training.best_model_path(), device="cpu").num_timesteps == 128


def test_a_stale_best_file_is_removed_at_train_start(stored_bars, monkeypatch, capsys):
    stale = training.best_model_path()
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")
    fake_scores(monkeypatch, [(1.0, 0), (1.0, 0), (1.0, 0)])
    run_train(monkeypatch, capsys)
    assert not stale.exists()


def test_saving_the_best_leaves_final_weights_byte_identical(stored_bars, tmp_path):
    market = training.load_market()
    kwargs: dict[str, Any] = {"total_timesteps": 320, "ppo": TINY_PPO}
    plain = training.fit(market, checkpoint_every=None, **kwargs)
    best = tmp_path / "best.zip"
    saved = training.fit(market, checkpoint_every=2, best_path=best, **kwargs)
    assert any(c.selected for c in saved.checkpoints) == best.exists()
    assert all(
        torch.equal(x, y)
        for x, y in zip(weights(plain.model), weights(saved.model), strict=True)
    )
