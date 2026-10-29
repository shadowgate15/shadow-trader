from datetime import UTC, datetime, timedelta
from pathlib import Path

from shadow_trader import config


def test_pinned_run_constants():
    assert datetime(2016, 1, 4, tzinfo=UTC) == config.HISTORY_START
    assert datetime(2024, 8, 23, tzinfo=UTC) == config.HISTORY_END
    assert config.SPLIT == (0.70, 0.15, 0.15)
    assert config.SPREAD_PIPS == 1.0
    assert timedelta(hours=4) == config.GAP_THRESHOLD
    assert config.GAP_THRESHOLD_BARS == 48
    assert config.WINDOW_SIZE == 24
    assert config.EPISODE_LENGTH == 288
    assert config.TOTAL_TIMESTEPS == 1_000_000
    assert config.SEED == 42
    assert config.N_BASELINE_DRAWS == 20
    assert config.DEVICE == "cpu"


def test_ppo_block_is_pinned_verbatim():
    assert config.PPO == {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        "device": "cpu",
        "seed": 42,
    }


def test_ent_coef_deviation_is_marked_in_source():
    lines = Path(config.__file__).read_text().splitlines()
    i = next(i for i, x in enumerate(lines) if '"ent_coef"' in x)
    assert "deviat" in "\n".join(lines[max(0, i - 3) : i + 1]).lower()
