"""Every value a run depends on, pinned explicitly in one flat module.

No CLI flag overrides any of these: a run's parameters have a single source.
The 1e4 observation scaling is a stated constant elsewhere, not a key here.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

# Root for everything written to disk: raw/ (vendor files verbatim), later
# derived datasets, models and metrics.
DATA_DIR = Path("data")

HISTORY_START = datetime(2016, 1, 4, tzinfo=UTC)
HISTORY_END = datetime(2024, 8, 23, tzinfo=UTC)

# Chronological train / validation / test fractions.
SPLIT = (0.70, 0.15, 0.15)

SPREAD_PIPS = 1.0

GAP_THRESHOLD = timedelta(hours=4)
GAP_THRESHOLD_BARS = 48  # 4 hours of M5 Bars

WINDOW_SIZE = 24  # N
EPISODE_LENGTH = 288  # Bars

TOTAL_TIMESTEPS = 1_000_000
SEED = 42
N_BASELINE_DRAWS = 20
DEVICE = "cpu"  # pinned, never automatic selection

# PPO block, written out verbatim (SB3 2.9). SDE parameters are omitted: they
# do not apply to a Discrete action space.
PPO = {
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
    # The one deliberate deviation from the SB3 default (0.0).
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    "device": DEVICE,
    "seed": SEED,
}
