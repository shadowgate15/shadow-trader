"""The reward: a pure function of two equity values (Gymnasium-free).

Because it needs only the equity series, any run can be re-scored from its Ledger
with a different ``reward_scale``. No shaping, no special treatment of closeouts
or clips in v1.
"""

from __future__ import annotations

import math

from .env_config import RewardConfig

__all__ = ["log_return_reward"]


# A wiped-out account (closeout on a gap can leave equity at or below zero) has no
# log; equity_after is floored at this fraction of equity_before so the reward stays
# finite. It is a numeric guard, not shaping: every other step is the plain log return.
WIPE_OUT_FLOOR = 1e-6


def log_return_reward(
    equity_before: float, equity_after: float, config: RewardConfig
) -> float:
    """``ln(equity_after / equity_before) x reward_scale``."""
    if equity_before <= 0:
        raise ValueError(f"equity before a step must be positive, got {equity_before}")
    floor = equity_before * WIPE_OUT_FLOOR
    return math.log(max(equity_after, floor) / equity_before) * config.reward_scale
