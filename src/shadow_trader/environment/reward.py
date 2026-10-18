"""The reward: a pure function of two equity values (Gymnasium-free).

Because it needs only the equity series, any run can be re-scored from its Ledger
with a different ``reward_scale``. No shaping, no special treatment of closeouts
or clips in v1.
"""

from __future__ import annotations

import math

from .env_config import RewardConfig

__all__ = ["log_return_reward"]


def log_return_reward(
    equity_before: float, equity_after: float, config: RewardConfig
) -> float:
    """``ln(equity_after / equity_before) x reward_scale``."""
    if equity_before <= 0 or equity_after <= 0:
        raise ValueError(
            f"equity must stay positive to take a log return, got {equity_before} "
            f"then {equity_after} (closeout is not implemented yet)"
        )
    return math.log(equity_after / equity_before) * config.reward_scale
