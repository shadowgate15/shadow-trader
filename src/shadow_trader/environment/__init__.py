"""Trading environment Subsystem.

This ticket's pieces are the Gymnasium-free engine and ledger layers:

- ``EngineConfig`` / ``InstrumentSpec``: frozen, validated, hashable configuration.
- ``initial_state``, ``step``, ``Market``, ``EngineState``, ``StepEvents``: the pure
  engine implementing the Execution model.
- ``market_from_row`` and the calendar helper (``session_calendar``): ``Market`` and
  its Session-final, rollover and fillable facts from timestamps.
- ``build_observation``, ``Observation``, ``ObservationConfig``: the pure,
  Gymnasium-free observation assembler shared by training and live.
- ``LedgerBuilder``, ``Ledger``, ``verify_ledger``: reward-independent tables and
  the invariant checker. Positions are FIFO lots; closed lots are in
  ``StepEvents.closed_lots`` and the ledger ``closed_lots`` table.
- ``EnvConfig`` (with ``RewardConfig`` and ``EpisodeConfig``) configures the env
  layer; ``make_env`` and ``TradingEnv`` build and are the Gymnasium env. Those two
  are loaded lazily so importing the package never imports Gymnasium.
"""

from typing import TYPE_CHECKING, Any

from .config import ConfigError, EngineConfig, InstrumentSpec
from .engine import (
    ClosedLot,
    EngineState,
    Fill,
    Lot,
    Market,
    StepEvents,
    initial_state,
    level_to_units,
    step,
)
from .env_config import EnvConfig, EpisodeConfig, RewardConfig
from .ledger import (
    CLOSED_LOTS_SCHEMA,
    FILLS_SCHEMA,
    STEPS_SCHEMA,
    Ledger,
    LedgerBuilder,
    LedgerError,
    verify_ledger,
)
from .observation import Observation, ObservationConfig, build_observation
from .session_calendar import market_from_row

if TYPE_CHECKING:
    from .env import TradingEnv, make_env


def __getattr__(name: str) -> Any:
    if name in ("TradingEnv", "make_env"):
        from . import env

        return getattr(env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CLOSED_LOTS_SCHEMA",
    "FILLS_SCHEMA",
    "STEPS_SCHEMA",
    "ClosedLot",
    "ConfigError",
    "EngineConfig",
    "EngineState",
    "EnvConfig",
    "EpisodeConfig",
    "Fill",
    "InstrumentSpec",
    "Ledger",
    "LedgerBuilder",
    "LedgerError",
    "Lot",
    "Market",
    "Observation",
    "ObservationConfig",
    "RewardConfig",
    "StepEvents",
    "TradingEnv",
    "build_observation",
    "initial_state",
    "level_to_units",
    "make_env",
    "market_from_row",
    "step",
    "verify_ledger",
]
