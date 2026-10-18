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
  the invariant checker.
"""

from .config import ConfigError, EngineConfig, InstrumentSpec
from .engine import (
    EngineState,
    Fill,
    Market,
    StepEvents,
    initial_state,
    level_to_units,
    step,
)
from .ledger import (
    FILLS_SCHEMA,
    STEPS_SCHEMA,
    Ledger,
    LedgerBuilder,
    LedgerError,
    verify_ledger,
)
from .observation import Observation, ObservationConfig, build_observation
from .session_calendar import market_from_row

__all__ = [
    "FILLS_SCHEMA",
    "STEPS_SCHEMA",
    "ConfigError",
    "EngineConfig",
    "EngineState",
    "Fill",
    "InstrumentSpec",
    "Ledger",
    "LedgerBuilder",
    "LedgerError",
    "Market",
    "Observation",
    "ObservationConfig",
    "StepEvents",
    "build_observation",
    "initial_state",
    "level_to_units",
    "market_from_row",
    "step",
    "verify_ledger",
]
