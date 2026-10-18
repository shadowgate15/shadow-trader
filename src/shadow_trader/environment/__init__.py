"""Trading environment Subsystem.

This ticket's pieces are the Gymnasium-free engine and ledger layers:

- ``EngineConfig`` / ``InstrumentSpec``: frozen, validated, hashable configuration.
- ``initial_state``, ``step``, ``Market``, ``EngineState``, ``StepEvents``: the pure
  engine implementing the Execution model.
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
    "StepEvents",
    "initial_state",
    "level_to_units",
    "step",
    "verify_ledger",
]
