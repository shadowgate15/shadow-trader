"""Trading environment Subsystem.

This ticket's pieces are the Gymnasium-free engine and ledger layers:

- ``EngineConfig`` / ``InstrumentSpec``: frozen, validated, hashable configuration.
- ``initial_state``, ``step``, ``Market``, ``EngineState``, ``StepEvents``: the pure
  engine implementing the Execution model.
- ``LedgerBuilder``, ``Ledger``, ``verify_ledger``: reward-independent tables and
  the invariant checker. Positions are FIFO lots; closed lots are in
  ``StepEvents.closed_lots`` and the ledger ``closed_lots`` table.
"""

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
from .ledger import (
    CLOSED_LOTS_SCHEMA,
    FILLS_SCHEMA,
    STEPS_SCHEMA,
    Ledger,
    LedgerBuilder,
    LedgerError,
    verify_ledger,
)

__all__ = [
    "CLOSED_LOTS_SCHEMA",
    "FILLS_SCHEMA",
    "STEPS_SCHEMA",
    "ClosedLot",
    "ConfigError",
    "EngineConfig",
    "EngineState",
    "Fill",
    "InstrumentSpec",
    "Ledger",
    "LedgerBuilder",
    "LedgerError",
    "Lot",
    "Market",
    "StepEvents",
    "initial_state",
    "level_to_units",
    "step",
    "verify_ledger",
]
