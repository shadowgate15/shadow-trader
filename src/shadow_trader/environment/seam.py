"""The versioned seam: the only tier Broker integration and Live orchestration import.

Everything else in the package is internal. ``SEAM_VERSION`` is an integer bumped on
any breaking change to a name exported here (and recorded in the ledger manifest).

Live restarts: the broker's ``AccountSnapshot`` plus the persisted ``TargetMemo``
rebuild an ``EngineState`` via ``engine_state_from_snapshot``. The snapshot wins on
position, cash and equity; the frozen target comes from the memo (a missing memo
means flat, nothing pending) and FIFO lots collapse to one lot per position at its
average price. These are reconciled, not reconstructed. Live orchestration owns the
memo and the live ledger; a Halt flatten uses ``halt_flatten``, which bypasses
``plan_order`` and flags its fills ``halt``.
"""

from __future__ import annotations

from .config import EngineConfig, InstrumentSpec
from .endings import HALT, STOPPED
from .engine import (
    AccountSnapshot,
    EngineState,
    Market,
    OrderPlan,
    StepEvents,
    TargetMemo,
    engine_state_from_snapshot,
    halt_flatten,
    initial_state,
    memo_from_state,
    plan_order,
    snapshot_from_state,
    step,
)
from .ledger import (
    CLOSED_LOTS_SCHEMA,
    EPISODES_SCHEMA,
    FILLS_SCHEMA,
    STEPS_SCHEMA,
    Ledger,
    LedgerBuilder,
)
from .observation import (
    Observation,
    ObservationConfig,
    build_observation,
    window_inputs,
)
from .session_calendar import (
    bars_to_session_final,
    is_fillable,
    is_session_final,
    market_from_row,
    rollover_days,
)

#: 2: ``EngineConfig.names``/``InstrumentSpec.name`` replace ``symbols``/``symbol``,
#: ``effective_leverage_cap`` is gone, and the episodes table carries counters.
SEAM_VERSION = 2

__all__ = [
    "CLOSED_LOTS_SCHEMA",
    "EPISODES_SCHEMA",
    "FILLS_SCHEMA",
    "HALT",
    "SEAM_VERSION",
    "STEPS_SCHEMA",
    "STOPPED",
    "AccountSnapshot",
    "EngineConfig",
    "EngineState",
    "InstrumentSpec",
    "Ledger",
    "LedgerBuilder",
    "Market",
    "Observation",
    "ObservationConfig",
    "OrderPlan",
    "StepEvents",
    "TargetMemo",
    "bars_to_session_final",
    "build_observation",
    "engine_state_from_snapshot",
    "halt_flatten",
    "initial_state",
    "is_fillable",
    "is_session_final",
    "market_from_row",
    "memo_from_state",
    "plan_order",
    "rollover_days",
    "snapshot_from_state",
    "step",
    "window_inputs",
]
