"""Episode ending reasons: the one definition every module shares.

``CLOSEOUT`` is the only ``terminated`` ending; the rest are time-based
(``truncated``). ``HALT`` and ``STOPPED`` are live only, so a simulation never
produces them. The engine's ``OrderPlan.override`` reuses ``SESSION_END`` and
``CLOSEOUT`` for the same two causes.
"""

from __future__ import annotations

__all__ = [
    "CLOSEOUT",
    "DATASET_END",
    "ENDINGS",
    "HALT",
    "SESSION_END",
    "STOPPED",
    "WINDOW_END",
]

CLOSEOUT = "closeout"
SESSION_END = "session_end"
DATASET_END = "dataset_end"
WINDOW_END = "window_end"
HALT = "halt"  # live only: the operator or watchdog halted the run
STOPPED = "stopped"  # live only: the run was stopped

ENDINGS = (CLOSEOUT, SESSION_END, DATASET_END, WINDOW_END, HALT, STOPPED)
