"""The evaluation sweep: one episode per Session, chronologically, chained.

``run_sweep(env, policy)`` plays ``policy`` (any callable from observation to a
level index, so a scripted baseline needs no model) through the same engine as
training, one episode per runnable Session of the Dataset. Each episode starts
flat on the Session's first usable Bar with the equity the previous one ended on
and ends at the Session's end (flat if the Session-final Bar was reached). A
closeout ends the whole sweep: the remaining Sessions are not run. Everything is
recorded in one Ledger, whose ``episodes`` table is the per-Session record.

The env must have been made with ``record_ledger=True``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .ledger import Ledger

__all__ = ["SweepResult", "run_sweep"]


@dataclass(frozen=True)
class SweepResult:
    """The sweep's Ledger and what it covered.

    ``sessions_run`` are Session ordinals in the order played, ``skipped`` the ones
    with nothing to trade, and ``closeout`` whether a closeout stopped the sweep
    (``sessions_run[-1]`` is then the Session it happened in).
    """

    ledger: Ledger
    sessions_run: list[int]
    skipped: list[int]
    closeout: bool


def run_sweep(
    env: Any,
    policy: Callable[[Any], int],
    *,
    sessions: Iterable[int] | None = None,
) -> SweepResult:
    """Run ``policy`` over ``sessions`` (default: every runnable Session).

    ``policy(observation)`` returns an action index into the env's level set.
    """
    if not env.records_ledger:
        raise ValueError("a sweep needs an env made with record_ledger=True")
    runnable = env.sweep_sessions()
    if sessions is None:
        wanted = list(range(env.session_count))
    else:
        wanted = sorted(set(sessions))
    skipped = [k for k in wanted if k not in runnable]
    wanted = [k for k in wanted if k in runnable]
    if not wanted:
        raise ValueError("no runnable Session to sweep")
    played: list[int] = []
    closeout = False
    equity: float | None = None
    for k in wanted:
        options: dict[str, Any] = {"session": k}
        if equity is not None:
            options.update(equity=equity, chain=True)
        obs, _ = env.reset(options=options)
        played.append(k)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        equity = env.equity
        if info["ending"] == "closeout":
            closeout = True
            break
    return SweepResult(env.ledger(), played, skipped, closeout)
