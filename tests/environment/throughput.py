"""Engine-only throughput: ``engine.step`` alone over a Dataset's Bars.

Everything the engine does not own (reading, scaling, building ``Market`` rows,
observations, the Ledger, Gymnasium) is done before the clock starts, so the number
is the cost of ``step`` itself. Unfillable Bars are skipped (the env carries them
inside one step), and a closeout restarts the account so the run goes on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from shadow_trader.environment import EngineConfig
from shadow_trader.environment.engine import initial_state, step
from shadow_trader.environment.market_data import MarketData

# Held in turn, 40 Bars each: long, flat, short (enough variety to open, hold,
# reduce and flip positions, so every branch of ``step`` is exercised).
LEVELS = (1.0, 0.0, -1.0)
HOLD = 40


@dataclass(frozen=True)
class Throughput:
    steps: int
    seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds


def measure_engine_throughput(
    data: MarketData, config: EngineConfig, *, max_steps: int | None = None
) -> Throughput:
    """Time ``step`` over every fillable Bar of ``data`` (up to ``max_steps``)."""
    rows = [i for i in range(len(data)) if data.fillable[i]]
    if max_steps is not None:
        rows = rows[:max_steps]
    markets = [data.market(i) for i in rows]
    targets = [LEVELS[(k // HOLD) % len(LEVELS)] for k in range(len(markets))]
    state = initial_state(config)
    start = time.perf_counter()
    for market, target in zip(markets, targets, strict=True):
        state, events = step(state, config, market, target)
        if events.closeout:
            state = initial_state(config)
    seconds = time.perf_counter() - start
    return Throughput(len(markets), seconds)
