"""Toy stateful per-Bar step loop in several implementations.

Same logic everywhere: a pending-order fill at next open, stop/target checks
against high/low, position/cash/equity/peak-drawdown state. Not a model of the
real executor; it only gives an order of magnitude for loop overhead.

Run: uv run python bench_loop.py [n_bars]
"""

import platform
import sys
import time
from dataclasses import dataclass

import numba
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(0)
close = 100.0 + np.cumsum(rng.normal(0, 0.1, N))
open_ = np.concatenate([[close[0]], close[:-1]])
high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.05, N))
low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.05, N))
sig = rng.integers(-1, 2, N).astype(np.int8)  # -1, 0, +1 desired side
STOP, TGT = 0.5, 0.8


def loop_py(o, h, l, c, s):  # noqa: E741
    """Pure Python, lists of floats, locals as state."""
    pos = 0
    entry = 0.0
    cash = 10_000.0
    pending = 0
    peak = cash
    maxdd = 0.0
    trades = 0
    for i in range(len(o)):
        if pending != 0:
            pos = pending
            entry = o[i]
            pending = 0
            trades += 1
        if pos != 0:
            if pos > 0:
                if l[i] <= entry - STOP:
                    cash += (-STOP) * pos
                    pos = 0
                elif h[i] >= entry + TGT:
                    cash += TGT * pos
                    pos = 0
            else:
                if h[i] >= entry + STOP:
                    cash += (-STOP) * (-pos)
                    pos = 0
                elif l[i] <= entry - TGT:
                    cash += TGT * (-pos)
                    pos = 0
        if pos == 0 and s[i] != 0:
            pending = s[i]
        eq = cash + (c[i] - entry) * pos if pos != 0 else cash
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > maxdd:
            maxdd = dd
    return cash, maxdd, trades


@dataclass(slots=True)
class Bar:
    o: float
    h: float
    l: float  # noqa: E741
    c: float
    s: int


@dataclass(slots=True)
class State:
    pos: int = 0
    entry: float = 0.0
    cash: float = 10_000.0
    pending: int = 0
    peak: float = 10_000.0
    maxdd: float = 0.0
    trades: int = 0


def step(st: State, b: Bar) -> None:
    """One step, object state and Bar object per call (executor-shaped)."""
    if st.pending != 0:
        st.pos = st.pending
        st.entry = b.o
        st.pending = 0
        st.trades += 1
    if st.pos != 0:
        if st.pos > 0:
            if b.l <= st.entry - STOP:
                st.cash += (-STOP) * st.pos
                st.pos = 0
            elif b.h >= st.entry + TGT:
                st.cash += TGT * st.pos
                st.pos = 0
        else:
            if b.h >= st.entry + STOP:
                st.cash += (-STOP) * (-st.pos)
                st.pos = 0
            elif b.l <= st.entry - TGT:
                st.cash += TGT * (-st.pos)
                st.pos = 0
    if st.pos == 0 and b.s != 0:
        st.pending = b.s
    eq = st.cash + (b.c - st.entry) * st.pos if st.pos != 0 else st.cash
    if eq > st.peak:
        st.peak = eq
    dd = st.peak - eq
    if dd > st.maxdd:
        st.maxdd = dd


def loop_objects(o, h, l, c, s):  # noqa: E741
    st = State()
    for i in range(len(o)):
        step(st, Bar(o[i], h[i], l[i], c[i], s[i]))
    return st.cash, st.maxdd, st.trades


# numba: same body as loop_py, compiled. Reuses the source function.
loop_nb = numba.njit(cache=False)(loop_py.__wrapped__ if hasattr(loop_py, "__wrapped__") else loop_py)


def timeit(fn, *args, repeat=3):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t)
    return best, out


def report(name, n, secs, out):
    print(f"{name:42s} {n/secs:>14,.0f} steps/s  ({secs*1e3:9.1f} ms for {n:,})  {out}")


if __name__ == "__main__":
    print(platform.platform(), platform.python_version(), "numpy", np.__version__,
          "numba", numba.__version__)
    # Python-level inputs
    lo, hi, op, cl, sg = low.tolist(), high.tolist(), open_.tolist(), close.tolist(), sig.tolist()
    n_py = min(N, 1_000_000)
    ref = loop_py(op[:n_py], hi[:n_py], lo[:n_py], cl[:n_py], sg[:n_py])
    secs, out = timeit(loop_py, op[:n_py], hi[:n_py], lo[:n_py], cl[:n_py], sg[:n_py])
    report("pure Python, lists, local-var state", n_py, secs, out)
    secs, out = timeit(loop_objects, op[:n_py], hi[:n_py], lo[:n_py], cl[:n_py], sg[:n_py])
    report("pure Python, State + Bar objects/step", n_py, secs, out)
    assert out == ref, (out, ref)
    # Python loop indexing numpy arrays directly (np.float64 scalars)
    n_np = min(N, 200_000)
    secs, out = timeit(loop_py, open_[:n_np], high[:n_np], low[:n_np], close[:n_np], sig[:n_np], repeat=2)
    report("pure Python loop indexing numpy arrays", n_np, secs, out)
    # numba
    t = time.perf_counter()
    loop_nb(open_[:10], high[:10], low[:10], close[:10], sig[:10])
    print(f"numba first-call compile: {(time.perf_counter()-t):.2f} s")
    secs, out = timeit(loop_nb, open_, high, low, close, sig, repeat=5)
    report("numba njit over numpy arrays", N, secs, out)
    ref_full = loop_py(op, hi, lo, cl, sg)
    print("numba == pure Python full run bitwise:", out == ref_full, out, ref_full)
    outs = {loop_nb(open_, high, low, close, sig) for _ in range(20)}
    print("numba 20 repeated runs distinct results:", len(outs))
