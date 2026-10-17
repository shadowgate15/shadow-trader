# Stepping loop performance options (issue #97)

Question: for a stateful per-Bar execution/step loop on a local Mac with Python 3.13, what are the realistic options (pure Python, numpy, numba, others), their steps-per-second order of magnitude, determinism properties, install and cp313 support, and the cost of handing a prepared Dataset (polars/Parquet) to the loop as arrays? Map: #95. Charter: #8. Researched 2026-10-17. Throwaway research branch `research/stepping-loop-performance`, never merged. Polars is the dataframe engine for features and is not under evaluation as the stepping engine; only the polars-to-array handoff is measured.

**Status:** facts and small local measurements only. No choice is made here.

Evidence labels: **[D]** official documentation read this session (numba docs via Context7); **[P]** PyPI JSON metadata (`pypi_meta.py`); **[T]** local measurement this session; **[U]** not verified.

Machine: Apple M1 Max, macOS 27.0.1 arm64, CPython 3.13.15, uv 0.13.0, numpy 2.5.4, numba 0.68.0 (llvmlite 0.50.0), Cython 3.3.0, polars 2.0.0, pyarrow 26.0.0 (scratch uv project outside the repo; repo `pyproject.toml` untouched). Single machine, best-of-N wall time, not a proof; absolute numbers will move with the real loop body.

## Method

`bench_loop.py` runs one toy stateful loop (pending order fills at next open, stop/target checks against high/low, position/cash/equity/peak-drawdown state, about 25 branch/arith operations per Bar) in several implementations on a 5M-bar random-walk series (pure-Python variants on the first 1M or 200k). The loop body is far smaller than a real executor (no fees, slippage, multiple orders, logging, event emission), so per-step rates are an upper bound for a given implementation style; the ratios between styles are the more portable fact. `bench_handoff.py` measures polars/Parquet to array conversion on a 5M-row, 17-column Dataset-shaped frame (ns UTC timestamp, 5 float64 price/volume columns, 10 float64 feature columns with leading warm-up nulls, int8 label; 597 MB Parquet).

## Steps per second (order of magnitude) [T]

| Implementation | Steps/s | Time for N bars | Notes |
|---|---|---|---|
| Pure Python, Python lists, state in local variables | ~5.6 M | 180 ms / 1M | Inputs first converted to lists (see handoff) |
| Pure Python, `State` (slots dataclass) + `Bar` object per step, function call per step | ~2.9 M | 343 ms / 1M | Shape closest to an executor with a `step(state, bar)` interface. Same results as the local-variable version (bitwise equal) |
| Pure Python loop indexing numpy arrays (np.float64 scalars) | ~2.1 M | 97 ms / 200k | Slower than lists; numpy scalar ops are costlier than float ops. Numpy gives no speedup for a per-Bar branching loop |
| numba `@njit` over numpy arrays | ~460 M | 10.8 ms / 5M | Same source function as the pure-Python version, compiled. First call compiles: 0.56 s in this probe |
| Cython (typed memoryviews, `cc -O2`) | ~490 M | 10.2 ms / 5M | Build 3.6 s (cython + cc). Needs a C compiler (`/usr/bin/cc` present); not packaged as a wheel for the loop itself |

Orders of magnitude: pure Python is millions of steps/s (10^6), compiled loops are hundreds of millions (10^8), about 80x to 170x apart in this toy. For reference, 5M one-minute bars is roughly 13 years of 24x5 FX-style minute bars (about 374k per year); the pure-Python object style would take about 1.7 s, the compiled loops about 10 ms. Whether either matters depends on how many times a loop is run (sweeps, walk-forward folds) and how much work the real step does. [U] Not measured: a real executor body, per-step Python callbacks from a compiled loop, event/log emission cost.

Not measured: Rust via PyO3/maturin (`rustc`/`cargo` exist on this machine; maturin 1.15.0 on PyPI [P], no wheel for it needed for use as a build tool) and mypyc. Both are listed as possible families only; no numbers were taken. [U]

## Determinism properties

- **Pure Python:** IEEE-754 double arithmetic evaluated in source order; no reassociation. Object-style and local-variable-style loops gave bitwise-identical results [T].
- **numba:** `fastmath` defaults to off; with it off, a reduction "must accumulate in strict order". `fastmath=True` permits floating-point reassociation and changes results in some cases (documented example: `(x - y) + y` returns nan with `fastmath=False` and 0.0 with `True` for `x=0, y=inf`) [D: numba.readthedocs.io/en/stable/user/performance-tips.html]. `parallel=True` with `prange` accumulates per-thread partial sums then combines them (the docs say it is only safe "assuming it is safe for the sum to be accumulated out of order") [D: same page and /user/parallel.html]. A sequential stateful step loop uses neither.
- **Measured:** numba result equals the pure-Python result bitwise on 5M bars (cash, max drawdown, trade count), and 20 repeated numba runs produced 1 distinct result [T]. Cython (default flags, no `-ffast-math`) also matched bitwise [T]. These are one platform, one input, one loop body; floating-point contraction (fused multiply-add) is a compiler decision and was not separately tested [U].
- **Integer/branch logic:** deterministic in all of the above; the only drift source found in the documentation is floating-point reassociation, which is opt-in in numba.
- **Cross-machine:** [U] no test across CPU architectures or numba/LLVM versions.

## Install and Python 3.13 support

[P] PyPI metadata read this session, macOS arm64 cp313 wheels:

| Package | Version (upload date) | requires-python | cp313 macOS arm64 wheel |
|---|---|---|---|
| numpy | 2.5.4 (2026-10-10) | >=3.12 | yes (`macosx_11_0_arm64`, `macosx_14_0_arm64`) |
| numba | 0.68.0 (2026-09-30) | >=3.10 | yes (`macosx_12_0_arm64`) |
| llvmlite (numba dependency) | 0.50.0 (2026-09-29) | >=3.10 | yes (`macosx_12_0_arm64`) |
| Cython | 3.3.0 (2026-08-22) | >=3.9 | yes (`macosx_11_0_arm64`) |
| pyarrow | 26.0.0 (2026-10-09) | >=3.11 | yes (`macosx_12_0_arm64`); already a project dependency |
| polars | 2.0.0 (2026-10-06) | >=3.10 | no cp313-tagged wheel listed on the main package; the compiled runtime ships as `polars-runtime-32` (installs and imports on 3.13 here [T]) |

- `uv add numpy numba polars pyarrow cython` in a scratch Python 3.13 project resolved and installed in well under a second with warm cache: numba 0.68.0 brought llvmlite 0.50.0 and numpy 2.5.4 [T]. No compiler or system LLVM needed for numba (llvmlite wheel bundles LLVM).
- Import cost (cold process, `uv run python -c "import X"` wall time incl. uv overhead): numpy 0.07 s, polars 0.13 s, numba 0.17 s (`-X importtime` shows ~142 ms for numba itself) [T].
- numba JIT compile of the toy loop: 0.56 s on first call per process [T]. numba's `cache=True` option exists to persist compiled functions on disk; not measured here [U].
- Cython requires a C compiler at build time (macOS: Xcode command line tools provide `cc`); the end user of the compiled extension does not. Build step adds packaging work to a uv_build-based project [U: not tried inside the repo].
- Pure Python needs nothing. numpy is not a direct dependency in this branch's `pyproject.toml` (which lists `pyarrow`, `rich`); whether it is present transitively was not checked [U].

## Handoff cost: polars/Parquet Dataset to arrays [T]

5M rows, 17 columns, 597 MB Parquet (651 MB in memory). Best of 5 unless noted.

| Operation | Time | Notes |
|---|---|---|
| `pl.read_parquet` all columns | 120 ms | Result columns have many chunks (41 chunks for one 5M float column in a separate probe) |
| `pl.scan_parquet(...).select(5 cols).collect()` | 27 ms | Column pruning; only needed columns read |
| `pyarrow.parquet.read_table` all columns | 118 ms | Same order as polars |
| `Series.to_numpy()` float64, no nulls, multi-chunk (as read) | 4.0 ms (40 MB) | A copy; `allow_copy=False` raises `RuntimeError: copy not allowed` because the Series is multi-chunk |
| `df.rechunk()` then `Series.to_numpy(allow_copy=False)` | rechunk 3.4 ms for one 5M column; to_numpy 0.04 ms | Zero-copy; resulting array is **read-only** (`writeable=False`); numba accepted the read-only array and gave the same result as on a copy |
| single-chunk Series (built from numpy) `to_numpy(allow_copy=False)` | succeeds | Read-only view |
| `Series.to_numpy()` float64 with nulls | 13.8 ms | Copy; nulls become NaN (50 nulls to 50 NaN); `allow_copy=False` raises. `fill_null(nan)` first then `to_numpy()` 8.8 ms |
| Datetime[ns, UTC] `Series.to_numpy()` | 4.9 ms | Gives `datetime64[ns]` (UTC tz dropped). `ts.dt.epoch("ns").to_numpy()` gives int64, 3.9 ms |
| int8 label `to_numpy()` | 0.13 ms | |
| 5 columns (OHLC+label) individually | 16.6 ms | |
| 10 feature columns `df.to_numpy()` as one 2D matrix | 21.8 ms | Returned Fortran-ordered (column-contiguous): each column slice is contiguous, but a row `m[i, :]` is strided. `order="c"` 18.1 ms |
| pyarrow column `.to_numpy(zero_copy_only=False)` | 14.0 ms | Copy (chunked) |
| `Series.to_list()` (for a pure-Python loop) | 54.7 ms per 5M-value column | About 11 ns per value; produces Python float objects (memory heavy: ~32 bytes each incl. list slot, [U] not measured) |
| `iter_rows()` over 5 columns, 1M rows | 197 ms | About 200 ns per row, comparable in magnitude to the pure-Python loop step itself (180 ns); row-wise iteration would roughly double the cost of a pure-Python loop |

Facts worth noting:
- The handoff is small relative to reading Parquet (tens of ms versus ~120 ms for a full read) and is negligible next to a pure-Python loop (hundreds of ms to seconds for millions of bars), but comparable to a compiled loop's own runtime (10 ms for 5M bars): at compiled speeds, the Parquet read and null/datetime conversions dominate wall time.
- Nulls become NaN on conversion. Warm-up nulls in feature columns therefore surface as NaN inside the loop; comparisons against NaN are false in every implementation tested (pure Python, numba, Cython), which can silently change branch behaviour if not handled [U: not exercised in the toy loop].
- Datetime handoff needs an explicit choice (`datetime64[ns]` array or int64 epoch). Time zone is dropped in the numpy array.
- Zero-copy requires a single chunk and no nulls; the result is read-only.

## Other options named but not measured

- **Rust (PyO3/maturin)**, **mypyc**, **C extension via cffi**: possible families for a compiled loop; no measurement, no PyPI cp313 check beyond maturin metadata. [U]
- **Vectorised numpy/polars formulations** of execution logic: not applicable to branching per-Bar state in general; the toy shows numpy scalar indexing is slower than lists in a Python loop. Fills that depend on prior state (open position, pending order) are sequentially dependent. [T for the slowdown, otherwise reasoning]
- **numba jitclass / structref** for object-style state inside a compiled loop, and Python callbacks per step from compiled code: not tried. [U]

## Reproduce

Files on this branch under `docs/research/stepping-loop-performance/`:
- `bench_loop.py` (pure Python, objects, numpy indexing, numba) and `loop_cy.pyx` + `run_cy.py` (Cython build and run).
- `bench_handoff.py` (polars/Parquet handoff).
- `pypi_meta.py` (PyPI version, requires-python, cp313 wheel listing).

They were run with `uv run --project <scratch project>` where the scratch project had numpy, numba, polars, pyarrow, cython added via `uv add`. The repo's own `pyproject.toml` does not include numba/polars/cython.
