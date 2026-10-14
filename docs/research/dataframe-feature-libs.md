# Dataframe engines and feature libraries (issue #43)

Question: which dataframe engines and feature/indicator libraries can run one feature definition both as a batch over a Dataset and incrementally on each newly closed Bar, with exact parity? Report on rolling-window/state handling, Parquet I/O, Python 3.13/uv support, null handling and determinism. Map: #42. Charter: resolution comment on #7. Researched 2026-10-14. Throwaway research branch, never merged.

**Status:** facts only. No choice is made here.

Evidence labels: **[D]** vendor/project documentation read this session; **[P]** PyPI JSON metadata (pypi.org/pypi/<pkg>/json) read this session; **[R]** `uv pip compile --python-version 3.13` resolve run in a scratch project this session; **[T]** observed in a small local test this session (Python 3.13, macOS arm64, random-walk float64 close series of 5000 values; single probe, not a proof); **[U]** not verified.

## Summary table

| Candidate | Role | Latest on PyPI [P] | Python 3.13 | Incremental/streaming API | Batch/incremental parity |
|---|---|---|---|---|---|
| polars | dataframe engine | 2.0.0 (2026-10-06) | cp313-compatible, classifier 3.13 | None found. Expressions run on whole frames/LazyFrames (batch). A streaming *engine* exists but it batches large inputs, it is not per-Bar state | Same expression on a longer frame: first-N rows bitwise equal [T]. Not equal to a naive per-window recompute (diff ~1e-16) [T] |
| pandas | dataframe engine | 3.0.6 (2026-09-17) | cp313 wheels | Only `ewm(...).online()`, needs numba [D][T] | Rolling prefix-invariant bitwise [T]; online EWM differed from batch by 1.2e-5 in one probe [T] |
| duckdb | SQL engine over Parquet | 1.5.6 | cp313 wheels | None found [U] | Not tested |
| pyarrow | Arrow/Parquet I/O, compute kernels | 26.0.0 (already a project dependency) | cp313 wheels | None | Not tested |
| TA-Lib (ta-lib) | indicator library, C core | 0.8.2 (2026-10-12) | cp313 wheels, requires-python >=3.9 | **Yes**: `talib.stream` handles with `update()`, `peek()`, `copy()` [D] | Documented identical to Function API per bar [D]; bitwise equal for SMA, EMA, RSI in test [T] |
| polars-talib | TA-Lib as polars expressions | 0.2.0 (2026-09-23) requires `polars<2` | requires-python >=3.10 | None found [U] | n/a |
| polars-ta | indicators as polars expressions | 0.5.17 | requires-python >=3.8; no 3.13 classifier | None found [U] | n/a |
| pandas-ta | indicators on pandas | 0.4.71b0 (pre-release, 2025-09-14) | requires >=3.12; no 3.13 classifier | None found [U] | n/a |
| pandas-ta-classic | indicators on pandas (fork) | 0.8.32 (2026-09-14) | requires >=3.10, classifier 3.13 | None found [U] | n/a |
| ta (bukosabino) | indicators on pandas | 0.11.0 (2023-11-02) | no requires-python, no 3.13 classifier | None found [U] | n/a |
| river | online ML / online stats | 0.26.1 (2026-08-21) | cp313 wheels (no 3.13 classifier) | **Yes**: stats objects with `update`; `utils.Rolling` wraps `update`/`revert` [D] | Not tested |
| numba | JIT for custom loops | 0.68.0 | cp313 wheels | n/a (tool for writing one loop used both ways) | n/a |
| bottleneck | fast NaN-aware moving-window numpy funcs | 1.6.0 (2025-09-08) | cp313 wheels | None | n/a |
| tulipy | indicators, C core | 0.4.0 (2019-04-11) | no wheels listed for 3.13 [P] | n/a | n/a |
| finta | indicators on pandas | 1.3 (2021-04-03) | n/a | n/a | n/a |
| tsfresh | bulk time-series feature extraction | 0.21.2 (2026-05-31) | no 3.13 classifier | n/a (batch extractor) | n/a |

## Python 3.13 and uv

All of these were run through `uv pip compile --python-version 3.13 --python-platform aarch64-apple-darwin` [R]. Facts:

- `polars`, `pandas`, `duckdb`, `pyarrow`, `numba`, `bottleneck`, `river`, `ta-lib`, `pandas-ta-classic`, `ta` each resolve alone. `polars` resolves to `polars 2.0.0` plus `polars-runtime-32 2.0.0`.
- `polars` and `pandas` both pull `numpy 2.5.4` in the same environment; numpy 2.5.4 requires Python >=3.12 [P].
- **`polars-talib` 0.2.0 requires `polars<2,>=1.20`** [P]. With `polars` 2.0.0 requested, uv resolved `polars-talib 0.1.6` instead [R]. Whether 0.1.6 behaves as 0.2.0 documents is not checked.
- **`pandas-ta` 0.4.71b0 pins `numba==0.61.2`, `numpy>=2.2.6`, `pandas>=2.3.2`** [P]; resolved alongside pandas 3.0.6 and polars 2.0.0 it forces `numpy 2.2.6` and `llvmlite 0.44.0` rather than the latest [R]. Only pre-releases exist on PyPI for it ([P], version string `0.4.71b0`). The package carries "Copyright (c) 2025 Pandas TA"; no license text was read.
- `polars-ta` 0.5.17 depends on `polars>=1.28.0`, `polars-ols>=0.3.0`, `numba`, `numpy`, `pandas`, `more-itertools` [P]; it resolves with polars 2.0.0 [R].
- `ta-lib` 0.8.2: PyPI lists cp313 wheels. README states binary wheels for Python 3.9-3.14 on Linux (x86_64, arm64), macOS (x86_64, arm64), Windows; the README also states the TA-Lib C library "must be installed separately" as a dependency [D]. In this session `uv add ta-lib` on macOS arm64 installed and imported without a separate C library step [T]. Version mapping from README: ta-lib-python 0.5 to 0.8.x support NumPy 2; 0.4.x supports NumPy 1 [D].
- `polars-talib` README: "Binary wheels include the C library" and bundles TA-Lib 0.8.1 [D].
- Project dependencies today: `pyarrow>=26.0.0`, `rich>=15.0.0`; dev: hypothesis, pytest, ruff, ty (pyproject.toml). Neither polars nor pandas is present yet.

## Dataframe engines

### polars 2.0.0

- **Rolling windows:** `Expr.rolling_mean(window_size, weights=None, *, min_samples=None, center=False)`. "The window at a given row will include the row itself, and the `window_size - 1` elements before it." `min_samples` is "the number of values in the window that should be non-null before computing a result" and defaults to `window_size`. `min_periods` was renamed `min_samples` in 1.21.0 [D]. Sibling expressions (`rolling_std`, `rolling_sum`, etc.), `ewm_mean` and time-based `rolling` exist [D, partially read].
- **Custom rolling:** `Expr.rolling_map(function, window_size, ...)` is documented as unstable ("may be changed at any point without it being considered a breaking change") and "extremely slow" [D].
- **State handling:** no persistent per-expression state object found; expressions are evaluated over a whole column. Computing the latest Bar incrementally means re-evaluating over a tail of enough history, or a separate implementation [inference from absence of such an API in docs read; U].
- **Streaming engine:** "Instead of processing all the data at once, Polars can execute the query in batches"; unsupported operations "fall back to the in-memory engine" [D]. This is out-of-core execution, not per-event state. The page read does not discuss result ordering or determinism of the streaming engine [D].
- **Null vs NaN:** "`NaN` values ... are not considered to be missing data in Polars"; null is the missing marker for all dtypes. Aggregations like `mean`/`sum` skip nulls while NaN "typically propagates". `fill_null` does not fill NaN and `fill_nan` does not fill null. `is_nan` needs real computation, null detection uses the validity bitmap [D].
- **Observed null behaviour [T]:** `rolling_mean(2)` over `[1, 2, null, 4, 5, 6]` gave `[null, 1.5, null, null, 4.5, 5.5]` (a null in the window nulls the output under default `min_samples`); with `min_samples=1` it gave `[1.0, 1.5, 2.0, 4.0, 4.5, 5.5]`. `ewm_mean(alpha=.5)` gave null at the null input row and continued afterwards; `ignore_nulls=True` changed later values (3.36 vs 3.00 at row 3).
- **Determinism [T]:** `rolling_mean`, `rolling_std`, `ewm_mean(span=20, adjust=False)` over the first 1000 rows of a 5000-row series were bitwise equal to the same expressions over all 5000 rows (rows 0..999). Compared with a per-window numpy recompute (`c[i-19:i+1].mean()`), `rolling_mean` differed by up to 4.4e-16 and `rolling_std` by 2.3e-16, not bitwise equal. Starting the same rolling mean at a different row offset (500) changed trailing values by up to 2.2e-16. So results depend on the algorithm's accumulation order, not only on the window contents.
- **Parquet I/O:** `scan_parquet` is lazy with predicate/projection pushdown, hive partition support, glob, `missing_columns='insert'`; some parameters are marked unstable (`parallel="prefiltered"`, `schema`, `hidden_file_prefix`) [D]. Round trip [T]: a UTC `Datetime` column came back as `timestamp[us, tz=UTC]` in pyarrow and `datetime64[us, UTC]` in pandas; a float column with `[1.0, null, NaN, 3.0, ...]` kept null and NaN as distinct values in polars and pyarrow (`null_count == 1`, NaN preserved).
- Packaging: the `polars` wheel is a pure-Python (`py3`) shim; the engine is in `polars-runtime-32` [P][R].

### pandas 3.0.6

- **Rolling windows:** `DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, closed=None, step=None, method='single')`. Integer windows default `min_periods` to the window size; offset (time) windows default to 1. NaN values are skipped; fewer than `min_periods` non-NaN values gives NaN [D]. Windowing always returns `float64` [D].
- **State handling:** only exponentially weighted windows support online operation, via `.ewm(...).online()`: the first call must "prime" the state by calling an aggregation, then `update=` takes new data [D]. Rolling, expanding and weighted windows are listed as not supporting online operation [D]. `method='table'` requires `engine='numba'` [D]. In this session `.online()` raised `ImportError: Import numba failed` until numba was installed [T].
- **Parity observed [T]:** `Series.rolling(20).mean()` over 1000 rows was bitwise equal to the first 1000 rows of the 5000-row result. Online EWM (`alpha=2/21`, `adjust=False`, primed on 100 rows, then one `update` row) differed from the batch `ewm(...).mean()` at that row by -1.19e-05 (not equal). Cause not investigated; one probe only.
- **Numerical precision:** docs state `mean`, `sum`, `var`, `std` "may suffer from numerical imprecision due to the underlying windowing algorithms accumulating sums" when magnitudes differ by about 4.5e15; Kahan summation is used for rolling sums [D].
- **Null handling:** NaN is the missing marker in default float64 columns; `ewm` has `ignore_na` (default False: weights follow absolute positions so nulls affect results) [D]. Parquet read: a null in a float column became NaN in default numpy-backed dtypes (null and NaN not distinguishable), but `dtype_backend="pyarrow"` kept `<NA>` and NaN distinct [T].
- **Parquet I/O:** `read_parquet`/`to_parquet` go through pyarrow (or fastparquet) [U, not read this session]; round trip above [T].

### duckdb 1.5.6 and pyarrow 26.0.0

- Both have cp313 wheels [P]. DuckDB window functions and Parquet scans exist in its documentation [U, not read this session]. PyArrow reads/writes Parquet and preserved null vs NaN and tz-aware timestamps in the round trip [T]. Neither was evaluated for incremental per-Bar state.

## Indicator and feature libraries

### TA-Lib (python wrapper `ta-lib` 0.8.2)

- **Three APIs [D]:** Function API (whole arrays), Abstract API (dict/DataFrame inputs), Streaming API. Inputs may be NumPy arrays, pandas Series/DataFrames, polars Series/DataFrames [D].
- **Streaming API, verbatim [D]:** "The Streaming API keeps a handle per indicator instead of recomputing from the whole array. Opening one costs a pass over the history; every bar after that is O(1), and each value it produces is identical to the one the Function API reports for that bar." Handle methods shown: `stream.SMA(close)` (handle positioned at history end, `.value`), `.update(price)` (one closed bar in, value out), `.peek(forming)` (what update would return, commits nothing), `.copy()` (independent handle). Single-output functions return `float`, multi-output the same tuple as the Function API. "Opening needs at least `lookback + 1` bars"; an `InsufficientHistory` exception is mentioned [D].
- **Observed parity [T]:** `stream.SMA(20)`, `stream.EMA(20)`, `stream.RSI(14)` opened on 100 bars then fed bar by bar produced output bitwise equal (`np.array_equal`) to `talib.SMA/EMA/RSI` over the full array for rows 99..4999. Running the batch function on a series starting at row 1000 instead of row 0: SMA differed by up to 4.4e-16 in the last 100 values; EMA and RSI showed 0.0 difference at the tail in that test.
- **Lookback and unstable period [D]:** "The lookback is function-specific and may be larger than the `timeperiod` parameter. For example, `RSI(timeperiod=14)` needs 15 price observations." Initial lookback outputs are `NaN`. Functions with an unstable period start with that setting at 0; `get_unstable_period()` returns non-zero only after `set_unstable_period()`. In the test `RSI(14)` first valid output was index 14 [T].
- **NaN handling [D]:** "the underlying TA-Lib C library handles NaN's in a sometimes surprising manner by typically propagating NaN's to the end of the output". Observed [T]: `SMA(5)` with one NaN at index 10 of 50 produced 44 NaN outputs out of 50 (the NaN did not just affect its window).
- **Null semantics:** NaN-only; polars nulls would need conversion [inference; U].
- **Threading:** extension releases the GIL during native calculation [D].
- **Determinism:** no statement found in docs read; compiled C code; same-array repeat runs not stress-tested.

### polars-talib 0.2.0

- Exposes TA-Lib functions as polars expressions via a `.ta` namespace (`pl.col("close").ta.ema(5)`, `.ta.macd(12, 26, 9)` returning a struct), "201 batch indicators", bundles TA-Lib 0.8.1, wheels include the C library [D]. Requires `polars<2` [P]; incompatible with the polars 2.0.0 current at research time, uv fell back to 0.1.6 [R]. No streaming/incremental API found in README [U].

### polars-ta 0.5.17

- "Technical Indicator Operators Rewritten in polars", function names prefixed `ts_` (e.g. `ts_mean`, `ts_RSI`), `min_samples` argument and a global `polars_ta.MIN_SAMPLES` [D]. Depends on `polars>=1.28.0`, `polars-ols`, `numba`, `pandas`, `numpy` [P]. Optional `ta-lib` extra. No streaming API found [U]. No Python 3.13 classifier [P].

### pandas-ta 0.4.71b0 and pandas-ta-classic 0.8.32

- pandas-ta: "leverages numba and numpy ... and pandas", "more than 150 indicators" and 60 candlestick patterns when TA-Lib installed [D]. Hard pins `numba==0.61.2` [P]. Pre-release only on PyPI [P].
- pandas-ta-classic: MIT [P]; requires `numpy>=2.0.0`, `pandas>=2.0.0`; optional `numba>=0.58.0` (performance extra) and `ta-lib>=0.6.8` (oracle extra) [P]. Both are DataFrame/Series-in, batch-oriented; no streaming API found in the metadata/READMEs read [U].

### ta (bukosabino) 0.11.0, finta 1.3, tulipy 0.4.0

- `ta`: pandas-based, last release 2023-11-02, no `requires_python` [P]. `finta`: last release 2021-04-03, LGPLv3+ [P]. `tulipy`: last release 2019-04-11, LGPL-3.0, no wheels listed for 3.13 [P]. Not evaluated further.

### river 0.26.1

- Online-learning library whose statistics are incremental by design. `river.utils.Rolling(cls, window_size)` "is a generic wrapper for performing rolling computations": it wraps any class that implements both `update` and `revert`, queues values and reverts the oldest once the window is full. `stats.Mean`, `stats.Var` and others support rolling when wrapped [D]. Streams one observation at a time (per-sample Python objects); no vectorised batch API was read [U]. Whether a batch pass over a Dataset using the same objects gives bitwise parity with a per-Bar loop is not tested, but both would execute the same `update` code path [inference].

### bottleneck 1.6.0, numba 0.68.0

- bottleneck: NaN-aware moving-window functions on NumPy arrays (cp313 wheels) [P]; functions not read this session [U].
- numba: JIT compiler. pandas documents numba as an optional engine for `apply`/`mean`/`median`/`max`/`min`/`sum` and for `method='table'` and `ewm.online` [D]. A hand-written loop compiled by numba is one way the same function body could be called over a whole array or once per Bar [inference; U].

## Cross-cutting facts

- **Where "one definition, two modes" is documented as a feature:** TA-Lib's Streaming API (identical to Function API per bar [D], reproduced bitwise for three indicators [T]); river's `update`-based statistics [D]; pandas `ewm().online()` (documented, not matched to batch in one probe [T]). For polars no per-Bar incremental API was found in the docs read.
- **Prefix invariance (the property that makes tail-only batch evaluation causal):** polars rolling mean/std and EWM and pandas rolling mean gave bitwise identical values at row i whether computed over rows 0..i or rows 0..N [T]. Offset-dependence: results for the same row differ at the 1e-16 level when the series is started at a different row (pandas, polars, TA-Lib SMA) [T]; TA-Lib EMA and RSI showed 0.0 difference in this test.
- **Null/NaN across the stack:** polars distinguishes null and NaN, and Parquet round trips keep both [T]. pandas default dtypes merge them to NaN on Parquet read; pyarrow-backed dtypes keep them apart [T]. TA-Lib sees NaN only and propagates it far past the affected window [D][T]. The charter requires Gap rows to survive with null features (#7).
- **Determinism statements:** pandas documents sum accumulation imprecision and Kahan summation [D]. No determinism guarantee (thread-count independence, run-to-run) was found in the polars pages read; polars streaming-engine page is silent on ordering [D]. No test of multithreaded run-to-run variation was done.
- **Not verified:** DuckDB window/Parquet behaviour, bottleneck function list, tsfresh and polars-ta internals, Windows/Linux behaviour (all tests ran on macOS arm64), float32 behaviour, performance.

## Sources

- Polars `Expr.rolling_mean`: https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_mean.html
- Polars `Expr.rolling_map`: https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_map.html
- Polars missing data: https://docs.pola.rs/user-guide/expressions/missing-data/
- Polars streaming: https://docs.pola.rs/user-guide/concepts/streaming/
- Polars `scan_parquet`: https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html
- pandas `DataFrame.rolling`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
- pandas windowing user guide: https://pandas.pydata.org/docs/user_guide/window.html
- TA-Lib Python README (Streaming API): https://raw.githubusercontent.com/TA-Lib/ta-lib-python/master/README.md and https://github.com/TA-Lib/ta-lib-python
- TA-Lib Python function docs: https://ta-lib.github.io/ta-lib-python/func.html
- River `utils.Rolling`: https://riverml.xyz/latest/api/utils/Rolling/
- PyPI JSON metadata: https://pypi.org/pypi/<package>/json for polars, pandas, duckdb, pyarrow, numpy, bottleneck, numba, ta-lib, pandas-ta, pandas-ta-classic, ta, river, tulipy, polars-ta, polars-talib, tsfresh, finta
- polars-ta and polars-talib READMEs as rendered in PyPI `description`
- pandas-ta README as rendered in PyPI `description`

## Local test method

Scratch uv project, Python 3.13, `uv add polars pandas ta-lib pyarrow numpy numba`; random-walk float64 series of 5000 values (seed 0). Compared: `talib.stream.*` vs `talib.*`; polars/pandas rolling and EWM over prefix vs full series and vs per-window numpy recompute; pandas `ewm.online`; null handling; Parquet round trip via polars write, read by polars, pyarrow, pandas. The scripts were throwaway and are not committed.
