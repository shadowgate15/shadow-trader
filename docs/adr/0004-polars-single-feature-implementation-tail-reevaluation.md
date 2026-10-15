# Polars is the only feature implementation; live features re-evaluate a history tail

Dataset preparation computes `feat_*` with polars expressions and nothing else. Batch runs evaluate them over the whole Dataset; the live run evaluates the same expressions over a fixed-length tail of recent Bars (at least the longest lookback plus a per-column warm-up) and keeps the last row. There is no separate per-Bar update code path. Parity is bitwise for finite-memory columns, a stated absolute tolerance for unbounded-memory columns (EMA, RSI, Wilder ATR), and the null pattern must match exactly. The manifest records the polars version, and a polars upgrade counts as a change in numerical behaviour for spec versioning.

## Considered Options

- **TA-Lib Streaming API for the live run**: rejected. It is the only candidate with a documented per-Bar API, but it is NaN-only (one NaN poisons output far past its window, conflicting with Gap rows carrying null features), has no rolling median, and would be a second implementation next to batch.
- **pandas**: rejected. Default dtypes merge null and NaN on Parquet read, and `ewm().online()` needs numba and differed from batch by 1.2e-5 in a probe.
- **Hand-written carried-state updates checked against batch**: rejected. Tighter parity for EWM columns, but two implementations break the charter's single-implementation guarantee.

## Consequences

- Unbounded-memory columns need a warm-up long enough that tail truncation error sits below the stated tolerance; the spec states warm-up per column.
- Live cost is one tail evaluation per Bar rather than O(1); acceptable at 5-minute Bars.
- Polars 2.0.0 has no determinism guarantee in its docs; parity and golden tests gate upgrades.
