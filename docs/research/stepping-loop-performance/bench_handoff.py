"""Cost of handing a polars/Parquet Dataset to a loop as arrays.

Run: uv run python bench_handoff.py [n_rows]
Writes a temp Parquet (timestamp ns, 5 float64 OHLCV-ish cols, 10 float64
features with leading warm-up nulls, 1 int8 label).
"""

import os
import sys
import tempfile
import time

import numba
import numpy as np
import polars as pl
import pyarrow.parquet as pq

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
rng = np.random.default_rng(1)
cols = {
    "ts": pl.Series(np.arange(N, dtype="int64") * 60_000_000_000).cast(
        pl.Datetime("ns", "UTC")
    )
}
for k in ("open", "high", "low", "close", "volume"):
    cols[k] = rng.normal(100, 1, N)
for j in range(10):
    a = rng.normal(0, 1, N)
    a[: 50 + j] = np.nan
    cols[f"f{j}"] = pl.Series(a, nan_to_null=True)  # warm-up as null
cols["label"] = pl.Series(rng.integers(-1, 2, N).astype(np.int8))
df = pl.DataFrame(cols)
path = os.path.join(tempfile.mkdtemp(), "ds.parquet")
df.write_parquet(path)
print(
    f"rows={N:,} cols={df.width} parquet={os.path.getsize(path) / 1e6:.0f} MB "
    f"in-memory={df.estimated_size() / 1e6:.0f} MB"
)


def t(name, fn, repeat=5):
    best, out = 1e9, None
    for _ in range(repeat):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    print(f"{name:58s} {best * 1e3:9.2f} ms")
    return out


d = t("pl.read_parquet (all cols)", lambda: pl.read_parquet(path))
t(
    "pl.scan_parquet().select(ohlc,label).collect()",
    lambda: pl.scan_parquet(path).select("open", "high", "low", "close", "label").collect(),
)
t("pyarrow pq.read_table (all cols)", lambda: pq.read_table(path))
c = d["close"]
arr = t("Series.to_numpy() float64 no nulls", lambda: c.to_numpy())
print("   OWNDATA:", arr.flags["OWNDATA"], "writeable:", arr.flags["WRITEABLE"])
try:
    c.to_numpy(allow_copy=False)
    print("   allow_copy=False OK for no-null float col")
except Exception as e:
    print("   allow_copy=False raised:", type(e).__name__, e)
f = d["f0"]
a2 = t("Series.to_numpy() float64 WITH nulls (-> NaN, copy)", lambda: f.to_numpy())
print("   null_count:", f.null_count(), "NaN count:", int(np.isnan(a2).sum()))
try:
    f.to_numpy(allow_copy=False)
    print("   allow_copy=False OK with nulls")
except Exception as e:
    print("   allow_copy=False with nulls raised:", type(e).__name__)
t("Series.fill_null(nan).to_numpy() w/ nulls", lambda: f.fill_null(float("nan")).to_numpy())
ts = d["ts"]
a3 = t("Datetime[ns,UTC] Series.to_numpy()", lambda: ts.to_numpy())
print("   dtype:", a3.dtype)
t("Datetime -> dt.epoch('ns') -> to_numpy (int64)", lambda: ts.dt.epoch("ns").to_numpy())
t("int8 label Series.to_numpy()", lambda: d["label"].to_numpy())
t(
    "5 OHLC+label cols individually to_numpy",
    lambda: [d[k].to_numpy() for k in ("open", "high", "low", "close", "label")],
)
fc = [f"f{j}" for j in range(10)]
m = t("df.select(10 feature cols).to_numpy() (2D)", lambda: d.select(fc).to_numpy())
print("   2D flags C/F:", m.flags["C_CONTIGUOUS"], m.flags["F_CONTIGUOUS"], m.dtype, m.shape)
t("df.select(10 feature cols).to_numpy(order='c')", lambda: d.select(fc).to_numpy(order="c"))
t(
    "pyarrow col .to_numpy() (zero_copy_only=False)",
    lambda: pq.read_table(path, columns=["close"]).column("close").to_numpy(),
)
t("Series.to_list() close (pure-Python handoff)", lambda: c.to_list(), repeat=3)
t("Series.to_numpy().tolist() close", lambda: c.to_numpy().tolist(), repeat=3)
t(
    "iter_rows() full consume, 5 cols, 1M rows",
    lambda: sum(
        1
        for _ in d.head(1_000_000).select("open", "high", "low", "close", "label").iter_rows()
    ),
    repeat=2,
)
t("np.ascontiguousarray(m[:,0]) strided column copy", lambda: np.ascontiguousarray(m[:, 0]))
sl = d.slice(1_000_000, 1_000_000)["close"].to_numpy()
print("slice to_numpy C-contiguous:", sl.flags["C_CONTIGUOUS"])


@numba.njit
def s(x):
    r = 0.0
    for i in range(x.shape[0]):
        r += x[i]
    return r


print("numba on polars-backed array == on copy:", s(arr) == s(arr.copy()),
      "readonly:", not arr.flags["WRITEABLE"])
