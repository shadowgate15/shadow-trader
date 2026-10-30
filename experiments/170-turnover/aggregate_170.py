"""Summarise out/*.json: per config, mean +/- sd over seeds of the final Checkpoint.

Run: python3 experiments/170-turnover/aggregate_170.py
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

out = Path(__file__).parent / "out"
runs = defaultdict(list)
for f in sorted(glob.glob(str(out / "*_s*.json"))):
    d = json.load(open(f))
    runs[d["config"]].append(d)

BASE = -14684  # #167 final-Checkpoint 5-seed mean
print(f"{'config':16}{'n':>2} {'net':>16} {'gross':>8} {'changes':>9} {'vs base':>8} {'best-ckpt net':>14} {'min/seed':>8}")
for name, rs in runs.items():
    fin = [r["checkpoints"][-1] for r in rs]
    best = [max(c["net"] for c in r["checkpoints"] if c["changes"] > 0) for r in rs]
    net = [c["net"] for c in fin]
    sd = st.stdev(net) if len(net) > 1 else 0
    print(f"{name:16}{len(rs):>2} {st.mean(net):>9,.0f} ±{sd:>5,.0f} "
          f"{st.mean(c['gross'] for c in fin):>8,.0f} {st.mean(c['changes'] for c in fin):>9,.0f} "
          f"{st.mean(net) - BASE:>+8,.0f} {st.mean(best):>14,.0f} {st.mean(r['seconds'] for r in rs) / 60:>8.1f}")
    print("   per-seed final net:", ", ".join(f"{r['seed']}:{c['net']:,.0f}/{c['changes']:,}" for r, c in zip(rs, fin)))
