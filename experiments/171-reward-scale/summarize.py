import json, glob
import numpy as np

def row(name, files):
    fin = [json.load(open(f))["checkpoints"] for f in files]
    last = [c[-1] for c in fin]
    first = [c[0] for c in fin]
    best = [max((x for x in c if x["changes"] > 0), key=lambda x: x["net"])["net"] for c in fin]
    f = lambda k, L: (np.mean([x[k] for x in L]), np.std([x[k] for x in L], ddof=1))
    n, g, ch = f("net", last), f("gross", last), f("changes", last)
    print(f"{name:12s} n={len(files)} final net {n[0]:9.0f} ± {n[1]:5.0f} | gross {g[0]:6.0f} ± {g[1]:5.0f} | changes {ch[0]:7.0f} ± {ch[1]:5.0f} | first-ckpt net {np.mean([x['net'] for x in first]):8.0f} | best-ckpt net {np.mean(best):8.0f} | per-seed final {[round(x['net']) for x in last]}")

row("baseline", sorted(glob.glob("experiments/167-baseline/out/seed*.json")))
for v in ("unit", "vecnorm", "vf2", "vecnorm_vf"):
    row(v, sorted(glob.glob(f"experiments/171-reward-scale/out/{v}_seed*.json")))
