#!/bin/sh
# Run every (config, seed) job, 9 at a time. Run from the repo root.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p experiments/170-turnover/out
for c in k6 g95 ent001 k3 ent0 g90 g95_ent001 k12 k6_g95_ent001; do
  for s in 42 1 2 3 4; do
    echo "$c $s"
  done
done | xargs -P 9 -L 1 sh -c 'uv run python experiments/170-turnover/sweep_170.py "$0" "$1" > "experiments/170-turnover/out/log_$0_$1.txt" 2>&1'
echo ALLDONE
