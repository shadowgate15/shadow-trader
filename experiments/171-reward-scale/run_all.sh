#!/bin/sh
# Run all variants x seeds, 10 at a time. From the repo root.
D=experiments/171-reward-scale
mkdir -p $D/logs
for v in unit vecnorm vf2 vecnorm_vf; do
  for s in 42 1 2 3 4; do
    uv run python $D/exp171.py $v $s > $D/logs/${v}_$s.log 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 10 ]; do sleep 5; done
  done
done
wait
echo finished
