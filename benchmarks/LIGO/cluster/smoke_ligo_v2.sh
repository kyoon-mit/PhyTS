#!/bin/bash
# Smoke test of configs/LIGO/v2: 2 train + 2 val steps per model on 100-sample jittered files.
F=/fast/barmstrong/LIGO_2048Hz_v2_smoke
(while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> /fast/barmstrong/outputs/smoke_v2_gpumem.$CONDOR_JOB_ID.txt; sleep 2; done) &
MON=$!; trap "kill $MON" EXIT
for m in s4d linoss conv1d; do
  echo "=== $m"; date
  uv run --no-sync python main.py fit --config configs/LIGO/v2/train_ligo_${m}_gaussnll_regression.yaml \
    --data.init_args.train_file $F/train.h5 --data.init_args.val_file $F/val.h5 --data.init_args.test_file $F/test.h5 \
    --trainer.fast_dev_run 2 --trainer.logger false && echo "PASS $m" || echo "FAIL $m"
  date; echo "peak MiB so far: $(sort -n /fast/barmstrong/outputs/smoke_v2_gpumem.$CONDOR_JOB_ID.txt | tail -1)"
done
