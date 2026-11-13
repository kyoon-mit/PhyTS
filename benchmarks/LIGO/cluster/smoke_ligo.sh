#!/bin/bash
# Smoke test: one fit step + val step per LIGO model on a 100-sample 2048 Hz file.
F=/fast/barmstrong/LIGO_2048Hz_smoke/train.h5
(while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> /fast/barmstrong/outputs/smoke_gpumem.$CONDOR_JOB_ID.txt; sleep 2; done) &
MON=$!; trap "kill $MON" EXIT
for m in conv1d s4d linoss; do
  echo "=== $m"; date
  uv run --no-sync python main.py fit --config configs/LIGO/train_ligo_${m}_gaussnll_regression.yaml \
    --data.init_args.train_file $F --data.init_args.val_file $F --data.init_args.test_file $F \
    --trainer.fast_dev_run 2 --trainer.logger false && echo "PASS $m" || echo "FAIL $m"
  echo "peak MiB so far: $(sort -n /fast/barmstrong/outputs/smoke_gpumem.$CONDOR_JOB_ID.txt | tail -1)"
done
