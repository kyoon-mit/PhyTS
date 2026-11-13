#!/bin/bash
# Check that the LinOSS task logs val/loss and val/rmse_chirp_mass (2 train + 3 val batches, CSV logger).
set -euo pipefail
OUT=/fast/barmstrong/outputs/smoke_linoss_val
rm -rf $OUT
uv run --no-sync python main.py fit --config configs/LIGO/linoss_sweep/train_ligo_linoss_h64.yaml \
  --trainer.max_epochs 1 --trainer.limit_train_batches 2 --trainer.limit_val_batches 3 --trainer.num_sanity_val_steps 0 \
  --trainer.logger '{"class_path":"lightning.pytorch.loggers.CSVLogger","init_args":{"save_dir":"'$OUT'"}}' \
  --trainer.callbacks '[]' --trainer.enable_checkpointing false
echo "=== metrics"; cat $OUT/lightning_logs/version_0/metrics.csv
