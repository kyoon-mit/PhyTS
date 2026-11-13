#!/bin/bash
# Evaluate existing best.ckpt files on the LIGO 2048 Hz test split (no training).
# Writes each config's csv_out (test_predictions.csv); one failure does not stop the rest.
# Usage (via condor): test_ligo_ckpts.sh
cd /home/barmstrong/TimeSeriesPhysics
CFGS=(
  configs/LIGO/train_ligo_s4d_gaussnll_regression.yaml
  configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml
  configs/LIGO/linoss_patch/train_ligo_linoss_h512_p8_lr3e-4.yaml
  configs/LIGO/linoss_patch/train_ligo_linoss_h512_p16_lr3e-4.yaml
  configs/LIGO/linoss_patch/train_ligo_linoss_h512_p32_conv_lr3e-4.yaml
)
for CFG in "${CFGS[@]}"; do
  CKPT_DIR=$(grep -m1 "dirpath:" $CFG | awk '{print $2}')
  echo "=== $CFG ($(date))"
  uv run --no-sync python main.py test --config $CFG --ckpt_path $CKPT_DIR/best.ckpt \
    --trainer.logger false && echo "PASS $CFG" || echo "FAIL $CFG"
done
echo "DONE $(date)"
