#!/bin/bash
# Train one LIGO 2048 Hz model, then evaluate the best checkpoint on the test split.
# Usage (via condor): train_ligo.sh <conv1d|s4d|linoss>
set -euo pipefail
M=${1:?model}
CFG=configs/LIGO/train_ligo_${M}_gaussnll_regression.yaml
uv run --no-sync python main.py fit  --config $CFG
uv run --no-sync python main.py test --config $CFG --ckpt_path checkpoints/ligo_${M}_gaussnll_regression/best.ckpt
echo "DONE $M $(date)"
