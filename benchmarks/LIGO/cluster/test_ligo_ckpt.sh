#!/bin/bash
# Evaluate one checkpoint of a LIGO config on the test split (no training, no W&B).
# Usage (via condor): test_ligo_ckpt.sh <config.yaml> <ckpt> <csv_out>
set -euo pipefail
CFG=${1:?config}; CKPT=${2:?ckpt}; CSV=${3:?csv_out}
NOTIFY=/home/barmstrong/cluster/htcondor/notify.sh
JOB=${CONDOR_JOB_ID:-local}
uv run --no-sync python main.py test --config "$CFG" --ckpt_path "$CKPT" \
  --model.init_args.csv_out "$CSV" --trainer.logger false
# LightningCLI (2.6) applies the checkpoint's saved hparams over the CLI, so the PyTorch tasks
# still write the config's csv_out; move that file to the requested name.
DEF=$(grep -m1 "csv_out:" "$CFG" | awk '{print $2}')
[[ -f $CSV ]] || mv "$DEF" "$CSV"
RES=$(uv run --no-sync python -c "
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); y, mu = d['true_chirp_mass'].to_numpy(), d['pred_chirp_mass_mean'].to_numpy()
e = mu - y
print(f'test RMSE {np.sqrt(np.mean(e**2)):.4f}, R2 {1 - np.sum(e**2) / np.sum((y - y.mean())**2):.3f}, n={len(d)}')
" "$CSV")
echo "$RES"
$NOTIFY "📊 ${JOB} $(basename "$CKPT") of $(basename "$CFG" .yaml): ${RES}"
