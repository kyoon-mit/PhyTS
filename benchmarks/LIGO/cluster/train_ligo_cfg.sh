#!/bin/bash
# Train from a LIGO config, then evaluate its best checkpoint on the test split.
# Resumes from <dirpath>/last.ckpt if it exists (eviction requeue or the next 4 h time slice).
# Exits 85 when JobTimer ended the slice; submit_runs.sh sets on_exit_remove so condor requeues.
# Telegram notifications: start/finish/fail/stop come from cuda_wrapper.sh; this script adds
# resume, fit-done and test-result messages.
# Usage (via condor): train_ligo_cfg.sh <config.yaml>
set -euo pipefail
CFG=${1:?config}
NOTIFY=/home/barmstrong/cluster/htcondor/notify.sh
NAME=$(basename "$CFG" .yaml)
JOB=${CONDOR_JOB_ID:-local}
CKPT_DIR=$(grep -m1 "dirpath:" "$CFG" | awk '{print $2}')
CSV=$(grep -m1 "csv_out:" "$CFG" | awk '{print $2}')
# JobTimer (configs/LIGO/v2) writes this when the per-job time slice, not early stopping, ended
# the fit; we then exit 85 and the submit file (on_exit_remove) puts the job back in the queue.
MARKER=$CKPT_DIR/.job_timeout
REQUEUE_RC=85
# One W&B run per condor job, shared by all its time slices (and the final test).
RUN_ID=$(printf '%s-%s' "$NAME" "${JOB%%.*}" | md5sum | cut -c1-12)
LOGGER=(--trainer.logger.init_args.id "$RUN_ID")
rm -f "$MARKER"

RESUME=()
if [[ -f $CKPT_DIR/last.ckpt ]]; then
  EPOCH=$(uv run --no-sync python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$CKPT_DIR/last.ckpt" 2>/dev/null || echo "?")
  RESUME=(--ckpt_path "$CKPT_DIR/last.ckpt")
  $NOTIFY "🔁 ${JOB} ${NAME}: resuming from last.ckpt (epoch ${EPOCH})"
fi

uv run --no-sync python main.py fit --config "$CFG" "${RESUME[@]}" "${LOGGER[@]}"
if [[ -f $MARKER ]]; then
  rm -f "$MARKER"
  EPOCH=$(uv run --no-sync python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$CKPT_DIR/last.ckpt" 2>/dev/null || echo "?")
  echo "time slice done at epoch $EPOCH, requeueing (exit $REQUEUE_RC) $(date)"
  exit $REQUEUE_RC
fi
$NOTIFY "🏁 ${JOB} ${NAME}: training done, running test on best.ckpt"

uv run --no-sync python main.py test --config "$CFG" --ckpt_path "$CKPT_DIR/best.ckpt" "${LOGGER[@]}"
RES=$(uv run --no-sync python -c "
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); y, mu = d['true_chirp_mass'].to_numpy(), d['pred_chirp_mass_mean'].to_numpy()
e = mu - y
print(f'test RMSE {np.sqrt(np.mean(e**2)):.4f}, R2 {1 - np.sum(e**2) / np.sum((y - y.mean())**2):.3f}, n={len(d)}')
" "$CSV" 2>/dev/null || echo "test metrics unavailable (see $CSV)")
$NOTIFY "📊 ${JOB} ${NAME}: ${RES}"
echo "DONE $CFG $(date)"
