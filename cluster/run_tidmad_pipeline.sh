#!/bin/bash
# TIDMAD benchmark pipeline — submits 4 chained SLURM jobs.
#
# Steps 1, 2, 3 are independent and run in parallel.
# Step 4 (eval) waits for all three to finish.
#
#   [1] train denoiser  ──┐
#   [2] train reg_raw   ──┼──> [4] eval
#   [3] train reg_clean ──┘
#
# Usage:
#   bash cluster/run_tidmad_pipeline.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /home/ilay.kamai/athena/logs/tidmad_*.out

set -e

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
SBATCH=${REPO}/cluster/tidmad_train.sbatch
EVAL_SBATCH=${REPO}/cluster/tidmad_eval.sbatch

# ── Step 1: Train denoiser ─────────────────────────────────────────────────
JOB1=$(sbatch --parsable \
  --job-name=tidmad_denoiser \
  --export=CONFIG=${REPO}/configs/TIDMAD/train_tidmad_s4d_denoising.yaml \
  ${SBATCH})
echo "[1/4] Denoiser submitted: job ${JOB1}"

# ── Step 2: Train raw regressor (independent) ──────────────────────────────
JOB2=$(sbatch --parsable \
  --job-name=tidmad_reg_raw \
  --export=CONFIG=${REPO}/configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml \
  ${SBATCH})
echo "[2/4] Raw regressor submitted: job ${JOB2}"

# ── Step 3: Train clean regressor (independent) ────────────────────────────
JOB3=$(sbatch --parsable \
  --job-name=tidmad_reg_clean \
  --export=CONFIG=${REPO}/configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml \
  ${SBATCH})
echo "[3/4] Clean regressor submitted: job ${JOB3}"

# ── Step 4: Eval (waits for all three) ────────────────────────────────────
JOB4=$(sbatch --parsable \
  --dependency=afterok:${JOB1}:${JOB2}:${JOB3} \
  ${EVAL_SBATCH})
echo "[4/4] Eval submitted: job ${JOB4} (depends on ${JOB1}, ${JOB2}, ${JOB3})"

echo ""
echo "Pipeline: [${JOB1}, ${JOB2}, ${JOB3}] --> ${JOB4}"
echo "Monitor:  squeue -u \$USER"
