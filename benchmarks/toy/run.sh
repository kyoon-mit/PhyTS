#!/bin/bash
# Full toy benchmark pipeline — submits chained SLURM jobs.
# Steps 2 & 3 run in parallel; step 4 waits for both to finish.
#
# Usage:
#   bash benchmarks/toy/run.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f benchmarks/toy/logs/toy_denoiser_<jobid>.out

set -e

WORKDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/TimeSeriesPhysics
LOGS=$WORKDIR/benchmarks/toy/logs

SBATCH_COMMON="
  --partition=gpu_requeue
  --nodes=1
  --ntasks=1
  --gpus=1
  --time=6:00:00
  --chdir=$WORKDIR
"

ACTIVATE="source ~/.bashrc && conda activate ts_cuda312"

# ── Step 1: Train denoiser ────────────────────────────────────────────────────
JOB1=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=toy_denoiser \
  --output=$LOGS/toy_denoiser_%j.out \
  --error=$LOGS/toy_denoiser_%j.err \
  --wrap="$ACTIVATE && python main.py fit \
    --config configs/toy/train_toy_s4d_denoising.yaml")
echo "[1/4] Denoiser submitted: job $JOB1"

# ── Steps 2 & 3: Train regressors in parallel (both wait for denoiser) ────────
JOB2=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB1 \
  --job-name=toy_reg_raw \
  --output=$LOGS/toy_reg_raw_%j.out \
  --error=$LOGS/toy_reg_raw_%j.err \
  --wrap="$ACTIVATE && python main.py fit \
    --config configs/toy/train_toy_mlp_regression_raw.yaml")
echo "[2/4] Raw regressor submitted: job $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB1 \
  --job-name=toy_reg_clean \
  --output=$LOGS/toy_reg_clean_%j.out \
  --error=$LOGS/toy_reg_clean_%j.err \
  --wrap="$ACTIVATE && python main.py fit \
    --config configs/toy/train_toy_mlp_regression_clean.yaml")
echo "[3/4] Clean regressor submitted: job $JOB3 (depends on $JOB1)"

# ── Step 4: Eval (waits for both regressors) ──────────────────────────────────
JOB4=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB2:$JOB3 \
  --job-name=toy_eval \
  --output=$LOGS/toy_eval_%j.out \
  --error=$LOGS/toy_eval_%j.err \
  --wrap="$ACTIVATE && python benchmarks/toy/eval_pipeline.py \
    --denoiser_ckpt        checkpoints/toy_s4d_denoising/best.ckpt \
    --regressor_raw_ckpt   checkpoints/toy_mlp_regression_raw/best.ckpt \
    --regressor_clean_ckpt checkpoints/toy_mlp_regression_clean/best.ckpt \
    --data_dir             data/toy/sinusoidal_signal_white_noise \
    --out_dir              benchmarks/toy")
echo "[4/4] Eval submitted: job $JOB4 (depends on $JOB2, $JOB3)"

echo ""
echo "Pipeline: $JOB1 → [$JOB2, $JOB3] → $JOB4"
echo "Monitor:  squeue -u $USER"
