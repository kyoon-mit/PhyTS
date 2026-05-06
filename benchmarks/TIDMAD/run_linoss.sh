#!/bin/bash
# TIDMAD benchmark pipeline with LinOSS denoiser — submits chained SLURM jobs.
# Steps 2 & 3 run in parallel after the denoiser; step 4 waits for both.
# Regressors are MLP (same as the baseline run) so reg_clean is shared.
#
# Usage:
#   bash benchmarks/TIDMAD/run_linoss.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f benchmarks/TIDMAD/logs/tidmad_linoss_denoiser_<jobid>.out

set -e

WORKDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/TimeSeriesPhysics
LOGS=$WORKDIR/benchmarks/TIDMAD/logs

SBATCH_COMMON="
  --partition=gpu_test
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gres=gpu:1
  --cpus-per-task=4
  --mem=40G
  --time=12:00:00
  --chdir=$WORKDIR
"

ACTIVATE="source ~/.bashrc && conda activate ts_cuda312"

# ── Step 1: Train LinOSS denoiser ─────────────────────────────────────────────
JOB1=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tidmad_linoss_denoiser \
  --output=$LOGS/tidmad_linoss_denoiser_%j.out \
  --error=$LOGS/tidmad_linoss_denoiser_%j.err \
  --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
    --config configs/TIDMAD/train_tidmad_linoss_denoising.yaml")
echo "[1/4] LinOSS denoiser submitted: job $JOB1"

# ── Steps 2 & 3: Train MLP regressors in parallel (wait for denoiser) ─────────
JOB2=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB1 \
  --job-name=tidmad_reg_raw \
  --output=$LOGS/tidmad_reg_raw_%j.out \
  --error=$LOGS/tidmad_reg_raw_%j.err \
  --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
    --config configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml")
echo "[2/4] Raw regressor submitted: job $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB1 \
  --job-name=tidmad_reg_clean \
  --output=$LOGS/tidmad_reg_clean_%j.out \
  --error=$LOGS/tidmad_reg_clean_%j.err \
  --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
    --config configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml")
echo "[3/4] Clean regressor submitted: job $JOB3 (depends on $JOB1)"

# ── Step 4: Eval (waits for both regressors) ──────────────────────────────────
JOB4=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB2:$JOB3 \
  --job-name=tidmad_linoss_eval \
  --output=$LOGS/tidmad_linoss_eval_%j.out \
  --error=$LOGS/tidmad_linoss_eval_%j.err \
  --wrap="$ACTIVATE && srun --cpu-bind=none python benchmarks/TIDMAD/eval_pipeline.py \
    --denoiser_ckpt        checkpoints/tidmad_linoss_denoising/best.ckpt \
    --denoiser_cfg         configs/TIDMAD/train_tidmad_linoss_denoising.yaml \
    --regressor_raw_ckpt   checkpoints/tidmad_mlp_regression_raw/best.ckpt \
    --regressor_raw_cfg    configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml \
    --regressor_clean_ckpt checkpoints/tidmad_mlp_regression_clean/best.ckpt \
    --regressor_clean_cfg  configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml \
    --data_dir             data/TIDMAD \
    --out_dir              benchmarks/TIDMAD/linoss")
echo "[4/4] Eval submitted: job $JOB4 (depends on $JOB2, $JOB3)"

echo ""
echo "Pipeline: $JOB1 → [$JOB2, $JOB3] → $JOB4"
echo "Monitor:  squeue -u $USER"
