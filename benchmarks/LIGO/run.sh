#!/bin/bash
# LIGO benchmarks — submit all training jobs.
#
# Usage:
#   bash benchmarks/LIGO/run.sh [--fast-dev-run]
#
# Flags:
#   --fast-dev-run   Add --trainer.fast_dev_run true to Lightning jobs
#
# Monitor:
#   squeue -u $USER

set -e

WORKDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/TimeSeriesPhysics
LOGS=$WORKDIR/slurm_logs/LIGO
mkdir -p $LOGS

# ── env activation strings ────────────────────────────────────────────────────
# Standard env: torch + Lightning (no JAX)
ACTIVATE="source ~/.bashrc && source $WORKDIR/.venv/bin/activate"
# JAX env: additionally loads cuDNN 9.10 so JAX (compiled against 9.8) can run
ACTIVATE_JAX="source ~/.bashrc && module load cudnn/9.10.2.21_cuda12-fasrc01 && source $WORKDIR/.venv/bin/activate"
# Foundation env (Chronos): separate Python 3.10 venv
ACTIVATE_FM="source ~/.bashrc && source $WORKDIR/benchmarks/foundation/.venv/bin/activate"

# ── parse flags ───────────────────────────────────────────────────────────────
FDR_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--fast-dev-run" ]; then
        FDR_FLAG="--trainer.fast_dev_run true"
    fi
done

# ── sbatch resource presets ───────────────────────────────────────────────────
SBATCH_GPU="--partition=gpu_requeue --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gres=gpu:1 --cpus-per-task=4 --mem=10G --time=2-00:00:00 --chdir=$WORKDIR"

SBATCH_JAX="--partition=gpu_requeue --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gres=gpu:1 --cpus-per-task=4 --mem=20G --time=2-00:00:00 --chdir=$WORKDIR"

SBATCH_SHARED="--partition=shared --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=20G --time=4:00:00 --chdir=$WORKDIR"

# ── 1. LinOSS GaussNLL regression (JAX) ──────────────────────────────────────
JOB_LINOSS=$(sbatch --parsable $SBATCH_JAX \
    --job-name=ligo_linoss_gaussnll \
    --output=$LOGS/ligo_linoss_gaussnll_%j.out \
    --error=$LOGS/ligo_linoss_gaussnll_%j.err \
    --wrap="$ACTIVATE_JAX && srun --cpu-bind=none python main.py fit \
        --config configs/LIGO/train_ligo_linoss_gaussnll_raw.yaml $FDR_FLAG")

# ── 2. ConvAE MSE denoising (PyTorch) ────────────────────────────────────────
JOB_CONV_AE=$(sbatch --parsable $SBATCH_GPU \
    --job-name=ligo_conv_ae_mse \
    --output=$LOGS/ligo_conv_ae_mse_%j.out \
    --error=$LOGS/ligo_conv_ae_mse_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/LIGO/train_ligo_conv_ae_denoising_mse.yaml $FDR_FLAG")

# ── 3. Chronos zero-shot ──────────────────────────────────────────────────────
SMOKE_FLAG=$( [ -n "$FDR_FLAG" ] && echo "--smoke_test" || echo "" )
JOB_ZS=$(sbatch --parsable $SBATCH_SHARED \
    --job-name=ligo_chronos_zs \
    --output=$LOGS/ligo_chronos_zs_%j.out \
    --error=$LOGS/ligo_chronos_zs_%j.err \
    --wrap="$ACTIVATE_FM && python benchmarks/LIGO/chronos_ligo.py \
        --mode zero_shot --model_size tiny \
        --out_dir results/LIGO/chronos_zeroshot $SMOKE_FLAG")

# ── 4. Chronos fine-tune (last-N blocks) ─────────────────────────────────────
JOB_FT=$(sbatch --parsable $SBATCH_SHARED \
    --job-name=ligo_chronos_ft \
    --output=$LOGS/ligo_chronos_ft_%j.out \
    --error=$LOGS/ligo_chronos_ft_%j.err \
    --wrap="$ACTIVATE_FM && python benchmarks/LIGO/chronos_ligo.py \
        --mode finetune --model_size tiny --finetune_strategy last_n --finetune_n_blocks 2 \
        --out_dir results/LIGO/chronos_finetune $SMOKE_FLAG")

echo "[linoss]      ligo_linoss_gaussnll : job $JOB_LINOSS"
echo "[conv_ae]     ligo_conv_ae_mse     : job $JOB_CONV_AE"
echo "[chronos_zs]  ligo_chronos_zs      : job $JOB_ZS"
echo "[chronos_ft]  ligo_chronos_ft      : job $JOB_FT"
echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    $LOGS/"
echo "Ckpts:   checkpoints/ligo_{linoss_gaussnll_raw,conv_ae_denoising_mse}/best.ckpt"
echo "Results: results/LIGO/chronos_{zeroshot,finetune}/metrics.json"
