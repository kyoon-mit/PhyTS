#!/bin/bash
# Full toy benchmark pipeline — all denoiser × loss combinations.
#
# Step 1: all denoisers train in parallel (classical filter: checkpoint saved inline).
# Step 2: raw + clean regressors train in parallel (independent of denoisers).
# Step 3: one eval job per denoiser variant (depends on its denoiser + both regressors).
#
# Usage:
#   bash benchmarks/toy/run.sh
#
# Monitor:
#   squeue -u $USER

set -e

WORKDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/TimeSeriesPhysics
LOGS=$WORKDIR/benchmarks/toy/logs

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

# ── Classical filter: no training needed — save checkpoint now ─────────────────
python tools/save_checkpoint.py \
    --cfg configs/toy/train_toy_classical_denoising.yaml \
    --out checkpoints/toy_classical_denoising/best.ckpt

# ── Step 1: Train all denoisers in parallel ────────────────────────────────────
JOB_S4D_MSE=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_s4d_mse \
    --output=$LOGS/toy_s4d_mse_%j.out --error=$LOGS/toy_s4d_mse_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_s4d_denoising_mse.yaml")
echo "[denoiser] s4d_mse:          job $JOB_S4D_MSE"

JOB_CONV_AE_MSE=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_conv_ae_mse \
    --output=$LOGS/toy_conv_ae_mse_%j.out --error=$LOGS/toy_conv_ae_mse_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_conv_ae_denoising_mse.yaml")
echo "[denoiser] conv_ae_mse:      job $JOB_CONV_AE_MSE"

JOB_CONV_AE_PSD=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_conv_ae_psd \
    --output=$LOGS/toy_conv_ae_psd_%j.out --error=$LOGS/toy_conv_ae_psd_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_conv_ae_denoising_psd.yaml")
echo "[denoiser] conv_ae_psd:      job $JOB_CONV_AE_PSD"

JOB_CONV_ATTN_MSE=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_conv_attn_mse \
    --output=$LOGS/toy_conv_attn_mse_%j.out --error=$LOGS/toy_conv_attn_mse_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_conv_attn_ae_denoising_mse.yaml")
echo "[denoiser] conv_attn_ae_mse: job $JOB_CONV_ATTN_MSE"

JOB_CONV_ATTN_PSD=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_conv_attn_psd \
    --output=$LOGS/toy_conv_attn_psd_%j.out --error=$LOGS/toy_conv_attn_psd_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_conv_attn_ae_denoising_psd.yaml")
echo "[denoiser] conv_attn_ae_psd: job $JOB_CONV_ATTN_PSD"

JOB_RNN_MSE=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_rnn_mse \
    --output=$LOGS/toy_rnn_mse_%j.out --error=$LOGS/toy_rnn_mse_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_rnn_denoising_mse.yaml")
echo "[denoiser] rnn_mse:          job $JOB_RNN_MSE"

JOB_RNN_PSD=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_rnn_psd \
    --output=$LOGS/toy_rnn_psd_%j.out --error=$LOGS/toy_rnn_psd_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_rnn_denoising_psd.yaml")
echo "[denoiser] rnn_psd:          job $JOB_RNN_PSD"

JOB_MLP_MSE=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_mlp_den_mse \
    --output=$LOGS/toy_mlp_den_mse_%j.out --error=$LOGS/toy_mlp_den_mse_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_mlp_denoiser_denoising_mse.yaml")
echo "[denoiser] mlp_denoiser_mse: job $JOB_MLP_MSE"

JOB_MLP_PSD=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_mlp_den_psd \
    --output=$LOGS/toy_mlp_den_psd_%j.out --error=$LOGS/toy_mlp_den_psd_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_mlp_denoiser_denoising_psd.yaml")
echo "[denoiser] mlp_denoiser_psd: job $JOB_MLP_PSD"

# ── Step 2: Train regressors (independent of which denoiser is used) ──────────
JOB_REG_RAW=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_reg_raw \
    --output=$LOGS/toy_reg_raw_%j.out --error=$LOGS/toy_reg_raw_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_mlp_regression_raw.yaml")
echo "[regressor] raw:   job $JOB_REG_RAW"

JOB_REG_CLEAN=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=toy_reg_clean \
    --output=$LOGS/toy_reg_clean_%j.out --error=$LOGS/toy_reg_clean_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/toy/train_toy_mlp_regression_clean.yaml")
echo "[regressor] clean: job $JOB_REG_CLEAN"

# ── Step 3: Eval for each denoiser variant ─────────────────────────────────────
# Shared eval args (regressors are the same for all denoisers)
EVAL_ARGS="
    --regressor_raw_ckpt   checkpoints/toy_mlp_regression_raw/best.ckpt
    --regressor_raw_cfg    configs/toy/train_toy_mlp_regression_raw.yaml
    --regressor_clean_ckpt checkpoints/toy_mlp_regression_clean/best.ckpt
    --regressor_clean_cfg  configs/toy/train_toy_mlp_regression_clean.yaml
    --data_dir             data/toy/sinusoidal_signal_white_noise
"

_eval() {
    local name=$1 ckpt=$2 cfg=$3 dep=$4
    sbatch --parsable $SBATCH_COMMON \
        --dependency=afterok:${dep}:${JOB_REG_RAW}:${JOB_REG_CLEAN} \
        --job-name=toy_eval_${name} \
        --output=$LOGS/toy_eval_${name}_%j.out \
        --error=$LOGS/toy_eval_${name}_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python benchmarks/toy/eval_pipeline.py \
            --denoiser_ckpt $ckpt \
            --denoiser_cfg  $cfg \
            $EVAL_ARGS \
            --out_dir plots/toy/${name}"
}

# Classical filter has no denoiser training job — use a dummy dep that's always satisfied
# (afterok: with only the regressor deps)
EVAL_CLASSICAL=$(sbatch --parsable $SBATCH_COMMON \
    --dependency=afterok:${JOB_REG_RAW}:${JOB_REG_CLEAN} \
    --job-name=toy_eval_classical \
    --output=$LOGS/toy_eval_classical_%j.out \
    --error=$LOGS/toy_eval_classical_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python benchmarks/toy/eval_pipeline.py \
        --denoiser_ckpt checkpoints/toy_classical_denoising/best.ckpt \
        --denoiser_cfg  configs/toy/train_toy_classical_denoising.yaml \
        $EVAL_ARGS \
        --out_dir plots/toy/classical")
echo "[eval] classical:      job $EVAL_CLASSICAL"

JOB_EVAL_S4D_MSE=$(_eval      s4d_mse          checkpoints/toy_s4d_denoising_mse/best.ckpt       configs/toy/train_toy_s4d_denoising_mse.yaml         $JOB_S4D_MSE)
JOB_EVAL_CONV_AE_MSE=$(_eval  conv_ae_mse      checkpoints/toy_conv_ae_denoising_mse/best.ckpt   configs/toy/train_toy_conv_ae_denoising_mse.yaml      $JOB_CONV_AE_MSE)
JOB_EVAL_CONV_AE_PSD=$(_eval  conv_ae_psd      checkpoints/toy_conv_ae_denoising_psd/best.ckpt   configs/toy/train_toy_conv_ae_denoising_psd.yaml      $JOB_CONV_AE_PSD)
JOB_EVAL_CONV_ATTN_MSE=$(_eval conv_attn_ae_mse checkpoints/toy_conv_attn_ae_denoising_mse/best.ckpt configs/toy/train_toy_conv_attn_ae_denoising_mse.yaml $JOB_CONV_ATTN_MSE)
JOB_EVAL_CONV_ATTN_PSD=$(_eval conv_attn_ae_psd checkpoints/toy_conv_attn_ae_denoising_psd/best.ckpt configs/toy/train_toy_conv_attn_ae_denoising_psd.yaml $JOB_CONV_ATTN_PSD)
JOB_EVAL_RNN_MSE=$(_eval      rnn_mse          checkpoints/toy_rnn_denoising_mse/best.ckpt       configs/toy/train_toy_rnn_denoising_mse.yaml          $JOB_RNN_MSE)
JOB_EVAL_RNN_PSD=$(_eval      rnn_psd          checkpoints/toy_rnn_denoising_psd/best.ckpt       configs/toy/train_toy_rnn_denoising_psd.yaml          $JOB_RNN_PSD)
JOB_EVAL_MLP_MSE=$(_eval      mlp_denoiser_mse checkpoints/toy_mlp_denoiser_denoising_mse/best.ckpt configs/toy/train_toy_mlp_denoiser_denoising_mse.yaml $JOB_MLP_MSE)
JOB_EVAL_MLP_PSD=$(_eval      mlp_denoiser_psd checkpoints/toy_mlp_denoiser_denoising_psd/best.ckpt configs/toy/train_toy_mlp_denoiser_denoising_psd.yaml $JOB_MLP_PSD)

echo "[eval] s4d_mse:          job $JOB_EVAL_S4D_MSE"
echo "[eval] conv_ae_mse:      job $JOB_EVAL_CONV_AE_MSE"
echo "[eval] conv_ae_psd:      job $JOB_EVAL_CONV_AE_PSD"
echo "[eval] conv_attn_ae_mse: job $JOB_EVAL_CONV_ATTN_MSE"
echo "[eval] conv_attn_ae_psd: job $JOB_EVAL_CONV_ATTN_PSD"
echo "[eval] rnn_mse:          job $JOB_EVAL_RNN_MSE"
echo "[eval] rnn_psd:          job $JOB_EVAL_RNN_PSD"
echo "[eval] mlp_denoiser_mse: job $JOB_EVAL_MLP_MSE"
echo "[eval] mlp_denoiser_psd: job $JOB_EVAL_MLP_PSD"

echo ""
echo "Monitor: squeue -u $USER"
echo "Plots will appear in: plots/toy/{variant}/"
