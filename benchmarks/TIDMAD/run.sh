#!/bin/bash
# TIDMAD benchmark pipeline — interactive mode selector.
#
# Usage:
#   bash benchmarks/TIDMAD/run.sh
#
# Monitor:
#   squeue -u $USER

set -e

WORKDIR=$(git rev-parse --show-toplevel)
LOGS=$WORKDIR/slurm_logs/TIDMAD
mkdir -p $LOGS

ACTIVATE="source ~/.bashrc && conda activate ts_cuda312"

SBATCH_TRAIN="--partition=gpu_requeue --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=3:00:00 --chdir=$WORKDIR"

SBATCH_EVAL="--partition=shared --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=20G --time=0:30:00 --chdir=$WORKDIR"

EVAL_ARGS="--regressor_raw_ckpt   checkpoints/tidmad_mlp_regression_raw/best.ckpt \
    --regressor_raw_cfg    configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml \
    --regressor_clean_ckpt checkpoints/tidmad_mlp_regression_clean/best.ckpt \
    --regressor_clean_cfg  configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml \
    --data_dir             data/TIDMAD/preprocessed \
    --sample_rate          10000000.0"

# ── Submission functions ───────────────────────────────────────────────────────

submit_denoisers() {
    python tools/save_checkpoint.py \
        --cfg configs/TIDMAD/train_tidmad_classical_denoising.yaml \
        --out checkpoints/tidmad_classical_denoising/best.ckpt

    JOB_S4D_MSE=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_s4d_mse \
        --output=$LOGS/tidmad_s4d_mse_%j.out --error=$LOGS/tidmad_s4d_mse_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_s4d_denoising_mse.yaml")

    JOB_CONV_AE_MSE=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_conv_ae_mse \
        --output=$LOGS/tidmad_conv_ae_mse_%j.out --error=$LOGS/tidmad_conv_ae_mse_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_conv_ae_denoising_mse.yaml")

    JOB_CONV_AE_PSD=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_conv_ae_psd \
        --output=$LOGS/tidmad_conv_ae_psd_%j.out --error=$LOGS/tidmad_conv_ae_psd_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_conv_ae_denoising_psd.yaml")

    JOB_CONV_ATTN_MSE=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_conv_attn_mse \
        --output=$LOGS/tidmad_conv_attn_mse_%j.out --error=$LOGS/tidmad_conv_attn_mse_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_conv_attn_ae_denoising_mse.yaml")

    JOB_CONV_ATTN_PSD=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_conv_attn_psd \
        --output=$LOGS/tidmad_conv_attn_psd_%j.out --error=$LOGS/tidmad_conv_attn_psd_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_conv_attn_ae_denoising_psd.yaml")

    JOB_RNN_MSE=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_rnn_mse \
        --output=$LOGS/tidmad_rnn_mse_%j.out --error=$LOGS/tidmad_rnn_mse_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_rnn_denoising_mse.yaml")

    JOB_RNN_PSD=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_rnn_psd \
        --output=$LOGS/tidmad_rnn_psd_%j.out --error=$LOGS/tidmad_rnn_psd_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_rnn_denoising_psd.yaml")

    JOB_MLP_MSE=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_mlp_den_mse \
        --output=$LOGS/tidmad_mlp_den_mse_%j.out --error=$LOGS/tidmad_mlp_den_mse_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_mlp_denoiser_denoising_mse.yaml")

    JOB_MLP_PSD=$(sbatch --parsable $SBATCH_TRAIN \
        --job-name=tidmad_mlp_den_psd \
        --output=$LOGS/tidmad_mlp_den_psd_%j.out --error=$LOGS/tidmad_mlp_den_psd_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_mlp_denoiser_denoising_psd.yaml")

    echo "[denoiser] s4d_mse:          job $JOB_S4D_MSE"
    echo "[denoiser] conv_ae_mse:      job $JOB_CONV_AE_MSE"
    echo "[denoiser] conv_ae_psd:      job $JOB_CONV_AE_PSD"
    echo "[denoiser] conv_attn_ae_mse: job $JOB_CONV_ATTN_MSE"
    echo "[denoiser] conv_attn_ae_psd: job $JOB_CONV_ATTN_PSD"
    echo "[denoiser] rnn_mse:          job $JOB_RNN_MSE"
    echo "[denoiser] rnn_psd:          job $JOB_RNN_PSD"
    echo "[denoiser] mlp_denoiser_mse: job $JOB_MLP_MSE"
    echo "[denoiser] mlp_denoiser_psd: job $JOB_MLP_PSD"
}

submit_regressors() {
    local dep=${1:-}
    local dep_flag=${dep:+--dependency=afterok:$dep}

    JOB_REG_RAW=$(sbatch --parsable $SBATCH_TRAIN $dep_flag \
        --job-name=tidmad_reg_raw \
        --output=$LOGS/tidmad_reg_raw_%j.out --error=$LOGS/tidmad_reg_raw_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml")

    JOB_REG_CLEAN=$(sbatch --parsable $SBATCH_TRAIN $dep_flag \
        --job-name=tidmad_reg_clean \
        --output=$LOGS/tidmad_reg_clean_%j.out --error=$LOGS/tidmad_reg_clean_%j.err \
        --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
            --config configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml")

    echo "[regressor] raw:   job $JOB_REG_RAW"
    echo "[regressor] clean: job $JOB_REG_CLEAN"
}

_eval() {
    local name=$1 ckpt=$2 cfg=$3 dep=${4:-}
    local dep_flag=${dep:+--dependency=afterok:$dep}
    sbatch --parsable $SBATCH_EVAL $dep_flag \
        --job-name=tidmad_eval_${name} \
        --output=$LOGS/tidmad_eval_${name}_%j.out \
        --error=$LOGS/tidmad_eval_${name}_%j.err \
        --wrap="$ACTIVATE && python benchmarks/TIDMAD/eval_pipeline.py \
            --denoiser_ckpt $ckpt --denoiser_cfg $cfg \
            $EVAL_ARGS --out_dir plots/TIDMAD/${name} --device cpu"
}

submit_evals() {
    local reg_dep=${1:-}

    # Classical filter: depends only on regressors (no denoiser training job)
    local e
    e=$(_eval classical checkpoints/tidmad_classical_denoising/best.ckpt \
        configs/TIDMAD/train_tidmad_classical_denoising.yaml "$reg_dep")
    echo "[eval] classical:      job $e"

    # Learned denoisers: dep = denoiser_job:reg_raw_job:reg_clean_job
    _dep() { local d=$1; echo "${d:+$d:}${reg_dep}"; }

    e=$(_eval s4d_mse          checkpoints/tidmad_s4d_denoising_mse/best.ckpt          configs/TIDMAD/train_tidmad_s4d_denoising_mse.yaml          "$(_dep "${JOB_S4D_MSE:-}")");      echo "[eval] s4d_mse:          job $e"
    e=$(_eval conv_ae_mse      checkpoints/tidmad_conv_ae_denoising_mse/best.ckpt      configs/TIDMAD/train_tidmad_conv_ae_denoising_mse.yaml      "$(_dep "${JOB_CONV_AE_MSE:-}")");  echo "[eval] conv_ae_mse:      job $e"
    e=$(_eval conv_ae_psd      checkpoints/tidmad_conv_ae_denoising_psd/best.ckpt      configs/TIDMAD/train_tidmad_conv_ae_denoising_psd.yaml      "$(_dep "${JOB_CONV_AE_PSD:-}")");  echo "[eval] conv_ae_psd:      job $e"
    e=$(_eval conv_attn_ae_mse checkpoints/tidmad_conv_attn_ae_denoising_mse/best.ckpt configs/TIDMAD/train_tidmad_conv_attn_ae_denoising_mse.yaml "$(_dep "${JOB_CONV_ATTN_MSE:-}")"); echo "[eval] conv_attn_ae_mse: job $e"
    e=$(_eval conv_attn_ae_psd checkpoints/tidmad_conv_attn_ae_denoising_psd/best.ckpt configs/TIDMAD/train_tidmad_conv_attn_ae_denoising_psd.yaml "$(_dep "${JOB_CONV_ATTN_PSD:-}")"); echo "[eval] conv_attn_ae_psd: job $e"
    e=$(_eval rnn_mse          checkpoints/tidmad_rnn_denoising_mse/best.ckpt          configs/TIDMAD/train_tidmad_rnn_denoising_mse.yaml          "$(_dep "${JOB_RNN_MSE:-}")");      echo "[eval] rnn_mse:          job $e"
    e=$(_eval rnn_psd          checkpoints/tidmad_rnn_denoising_psd/best.ckpt          configs/TIDMAD/train_tidmad_rnn_denoising_psd.yaml          "$(_dep "${JOB_RNN_PSD:-}")");      echo "[eval] rnn_psd:          job $e"
    e=$(_eval mlp_denoiser_mse checkpoints/tidmad_mlp_denoiser_denoising_mse/best.ckpt configs/TIDMAD/train_tidmad_mlp_denoiser_denoising_mse.yaml "$(_dep "${JOB_MLP_MSE:-}")");     echo "[eval] mlp_denoiser_mse: job $e"
    e=$(_eval mlp_denoiser_psd checkpoints/tidmad_mlp_denoiser_denoising_psd/best.ckpt configs/TIDMAD/train_tidmad_mlp_denoiser_denoising_psd.yaml "$(_dep "${JOB_MLP_PSD:-}")");     echo "[eval] mlp_denoiser_psd: job $e"
}

# ── Interactive menu ───────────────────────────────────────────────────────────

echo ""
echo "TIDMAD benchmark pipeline"
echo "-------------------------"
PS3=$'\nSelect: '
select choice in \
    "Full pipeline (denoisers + regressors + eval)" \
    "Denoisers only" \
    "Regressors only (raw + clean)" \
    "Eval only (checkpoints must exist)" \
    "Exit"
do
    case $choice in
        "Full pipeline (denoisers + regressors + eval)")
            submit_denoisers
            submit_regressors
            submit_evals "${JOB_REG_RAW}:${JOB_REG_CLEAN}"
            break ;;
        "Denoisers only")
            submit_denoisers
            break ;;
        "Regressors only (raw + clean)")
            submit_regressors
            break ;;
        "Eval only (checkpoints must exist)")
            submit_evals ""
            break ;;
        "Exit")
            break ;;
        *)
            echo "Invalid option" ;;
    esac
done

echo ""
echo "Monitor: squeue -u $USER"
echo "Plots:   plots/TIDMAD/{variant}/"
