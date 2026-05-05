#!/usr/bin/env bash
# Submit the *pending* TESS benchmark cells to Condor (NiceUser).
#
# Skips cells already produced or actively running on krakas/brigand:
#   - Cls LoRA: moment, chronos, timesfm (done), timemoe + moirai (running)
#   - Cls Full FT: timesfm + timemoe + granite_ttm (queued on brigand chain)
#   - Reg LoRA: moment, chronos, timesfm (done), granite_ttm LoRA blocked
#
# Submitted here:
#   Cls Full FT:    moment, chronos, moirai          (3 jobs)
#   Reg LoRA:       timemoe, moirai                  (2 jobs)
#   Reg Full FT:    moment, chronos, timesfm,
#                   timemoe, granite_ttm             (5 jobs)
# Total: 10 jobs.
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT_JOB="${JOB_DIR}/tess_finetune.job"
REG_JOB="${JOB_DIR}/tess_regression_finetune.job"

# ─── Classification: Full FT ──────────────────────────────────────────────
# bs=4, 30 epochs, patience 5, low backbone LR (matches the brigand chain).
FULL_OUT="plots/tess/full_condor"
FULL_EXTRA="--adapter full --epochs 30 --patience 5 --backbone_lr 1e-5 --out_dir ${FULL_OUT}"

for m in moment chronos moirai; do
  echo "==> Submitting Cls Full-FT [$m]"
  condor_submit \
    -a "tag=full_${m}" \
    -a "model=${m}" \
    -a "model_size=base" \
    -a "extra_args=${FULL_EXTRA}" \
    "${FT_JOB}"
done

# ─── Regression: LoRA ──────────────────────────────────────────────────────
# bs=4 (Condor default in the job file). Smaller dataset → bs=4 is fine.
LORA_REG_OUT="plots/tess/lora_regression_condor"
LORA_REG_EXTRA="--adapter lora --epochs 50 --patience 10 --lora_r 8 --lora_alpha 16 --out_dir ${LORA_REG_OUT}"

for m in timemoe moirai; do
  echo "==> Submitting Reg LoRA [$m]"
  condor_submit \
    -a "tag=lora_reg_${m}" \
    -a "model=${m}" \
    -a "model_size=base" \
    -a "extra_args=${LORA_REG_EXTRA}" \
    "${REG_JOB}"
done

# ─── Regression: Full FT ───────────────────────────────────────────────────
FULL_REG_OUT="plots/tess/full_regression_condor"
FULL_REG_EXTRA="--adapter full --epochs 30 --patience 5 --backbone_lr 1e-5 --out_dir ${FULL_REG_OUT}"

for m in moment chronos timesfm timemoe granite_ttm; do
  echo "==> Submitting Reg Full-FT [$m]"
  condor_submit \
    -a "tag=full_reg_${m}" \
    -a "model=${m}" \
    -a "model_size=base" \
    -a "extra_args=${FULL_REG_EXTRA}" \
    "${REG_JOB}"
done

echo
echo "Submitted. Track with: condor_q \$USER"
