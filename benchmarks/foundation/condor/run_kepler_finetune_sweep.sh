#!/usr/bin/env bash
# Sequential LoRA / last_n fine-tuning sweep for MOMENT on Kepler Q9v3.
#
# Runs four variants back-to-back, each writing to its own out_dir and
# log file.  Intended to be launched once via nohup on a remote host.
#
#   A) lora_r16       : higher LoRA rank (r=16, α=32).
#   B) lora_short     : fewer epochs, lower LR (earlier best-val checkpoint).
#   C) lora_balanced  : inverse-frequency class weights on CE loss.
#   D) last_n         : unfreeze last 2 transformer blocks instead of LoRA.
set -u  # don't set -e; we want subsequent variants to run even if one fails

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

COMMON="--zip data/keplerq9v3.zip --models moment --model_size base \
  --batch_size 16 --seed 42"

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/ramee_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "=========================================================================="
  echo "[sweep] $(date)  start  ${name}" | tee -a "${log}"
  "${PY}" benchmarks/foundation/run_kepler_finetune.py ${COMMON} "$@" >>"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}" | tee -a "${log}"
}

# A: higher LoRA rank
run_variant "lora_r16" \
  --adapter lora --lora_r 16 --lora_alpha 32 --epochs 10 \
  --out_dir plots/kepler_q9v3/lora_r16

# B: fewer epochs, lower LR
run_variant "lora_short" \
  --adapter lora --lora_r 8 --lora_alpha 16 --epochs 5 \
  --head_lr 5e-4 --backbone_lr 5e-5 \
  --out_dir plots/kepler_q9v3/lora_short

# C: class-balanced CE loss
run_variant "lora_balanced" \
  --adapter lora --lora_r 8 --lora_alpha 16 --epochs 10 \
  --class_weights balanced \
  --out_dir plots/kepler_q9v3/lora_balanced

# D: last_n unfreezing (no LoRA)
run_variant "last_n" \
  --adapter last_n --unfreeze_last_n 2 --epochs 10 \
  --out_dir plots/kepler_q9v3/last_n

echo "[sweep] all variants done at $(date)"
