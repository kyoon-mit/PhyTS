#!/usr/bin/env bash
# Chronos LoRA sweep on Kepler Q9v3 (r=8 then r=16), to fill the cell that
# OOMed during the original overnight LoRA sweep (bs=16). Uses bs=4 with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; matches the rest of the
# overnight sweep (100 ep, patience 20, head_lr=1e-3, backbone_lr=1e-4).
#
# Waits for any in-flight run_kepler_finetune.py process to clear before
# starting, so it can be queued behind the current chronos+timemoe sweep.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"

# Wait for the in-flight sweep to finish before grabbing the GPU.
echo "[queue] $(date)  waiting for any running run_kepler_finetune.py..."
while pgrep -f "run_kepler_finetune.py" > /dev/null; do
  sleep 60
done
echo "[queue] $(date)  GPU free; starting chronos LoRA sweep."

TS="$(date +%Y%m%d_%H%M%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--zip data/keplerq9v3.zip --models chronos --model_size base \
  --batch_size 4 --seed 42 \
  --adapter lora --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-4"

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/ramee_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "=========================================================================="
  echo "[sweep] $(date)  start  ${name}"
  "${PY}" -u benchmarks/foundation/run_kepler_finetune.py ${COMMON} "$@" >"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}"
}

for r in 8 16; do
  alpha=$((2 * r))
  run_variant "lora_chronos_r${r}_bs4" \
    --lora_r ${r} --lora_alpha ${alpha} \
    --out_dir "plots/kepler_q9v3/lora_chronos_r${r}_bs4"
done

echo "[sweep] all variants done at $(date)"
