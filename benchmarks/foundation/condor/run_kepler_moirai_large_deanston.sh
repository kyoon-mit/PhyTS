#!/usr/bin/env bash
# MOIRAI-large (1.1-R-large, 311M) LoRA sweep on Kepler Q9v3 for deanston (24 GiB).
# Sequence: LoRA r=8 → LoRA r=16. (Full FT skipped: needs >20 GiB optim + activations.)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/deanston_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "[sweep] $(date)  start  ${name}"
  "${PY}" -u benchmarks/foundation/run_kepler_finetune.py "$@" >"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}"
}

COMMON="--zip data/keplerq9v3.zip --models moirai --model_size large \
  --batch_size 4 --seed 42 \
  --adapter lora --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-4"

for r in 8 16; do
  alpha=$((2 * r))
  run_variant "lora_moirai_large_r${r}" ${COMMON} \
    --lora_r ${r} --lora_alpha ${alpha} \
    --out_dir "plots/kepler_q9v3/lora_moirai_large_r${r}_deanston"
done

echo "[sweep] all variants done at $(date)"
