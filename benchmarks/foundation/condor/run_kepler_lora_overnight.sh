#!/usr/bin/env bash
# Overnight LoRA sweep on Kepler Q9v3: 3 models × 2 ranks.
# Each run uses --epochs 100 --patience 20 so it stops when val_acc plateaus.
#
# Model-size choices match prior benchmarks: MOMENT/Chronos at 'base',
# TimeMoE at 'small' (the only published size).
# Granite-TTM is excluded: it is a pure mixer with no q/k/v/o projections,
# so LoRA matches zero modules -- only the head trains, which isn't a
# meaningful "LoRA" comparison.
set -u  # don't exit on a single failed run; let the rest of the sweep continue

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

COMMON="--zip data/keplerq9v3.zip --batch_size 16 --seed 42 \
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

# Order: known-good (moment) → unknowns (chronos, timemoe).
# Within each model: r=8 then r=16.
for size_args in \
  "moment:base" \
  "chronos:base" \
  "timemoe:small" \
; do
  model="${size_args%%:*}"
  msize="${size_args##*:}"
  for r in 8 16; do
    alpha=$((2 * r))
    run_variant "lora_${model}_r${r}" \
      --models "${model}" --model_size "${msize}" \
      --lora_r ${r} --lora_alpha ${alpha} \
      --out_dir "plots/kepler_q9v3/lora_${model}_r${r}"
  done
done

echo "[sweep] all variants done at $(date)"
