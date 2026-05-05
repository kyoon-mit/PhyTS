#!/usr/bin/env bash
# Full fine-tuning of Granite-TTM (base) on Kepler Q9v3.
# Matches the MOMENT full-FT recipe: 100 epochs, patience 20,
# head_lr=1e-3, backbone_lr=1e-5.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/ramee_full_ft_granite_ttm_${TS}.log"

echo "[run] starting granite_ttm full FT  -> ${LOG}"
echo "[run] $(date)  start"
"${PY}" -u benchmarks/foundation/run_kepler_finetune.py \
  --zip data/keplerq9v3.zip \
  --models granite_ttm --model_size base \
  --batch_size 16 --seed 42 \
  --adapter full --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-5 \
  --out_dir plots/kepler_q9v3/full_ft_granite_ttm \
  >"${LOG}" 2>&1
rc=$?
echo "[run] $(date)  end  rc=${rc}"
