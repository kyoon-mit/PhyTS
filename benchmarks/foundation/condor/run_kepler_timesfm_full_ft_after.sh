#!/usr/bin/env bash
# TimesFM full fine-tuning on Kepler Q9v3, queued after any in-flight runs.
# Same recipe as the MOMENT/Chronos/Granite full-FT on ramee:
# bs=4, 100 ep, patience 20, head_lr=1e-3, backbone_lr=1e-5.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"

echo "[queue] $(date)  waiting for any running run_kepler_finetune.py..."
while pgrep -f "run_kepler_finetune.py" > /dev/null; do
  sleep 60
done
echo "[queue] $(date)  GPU free; starting timesfm full FT."

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/ramee_full_ft_timesfm_${TS}.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================================================="
echo "[run] starting timesfm full FT  -> ${LOG}"
echo "=========================================================================="
echo "[run] $(date)  start"
"${PY}" -u benchmarks/foundation/run_kepler_finetune.py \
  --zip data/keplerq9v3.zip \
  --models timesfm --model_size base \
  --batch_size 4 --seed 42 \
  --adapter full --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-5 \
  --out_dir plots/kepler_q9v3/full_ft_timesfm \
  >"${LOG}" 2>&1
rc=$?
echo "[run] $(date)  end  rc=${rc}"
