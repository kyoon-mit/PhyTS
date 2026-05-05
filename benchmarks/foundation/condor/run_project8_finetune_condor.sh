#!/usr/bin/env bash
# Condor executable for Project 8 LoRA / last-N / full fine-tune.
# Forwards all arguments to run_project8_finetune.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[ft-p8] host=$(hostname)  pid=$$"
echo "[ft-p8] python=${PYTHON_BIN}"
echo "[ft-p8] HF_HOME=${HF_HOME:-}"
echo "[ft-p8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[ft-p8] args: $*"
echo "[ft-p8] $(date)  start"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -u benchmarks/foundation/run_project8_finetune.py "$@"

echo "[ft-p8] $(date)  done"
