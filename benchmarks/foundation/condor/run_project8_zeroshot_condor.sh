#!/usr/bin/env bash
# Condor executable for Project 8 zero-shot linear probe.
# Forwards all arguments to run_project8.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[zs-p8] host=$(hostname)  pid=$$"
echo "[zs-p8] python=${PYTHON_BIN}"
echo "[zs-p8] PYTHONPATH=${PYTHONPATH:-}"
echo "[zs-p8] HF_HOME=${HF_HOME:-}"
echo "[zs-p8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[zs-p8] args: $*"
echo "[zs-p8] $(date)  start"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# run_project8.py picks up LAGLLAMA_CKPT from the environment (set by the
# Condor job file).  No CLI arg needed.
"${PYTHON_BIN}" -u benchmarks/foundation/run_project8.py "$@"

echo "[zs-p8] $(date)  done"
