#!/usr/bin/env bash
# Condor executable: train a Lightning-CLI config (Python 3.12 tsenv)
#
# Usage:
#   condor_submit -a "config=<yaml>" benchmarks/foundation/condor/train.job
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv/bin/python"

echo "[train] host=$(hostname)  pid=$$"
echo "[train] python=${PYTHON_BIN}"
echo "[train] config=$1"
date

"${PYTHON_BIN}" main.py fit --config "$1"

echo "[train] DONE"
date
