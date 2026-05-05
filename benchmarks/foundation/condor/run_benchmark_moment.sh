#!/usr/bin/env bash
# Condor executable: MOMENT + Chronos benchmark (Python 3.12 tsenv)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${MOMENT_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv/bin/python}"

echo "[moment_bench] host=$(hostname)  pid=$$"
echo "[moment_bench] python=${PYTHON_BIN}"
echo "[moment_bench] args: $*"
date

"${PYTHON_BIN}" benchmarks/foundation/run_benchmark.py "$@"

echo "[moment_bench] DONE"
date
