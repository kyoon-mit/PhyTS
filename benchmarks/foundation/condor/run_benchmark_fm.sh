#!/usr/bin/env bash
# Condor executable: foundation model benchmark (Python 3.10 env)
#
# Arguments are passed through to run_benchmark.py.
# The environment variable FM_PYTHON can override the Python binary.
#
# Example:
#   FM_PYTHON=/path/to/tsenv_fm/bin/python
#   condor_submit benchmarks/foundation/condor/bench_fm.job
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[fm_bench] host=$(hostname)  pid=$$"
echo "[fm_bench] python=${PYTHON_BIN}"
echo "[fm_bench] args: $*"
date

"${PYTHON_BIN}" benchmarks/foundation/run_benchmark.py "$@"

echo "[fm_bench] DONE"
date
