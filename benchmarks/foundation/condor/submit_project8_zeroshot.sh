#!/usr/bin/env bash
# Submit one Condor job per foundation model for Project 8 zero-shot.
#
# Usage:
#   bash benchmarks/foundation/condor/submit_project8_zeroshot.sh
#   bash benchmarks/foundation/condor/submit_project8_zeroshot.sh chronos moment    # subset
#   NOISE=gauss bash benchmarks/foundation/condor/submit_project8_zeroshot.sh        # safety baseline
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOB="${REPO}/benchmarks/foundation/condor/project8_zeroshot.job"

DEFAULT_MODELS=(granite_ttm lagllama moirai chronos timemoe moment timesfm)
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

NOISE="${NOISE:-cav}"
OUT_DIR="${OUT_DIR:-plots/Project8/linear_probe_${NOISE}}"
MODEL_SIZE="${MODEL_SIZE:-base}"

mkdir -p "${REPO}/benchmarks/foundation/condor/logs"

for m in "${MODELS[@]}"; do
    echo "[submit] model=${m}  noise=${NOISE}  size=${MODEL_SIZE}  out=${OUT_DIR}"
    condor_submit \
        -a "tag=${m}" \
        -a "model=${m}" \
        -a "model_size=${MODEL_SIZE}" \
        -a "noise_type=${NOISE}" \
        -a "out_dir=${OUT_DIR}" \
        "${JOB}"
done

echo
echo "Submitted. Watch with:  condor_q $USER"
echo "Logs in:                ${REPO}/benchmarks/foundation/condor/logs/p8_zs_*"
