#!/usr/bin/env bash
# Submit one Condor job per foundation model for Project 8 fine-tuning.
#
# Usage:
#   bash benchmarks/foundation/condor/submit_project8_finetune.sh
#       → defaults: --adapter lora, all models that support LoRA (granite_ttm dropped)
#   bash benchmarks/foundation/condor/submit_project8_finetune.sh moment chronos
#       → subset
#   ADAPTER=full bash benchmarks/foundation/condor/submit_project8_finetune.sh
#       → full fine-tune (heavier; expensive)
#   ADAPTER=last_n UNFREEZE=2 bash benchmarks/foundation/condor/submit_project8_finetune.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOB="${REPO}/benchmarks/foundation/condor/project8_finetune.job"

ADAPTER="${ADAPTER:-lora}"
NOISE="${NOISE:-cav}"
MODEL_SIZE="${MODEL_SIZE:-base}"
EPOCHS="${EPOCHS:-10}"
OUT_DIR="${OUT_DIR:-plots/Project8/${ADAPTER}_${NOISE}}"

case "${ADAPTER}" in
    lora)         DEFAULT_MODELS=(moment chronos timemoe timesfm moirai) ;;
    last_n|full)  DEFAULT_MODELS=(moment chronos timemoe timesfm moirai granite_ttm) ;;
    frozen)       DEFAULT_MODELS=(moment chronos timemoe timesfm moirai granite_ttm) ;;
    *) echo "ADAPTER must be lora|last_n|full|frozen" >&2; exit 1 ;;
esac
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

mkdir -p "${REPO}/benchmarks/foundation/condor/logs"

for m in "${MODELS[@]}"; do
    if [[ "${ADAPTER}" == "lora" && "${m}" == "granite_ttm" ]]; then
        echo "[skip] granite_ttm + lora is a no-op (no attention projections)"
        continue
    fi
    TAG="${m}_${ADAPTER}"
    echo "[submit] model=${m}  adapter=${ADAPTER}  size=${MODEL_SIZE}  out=${OUT_DIR}"
    condor_submit \
        -a "tag=${TAG}" \
        -a "model=${m}" \
        -a "adapter=${ADAPTER}" \
        -a "model_size=${MODEL_SIZE}" \
        -a "noise_type=${NOISE}" \
        -a "out_dir=${OUT_DIR}" \
        -a "epochs=${EPOCHS}" \
        "${JOB}"
done

echo
echo "Submitted. Watch with:  condor_q $USER"
echo "Logs in:                ${REPO}/benchmarks/foundation/condor/logs/p8_ft_*"
