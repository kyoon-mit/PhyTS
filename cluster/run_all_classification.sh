#!/bin/bash
# Launch all TESS classification models on SLURM.
# Usage: bash cluster/run_all_classification.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODEL in s4d rnn conv; do
    echo "Submitting ${MODEL}..."
    sbatch --export=ALL,MODEL="${MODEL}" \
           --job-name="tess_cls_${MODEL}" \
           "${SCRIPT_DIR}/tess_classification.sbatch"
done

echo "All jobs submitted. Check status with: squeue -u \$USER"
