#!/bin/bash
# Download TESS PhyTS-bench parquet files from Hugging Face.
# Final location: data/TESS/tess_classification.parquet
#
# Usage:
#   bash benchmarks/TESS/setup_data.sh

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)

if [ -f "$REPO_ROOT/data/TESS/tess_classification.parquet" ]; then
    echo "Data already present at data/TESS/tess_classification.parquet — nothing to do."
    exit 0
fi

echo "Downloading TESS data from PhyTS-team/PhyTS-bench ..."
python "$REPO_ROOT/data/TESS/download_tess.py"

echo "Done. File: data/TESS/tess_classification.parquet"
