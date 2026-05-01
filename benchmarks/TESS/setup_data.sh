#!/bin/bash
# Download TESS PhyTS-bench Parquet files from Hugging Face.
#
# Run this ONCE from the Engaging login node before launching run.sh.
# Login nodes have internet access; compute nodes generally do not.
#
# Usage (from anywhere):
#   bash benchmarks/TESS/setup_data.sh
#
# The two files (~225 MB total) are saved under the pool tree (default), e.g.:
#   $TESS_POOL_ROOT/data_engaging/TESS/.cache/TESS/tess_regression.parquet
#
# Optional: ``export TESS_POOL_ROOT=...`` when your pool path differs from the default.
#
# Optional: set HF_TOKEN for better HuggingFace rate limits.
#   export HF_TOKEN=hf_...

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
POOL="${TESS_POOL_ROOT:-/home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics}"
DATA_DIR="$POOL/data_engaging/TESS/.cache"

# Skip if both files already present
if [ -f "$DATA_DIR/TESS/tess_regression.parquet" ] && [ -f "$DATA_DIR/TESS/tess_classification.parquet" ]; then
    echo "Data already present at $DATA_DIR — nothing to do."
    exit 0
fi

echo "Downloading TESS PhyTS-bench data to $DATA_DIR ..."
module load cuda miniforge

# huggingface_hub is in the jax extra; pyarrow is a core dep
uv run --extra jax python "$REPO_ROOT/data/TESS/download_tess.py" \
    --cache-dir "$DATA_DIR"

echo ""
echo "Files:"
ls -lh "$DATA_DIR"/TESS/*.parquet
