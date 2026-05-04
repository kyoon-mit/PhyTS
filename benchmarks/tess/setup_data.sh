#!/bin/bash
# Download the pre-split TESS parquet files needed for classification training.

set -e

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_DIR=$WORKDIR/data/TESS/.cache/TESS

mkdir -p "$DATA_DIR"

DATA_DIR="$DATA_DIR" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "PhyTS-team/PhyTS-bench"
repo_type = "dataset"
dest = Path(os.environ["DATA_DIR"])

for filename in [
    "TESS/split/tess_classification_train.parquet",
    "TESS/split/tess_classification_val.parquet",
    "TESS/split/tess_classification_test.parquet",
    "TESS/split/tess_regression_train.parquet",
    "TESS/split/tess_regression_val.parquet",
    "TESS/split/tess_regression_test.parquet",
]:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    target = dest / Path(filename).name
    source = Path(path)
    if source != target:
        source.replace(target)
    print(target)
PY

echo "Downloaded pre-split TESS parquet files into: $DATA_DIR"