#!/usr/bin/env bash
# Download the Project 8 split of PhyTS-team/PhyTS-bench (~49.7 GB) into
# data/Project8/{train,val,test}.  Idempotent: huggingface-cli skips files
# already present, and we only rename `valid` -> `val` once.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found.  Install with: pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

huggingface-cli download PhyTS-team/PhyTS-bench \
    --repo-type dataset \
    --include 'Project8/*' \
    --local-dir data

if [ -d data/Project8/valid ] && [ ! -d data/Project8/val ]; then
    mv data/Project8/valid data/Project8/val
fi

du -sh data/Project8/{train,val,test}
ls data/Project8/{train,val,test} | sed 's/^/  /'
