#!/bin/bash
# Download the re-uploaded LIGO_2048Hz subset of PhyTS-bench (HF commit a0064e5c, 2026-10-27) to /fast.
# Layout: train/bns_snr_*/train_XXXXXX-XXXXXX.h5, val/snr_5_50/*.h5,
#         test/snr_5_50/<row-range>/*.h5  (~4.0 TB total, 19k shards of 100 samples).
# Other subsets (Project8, TESS, TIDMAD, sample) were downloaded earlier and are unchanged.
# Resumable: re-submitting skips files that are already complete.
set -euo pipefail
export PATH=/home/barmstrong/.local/bin:/usr/local/bin:/usr/bin:/bin
export HF_HOME=/fast/barmstrong/hf
export HF_XET_HIGH_PERFORMANCE=1
cd /fast/barmstrong
/home/barmstrong/.local/bin/uvx --from 'huggingface_hub[hf_xet]' hf download PhyTS-team/PhyTS-bench \
    --repo-type dataset \
    --revision a0064e5c80319f960b76c7baf575469a58f056ef \
    --local-dir /fast/barmstrong/PhyTS-bench \
    --include 'LIGO_2048Hz/*' \
    --max-workers 8
echo "DONE $(date)"
