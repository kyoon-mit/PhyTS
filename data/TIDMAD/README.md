# TIDMAD Data

This directory holds the raw TIDMAD dataset files used by
`src/dataloader/tidmad_dataloader.py`.

## Directory layout

```
data/TIDMAD/
  original/              # symlinks (or copies) to raw H5 files — not tracked by git
    abra_training_*.h5   # 20 files, ~2 GB each — two channels, injected signal
    abra_validation_*.h5 # 40 files, ~2 GB each — two channels, injected signal
    abra_science_*.h5    # 208 files, ~2.5 GB each — one channel, real DM search data
  README.md
```

## About the dataset

**TIDMAD** (Time-Domain Dark Matter Detection) is a 689 GB dataset from the ABRACADABRA
experiment, recording SQUID magnetometer voltage during an axion dark matter search.

- **Paper:** https://arxiv.org/abs/2406.04378 (NeurIPS 2025 Spotlight)
- **Repository:** https://github.com/jessicafry/TIDMAD
- **HuggingFace:** https://huggingface.co/datasets/jessicafry/TIDMAD

Each training/validation H5 file has two channels at 10 MHz, int8, ~2×10⁹ samples (~200 s):
- `timeseries/channel0001/timeseries` — SQUID readout (noisy input)
- `timeseries/channel0002/timeseries` — injected reference signal (clean ground truth)

Science files have only `channel0001` (the real DM search data, no ground truth).

## Getting the raw H5 files

### Option A — Symlink from existing copy (IAIFI cluster)

```bash
for f in /n/holystore01/LABS/iaifi_lab/Lab/creissel/TIDMAD/*.h5; do
    ln -s "$f" data/TIDMAD/original/$(basename "$f")
done
```

### Option B — Download from OSDF (official)

```bash
git clone https://github.com/jessicafry/TIDMAD /tmp/tidmad_repo
cd /tmp/tidmad_repo
python download_data.py --train_files 20 --validation_files 40 --science_files 0 \
  --output_dir /path/to/TimeSeriesPhysics/data/TIDMAD/original
```

### Option C — HuggingFace

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('jessicafry/TIDMAD', repo_type='dataset',
                  local_dir='data/TIDMAD/original')
"
```
