# TIDMAD Data

This directory should contain the TIDMAD dataset files (not tracked by git — see below).

## About the dataset

**TIDMAD** (Time-Domain Dark Matter Detection) is a 689 GB dataset from the ABRACADABRA
experiment, recording SQUID magnetometer voltage during an axion dark matter search.

- **Paper:** https://arxiv.org/abs/2406.04378 (NeurIPS 2025 Spotlight)
- **Repository:** https://github.com/jessicafry/TIDMAD
- **HuggingFace:** https://huggingface.co/datasets/jessicafry/TIDMAD

## File structure

```
abra_training_{0000-0019}.h5       # 20 files, ~2 GB each
abra_validation_{0000-0019}.h5     # 20 files, ~2 GB each
abra_science_{0000-0207}.h5        # 208 files, ~2.5 GB each
```

Each training/validation file has two channels at 10 MHz, int8, 2×10⁹ samples (~200 s):
- `timeseries/channel0001/timeseries` — SQUID readout (noisy input)
- `timeseries/channel0002/timeseries` — injected reference signal (clean ground truth)

Science files have only `channel0001` (the real DM search data, no ground truth).

## How to get the data

### Option A — Symlink from existing copy (IAIFI cluster)

```bash
rmdir data/TIDMAD
ln -s /n/holystore01/LABS/iaifi_lab/Lab/creissel/TIDMAD data/TIDMAD
```

### Option B — Download from OSDF (official)

```bash
git clone https://github.com/jessicafry/TIDMAD /tmp/tidmad_repo
cd /tmp/tidmad_repo
# Small subset for development (2 train + 2 val files)
python download_data.py --train_files 2 --validation_files 2 --science_files 0 \
  --output_dir /path/to/TimeSeriesPhysics/data/TIDMAD
# Full dataset (~689 GB)
python download_data.py --train_files 20 --validation_files 20 --science_files 208 \
  --output_dir /path/to/TimeSeriesPhysics/data/TIDMAD
```

### Option C — HuggingFace

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('jessicafry/TIDMAD', repo_type='dataset', local_dir='data/TIDMAD')
"
```
