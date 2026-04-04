# TIDMAD Data

This directory holds the TIDMAD dataset files and the preprocessing script.

## Directory layout

```
data/TIDMAD/
  original/              # symlinks (or copies) to raw H5 files — not tracked by git
    abra_training_*.h5   # 20 files, ~2 GB each — two channels, injected signal
    abra_validation_*.h5 # 40 files, ~2 GB each — two channels, injected signal
    abra_science_*.h5    # 208 files, ~2.5 GB each — one channel, real DM search data
  preprocessed/          # output of preprocess_tidmad.py — not tracked by git
    train_ch1.npy        # int8  (79968, 100000)
    train_ch2.npy        # int8  (79968, 100000)
    train_params.npy     # float32 (79968, 3)
    val_ch1.npy  ...
    test_ch1.npy ...
    scale.npy            # float32 scalar: voltage_range_mV / 256.0
  preprocess_tidmad.py   # one-time preprocessing script
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

## Step 1 — Get the raw H5 files

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

## Step 2 — Preprocess

```bash
python data/TIDMAD/preprocess_tidmad.py \
    --data_dir  data/TIDMAD/original \
    --out_dir   data/TIDMAD/preprocessed \
    --n_per_file 1666 \
    --seed      42
```

This takes ~1 hour and produces ~20 GB of int8 `.npy` files.

### What the script does

1. Collects all 60 training + validation H5 files, shuffles with `--seed`.
2. Splits by file: **48 train / 6 val / 6 test** (80/10/10).
3. For each file:
   - Reads the first 1-second segment of `channel0002` at full 10 MHz resolution.
   - Computes FFT labels (`frequency_hz`, `amplitude`, `snr`).
   - Snaps `frequency_hz` to the nearest value in the 309-point known frequency grid
     (1.1 kHz – 4.9 MHz, log-uniform steps).
   - Randomly samples `--n_per_file` non-overlapping 0.01-second windows.
4. Saves each split as memory-mappable `.npy` files (int8 for channel data).

### Preprocessed format

| File | dtype | Shape | Description |
|------|-------|-------|-------------|
| `{split}_ch1.npy` | int8 | (N, 100000) | noisy SQUID channel |
| `{split}_ch2.npy` | int8 | (N, 100000) | clean injected reference |
| `{split}_params.npy` | float32 | (N, 3) | [frequency_hz, amplitude_mV, snr] |
| `scale.npy` | float32 | scalar | `voltage_range_mV / 256.0` |

- **Window:** 100,000 samples = 0.01 s at 10 MHz (no downsampling)
- **FFT resolution:** 100 Hz (matches the minimum step in the injection frequency grid)
- **Nyquist:** 5 MHz (covers the full 1.1 kHz – 4.9 MHz injection range)
- **Total size:** ~20 GB int8 for all splits combined

## Verification

```python
from dataloader.tidmad_dataloader import TIDMADDataModule, Param
dm = TIDMADDataModule('data/TIDMAD/preprocessed')
dm.setup('fit')
ch1, ch2, params = next(iter(dm.train_dataloader()))
print(ch1.shape)     # (32, 100000)
print(ch2.shape)     # (32, 100000)
print(params.shape)  # (32, 3)
print(params[:, Param.frequency_hz])  # should be values from the known freq grid
```
