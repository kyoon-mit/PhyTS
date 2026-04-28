# TIDMAD Benchmark: Denoising + Parameter Regression

Evaluates denoising quality and dark matter signal parameter recovery on the
[TIDMAD dataset](https://arxiv.org/abs/2406.04378) (ABRACADABRA experiment).

## Pipeline

```bash
bash benchmarks/TIDMAD/run.sh
```

Submits 4 chained SLURM jobs:

```
[1] train denoiser ──┬──> [2] train reg_raw   ──┬──> [4] eval
                     └──> [3] train reg_clean ──┘
```

| Pipeline | Training input | Eval input | Purpose |
|----------|---------------|------------|---------|
| **raw** | `channel0001` (noisy) | `channel0001` | baseline |
| **denoised** | `channel0002` (clean) | `denoiser(channel0001)` | benefit of denoising |
| **oracle** | `channel0002` (clean) | `channel0002` | upper bound |

Results saved to `benchmarks/TIDMAD/regression_benchmark.png`.

## Models & Configs

| Step | Config | Checkpoint |
|------|--------|-----------|
| Denoiser | `configs/TIDMAD/train_tidmad_s4d_denoising.yaml` | `checkpoints/tidmad_s4d_denoising/best.ckpt` |
| Raw regressor | `configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml` | `checkpoints/tidmad_mlp_regression_raw/best.ckpt` |
| Clean regressor | `configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml` | `checkpoints/tidmad_mlp_regression_clean/best.ckpt` |

## Data format

- **Input**: 20 training + 20 validation H5 files at 10 MHz, int8
- **channel0001**: SQUID magnetometer readout (noisy)
- **channel0002**: injected reference signal (clean ground truth)
- **Normalization**: `int8 × (volt_range_mV / 256)` → physical millivolts
- **Window size**: 1 second = 10,000,000 samples (non-overlapping)
- **Downsampling**: `downsample_factor=1000` → `seq_len=10,000`, effective Nyquist = 5 kHz

> **Note**: with `downsample_factor=1000`, only injected frequencies below 5 kHz
> are accurately represented. The full injection range is 1.1 kHz–4.9 MHz.
> Lower `downsample_factor` in the YAML to cover higher frequencies (at higher
> memory and compute cost).

## Regression targets

Labels are derived per window from `channel0002` FFT (same method as the
official TIDMAD benchmark script):

- `frequency_hz` — dominant FFT peak frequency (Hz)
- `amplitude` — peak amplitude in millivolts
- `snr` — signal power / noise power in ±1 vs ±50 PSD bins around the peak

SNR bins: `low < 1`, `1 ≤ mid < 10`, `high ≥ 10`.
