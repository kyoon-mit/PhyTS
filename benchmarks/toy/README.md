# Toy Benchmark: Raw vs Denoised Regression

Compares how well physical signal parameters are recovered from (a) the raw noisy
signal and (b) the denoiser's output, as a function of SNR.

---

## Setup

Two models are needed:

| Model | Config | Task |
|-------|--------|------|
| Denoiser | `configs/train_toy.yaml` | `DenoisingMSE` (S4D seq2seq) |
| Regressor | `configs/train_toy_regression.yaml` | `RegressionMSE` (S4Model) |

### Train the denoiser
```bash
python main.py fit --config configs/train_toy.yaml
```

### Train the regressor (on raw noisy signals)
```bash
python main.py fit --config configs/train_toy_regression.yaml
```

Lightning saves checkpoints to `lightning_logs/` by default.

---

## Running the Benchmark

```bash
python benchmarks/toy/eval_pipeline.py \
    --denoiser_ckpt  <path/to/denoising.ckpt> \
    --regressor_ckpt <path/to/regression.ckpt> \
    --data_dir       data/toy/sinusoidal_signal_white_noise \
    --out_dir        benchmarks/toy
```

Outputs `benchmarks/toy/regression_benchmark.png`.

---

## What the Plot Shows

Bar chart of RMSE per SNR bin (low / mid / high), one panel per target parameter
(`amplitude`, `frequency_hz`, `phase_rad`).

- **Blue bars** — regression on raw noisy signal
- **Red bars** — regression on denoised signal

The key result: denoising should reduce RMSE most in the **low-SNR** bin (SNR < 0.1),
and the benefit should shrink as SNR increases.

---

## SNR Bins

| Bin | SNR range |
|-----|-----------|
| low | < 0.1 |
| mid | 0.1 – 0.3 |
| high | > 0.3 |

The toy dataset has typical SNR ≈ 0.14 (A ≈ 1, σ ≈ 5), so most samples fall in the
mid bin, with a spread across all three.

---

## Notes on Phase

Phase is circular (0 to 2π). MSE treats it as a linear quantity, so errors near the
0/2π wrap-around are overestimated. This is acceptable for a toy benchmark — treat
phase RMSE as indicative rather than exact.
