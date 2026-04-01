# Toy Benchmark: Raw vs Denoised Regression

Compares how well physical signal parameters are recovered across three pipelines,
as a function of SNR.

---

## Quickstart

```bash
bash benchmarks/toy/run.sh
```

That's it. The script submits 4 SLURM jobs with automatic dependencies:

```
[1] toy_denoiser   ──┬──> [2] toy_reg_raw   ──┬──> [4] toy_eval
                     └──> [3] toy_reg_clean ──┘
```

Steps 2 and 3 run in parallel. Step 4 starts only after both finish.

Monitor progress:
```bash
squeue -u $USER
tail -f benchmarks/toy/logs/toy_denoiser_<jobid>.out
```

Results are saved to `benchmarks/toy/regression_benchmark.png`.

---

## Pipelines

| Pipeline | Input to regressor | Regressor trained on |
|----------|-------------------|----------------------|
| **raw** | `sig_bkg` (noisy) | `sig_bkg` (noisy) |
| **denoised** | `denoiser(sig_bkg)` | `sig` (clean) |
| **oracle** | `sig` (clean) | `sig` (clean) |

The gap between **denoised** and **oracle** measures how much regression error
comes from imperfect denoising. The gap between **raw** and **denoised** measures
the benefit of denoising.

---

## Models & Configs

| Step | Config | Checkpoint |
|------|--------|-----------|
| Denoiser | `configs/toy/train_toy_s4d_denoising.yaml` | `checkpoints/toy_s4d_denoising/best.ckpt` |
| Raw regressor | `configs/toy/train_toy_mlp_regression_raw.yaml` | `checkpoints/toy_mlp_regression_raw/best.ckpt` |
| Clean regressor | `configs/toy/train_toy_mlp_regression_clean.yaml` | `checkpoints/toy_mlp_regression_clean/best.ckpt` |

The denoiser is S4D seq2seq (`S4ModelSeq2Seq`). Both regressors are MLPs
(`MLPRegressor`, ~205K params): flatten the 640-step sequence then pass through
FC layers `[256, 128, 64]` with LayerNorm + GELU + Dropout.

### How denoising is applied before regression

The **raw** and **clean** regressors are trained separately on their respective
input domains. At eval time, the denoiser bridges them:

```
raw pipeline:      sig_bkg ──────────────────────────> reg_raw   -> y_hat
denoised pipeline: sig_bkg -> denoiser -> sig_approx -> reg_clean -> y_hat
oracle pipeline:   sig     ──────────────────────────> reg_clean -> y_hat
```

`reg_clean` is trained on clean `sig`, so feeding it `denoiser(sig_bkg)` tests
how well the denoiser recovers the true signal distribution. The two gaps are:

- **oracle − denoised**: regression error from imperfect denoising
- **denoised − raw**: benefit of denoising over using the raw noisy signal

---

## Target Parameters

`[amplitude, frequency_hz, phase_rad]` — the physical signal parameters.
`noise_amplitude` and `snr` are excluded (they describe the noise, not the signal).

> **Note on phase:** Phase is circular (0 to 2π). MSE treats it as linear,
> so errors near the 0/2π boundary are overestimated. Treat phase RMSE as indicative.

---

## SNR Bins

| Bin | SNR range |
|-----|-----------|
| low | < 0.1 |
| mid | 0.1 – 0.3 |
| high | > 0.3 |

The toy dataset has typical SNR ≈ 0.14, so most samples fall in the mid bin.
The benefit of denoising should be largest in the low-SNR bin.
