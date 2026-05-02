# TIDMAD Denoising Benchmark — Experiment Summary

## Overview

This experiment benchmarks two deep learning architectures for signal denoising on the TIDMAD dataset
(ABRACADABRA dark matter experiment). The task is to recover a clean SQUID signal from a noisy input,
evaluated using a PSD-based SNR metric from the official TIDMAD benchmark.

---

## Dataset

- **Source**: ABRACADABRA dark matter detector data
- **Input (ch1)**: Noisy SQUID magnetometer signal
- **Reference (ch2)**: Clean signal used for synthetic injection and SNR normalization
- **Sampling rate**: 10 MHz
- **Window size**: 100,000 samples per window
- **Test set**: 8 H5 files (~160,000 windows total)
  - `abra_validation_0013.h5`, `abra_training_0014.h5`, `abra_training_0013.h5`
  - `abra_validation_0026.h5`, `abra_training_0036.h5`, `abra_validation_0025.h5`
  - `abra_validation_0037.h5`, `abra_training_0008.h5`
- **Preprocessed data**: `data/TIDMAD/preprocessed/` (normalized NPY shards)
- **Original data**: `data/TIDMAD/original/` (raw H5 files)

---

## Benchmark Metric

The official TIDMAD Benchmark 1 score is:

```
score = log_5.27( mean( norm_ch2_SNR * denoised_ch1_SNR ) )
```

Where:
- A synthetic narrow-band signal is injected into ch1 at a frequency identified from ch2
- The model denoises the injected ch1
- `denoised_ch1_SNR` measures how well the denoised signal recovers the injection
- `norm_ch2_SNR` is a normalization factor from the clean ch2 reference
- Higher score = better denoising

Implementation adapted from the official TIDMAD team code
(`src/tasks/TIDMAD/tidmad_denoising.py`, `@author: TIDMAD Team`).

---

## Models

### ConvAE-L (~190K parameters)
- **Architecture**: 1D convolutional autoencoder
  - `n_layers=5`, `latent_channels=64`, `kernel_size=9`, `pool_stride=2`
- **Framework**: PyTorch + PyTorch Lightning
- **Optimizer**: AdamW with exponential LR decay (lr=1e-3, lr_decay=0.99)
- **Source**: `src/models/conv_ae.py`

### LinOSS-190K (~200K parameters)
- **Architecture**: Linear Oscillatory State Space Model
  - `num_blocks=4`, `ssm_size=32`, `H=128`, `discretization=damped_IMEX`
- **Framework**: JAX + Equinox + PyTorch Lightning (JAXLightningModule)
- **Optimizer**: AdamW with grad clipping (lr=1e-4, clip_grad_norm=1.0)
- **Note**: lr=1e-3 caused NaN gradients; reduced to 1e-4 for stability
- **Source**: `src/models/linoss.py`, `src/tasks/TIDMAD/linoss_denoising.py`

---

## Loss Functions

### Time-Domain MSE (baseline)
```
L = mean( (y_pred - y_true)^2 )
```
- Standard mean squared error in the time domain
- **Problem**: Dominated by broadband noise floor; completely ignores the narrow-band injection signal
- The benchmark metric is PSD-based, so MSE training is misaligned with the evaluation criterion

### PSD Loss (improved)
```python
# PyTorch (ConvAE)
psd_pred = torch.fft.rfft(y_pred).abs().pow(2)
psd_true = torch.fft.rfft(y_true).abs().pow(2)
L = F.mse_loss(psd_pred, psd_true)

# JAX (LinOSS)
psd_pred = jnp.abs(jnp.fft.rfft(y_pred, axis=1)) ** 2
psd_true = jnp.abs(jnp.fft.rfft(y_true, axis=1)) ** 2
L = jnp.mean((psd_pred - psd_true) ** 2)
```
- MSE over power spectral density (frequency domain)
- Directly penalizes discrepancies at each frequency bin
- Much better aligned with the PSD-based SNR benchmark metric

---

## Architecture Ablation (max 5 epochs, plot_only — no test scores)

| Variant     | Params  | Val Loss | Epochs | Notes              |
|-------------|---------|----------|--------|--------------------|
| conv_abl_s  | 4,481   | 52.46    | 4      | Smallest ConvAE    |
| conv_abl_m  | 36,929  | 52.61    | 4      |                    |
| conv_abl_l  | 190,977 | **52.38**| 5      | Best ablation      |
| conv_abl_w  | 67,073  | 52.41    | 5      | Wider latent       |
| conv_abl_xl | 758,785 | 52.42    | 2      | 4x overparams      |
| linoss_abl_s| 4,401   | 52.45    | 4      | Smallest LinOSS    |

**Conclusion**: `conv_abl_l` (190K params) achieves the best ablation val loss. Increasing to conv_xl
(759K) does not improve and requires more compute. conv_l was selected as the full training baseline.

---

## Full Training Results

| Variant         | Model    | Loss | Params  | Best Val   | Epochs | Benchmark Score |
|-----------------|----------|------|---------|------------|--------|-----------------|
| conv_l_mse      | ConvAE-L | MSE  | 190,977 | 52.36      | 17     | **-5.526**      |
| conv_l_full     | ConvAE-L | PSD  | 190,977 | ~5.3e18    | 16     | **-3.685**      |
| linoss_190k_mse | LinOSS   | MSE  | ~200K   | 52.43      | 9      | _(pending)_     |
| linoss_190k_full| LinOSS   | PSD  | ~200K   | (epoch 3)  | 9*     | _(pending)_     |

*LinOSS PSD run hit 12h wall time limit at epoch 9; best checkpoint saved at epoch 3.

### Chronos Zero-Shot Baseline

| Variant       | Model          | Params | Benchmark Score |
|---------------|----------------|--------|-----------------|
| chronos_tiny  | Chronos (tiny) | 8M     | **-3.071**      |

---

## Key Findings

### 1. MSE Loss is Wrong for TIDMAD

Time-domain MSE training (conv_l_mse) produced a score of **-5.526**, significantly worse than the
Chronos zero-shot baseline (**-3.071**). The model learns to suppress all signal components equally,
destroying the narrow-band injection that the benchmark measures.

### 2. PSD Loss Substantially Improves Performance

Switching to PSD loss (conv_l_full) raised the score from -5.526 to **-3.685** — a large improvement,
though still below the Chronos zero-shot baseline (-3.071). This confirms that the training objective
must be aligned with the frequency-domain evaluation metric.

### 3. Chronos Zero-Shot is a Strong Baseline

Chronos tiny (8M params, no fine-tuning) scores -3.071. This is better than MSE-trained models
and competitive with PSD-trained ConvAE-L (-3.685). The difference is small; LinOSS PSD results
may shed more light on whether a purpose-trained model can clearly outperform zero-shot forecasters.

### 4. LinOSS Stability Required damped_IMEX

Standard IMEX discretization caused NaN gradients at high learning rates. Switching to
`damped_IMEX` discretization resolved the training instability. Learning rate was also reduced
from 1e-3 to 1e-4.

---

## Score Interpretation

Higher (less negative) = better. Score = 0 means perfect denoising.

| Score  | Model              | Notes                                      |
|--------|--------------------|--------------------------------------------|
| 0.0    | —                  | Perfect denoising (theoretical ceiling)    |
| -3.07  | Chronos zero-shot  | Best result so far (no fine-tuning)        |
| -3.68  | ConvAE-L PSD       | Worse than Chronos; better than MSE        |
| -5.53  | ConvAE-L MSE       | Worst; time-domain loss misaligned with metric |

All scores are negative because no model yet enhances the injection SNR — they all suppress some
signal along with the noise. Chronos zero-shot currently leads despite having no task-specific
training, which suggests the trained models are not yet learning to preserve the narrow-band
injection frequency.

---

## Pending Results

- LinOSS-190K MSE benchmark score
- LinOSS-190K PSD benchmark score
- Chronos medium/large zero-shot scores (optional)

---

## File Locations

| Artifact                     | Path                                                           |
|------------------------------|----------------------------------------------------------------|
| Master results CSV           | `benchmarks/TIDMAD/results/master_results.csv`                 |
| Conv MSE scores              | `benchmarks/TIDMAD/results/conv_l_mse/test_scores.csv`         |
| Conv PSD scores              | `benchmarks/TIDMAD/results/conv_l_full/test_scores.csv`        |
| Conv MSE loss curve          | `benchmarks/TIDMAD/results/conv_l_mse/loss_curve.png`          |
| Conv PSD loss curve          | `benchmarks/TIDMAD/results/conv_l_full/loss_curve.png`         |
| Conv MSE checkpoint          | `checkpoints/tidmad_conv_l/best.ckpt`                          |
| Conv PSD checkpoint          | `checkpoints/tidmad_conv_l_psd/best.ckpt`                      |
| LinOSS MSE checkpoint dir    | `logs/tidmad_linoss_190k/version_0/checkpoints/`               |
| LinOSS PSD checkpoint dir    | `logs/tidmad_linoss_190k_psd/version_0/checkpoints/`           |
| SLURM logs                   | `/home/ilay.kamai/athena/logs/`                                |

---

## Training Configs

| Variant          | Config                                                  |
|------------------|---------------------------------------------------------|
| ConvAE-L MSE     | `configs/TIDMAD/train_tidmad_conv_l_denoising.yaml` (original, DenoisingMSE) |
| ConvAE-L PSD     | `configs/TIDMAD/train_tidmad_conv_l_denoising.yaml` (DenoisingPSD, ckpt=_psd) |
| LinOSS-190K MSE  | `configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml` (original, _jax_mse) |
| LinOSS-190K PSD  | `configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml` (_jax_psd_loss, log=_psd) |
