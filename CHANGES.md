# Changes — LIGO Benchmark Migration (2026-04-30)

## Environment

- **Removed** `ts_cuda312` conda environment.
- **Created** `.venv` via `uv sync --extra jax --extra cu12` (Python 3.12, torch ≥ 2.6 cu124, JAX 0.10 with CUDA 12 backend).
  - `pyproject.toml` was updated: `torch>=2.11.0` → `torch>=2.6.0`, added `[tool.uv.sources]` pointing to `https://download.pytorch.org/whl/cu124` so uv resolves a CUDA 12 wheel (cluster max: CUDA 12.9).
  - Activate: `source .venv/bin/activate`
- Foundation-model experiments use a **separate** Python 3.10 env at `benchmarks/foundation/.venv`:
  - Create: `make env-fm`
  - Activate: `source benchmarks/foundation/.venv/bin/activate`

## Dataset

All new LIGO experiments target:

```
/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/
  ai4gw@cern/bns_snr_5_50_powerlaw_256Hz_80K_10K_100K/
    train/sig_combined_train.h5   (50 300 samples)
    val/sig_combined_val.h5       (10 000 samples)
    test/sig_combined_test.h5     (100 000 samples)
```

- Shape: `(N, 2, 16384)` — 2-channel (H1 + L1), 64 s at 256 Hz.
- Active window: **59–63 s** → `L = 1024` samples.
- Keys used: `whitened_injected`, `whitened_signal`, `chirp_mass`, `snr`.

## Modified files

### `src/dataloader/LIGO_dataloader.py` (rewrite)

| Before | After |
|--------|-------|
| Single `LIGODataset` / `LIGODataModule` (regression only, scalar targets) | + `LIGODenoisingDataset` / `LIGODenoisingDataModule` (returns waveform pairs) |
| `injected_data_key = 'injected_data'` | `injected_data_key = 'whitened_injected'` |
| No `signal_data_key` | `signal_data_key = 'whitened_signal'` in denoising classes |
| Inconsistent `strain_frequency` default (512 Hz) | Corrected to 256 Hz |

Batch formats:

```python
# LIGODataModule (regression)
X_injected   (B, n_ifos, L)   # noisy strain
y_targets    (B, n_targets)   # e.g. chirp_mass
z_observed   (B, n_obs)       # e.g. snr

# LIGODenoisingDataModule (denoising)
X_injected   (B, n_ifos, L)   # noisy strain
X_signal     (B, n_ifos, L)   # clean template
y_params     (B, n_params)    # optional scalar params
```

### `src/models/conv_ae.py`

Added `d_input: int = 1` parameter.  
Setting `d_input=2` enables 2-channel (H1+L1) LIGO denoising without any other change.  
The encoder first-layer `in_ch` and decoder output `Conv1d(latent_channels, d_input)` are now parameterised.

### `src/tasks/LIGO/denoising.py` (rewrite)

- `DenoisingMSE`: updated batch unpack to `(X, y, _)` matching new denoising DataModule; forward now uses `x.transpose(1, 2)` so any `(B, L, d_input)` model works.
- `DenoisingPSD`: new class — same as `DenoisingMSE` but with PSD-domain MSE loss (`torch.fft.rfft`).

## New files

### `src/tasks/LIGO/regression_jax.py`

`LinOSSLIGOGaussNLL` — JAX/equinox Lightning module:

- Extends `JAXLightningModule`.
- Overrides `_prepare_batch`: transposes `(B, n_ifos, L)` → `(B, L, n_ifos)` for LinOSS.
- Loss: **Gaussian NLL** — model outputs `(2*n_targets,)` per sample (first half = means, second = log-variances); log-variance clamped to `[-10, 10]`.

### `configs/LIGO/train_ligo_linoss_gaussnll_raw.yaml`

LinOSS regression config:
- `LinOSS(num_blocks=4, N=2, ssm_size=64, H=256, output_dim=2, discretization=damped_IMEX)`
- Target: `chirp_mass`; `d_output=2` (mean + log-var)
- `accelerator: cpu` — JAX manages its own GPU device

### `configs/LIGO/train_ligo_conv_ae_denoising_mse.yaml`

ConvAE denoising config:
- `ConvAE(n_layers=4, latent_channels=64, kernel_size=5, d_input=2)`
- Uses `LIGODenoisingDataModule` with `whitened_injected` → `whitened_signal`

### `benchmarks/LIGO/chronos_ligo.py`

Standalone Chronos benchmark script (requires `benchmarks/foundation/.venv`):

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `zero_shot` | `zero_shot` or `finetune` |
| `--model_size` | `small` | `tiny/small/base/large` |
| `--finetune_strategy` | `lora` | `frozen/last_n/lora` |
| `--finetune_n_blocks` | `2` | Blocks to unfreeze / LoRA target range |
| `--smoke_test` | false | Use 2×batch samples per split |

Embeds each IFO independently with Chronos, concatenates embeddings (`2 × d_model`), trains an MLP regression head for `chirp_mass`.

## Fast-dev-run results (2026-04-30)

All 4 experiments passed fast-dev-run / smoke tests.

| Job ID | Experiment | Status | Notes |
|--------|-----------|--------|-------|
| 9466140 | `ligo_linoss_gaussnll_raw` | ✅ PASSED | `train/loss=2.55, val/loss=0.673` |
| 9464239 | `ligo_conv_ae_denoising_mse` | ✅ PASSED | `train/loss=0.259, val/loss=0.056` |
| 9464399 | Chronos zero-shot smoke test | ✅ PASSED | `RMSE=0.287, MAE=0.241` |
| 9464400 | Chronos fine-tune (last_n) smoke test | ✅ PASSED | `RMSE=0.288, MAE=0.241` |

Earlier failed attempts: 9458939, 9458951 (missing CUDA modules), 9459018/9459019 (foundation env not yet created), 9464226 (cuDNN mismatch, see below).

Logs: `slurm_logs/LIGO/`

## Fixes applied

### cuDNN mismatch for JAX jobs

JAX in `.venv` was compiled against cuDNN 9.8.0, but the cluster default is cuDNN 9.1.0.  
**Fix**: `benchmarks/LIGO/run.sh` now uses a separate `ACTIVATE_JAX` string that explicitly loads `cudnn/9.10.2.21_cuda12-fasrc01` before activating the venv:

```bash
ACTIVATE_JAX="source ~/.bashrc && module load cudnn/9.10.2.21_cuda12-fasrc01 && source $WORKDIR/.venv/bin/activate"
```

### Missing CUDA modules in earlier jobs

`source $WORKDIR/.venv/bin/activate` alone is insufficient on the cluster — CUDA modules are not in the system path by default. All SLURM `--wrap` commands must start with `source ~/.bashrc` (which initialises Lmod) before any `module load` or venv activation.

### Foundation env conflicts

`make env-fm` failed due to `momentfm` vs `transformers` version conflict. Created a minimal Chronos-only env instead:

```bash
uv venv benchmarks/foundation/.venv --python 3.10
uv pip install --python benchmarks/foundation/.venv/bin/python \
    "chronos-forecasting" "torch==2.4.1" h5py numpy
```

### torch cu130 → cu124

`pyproject.toml` had `torch>=2.11.0` which resolved to a CUDA 13.0 nightly wheel. The cluster maximum is CUDA 12.9.  
**Fix**: installed `torch==2.6.0+cu124` directly via:

```bash
uv pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

Monitor: `squeue -u kyoon`
