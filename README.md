# TimeSeriesPhysics

Deep learning for physics time series across four experimental domains. Built with PyTorch Lightning and configurable via YAML CLI. JAX-based models (LinOSS) use a separate `uv`-managed environment.

## Domains

### TIDMAD — Dark Matter Direct Detection
Time series from dark matter axion detection experiment (ABRACADABRA, [arXiv:2406.04378](https://arxiv.org/abs/2406.04378)).

### Project 8 — Neutrino Mass Spectroscopy
Cyclotron Radiation Emission Spectroscopy (CRES) signals for tritium beta-decay spectroscopy.

### TESS — Exoplanet Transit Photometry
Stellar light curves from the Transiting Exoplanet Survey Satellite.

### LIGO — Gravitational Wave Denoising
Gravitational wave strain from LIGO detectors (H1 + L1).

### Toy — Sinusoidal Signal + White Noise
Synthetic dataset for denoising benchmarks: `s(t) = A·sin(2πft + φ)` buried in white noise.
See [`data/toy/sinusoidal_signal_white_noise/README.md`](data/toy/sinusoidal_signal_white_noise/README.md).

---

## Setup

```bash
conda env create -f env.yaml
conda activate tsenv
pip install -e .
```

> Local dev env on this machine: `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm` (Python 3.10, torch + lightning + h5py + scipy + huggingface_hub).

### Project 8 dataset

Download the ~50 GB HDF5 split from Hugging Face into `data/Project8/{train,val,test}/`:

```bash
bash tools/download_project8.sh
```

Then exercise the foundation-model linear probe:

```bash
python benchmarks/foundation/run_project8.py --models granite_ttm --smoke_test
```

---

## Structure

```
src/
  models/
    s4d.py              # S4D kernel + S4Model (sequence classifier)      [Apache 2.0]
    s4d_seq2seq.py      # S4ModelSeq2Seq (denoising, no pooling)
    linoss.py           # LinOSS (JAX/equinox) — IM and IMEX discretizations
    mlp.py              # MLPRegressor (flatten → FC)
    mlp_denoiser.py     # MLP-based denoiser
    conv_ae.py          # Convolutional autoencoder
    conv_attn_ae.py     # Conv autoencoder + attention
    rnn_seq2seq.py      # RNN seq2seq denoiser
    classical_filter.py # Butterworth / matched-filter baselines
    utils/jax/
      wrapper.py        # JAXLightningModule — bridges equinox models into Lightning
      training.py       # Pure-JAX train step (vmap + grad)
  tasks/
    toy/
      toy_denoising.py      # DenoisingMSE (PyTorch)
      toy_regression.py     # RegressionMSE (PyTorch)
      toy_regression_jax.py # LinOSSToyRegression (JAX via JAXLightningModule)
    TIDMAD/
      tidmad_denoising.py   # Denoising task: Main task for TIDMAD
      tidmad_regression.py  # RegressionMSE for TIDMAD
    LIGO/
      denoising.py          # DenoisingMSE for LIGO
      classification.py     # classification task for LIGO
configs/
  toy/         # per-model YAML configs (LightningCLI)
  TIDMAD/
  LIGO/
benchmarks/
  toy/         # SLURM pipeline: denoiser → regressors → eval
  TIDMAD/      # SLURM pipeline: denoiser → regressors → eval
  foundation/  # Foundation-model benchmark (7 models, 3 tasks)
data/
  LIGO/sample_dataset/
  toy/sinusoidal_signal_white_noise/
  TIDMAD/
    tidmad_dataset.py      # Dataset reader for TIDMAD (PyTorch)
tools/
  save_checkpoint.py
main.py          # LightningCLI entry point
pyproject.toml   # uv-compatible; optional [jax] and [cu12] extras
```

---

## Setup

### Core (PyTorch models, all domains)

Requires Python ≥ 3.12.  Use [uv](https://github.com/astral-sh/uv):

```bash
uv sync
source .venv/bin/activate   # or: uv run python main.py ...
```

To install in editable mode inside an existing environment:

```bash
pip install -e .
```

### JAX models (LinOSS)

LinOSS depends on JAX + Equinox, which are not in the default dependency set to keep
the PyTorch environment clean. Install the optional `jax` extras:

```bash
# CPU
uv sync --extra jax

# GPU (CUDA 12)
uv sync --extra jax --extra cu12
```

Verify JAX sees the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

The `uv.lock` file pins all transitive dependencies. After adding a new package to
`pyproject.toml`, run `uv lock` to update the lockfile and commit both files.

### Foundation-model benchmark (separate conda env)

The seven foundation-model wrappers (MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI,
Lag-Llama, Granite TTM) require Python 3.10 and have conflicting version requirements
that are incompatible with the main env. Use a dedicated conda environment:

```bash
conda create -p envs/fm python=3.10 -y
envs/fm/bin/pip install -r benchmarks/foundation/requirements_fm.txt

# Lag-Llama (source install + legacy setuptools)
envs/fm/bin/pip install \
    'git+https://github.com/time-series-foundation-models/lag-llama.git#egg=lag-llama'
envs/fm/bin/pip install 'setuptools<72'

# Download Lag-Llama checkpoint
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt \
    --local-dir checkpoints/lag-llama/
```

---

## Training (individual runs)

All training goes through `main.py` (LightningCLI). Pick any config and run:

(Make sure to activate the python environment or prefix with `uv run` as described in the setup section.)

```bash
python main.py fit --config <path/to/config.yaml>
```

Override any config value on the command line:

```bash
python main.py fit --config configs/toy/train_toy_mlp_regression_raw.yaml \
  --model.init_args.lr 5e-4 \
  --data.init_args.batch_size 128
```

### Example configs

| Model | Domain | Config |
|-------|--------|--------|
| S4D denoiser | Toy | `configs/toy/train_toy_s4d_denoising.yaml` |
| MLP regressor (raw) | Toy | `configs/toy/train_toy_mlp_regression_raw.yaml` |
| MLP regressor (clean) | Toy | `configs/toy/train_toy_mlp_regression_clean.yaml` |
| LinOSS regressor (raw) | Toy | `configs/toy/train_toy_linoss_regression_raw.yaml` |
| Conv-AE denoiser (MSE) | Toy | `configs/toy/train_toy_conv_ae_denoising_mse.yaml` |
| RNN seq2seq denoiser | Toy | `configs/toy/train_toy_rnn_denoising_mse.yaml` |
| S4D denoiser | TIDMAD | `configs/TIDMAD/train_tidmad_s4d_denoising.yaml` |
| MLP regressor | TIDMAD | `configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml` |
| LIGO denoiser | LIGO | `configs/LIGO/train_LIGO.yaml` |

---

## LinOSS (JAX) — contributor notes

LinOSS is implemented in JAX + Equinox ([paper](https://openreview.net/pdf?id=GRMfXcAAFh)).
The model lives in [src/models/linoss.py](src/models/linoss.py); the Lightning integration
lives in [src/models/utils/jax/wrapper.py](src/models/utils/jax/wrapper.py).

**Adding a new JAX task:**

1. Subclass `JAXLightningModule` (see [src/tasks/toy/toy_regression_jax.py](src/tasks/toy/toy_regression_jax.py)).
2. Implement `_prepare_batch(batch) -> (x, y)` to extract inputs and targets.
3. Pass an `optax` optimizer and a `loss_fn: (y_hat, y) -> scalar` to `super().__init__`.
4. Write a YAML config pointing `model.class_path` at your new task class and
   `model.init_args.model.class_path` at the equinox model.

**Adding a new equinox model:**

1. Implement it as an `eqx.Module` with `__call__(self, x, state, key)` returning
   `(output, state)` — BatchNorm state must pass through.
2. Register it under `src/models/` and export from [src/models/__init__.py](src/models/__init__.py).
3. Activate the JAX extras before running: `uv sync --extra jax` (or `--extra cu12`).

---

## Foundation-model benchmark — contributor notes

The benchmark lives in [benchmarks/foundation/](benchmarks/foundation/) and evaluates
pretrained time-series foundation models on the Toy dataset across three tasks:
**forecasting**, **denoising** (via downstream regressor), and **embedding regression**.

Supported models: MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI, Lag-Llama, Granite TTM.

### Running the benchmark

Activate the foundation env, then:

```bash
# Zero-shot forecasting (smoke test: 2 batches only)
python benchmarks/foundation/run_benchmark.py \
    --models moment \
    --tasks  forecasting \
    --mode   zero_shot \
    --smoke_test

# Full zero-shot run, multiple models
python benchmarks/foundation/run_benchmark.py \
    --data_dir data/toy/sinusoidal_signal_white_noise \
    --out_dir  plots/toy/foundation \
    --models   moment chronos timesfm moirai \
    --tasks    forecasting denoising embedding \
    --mode     zero_shot \
    --baseline_regressor_raw_ckpt   checkpoints/toy_mlp_regression_raw/best.ckpt \
    --baseline_regressor_raw_cfg    configs/toy/train_toy_mlp_regression_raw.yaml \
    --baseline_regressor_clean_ckpt checkpoints/toy_mlp_regression_clean/best.ckpt \
    --baseline_regressor_clean_cfg  configs/toy/train_toy_mlp_regression_clean.yaml

# Lag-Llama (requires checkpoint path)
python benchmarks/foundation/run_benchmark.py \
    --models lagllama \
    --tasks  forecasting \
    --mode   zero_shot \
    --lagllama_ckpt checkpoints/lag-llama/lag-llama.ckpt
```

Results are written to `--out_dir/summary.csv`. The file is append-only so that
concurrent jobs do not overwrite each other; delete it manually to start fresh.

### Adding a new foundation model

1. Create `benchmarks/foundation/wrappers/<name>_wrapper.py` subclassing
   [`BaseWrapper`](benchmarks/foundation/wrappers/base.py).
2. Implement `load(...)`, `forecast(...)`, `embed(...)` (optional), and `unload()`.
   Set `self.name` and `self.supports_embed`.
3. Register the model name in `_get_wrapper()` in
   [`run_benchmark.py`](benchmarks/foundation/run_benchmark.py).
4. Add its pip dependency to
   [`benchmarks/foundation/requirements_fm.txt`](benchmarks/foundation/requirements_fm.txt).

---

## Toy benchmark

```bash
bash benchmarks/toy/run.sh
```

Submits 4 chained SLURM jobs and compares three regression pipelines:

```
[1] train denoiser ──┬──> [2] train reg_raw   ──┬──> [4] eval
                     └──> [3] train reg_clean ──┘
```

| Pipeline | Training input | Eval input | Purpose |
|----------|---------------|------------|---------|
| **raw** | `sig_bkg` (noisy) | `sig_bkg` | baseline |
| **denoised** | `sig` (clean) | `denoiser(sig_bkg)` | benefit of denoising |
| **oracle** | `sig` (clean) | `sig` (clean) | upper bound |

Results saved to `benchmarks/toy/regression_benchmark.png`.

---

## TIDMAD benchmark

To obtain a denoising score for TIDMAD data, run inference over the validation data to produce denoised time series. 
The denoised time series should be saved in the equivalent .h5 file format as the validation data.
Run `tidmad_denoising.py` over denoised data files. 
### Denoising Score — Usage
This script evaluates denoised time-series data stored in denoised HDF5 (`.h5`) files. 
For formulation of the denoising score, please see [TIDAMD NeurIPS Paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/121748).
Input files must contain one-second, chunked time series (`time_series_ch1`, `time_series_ch2`) and a corresponding `signal_frequency` dataset with precomputed peak frequencies, along with attributes such as `sample_rate_hz` and `n_chunks` all contained in the original validation file.
To run the benchmark on denoised data, place the `.h5` files in a directory and execute the script via the command line: `python script.py --data_dir <path_to_h5_files> --output_dir <path_to_save_results>`. 
Optionally, specify particular files using `--files file1.h5 file2.h5`, enable multiprocessing with `--parallel --num_workers <N>`, or run a coarse approximation using `--coarse` (processes every 10th chunk). 
The script computes the power spectral density (PSD) per chunk, evaluates signal-to-noise ratios (SNR) at the provided peak frequencies for both channels, and aggregates these into a normalized, log-scaled denoising score. 
Results are written to `benchmark_results.csv` in the output directory, including the final score and per-chunk SNR values.


## Task architecture

Tasks are organized **by domain**, not by model. Each `LightningModule` (or
`JAXLightningModule`) accepts any compatible model as a constructor argument —
swapping models only requires changing the YAML config.

---

## Models

| Model | Class | Backend | Description |
|-------|-------|---------|-------------|
| S4D | `models.s4d.S4Model` | PyTorch | Diagonal SSM — sequence classifier |
| S4D Seq2Seq | `models.s4d_seq2seq.S4ModelSeq2Seq` | PyTorch | Diagonal SSM — sequence-to-sequence |
| LinOSS | `models.linoss.LinOSS` | JAX/equinox | Linear operator SSM — IM and IMEX |
| MLP | `models.mlp.MLPRegressor` | PyTorch | Flatten → FC layers |
| MLP Denoiser | `models.mlp_denoiser.MLPDenoiser` | PyTorch | MLP for denoising |
| Conv-AE | `models.conv_ae.ConvAE` | PyTorch | Convolutional autoencoder |
| Conv-Attn-AE | `models.conv_attn_ae.ConvAttnAE` | PyTorch | Conv-AE + attention |
| RNN Seq2Seq | `models.rnn_seq2seq.RNNSeq2Seq` | PyTorch | RNN encoder-decoder |
| Classical filter | `models.classical_filter` | NumPy | Butterworth / matched-filter |

---

## Acknowledgements

S4D implementation derived from [state-spaces/s4](https://github.com/state-spaces/s4) (Apache 2.0).
See [NOTICE](NOTICE) for full attribution.
LIGO dataset generated with [GWDatasetGeneration](https://github.com/ML4GW).
LinOSS implementation adapted from [tk-rusch/linoss](https://github.com/tk-rusch/linoss/tree/main).
