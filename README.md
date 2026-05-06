# PhyTS

PhyTS is a machine learning benchmark built around four precision physics experiments: LIGO (gravitational waves), ABRACADABRA (axion dark matter), TESS (stellar variability), and Project 8 (neutrino mass). Each dataset is drawn from real or high-fidelity simulated detector data and defines a concrete inference task—chirp-mass regression, time-series denoising, variability classification, or electron energy regression—where better model performance maps directly to improved scientific sensitivity.

The datasets span twelve orders of magnitude in sampling rate, sequence lengths from hundreds to millions of samples, and noise backgrounds that are non-Gaussian, non-stationary, and shaped by detector hardware. They are a poor fit for standard time-series benchmarks in all of these respects.

**Paper:** PhyTS: A Benchmark for Scientific Time Series (NeurIPS 2026)  
**Data:** [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench) on Hugging Face

---

## Datasets

| Experiment | Domain | Task | Sampling rate | Seq. length | SNR |
|-----------|--------|------|:-------------:|:-----------:|:---:|
| LIGO | Gravitational waves | Chirp-mass regression | 256 Hz | 1,024 (4 s) | 5–50 |
| ABRACADABRA | Axion dark matter | Time-series denoising | 10 MHz | 100,000 (1 s) | 0.02–200 |
| TESS | Stellar variability | 8-class classification | 0.56–1.67 mHz | 1,300–3,670 | 0.3–140 |
| Project 8 | Neutrino mass | Energy regression | 403 MHz | 24,576 (61 μs) | 3–25 |

Detailed dataset descriptions and preprocessing steps are in the paper (Section 3).

---

## Setup

Three environments are needed depending on which models you run.
All main environments are managed by [uv](https://github.com/astral-sh/uv).

```bash
# PyTorch models (S4D, CNN, RNN, MLP, Conv-AE) — covers all four datasets
make env
source .venv/bin/activate

# LinOSS (JAX/Equinox) — same environment with JAX + CUDA 12 added
make env-jax
source .venv/bin/activate

# Foundation models (MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI, Granite TTM)
# Uses a separate Python 3.10 environment due to conflicting dependencies
make env-fm
source benchmarks/foundation/.venv/bin/activate
```

CUDA 13 (driver ≥ 580): replace `env-jax` with `uv sync --extra jax --extra cu13`.

---

## Data

Download each dataset from [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench) on Hugging Face and place it as follows:

| Dataset | Location |
|---------|----------|
| LIGO | `data/LIGO/{train,val,test}/sig_combined_{split}.h5` |
| ABRACADABRA | Raw HDF5 → `python data/TIDMAD/preprocess_tidmad.py` (see `data/TIDMAD/README.md`) |
| TESS | `python data/TESS/download_tess.py` → `data/TESS/tess_classification.parquet` |
| Project 8 | `data/Project8/{train,valid,test}/` |

---

## Training

All training goes through `main.py` (LightningCLI). Pick any config from `configs/`:

```bash
python main.py fit --config <path/to/config.yaml>
```

One example per domain:

```bash
# LIGO — chirp-mass regression, S4D
python main.py fit --config configs/LIGO/train_ligo_s4d_gaussnll_regression.yaml

# LIGO — chirp-mass regression, 1D CNN
python main.py fit --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml

# ABRACADABRA — denoising, LinOSS  (requires env-jax)
python main.py fit --config configs/TIDMAD/train_tidmad_linoss_denoising.yaml

# TESS — variability classification, S4D
python main.py fit --config configs/TESS/train_tess_s4d_classification.yaml

# Project 8 — energy regression, S4D
python main.py fit --config configs/Project8/train_project8_s4d_regression_energy_gaussiannll.yaml
```

Any config value can be overridden on the command line:

```bash
python main.py fit --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml \
  --model.init_args.lr 1e-4 \
  --data.init_args.batch_size 64
```

The full set of configs is in `configs/`.

---

## Benchmarks

Each domain has a pipeline script that trains all models and runs evaluation end-to-end:

```bash
bash benchmarks/LIGO/run.sh
bash benchmarks/TIDMAD/run.sh
bash benchmarks/TESS/run.sh
bash benchmarks/Project8/run.sh
```

For foundation models (zero-shot evaluation):

```bash
source benchmarks/foundation/.venv/bin/activate
python benchmarks/foundation/run_benchmark.py \
  --models moment chronos timesfm moirai granite_ttm \
  --tasks  forecasting denoising embedding \
  --mode   zero_shot
```

ABRACADABRA denoising score — after training, evaluate all variants:

```bash
PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py
```

---

## Results

Numbers from the paper (Table 2). Foundation models evaluated zero-shot.

| Model | LIGO RMSE [M☉] | LIGO R² | TIDMAD score | P8 RMSE [eV] | P8 R² |
|-------|:--------------:|:-------:|:------------:|:------------:|:-----:|
| Mean baseline | 0.271 | 0.000 | 1.00 | 28.83 | 0.000 |
| S4D | **0.254** | **0.125** | — | **15.68** | **0.704** |
| LinOSS | 0.259 | 0.081 | **1.30** | 20.88 | 0.476 |
| CNN | 0.280 | −0.068 | −0.11 | 20.11 | 0.514 |
| MOMENT | 0.284 | −0.096 | 0.46 | 25.22 | 0.236 |
| Chronos | 0.278 | −0.052 | −0.88 | 25.22 | 0.235 |

---

## Repository layout

```
src/
  models/         # S4D, LinOSS, CNN, RNN, MLP, Conv-AE, classical filter
  tasks/          # LightningModules per domain (LIGO, TIDMAD, TESS, Project8)
  dataloader/     # PyTorch DataModules per domain
  functions/      # Loss functions, dropout, learning-rate schedules
configs/          # YAML training configs (one per model × domain × task)
benchmarks/
  LIGO/           # run.sh + chronos scripts
  TIDMAD/         # run.sh + evaluation pipeline
  TESS/           # run.sh + evaluation pipeline
  Project8/       # run.sh + evaluation pipeline
  foundation/     # Foundation-model wrappers and benchmark runner
data/
  LIGO/           # HDF5 strain files (gitignored; download from HuggingFace)
  TIDMAD/         # Raw HDF5 + preprocess_tidmad.py
  TESS/           # Parquet files (gitignored; python data/TESS/download_tess.py)
  Project8/       # HDF5 files (gitignored; download from HuggingFace)
main.py           # LightningCLI entry point
pyproject.toml    # Dependencies (uv)
Makefile          # Environment setup targets
```

---

## Models

| Model | Class | Backend |
|-------|-------|---------|
| S4D | `models.s4d.S4Model` | PyTorch |
| S4D Seq2Seq | `models.s4d_seq2seq.S4ModelSeq2Seq` | PyTorch |
| LinOSS | `models.linoss.LinOSS` | JAX / Equinox |
| 1D CNN | `models.conv1d_regressor.Conv1DRegressor` | PyTorch |
| Conv-AE | `models.conv_ae.ConvAE` | PyTorch |
| Conv-Attn-AE | `models.conv_attn_ae.ConvAttnAE` | PyTorch |
| RNN Seq2Seq | `models.rnn_seq2seq.RNNSeq2Seq` | PyTorch |
| MLP | `models.mlp.MLPRegressor` | PyTorch |
| Classical filter | `models.classical_filter` | NumPy |

Tasks are organized by domain, not by model. Any model can be swapped into any compatible task by changing the `model.class_path` in the YAML config.

---

## Citation

```bibtex
@inproceedings{phyts2026,
  title     = {PhyTS: A Benchmark for Scientific Time Series},
  author    = {...},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## License

Apache 2.0. The S4D implementation is derived from [state-spaces/s4](https://github.com/state-spaces/s4) (Apache 2.0); see [NOTICE](NOTICE) for full attribution. LinOSS adapted from [tk-rusch/linoss](https://github.com/tk-rusch/linoss).
