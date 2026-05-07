# PhyTS

PhyTS is a benchmark suite of precision scientific time series datasets for machine learning, spanning experiments in gravitational-wave detection, dark matter searches, neutrino mass determination, and stellar variability detection. Despite their diverse scientific goals, these domains share a common challenge: recovering weak, structured signals and estimating underlying physical parameters from noise-dominated measurements.

Unlike standard sequence modeling benchmarks such as audio and speech, these data exhibit non-Gaussian and nonstationary noise, long-range temporal correlations, detector-specific systematics, irregular sampling, and signals that are sparse, weak, or only partially modeled. As a result, they provide a challenging testbed for evaluating whether modern AI methods can support downstream scientific inference.

Existing time series benchmarks inadequately prepare models for scientific applications due to three critical limitations: (1) **unrealistic noise assumptions** — most benchmarks use Gaussian noise rather than the complex, frequency-dependent backgrounds found in real detectors; (2) **missing physics constraints** — standard metrics ignore that scientific applications require interpretable confidence estimates and respect for physical laws; and (3) **simplified temporal structure** — scientific signals often exhibit multi-scale dependencies and rare transient events that are poorly represented in current datasets.

PhyTS addresses these gaps across four dimensions:
- **Realistic complexity**: physics domains spanning 12 orders of magnitude in sampling rate with authentic detector noise, non-stationary backgrounds, and scientifically meaningful signal-to-noise ratios.
- **Physics-informed evaluation**: task formulations that reflect real experimental constraints, including parameter inference under uncertainty and noise reduction across orders of magnitude of signal frequency.
- **Systematic baselines**: comprehensive comparisons of supervised models and six zero-shot foundation models, revealing fundamental limitations of current architectures on scientific data.
- **Grounding in scientific measurement**: each predicted value propagates transparently into a downstream physics result, enabling evaluation by scientific impact.

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
# PyTorch models (S4D, CNN, RNN, MLP, Conv-AE) and data download — covers all four datasets
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

All datasets are on Hugging Face at [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench).

For pipeline verification (NeurIPS reproducibility), sample files covering all four domains can be downloaded in one step:

```bash
python data/download.py --sample
```

To download a full dataset:

```bash
python data/download.py --domain tess        # 194 MB
python data/download.py --domain ligo        # ~157 GB
python data/download.py --domain project8    # ~44 GB
python data/download.py --domain tidmad      # ~163 GB; then run:
python data/TIDMAD/preprocess_tidmad.py \
    --data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed
```

See `data/download.py --help` for options.

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

Numbers from the paper (Table 2). Results are written to `results/` after running the benchmark scripts. Foundation models evaluated zero-shot.

| Model | LIGO RMSE [M☉] | LIGO R² | TESS Bal. Acc. | TESS R² | TIDMAD score | P8 RMSE [eV] | P8 R² |
|-------|:--------------:|:-------:|:--------------:|:-------:|:------------:|:------------:|:-----:|
| Mean baseline | 0.271 | 0.000 | 0.125 | −0.017 | 1.00 | 28.83 | 0.000 |
| S4D | **0.254** | **0.125** | **0.887** | **0.665** | — | **15.68** | **0.704** |
| LinOSS | 0.259 | 0.081 | 0.843 | 0.612 | **1.30** | 20.88 | 0.476 |
| CNN | 0.280 | −0.068 | 0.851 | 0.617 | −0.11 | 20.11 | 0.514 |
| MOMENT | 0.284 | −0.096 | 0.828 | 0.263 | 0.46 | 25.22 | 0.236 |
| Chronos | 0.278 | −0.052 | 0.812 | 0.305 | −0.88 | 25.22 | 0.235 |

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
  LIGO/           # run.sh + evaluation pipeline
  TIDMAD/         # run.sh + evaluation pipeline
  TESS/           # run.sh + evaluation pipeline
  Project8/       # run.sh + evaluation pipeline
  foundation/     # Foundation-model wrappers and benchmark runner
data/
  LIGO/           # HDF5 strain files (gitignored; download from HuggingFace)
  TIDMAD/         # Raw HDF5 + preprocess_tidmad.py
  TESS/           # Parquet files (gitignored; bash benchmarks/TESS/setup_data.sh)
  Project8/       # HDF5 files (gitignored; download from HuggingFace)
plots/            # Generated by benchmark scripts (gitignored)
results/          # Generated by benchmark scripts (gitignored)
main.py           # LightningCLI entry point
pyproject.toml    # Dependencies (uv)
Makefile          # Environment setup targets
```

---

## Models

### Supervised (trained per domain)

| Model | Class | Backend |
|-------|-------|---------|
| S4D | `models.s4d.S4Model` | PyTorch |
| LinOSS | `models.linoss.LinOSS` | JAX / Equinox |
| 1D CNN (LIGO) | `models.conv1d_regressor.ResNet1DRegressor` | PyTorch |
| 1D CNN (TESS) | `models.conv.ConvClassifier` | PyTorch |
| Conv-AE (TIDMAD) | `models.conv_ae.ConvAE` | PyTorch |
| 1D CNN (Project 8) | `models.conv_regressor.Conv1DRegressor` | PyTorch |

Tasks are organized by domain; configs live in `configs/<domain>/`. Any model can be swapped into any compatible task by changing `model.class_path` in the YAML.

### Foundation models (zero-shot and fine-tuned)

| Model | Reference |
|-------|-----------|
| MOMENT | `benchmarks/foundation/wrappers/moment_wrapper.py` |
| Chronos | `benchmarks/foundation/wrappers/chronos_wrapper.py` |
| TimesFM | `benchmarks/foundation/wrappers/timesfm_wrapper.py` |
| Time-MoE | `benchmarks/foundation/wrappers/timemoe_wrapper.py` |
| MOIRAI | `benchmarks/foundation/wrappers/moirai_wrapper.py` |
| Granite TTM | `benchmarks/foundation/wrappers/granite_ttm_wrapper.py` |

Foundation models are evaluated zero-shot and with lightweight fine-tuning via `benchmarks/foundation/run_benchmark.py`. Each wrapper exposes a uniform interface (`forecast`, `denoise`, `embed`) so the same evaluators run across all models.

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
