# TimeSeriesPhysics

Deep learning for physics time series across four experimental domains. Built with PyTorch Lightning and configurable via YAML CLI.

## Domains

### TIDMAD — Dark Matter Direct Detection
Time series from dark matter axion detection experiments.

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

---

## Structure

```
src/
  models/
    s4d.py           # S4D kernel + S4Model (sequence classifier)       [Apache 2.0, derived from state-spaces/s4]
    s4d_seq2seq.py   # S4ModelSeq2Seq (no pooling, for denoising)       [Apache 2.0, derived from s4d.py]
    mlp.py           # MLPRegressor (flatten → FC layers)
  dataloader/
    LIGO_dataloader.py
    tidmad_dataloader.py
    toy_dataloader.py
  tasks/
    toy/
      toy_denoising.py    # DenoisingMSE task for toy dataset
      toy_regression.py   # RegressionMSE task for toy dataset
    TIDMAD/
      tidmad_regression.py # RegressionMSE task for TIDMAD
    LIGO/
      LIGO_denoising.py   # DenoisingMSE task for LIGO
configs/toy/           # PyTorch Lightning YAML configs
configs/TIDMAD/        # PyTorch Lightning YAML configs for TIDMAD
benchmarks/toy/        # Benchmark scripts and results
benchmarks/TIDMAD/     # Benchmark scripts and results for TIDMAD
data/
  LIGO/sample_dataset/
  toy/sinusoidal_signal_white_noise/
  TIDMAD/
    original/          # raw H5 files (symlinks or downloads) — not tracked by git
    preprocessed/      # .npy arrays from preprocess_tidmad.py — not tracked by git
    preprocess_tidmad.py
main.py                # LightningCLI entry point
```

---

## Task architecture

Tasks are organized **by domain**, not by model. Each LightningModule accepts
any compatible `nn.Module` as a constructor argument — swapping models only
requires changing the YAML config.

---

## TIDMAD benchmark: denoising pipeline

**One-time setup** — preprocess raw H5 files into memory-mappable `.npy` arrays
(see [`data/TIDMAD/README.md`](data/TIDMAD/README.md) for how to obtain the H5 files):

```bash
python data/TIDMAD/preprocess_tidmad.py \
    --data_dir data/TIDMAD/original \
    --out_dir  data/TIDMAD/preprocessed
```

Then run the benchmark:

```bash
bash benchmarks/TIDMAD/run.sh
```

---

## Toy benchmark: denoising pipeline

```bash
bash benchmarks/toy/run.sh
```

This submits 4 chained SLURM jobs and compares three regression pipelines:

```
[1] train denoiser ──┬──> [2] train reg_raw   ──┬──> [4] eval
                     └──> [3] train reg_clean ──┘
```

| Pipeline | Training input | Eval input | Purpose |
|----------|---------------|------------|---------|
| **raw** | `sig_bkg` (noisy) | `sig_bkg` | baseline |
| **denoised** | `sig` (clean) | `denoiser(sig_bkg)` | benefit of denoising |
| **oracle** | `sig` (clean) | `sig` (clean) | upper bound |

`reg_clean` is trained on clean signals so it "knows" what clean signals look
like. At eval time, the denoiser approximates `sig` from `sig_bkg`, and that
approximation is fed into `reg_clean`. The gap **oracle − denoised** measures
imperfect denoising; the gap **denoised − raw** measures the benefit of denoising.

Results are saved to `benchmarks/toy/regression_benchmark.png`.

---

## Training (individual steps)

```bash
# Toy denoising
python main.py fit --config configs/toy/train_toy_s4d_denoising.yaml

# Toy regression (MLP, raw noisy input)
python main.py fit --config configs/toy/train_toy_mlp_regression_raw.yaml

# Toy regression (MLP, clean input)
python main.py fit --config configs/toy/train_toy_mlp_regression_clean.yaml
```

Override config values from the command line:

```bash
python main.py fit --config configs/toy/train_toy_mlp_regression_raw.yaml \
  --model.init_args.lr 5e-4 \
  --data.init_args.batch_size 128
```

---

## Models

| Model | Class | Description |
|-------|-------|-------------|
| S4D | `models.s4d.S4Model` | Diagonal SSM — sequence classifier (mean pooling) |
| S4D Seq2Seq | `models.s4d_seq2seq.S4ModelSeq2Seq` | Diagonal SSM — sequence-to-sequence (denoising) |
| MLP | `models.mlp.MLPRegressor` | Flatten → FC layers (~205K params for seq_len=640) |

---

## Tasks

| Domain | Task | Class |
|--------|------|-------|
| Toy | Denoising (MSE) | `tasks.toy.toy_denoising.DenoisingMSE` |
| Toy | Regression (MSE) | `tasks.toy.toy_regression.RegressionMSE` |
| LIGO | Denoising (MSE) | `tasks.LIGO.LIGO_denoising.DenoisingMSE` |

---

## Acknowledgements

S4D implementation derived from [state-spaces/s4](https://github.com/ML4GW) (Apache 2.0).
See [NOTICE](NOTICE) for full attribution.
LIGO dataset generated with [GWDatasetGeneration](https://github.com/ML4GW).
