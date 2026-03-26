# TimeSeriesPhysics

Deep learning for physics time series across four experimental domains. Built with PyTorch Lightning and configurable via YAML CLI.

## Domains

### TIDMAD — Dark Matter Direct Detection
Time series from dark matter axion detection experiments.

### Project 8 — Neutrino Mass Spectroscopy
Cyclotron Radiation Emission Spectroscopy (CRES) signals for tritium beta-decay spectroscopy.

### TESS — Exoplanet Transit Photometry
Stellar light curves from the Transiting Exoplanet Survey Satellite.

### LIGO — Gravitational Wave Parameter Estimation
Gravitational Wave signals from LIGO detectors.

---

## Setup

```bash
mamba env create -f env.yaml
conda activate tsenv
pip install -e .
```

---

## Structure

```
src/
  models/
  dataloader/     # LightningDataModules per domain
  tasks/          # LightningModules (loss + optimizer) per domain
  functions/      # Utility functions
configs/          # PyTorch Lightning config files
data/             # Dataset storage folder
main.py           # Contains LightningCLI
```

## Training

Examples are shown below for LIGO.
```bash
python main.py fit --config configs/train_LIGO.yaml
```

Override config values from the command line:

```bash
python main.py fit --config configs/train_LIGO.yaml \
  --model.init_args.d_model 512 \
  --data.init_args.train_batch_size 128
```

## Models

| Model | Class | Description |
|-------|-------|-------------|
| S4D | `models.s4d.S4Model` | Diagonal state-space sequence model |

## Tasks

| Domain | Task | Class |
|--------|------|-------|
| LIGO | Regression (MSE) | `tasks.LIGO_regression.S4DMSELoss` |

## Acknowledgements

S4D implementation derived from [state-spaces/s4](https://github.com/state-spaces/s4) (Apache 2.0).
LIGO dataset generated with [GWDatasetGeneration](https://github.com/ML4GW).
