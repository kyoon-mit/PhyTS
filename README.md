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
  dataloader/
    LIGO_dataloader.py
    toy_dataloader.py
  tasks/
    LIGO_denoising.py   # DenoisingMSE task for LIGO
    toy_denoising.py    # DenoisingMSE task for toy dataset
  functions/
configs/               # PyTorch Lightning YAML configs
data/
  LIGO/sample_dataset/
  toy/sinusoidal_signal_white_noise/
main.py                # LightningCLI entry point
```

---

## Task architecture

Tasks are organized **by domain**, not by model. Each task file exposes a
`DenoisingMSE` LightningModule that accepts any seq2seq `nn.Module` as a
constructor argument. Swapping models only requires changing the YAML config.

```
tasks/toy_denoising.py   DenoisingMSE(model=..., lr=..., lr_decay=...)
tasks/LIGO_denoising.py  DenoisingMSE(model=..., lr=..., lr_decay=...)
```

---

## Training

```bash
# Toy denoising (S4D seq2seq)
python main.py fit --config configs/train_toy.yaml

# LIGO denoising
python main.py fit --config configs/train_LIGO.yaml
```

Override config values from the command line:

```bash
python main.py fit --config configs/train_toy.yaml \
  --model.init_args.model.init_args.d_model 512 \
  --data.init_args.batch_size 128
```

---

## Models

| Model | Class | Description |
|-------|-------|-------------|
| S4D | `models.s4d.S4Model` | Diagonal SSM — sequence classifier (mean pooling) |
| S4D Seq2Seq | `models.s4d_seq2seq.S4ModelSeq2Seq` | Diagonal SSM — sequence-to-sequence (denoising) |

---

## Tasks

| Domain | Task | Class |
|--------|------|-------|
| Toy | Denoising (MSE) | `tasks.toy_denoising.DenoisingMSE` |
| LIGO | Denoising (MSE) | `tasks.LIGO_denoising.DenoisingMSE` |

---

## Acknowledgements

S4D implementation derived from [state-spaces/s4](https://github.com/ML4GW) (Apache 2.0).
See [NOTICE](NOTICE) for full attribution.
LIGO dataset generated with [GWDatasetGeneration](https://github.com/ML4GW).
