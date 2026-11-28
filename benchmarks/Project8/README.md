# Project 8 benchmark

Regression of the electron kinetic energy (in eV) from the I/Q time series of a simulated CRES event, the quantity behind the tritium endpoint fit that constrains the neutrino mass. Metrics: RMSE in eV and R² on the test split.

## Run

Download the data (see [data/Project8](../../data/Project8/README.md)), then:

```bash
bash benchmarks/Project8/run.sh
```

| Step | Command | Output |
|---|---|---|
| Train, test | `main.py fit` / `main.py test` | `results/Project8/<model>_gaussnll_regression/test_predictions.csv` |
| Foundation models | `fm_zeroshot_project8.py --model <model>` | `results/Project8/fm_<model>/test_predictions.csv` |
| Score | `score.py` | Project 8 columns of Table 2 |

## Models

| Model | Config or `--model` | Environment |
|---|---|---|
| S4D | `configs/Project8/train_project8_s4d_gaussnll_regression.yaml` | `make env` |
| LinOSS | `configs/Project8/train_project8_linoss_gaussnll_regression.yaml` | `make env-jax` |
| CNN | `configs/Project8/train_project8_conv_gaussnll_regression.yaml` | `make env` |
| MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI, Granite-TTM (base) | `moment`, `chronos`, `timesfm`, `timemoe`, `moirai`, `granite_ttm` | `make env-fm` |

## Protocol

- **Input:** signal plus cavity noise for I and Q, each channel normalized; the energy is z-scored with training statistics.
- **Supervised models:** predict the mean and variance of the energy with a Gaussian negative log-likelihood; early stopping on validation loss.
- **Foundation models:** zero-shot. Frozen embeddings of overlapping 512-sample windows are averaged per channel, and a ridge probe is fitted on the training split.
- **Mean baseline:** predicts the training-split mean energy.
