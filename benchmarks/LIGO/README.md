# LIGO benchmark

Chirp-mass estimation for binary-neutron-star mergers from two-detector (H1, L1) strain. Metrics: RMSE in solar masses and R² on the test split.

## Run

Download and extract the data (see [data/LIGO](../../data/LIGO/README.md)), then:

```bash
bash benchmarks/LIGO/run.sh
```

| Step | Command | Output |
|---|---|---|
| Train, test | `main.py fit` / `main.py test` | `results/LIGO/<model>_gaussnll_regression/test_predictions.csv` |
| Foundation models | `fm_zeroshot_ligo.py --model <model>` | `results/LIGO/fm_<model>/test_predictions.csv` |
| Score | `score.py` | LIGO columns of Table 2 |

## Models

| Model | Config or `--model` | Environment |
|---|---|---|
| S4D | `configs/LIGO/train_ligo_s4d_gaussnll_regression.yaml` | `make env` |
| LinOSS | `configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml` | `make env-jax` |
| CNN (ResNet1D) | `configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml` | `make env` |
| MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI, Granite-TTM (base) | `moment`, `chronos`, `timesfm`, `timemoe`, `moirai`, `granite_ttm` | `make env-fm` |

## Protocol

- **Input:** the 4 s of strain before the merger from both detectors, with a small time shift (random in training, fixed in validation and test).
- **Supervised models:** predict the mean and variance of the chirp mass with a Gaussian negative log-likelihood; test uses the best-validation checkpoint.
- **Foundation models:** zero-shot. Frozen embeddings of 512-sample chunks are averaged per detector, and a ridge probe fitted on the validation split is scored on the test split.
- **Mean baseline:** predicts the mean chirp mass.
