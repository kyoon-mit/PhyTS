# TESS benchmark

Stellar variability from single-sector TESS light curves, in two tasks: classification into 8 variability classes (metric: balanced accuracy) and regression of the near-core rotation frequency frot of g-mode pulsators (metric: R²).

## Run

Download the data (see [data/TESS](../../data/TESS/README.md)), then:

```bash
bash benchmarks/TESS/run.sh
```

| Step | Command | Output |
|---|---|---|
| Train, test | `main.py fit` / `main.py test` | `results/TESS/<model>_<task>/test_predictions.csv` |
| Foundation models | `fm_zeroshot_tess.py --model <model>` | `results/TESS/fm_<model>_<task>/test_predictions.csv` |
| Score | `score.py` | TESS columns of Table 2 |

## Models

`<task>` is `classification` or `regression`.

| Model | Config or `--model` | Environment |
|---|---|---|
| CNN | `configs/TESS/train_tess_conv_<task>.yaml` | `make env` |
| S4D | `configs/TESS/train_tess_s4d_<task>.yaml` | `make env` |
| LinOSS | `configs/TESS/train_tess_linoss_<task>.yaml` | `make env-jax` |
| MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI, Granite-TTM (base) | `moment`, `chronos`, `timesfm`, `timemoe`, `moirai`, `granite_ttm` | `make env-fm` |

## Protocol

- **Splits:** the Hub's fixed splits, grouped by Gaia DR3 source.
- **Supervised models:** each light curve is z-scored and cropped or zero-padded to a fixed length; early stopping on validation loss.
- **Foundation models:** zero-shot. Frozen embeddings of two 512-point windows are averaged, and a logistic (class) or ridge (frot) probe is fitted on the training split.
- **Hyperparameters:** Table 2 reports the best runs of a hyperparameter sweep per model (Appendix C), over learning rate 5e-5 to 2e-2, dropout 0 to 0.5, weight decay 1e-6 to 1e-2 and batch size 128 to 512. The configs use one setting in that range.
