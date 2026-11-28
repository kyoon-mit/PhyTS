# TIDMAD benchmark

Denoising of the ABRACADABRA SQUID readout: a model maps the noisy channel to the injected axion-like signal. Metric: the TIDMAD denoising score ([arXiv:2406.04378](https://arxiv.org/abs/2406.04378)); higher is better.

## Run

Download and preprocess the data (see [data/TIDMAD](../../data/TIDMAD/README.md)), then:

```bash
bash benchmarks/TIDMAD/run.sh
```

| Step | Command | Output |
|---|---|---|
| Train, test | `main.py fit` / `main.py test` | `checkpoints/tidmad_<model>_denoising/best.ckpt` |
| Denoise the test file | `denoise.py --model <model>` | `results/TIDMAD/<model>/tidmad_test_file.h5` |
| Score | `score.py` | TIDMAD column of Table 2 |

## Models

| Model | `--model` | Config | Environment |
|---|---|---|---|
| No denoising (baseline) | `raw` | – | `make env` |
| ConvAE | `conv` | `configs/TIDMAD/train_tidmad_conv_denoising.yaml` | `make env` |
| S4D | `s4d` | `configs/TIDMAD/train_tidmad_s4d_denoising.yaml` | `make env` |
| LinOSS | `linoss` | `configs/TIDMAD/train_tidmad_linoss_denoising.yaml` | `make env-jax` |
| MOMENT-base | `moment` | zero-shot | `make env-fm` |
| Chronos-tiny | `chronos` | zero-shot | `make env-fm` |
| Granite-TTM | `granite_ttm` | zero-shot | `make env-fm` |
| MOIRAI-base | `moirai` | zero-shot | `make env-fm` |

Foundation-model outputs go to `results/TIDMAD/fm_<model>/`.

## Protocol

- **Supervised models:** train on 100,000-sample windows with a spectral (PSD) loss and early stopping on validation loss.
- **Test:** the Hub test file is denoised window by window and scored on whole one-second chunks.
- **Foundation models:** zero-shot on short chunks. MOMENT uses its reconstruction head; the others forecast each half of a chunk from the other half.
