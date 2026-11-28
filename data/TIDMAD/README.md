# TIDMAD data

SQUID magnetometer readout from the ABRACADABRA axion search with a known hardware-injected signal (TIDMAD, [arXiv:2406.04378](https://arxiv.org/abs/2406.04378)). Hosted in `TIDMAD/` of [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench).

## Download

```bash
python data/download.py --domain tidmad            # full dataset (~163 GB)
python data/download.py --domain tidmad --sample   # one file, copied to train, val and test
python data/TIDMAD/preprocess_tidmad.py --data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed
```

## Files

| Split | Hub files (`data/TIDMAD/original/`) | Preprocessed (`data/TIDMAD/preprocessed/`) |
|---|---|---|
| train | `tidmad_training_*.h5` | `train_{ch1,ch2,params}.npy` |
| val | `tidmad_validation_*.h5` | `val_{ch1,ch2,params}.npy` |
| test | `tidmad_test_file.h5` | `test_{ch1,ch2,params}.npy` |

## Format

Each Hub file holds one-second chunks sampled at 10 MHz:

| Dataset | Description |
|---|---|
| `time_series_ch1` | SQUID readout (noisy input) |
| `time_series_ch2` | injected reference signal (clean target) |
| `signal_frequency` | injection frequency of each chunk [Hz] |

Preprocessing cuts the chunks into 100,000-sample windows (`{split}_ch1.npy`, `{split}_ch2.npy`) with per-window parameters (`{split}_params.npy`), and stores the DAQ-to-mV scale in `scale.npy`.
