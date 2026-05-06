# TIDMAD Data

This directory holds the TIDMAD dataset reader. For data, please see the [PhyTS Hugging face](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench).

## Directory layout

```
data/TIDMAD/
  tidmad_dataset.py
  download_tidmad.py
  README.md
```

## About the dataset

**TIDMAD** (Time-Domain Dark Matter Detection) is a 689 GB dataset from the ABRACADABRA
experiment, recording SQUID magnetometer voltage during an axion dark matter search.

- **Paper:** https://arxiv.org/abs/2406.04378 (NeurIPS 2025 Spotlight)
- **Original Repository:** https://github.com/TIDMAD/TIDMAD
- **PhyTS HuggingFace:** https://huggingface.co/datasets/PhyTS-team/PhyTS-bench/TIDMAD

There are 20 training files and 20 validation files.
Each training/validation H5 file has 200 examples. 
The files also contain metadata including number of examples (chunks), example time sereis length (chunk length), and sampling rate.  
Within each example there are two time series channels sampled at 10 MHz, int8, 1×10^6 samples (1 s):
- `time_series_ch1` — SQUID readout (noisy input)
- `time_series_ch2` — injected reference signal (clean ground truth)
Each example also contains metadata including the signal frequency for the hardware injected signal
- `signal_frequency` — int8 hardware injected signal frequency.

## Downloading the dataset

To download the dataset from hugging face, use the provided `download_tidmad.py` script.
Required dependancy is hugging-hub:
```
pip install huggingface-hub
```
Usage of `download_tidmad.py` includes a few option:
```
# Download all splits (default)
python tidmad_downloader.py

# Download only training and validation
python tidmad_downloader.py --splits training validation

# Download only test set
python tidmad_downloader.py --splits test

# Download all with verification
python tidmad_downloader.py --verify
```
The directory layout after downloading will be:
```angular2html
tidmad_data/
├── training/
│   ├── tidmad_training_0000.h5
│   ├── ...
│   └── tidmad_training_0019.h5
├── validation/
│   ├── tidmad_validation_0000.h5
│   ├── ...
│   └── tidmad_validation_0019.h5
└── test/
    ├── tidmad_test_file.h5
    └── ...
```
