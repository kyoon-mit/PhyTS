# LIGO BNS Dataset — SNR 20–30, Uniform, 512 Hz, 10K

Gravitational-wave dataset of simulated Binary Neutron Star (BNS) signals injected into LIGO detector noise.

## Downloading Files

```bash
uv run python data/LIGO/download.py
# or to a custom location:
uv run python data/LIGO/download.py --dest /fast/barmstrong/LIGO
```

The script downloads sharded H5 files from `PhyTS-team/PhyTS-bench` on HuggingFace,
merges each split into a single file, and symlinks `data/LIGO/sample_dataset/` into
the repo. Requires `huggingface_hub` and `h5py` (both in the uv environment).

| File | Samples | Size |
|------|---------|------|
| `train.h5` | 40 000 | ~30 GB |
| `val.h5` | ~10 000 | ~8 GB |
| `test.h5` | ~10 000 | ~8 GB |

Train/val use SNR 45–50 (`bns_snr_45_50`); test uses SNR 5–50 (`snr_5_50`, out-of-distribution).

## Data Format

Each HDF5 file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `whitened_injected` | (N, 2, 16384) | Whitened strain: signal + noise, both detectors |
| `whitened_signal` | (N, 2, 16384) | Whitened signal only (no noise) |
| `whitened_bkg` | (N, 2, 16384) | Whitened noise only (no signal) |
| `snr` | (N,) | Network SNR |
| `mass_1`, `mass_2` | (N,) | Component masses [M☉] |
| `chirp_mass` | (N,) | Chirp mass [M☉] |
| `mass_ratio` | (N,) | mass_2 / mass_1 |
| `distance` | (N,) | Luminosity distance [Mpc] |
| `inclination` | (N,) | Inclination angle [rad] |
| `s1z`, `s2z`, `a_1`, `a_2` | (N,) | Spin parameters |
| `phic`, `phi`, `psi`, `dec` | (N,) | Phase and sky location |
| `tilt_1`, `tilt_2`, `phi_12`, `phi_jl` | (N,) | Precession angles |

- Channel axis: index 0 = H1 (Hanford), index 1 = L1 (Livingston)
- Time axis: 16384 samples = 32 s at 512 Hz
- Coalescence is 1 s from the right edge of the window

## Waveform Parameters

| Parameter | Distribution |
|-----------|-------------|
| mass_1 | Triangular(1.0, 2.5, peak=2.5) M☉ |
| mass_2 | Uniform(1.0, mass_1) M☉ |
| s1z, s2z | 0 (non-spinning) |
| distance | PowerLaw(100, 1000, α=2) Mpc |
| inclination | Sine |
| phic | 0 (fixed) |
| SNR (reweighted) | Uniform(20, 30) |

## Signal Generation

- **Approximant:** IMRPhenomPv2 (default for BNS in ml4gw)
- **Sample rate:** 512 Hz
- **Duration:** 64 s
- **f_min / f_max / f_ref:** 20 / 1024 / 50 Hz
- **Detectors:** H1, L1
- **Whitening:** FFT length 2 s, PSD estimated over 64 s, median averaging
