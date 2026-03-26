# LIGO BNS Dataset — SNR 20–30, Uniform, 512 Hz, 10K

Gravitational-wave dataset of simulated Binary Neutron Star (BNS) signals injected into LIGO detector noise.

## (TODO) Downloading Files
(TODO) Create script to download data from cloud.

| File | Samples | Size |
|------|---------|------|
| `train.h5` | 8 000 | 5.9 GB |
| `val.h5` | 1 000 | 751 MB |
| `test.h5` | 1 000 | 751 MB |

**Total: 10 000 samples** (80 / 10 / 10 split)

## Data Format

Each HDF5 file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `injected_data` | (N, 2, 32768) | Whitened strain: signal + noise, both detectors |
| `signal_only_data` | (N, 2, 32768) | Whitened signal only (no noise) |
| `bkg_only_data` | (N, 2, 32768) | Whitened noise only (no signal) |
| `snr` | (N,) | Network SNR after reweighting |
| `mass_1`, `mass_2` | (N,) | Component masses [M☉] |
| `chirp_mass` | (N,) | Chirp mass [M☉] |
| `mass_ratio` | (N,) | mass_2 / mass_1 |
| `distance` | (N,) | Luminosity distance [Mpc] |
| `inclination` | (N,) | Inclination angle [rad] |
| `s1z`, `s2z`, `chi1`, `chi2` | (N,) | Spin parameters |
| `phic`, `phi`, `psi`, `dec` | (N,) | Phase and sky location |

- Channel axis: index 0 = H1 (Hanford), index 1 = L1 (Livingston)
- Time axis: 32768 samples = 64 s at 512 Hz
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
