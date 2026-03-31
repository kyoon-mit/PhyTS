# Sinusoidal Signal + White Noise — Toy Dataset

A synthetic dataset for benchmarking time series denoising models.

---

## Signal Model

```
s(t) = A · sin(2π f t + φ)          (clean signal)
x(t) = s(t) + ε(t)                  (observed signal)
ε(t) ~ N(0, σ²)                     (white / additive Gaussian noise)
```

White noise has a flat power spectral density — each sample is drawn i.i.d. from a Gaussian.

---

## Priors

| Parameter | Symbol | Distribution | Units |
|-----------|--------|-------------|-------|
| Amplitude | A | Uniform(0.5, 1.5) | — |
| Frequency | f | Uniform(1.0, 10.0) | Hz |
| Phase | φ | Uniform(0, 2π) | rad |
| Noise amplitude | σ | Normal(5.0, 1.0), clipped > 0 | — |

**SNR** = RMS(signal) / RMS(noise) = (A / √2) / σ

With typical A ≈ 1, σ ≈ 5, the expected SNR ≈ 0.14 — a challenging low-SNR regime.

---

## Dataset

| Split | Samples |
|-------|--------:|
| train | 80,000 |
| val   | 10,000 |
| test  | 10,000 |

- Sample rate: 64 Hz
- Duration: 10 s → **640 samples** per sequence
- Precision: float32
- Random seed: 42

---

## File Layout

```
sinusoidal_signal_white_noise/
├── generate.py
├── plot.py
├── README.md
├── train.npz    # sig (80000,640), sig_bkg (80000,640), params (80000,5)
├── val.npz      # sig (10000,640), sig_bkg (10000,640), params (10000,5)
└── test.npz     # sig (10000,640), sig_bkg (10000,640), params (10000,5)
```

Each `.npz` contains three float32 arrays: `sig`, `sig_bkg`, `params`.

`params` columns: `[amplitude, frequency_hz, phase_rad, noise_amplitude, snr]`

---

## Usage

**Generate dataset:**
```bash
python data/toy/sinusoidal_signal_white_noise/generate.py
```

**Plot parameter distributions + random sample:**
```bash
python data/toy/sinusoidal_signal_white_noise/plot.py [--split train] [--seed 0]
```

Outputs `param_distributions.png` and `sample.png` in this directory.
