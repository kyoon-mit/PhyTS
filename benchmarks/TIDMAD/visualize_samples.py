"""
Visualize ch1 (noisy SQUID), ch2 (clean reference), and denoised ch1 samples.
Injection frequency detected per-window from ch2 FFT (correct approach).

Usage:
    cd /path/to/TimeSeriesPhysics
    PYTHONPATH=src python benchmarks/TIDMAD/visualize_samples.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)

from run_inference import get_injection_freq, load_conv_denoiser
import torch
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = 'data/TIDMAD/preprocessed'
CKPT_CONV  = 'checkpoints/tidmad_conv_l_psd/best.ckpt'
CFG_CONV   = 'configs/TIDMAD/train_tidmad_conv_l_denoising.yaml'
OUT_DIR    = 'benchmarks/TIDMAD/results/visualizations'
FS         = 10_000_000
N_SAMPLES  = 6
SEED       = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
scale  = float(np.load(f'{DATA_DIR}/scale.npy'))
ch1    = np.load(f'{DATA_DIR}/val_ch1.npy', mmap_mode='r')
ch2    = np.load(f'{DATA_DIR}/val_ch2.npy', mmap_mode='r')

rng     = np.random.default_rng(SEED)
indices = rng.choice(len(ch1), size=N_SAMPLES, replace=False)
indices.sort()

t     = np.arange(ch1.shape[1]) / FS * 1e3   # ms
freqs = np.fft.rfftfreq(ch1.shape[1], d=1.0/FS) / 1e3  # kHz

def compute_psd(x):
    dt = 1.0 / FS
    N  = len(x)
    return dt / N * np.abs(np.fft.rfft(x))[1:] ** 2, freqs[1:]

# ---------------------------------------------------------------------------
# Load ConvAE-L (PSD) denoiser — best trained model
# ---------------------------------------------------------------------------
print('Loading ConvAE-L (PSD) denoiser...')
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
denoise_fn = load_conv_denoiser(CKPT_CONV, CFG_CONV, device)

# Precompute denoised outputs for selected windows
x1_batch = np.stack([ch1[i].astype(np.float32) * scale for i in indices])  # (N, L)
x1_den   = denoise_fn(x1_batch)  # (N, L)

# Detect injection frequency per window from ch2 (correct approach)
x2_batch   = np.stack([ch2[i].astype(np.float32) * scale for i in indices])
freq_kHz_list = [get_injection_freq(x2_batch[r], 1.0) / 1e3 for r in range(N_SAMPLES)]

print(f'Detected injection frequencies (kHz): {[f"{f:.1f}" for f in freq_kHz_list]}')

# ---------------------------------------------------------------------------
# Plot 1: time domain — noisy ch1 vs denoised ch1 vs clean ch2
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(18, 3 * N_SAMPLES))
fig.suptitle('TIDMAD samples — time domain', fontsize=14, fontweight='bold')

for row in range(N_SAMPLES):
    freq_kHz = freq_kHz_list[row]
    axes[row, 0].plot(t, x1_batch[row], lw=0.3, color='steelblue')
    axes[row, 0].set_title(f'ch1 noisy  |  inj={freq_kHz:.1f} kHz', fontsize=9)
    axes[row, 0].set_ylabel('mV')

    axes[row, 1].plot(t, x1_den[row], lw=0.3, color='seagreen')
    axes[row, 1].set_title(f'ch1 denoised (ConvAE-L PSD)  |  inj={freq_kHz:.1f} kHz', fontsize=9)
    axes[row, 1].set_ylabel('mV')

    axes[row, 2].plot(t, x2_batch[row], lw=0.3, color='darkorange')
    axes[row, 2].set_title(f'ch2 clean ref  |  inj={freq_kHz:.1f} kHz', fontsize=9)
    axes[row, 2].set_ylabel('mV')

for ax in axes[-1]:
    ax.set_xlabel('Time (ms)')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'samples_time_domain.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ---------------------------------------------------------------------------
# Plot 2: PSD — noisy ch1 vs denoised ch1 vs clean ch2 (zoomed ±5 kHz)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(18, 3 * N_SAMPLES))
fig.suptitle('TIDMAD samples — PSD (zoomed ±5 kHz around injection)', fontsize=14, fontweight='bold')

for row in range(N_SAMPLES):
    freq_kHz = freq_kHz_list[row]
    zoom = 5
    p1, f = compute_psd(x1_batch[row])
    pd, _ = compute_psd(x1_den[row])
    p2, _ = compute_psd(x2_batch[row])
    mask  = (f >= freq_kHz - zoom) & (f <= freq_kHz + zoom)

    for ax, p, label, color in [
        (axes[row, 0], p1, 'ch1 noisy',             'steelblue'),
        (axes[row, 1], pd, 'ch1 denoised',           'seagreen'),
        (axes[row, 2], p2, 'ch2 clean ref',          'darkorange'),
    ]:
        ax.semilogy(f[mask], p[mask], lw=0.8, color=color)
        ax.axvline(freq_kHz, color='red', lw=1, ls='--', label=f'{freq_kHz:.1f} kHz')
        ax.set_title(f'{label}  |  inj={freq_kHz:.1f} kHz', fontsize=9)
        ax.set_ylabel('PSD')
        ax.legend(fontsize=7)

for ax in axes[-1]:
    ax.set_xlabel('Frequency (kHz)')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(OUT_DIR, 'samples_psd.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ---------------------------------------------------------------------------
# Plot 3: single-window full PSD comparison (log scale, full range)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle('Full PSD — single window', fontsize=13, fontweight='bold')

row = 0
freq_kHz = freq_kHz_list[row]
p1, f = compute_psd(x1_batch[row])
pd, _ = compute_psd(x1_den[row])
p2, _ = compute_psd(x2_batch[row])

for ax, p, label, color in [
    (axes[0], p1, 'ch1 noisy',        'steelblue'),
    (axes[1], pd, 'ch1 denoised',     'seagreen'),
    (axes[2], p2, 'ch2 clean ref',    'darkorange'),
]:
    ax.semilogy(f, p, lw=0.5, color=color)
    ax.axvline(freq_kHz, color='red', lw=1.5, ls='--', label=f'inj={freq_kHz:.1f} kHz')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('PSD')
    ax.set_title(label, fontsize=11)
    ax.legend()

plt.tight_layout()
out = os.path.join(OUT_DIR, 'sample_psd_full.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print('\nDone.')
