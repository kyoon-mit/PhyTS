"""Classical spectral denoiser (no learnable parameters).

Uses a Wiener-inspired approach: estimate the per-sample noise floor from the
median PSD and apply a gain H(f) = max(PSD - noise, 0) / PSD.

Interface: (B, L, 1) -> (B, L, 1)
"""

import torch
import torch.nn as nn


class ClassicalFilter(nn.Module):
    """Frequency-domain Wiener-like filter.

    For each sample in the batch:
      1. Compute one-sided power spectral density via rfft.
      2. Estimate noise PSD as the median of all frequency bins.
      3. Apply spectral gain H(f) = max(PSD(f) - noise_psd, 0) / PSD(f).
      4. Return to time domain via irfft.
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters

    def forward(self, x):
        # x: (B, L, 1)
        B, L, _ = x.shape
        x_sq = x.squeeze(-1)                          # (B, L)
        X = torch.fft.rfft(x_sq)                      # (B, L//2+1)
        psd = X.abs().pow(2)                           # (B, F)
        noise_psd = psd.median(dim=-1, keepdim=True).values  # (B, 1)
        gain = (psd - noise_psd).clamp(min=0) / (psd + 1e-12)
        y = torch.fft.irfft(X * gain, n=L)            # (B, L)
        return y.unsqueeze(-1)                         # (B, L, 1)
