"""Shared loss modules for sequence denoising.

``MixtureMSESpectralLoss`` is a verbatim port of the loss in the original
neutrino_project ``denoising`` branch
(https://github.com/chreissel/neutrino_project/blob/denoising/src/models/losses.py):
a convex mixture of time-domain MSE and the MSE between the magnitudes of
the real FFT (along the sequence axis ``dim=1``) of prediction and target.
"""

import torch
import torch.nn as nn


class MixtureMSESpectralLoss(nn.Module):
    """Mixture of MSE loss (time domain) and spectral loss (frequency domain).

    The spectral component compares the magnitude of the real FFT of the
    predicted and target sequences along the sequence dimension (``dim=1``).

    Args:
        alpha: Weight for the MSE term. The spectral term is weighted by
        ``(1 - alpha)``. Defaults to ``0.5``.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(inputs, targets)
        inputs_mag = torch.abs(torch.fft.rfft(inputs, dim=1))
        targets_mag = torch.abs(torch.fft.rfft(targets, dim=1))
        spectral_loss = self.mse(inputs_mag, targets_mag)
        return self.alpha * mse_loss + (1.0 - self.alpha) * spectral_loss
