"""Shared loss modules for sequence denoising / regression.

``PSDLoss`` and ``MixtureMSEPSDLoss`` operate on tensors with a dedicated
time axis (default ``dim=1``, matching ``(B, L, C)`` batches) and use the
power-spectral-density formulation already used by ``tasks.toy.toy_denoising``
(MSE between ``|rfft|^2`` of prediction and target).

``MixtureMSEPSDLoss`` mirrors ``MixtureMSESpectralLoss`` from the original
neutrino_project ``denoising`` branch
(https://github.com/chreissel/neutrino_project/blob/denoising/src/models/losses.py),
but with the magnitude term replaced by the squared-magnitude (PSD) so that
the spectral component matches ``PSDLoss`` used elsewhere in this repo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSDLoss(nn.Module):
    """MSE loss in the power-spectral-density domain.

    Computes ``MSE(|rfft(y_pred)|^2, |rfft(y_true)|^2)`` along ``dim`` (the
    time axis).  Default ``dim=1`` matches ``(B, L, C)`` Project 8 / Project 8
    combined batches; pass ``dim=-1`` for ``(B, L)`` toy batches.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        psd_pred = torch.fft.rfft(y_pred, dim=self.dim).abs().pow(2)
        psd_true = torch.fft.rfft(y_true, dim=self.dim).abs().pow(2)
        return F.mse_loss(psd_pred, psd_true)


class MixtureMSEPSDLoss(nn.Module):
    """Convex mixture of time-domain MSE and PSD loss.

    ``alpha * MSE(y_pred, y_true) + (1 - alpha) * PSDLoss(y_pred, y_true)``.
    """

    def __init__(self, alpha: float = 0.5, dim: int = 1):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.psd = PSDLoss(dim=dim)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(y_pred, y_true) + (1.0 - self.alpha) * self.psd(y_pred, y_true)
