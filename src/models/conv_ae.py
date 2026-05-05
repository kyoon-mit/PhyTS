"""1D Convolutional Autoencoder for seq2seq denoising, with optional classification mode.

Autoencoder mode (num_classes=0, default):
    Interface: (B, L, 1) → (B, L, 1)
    Encoder: Conv1d + GroupNorm + LeakyReLU + MaxPool1d  (× n_layers)
    Decoder: ConvTranspose1d + GroupNorm + LeakyReLU     (× n_layers) + Conv1d(→1)

Classification mode (num_classes > 0):
    Interface: (B, L, 1) → (B, num_classes)
    Encoder only + global average pool + Linear MLP head.
    Optional ``dropout`` after each encoder activation (training only).

    Parameter count (kernel_size=k, n_layers=L, num_classes=nc):
        layer 0 (1→C): (k+3)*C
        layers 1..L-1 (C→C): k*C² + 3C   [L-1 of these]
        head Linear(C, nc): C*nc + nc
        Total = (L-1)*k*C² + [k+3 + 3*(L-1) + nc]*C + nc
    With k=5, L=4, nc=8:
        Total = 15*C² + 25*C + 8
        Inverse: C = round((-25 + sqrt(625 + 60*(target − 8))) / 30)
        Tiers (latent_channels=C):
            xs (~10K):  C=25  → 10,008
            sm (~100K): C=80  → 98,008
            md (~300K): C=140 → 297,508
            lg (~700K): C=216 → 705,248
"""

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvAE(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        latent_channels: int = 32,
        kernel_size: int = 5,
        pool_stride: int = 2,
        num_classes: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be in [0, 1]')
        pad = kernel_size // 2
        self._classify = num_classes > 0

        enc = OrderedDict()
        for i in range(n_layers):
            in_ch = 1 if i == 0 else latent_channels
            enc[f'conv{i}'] = nn.Conv1d(in_ch, latent_channels, kernel_size, padding=pad)
            enc[f'norm{i}'] = nn.GroupNorm(1, latent_channels)
            enc[f'act{i}']  = nn.LeakyReLU()
            enc[f'drop{i}'] = nn.Dropout(dropout)
            enc[f'pool{i}'] = nn.MaxPool1d(pool_stride, stride=pool_stride)
        self.encoder = nn.Sequential(enc)

        if self._classify:
            self.head = nn.Linear(latent_channels, num_classes)
        else:
            dec = OrderedDict()
            for i in range(n_layers):
                dec[f'deconv{i}'] = nn.ConvTranspose1d(latent_channels, latent_channels,
                                                        pool_stride, stride=pool_stride)
                dec[f'norm{i}']   = nn.GroupNorm(1, latent_channels)
                dec[f'act{i}']    = nn.LeakyReLU()
                dec[f'drop{i}']   = nn.Dropout(dropout)
            dec['out'] = nn.Conv1d(latent_channels, 1, kernel_size=1)
            self.decoder = nn.Sequential(dec)

    def forward(self, x, mask=None):
        # x: (B, L, 1); mask: (B, L) bool, True = valid cadence.
        L = x.shape[1]
        x = x.transpose(1, 2)          # (B, 1, L)
        z = self.encoder(x)            # (B, C, L')

        if self._classify:
            if mask is not None:
                # Downsample mask to bottleneck length L' (valid if any covered position is valid).
                L_prime = z.shape[-1]
                mask_ds = F.adaptive_max_pool1d(
                    mask.float().unsqueeze(1), L_prime
                ).squeeze(1)           # (B, L')
                pooled = (z * mask_ds.unsqueeze(1)).sum(dim=-1) / mask_ds.sum(dim=-1, keepdim=True).clamp(min=1.0)
            else:
                pooled = z.mean(dim=-1)
            return self.head(pooled)   # (B, num_classes)

        y = self.decoder(z)            # (B, 1, L'')
        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = F.pad(y, (0, L - y.shape[-1]))
        return y.transpose(1, 2)       # (B, L, 1)
