"""1D Convolutional Autoencoder for seq2seq denoising.

Interface: (B, L, 1) -> (B, L, 1)

Encoder: Conv1d + LeakyReLU + MaxPool1d  (× n_layers)
Decoder: ConvTranspose1d + LeakyReLU     (× n_layers) + Conv1d(→1)
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
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        pad = kernel_size // 2

        enc = OrderedDict()
        for i in range(n_layers):
            in_ch = 1 if i == 0 else latent_channels
            enc[f'conv{i}'] = nn.Conv1d(in_ch, latent_channels, kernel_size, padding=pad)
            enc[f'norm{i}'] = nn.GroupNorm(1, latent_channels)
            enc[f'act{i}']  = nn.LeakyReLU()
            enc[f'pool{i}'] = nn.MaxPool1d(pool_stride, stride=pool_stride)
        self.encoder = nn.Sequential(enc)

        dec = OrderedDict()
        for i in range(n_layers):
            dec[f'deconv{i}'] = nn.ConvTranspose1d(latent_channels, latent_channels,
                                                    pool_stride, stride=pool_stride)
            dec[f'norm{i}']   = nn.GroupNorm(1, latent_channels)
            dec[f'act{i}']    = nn.LeakyReLU()
        dec['out'] = nn.Conv1d(latent_channels, 1, kernel_size=1)
        self.decoder = nn.Sequential(dec)

    def forward(self, x):
        # x: (B, L, 1)
        B, L, _ = x.shape
        x = x.transpose(1, 2)          # (B, 1, L)
        z = self.encoder(x)            # (B, C, L')
        y = self.decoder(z)            # (B, 1, L'')
        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = F.pad(y, (0, L - y.shape[-1]))
        return y.transpose(1, 2)       # (B, L, 1)
