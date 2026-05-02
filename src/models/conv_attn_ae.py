"""1D Convolutional Autoencoder with Multi-Head Attention in the bottleneck.

Autoencoder mode (num_classes=0, default):
    Interface: (B, L, 1) → (B, L, 1)
    Same encoder/decoder as ConvAE; attention operates on the bottleneck
    representation before decoding.

Classification mode (num_classes > 0):
    Interface: (B, L, 1) → (B, num_classes)
    Encoder + bottleneck attention + global average pool + Linear MLP head.
    Optional ``dropout`` after each conv activation; ``attn_dropout`` on MHA.

    Parameter count (kernel_size=k, n_layers=L, num_heads=H, num_classes=nc):
        Conv backbone (same as ConvAE classify mode): (L-1)*k*C² + [k+3+3*(L-1)]*C
        MHA (4C²+4C) + LayerNorm (2C): 4C²+6C
        head Linear(C, nc): C*nc + nc
        Total = [(L-1)*k + 4]*C² + [k+3+3*(L-1)+6+nc]*C + nc
    With k=5, L=4, H=4, nc=8:
        Total = 19*C² + 31*C + 8   (C must be divisible by num_heads)
        Inverse: C = round((-31 + sqrt(961 + 76*(target − 8))) / 38)
        Tiers (latent_channels=C, divisible by 4):
            xs (~10K):  C=24  → 11,696
            sm (~100K): C=72  → 100,736
            md (~300K): C=124 → 295,996
            lg (~700K): C=192 → 706,376
"""

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvAttnAE(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        latent_channels: int = 32,
        kernel_size: int = 5,
        pool_stride: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.1,
        num_classes: int = 0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be in [0, 1]')
        if not 0.0 <= attn_dropout <= 1.0:
            raise ValueError('attn_dropout must be in [0, 1]')
        if latent_channels % num_heads != 0:
            raise ValueError(f'latent_channels ({latent_channels}) must be divisible by num_heads ({num_heads})')
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

        # Bottleneck attention: operates on (B, L', latent_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(latent_channels)

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

    def forward(self, x):
        # x: (B, L, 1)
        B, L, _ = x.shape
        x = x.transpose(1, 2)                    # (B, 1, L)
        z = self.encoder(x)                       # (B, C, L')
        # Attention in bottleneck
        z_t = z.transpose(1, 2)                   # (B, L', C)
        h, _ = self.attn(z_t, z_t, z_t)
        z_t = self.attn_norm(z_t + h)            # residual
        z = z_t.transpose(1, 2)                   # (B, C, L')

        if self._classify:
            return self.head(z.mean(dim=-1))       # (B, num_classes)

        y = self.decoder(z)                       # (B, 1, L'')
        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = F.pad(y, (0, L - y.shape[-1]))
        return y.transpose(1, 2)                  # (B, L, 1)
