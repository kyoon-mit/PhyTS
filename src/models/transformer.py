"""Transformer encoder classifier for 1D time series.

Interface:
    forward(x: (B, L, 1), mask: (B, L)) -> (B, d_output)

The mask uses True for valid cadence positions and False for padding. It is
passed as a key-padding mask to the Transformer encoder and also used for
masked mean pooling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_output: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_output),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, L, 1); got {tuple(x.shape)}")
        if x.size(-1) != 1:
            raise ValueError(f"Expected last dimension 1; got {x.size(-1)}")
        if x.size(1) > self.seq_len:
            raise ValueError(f"Input length {x.size(1)} exceeds configured seq_len={self.seq_len}")

        x = self.input_proj(x)
        x = self.positional_encoding(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = mask.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)