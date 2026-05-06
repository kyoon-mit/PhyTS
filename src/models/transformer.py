"""Transformer encoder for time-series classification.

Architecture:
    Linear encoder  (B, L, d_input) -> (B, L, d_model)
    + learnable positional encoding
    TransformerEncoder  n_layers × (MultiheadAttention + FFN)
    Masked mean pool  (B, L, d_model) -> (B, d_model)
    Linear decoder  (B, d_model) -> (B, d_output)

Interface matches S4Model: forward(x, mask=None) where x is (B, L, d_input).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_input: int = 1,
        d_output: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model

        self.encoder = nn.Linear(d_input, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        x:    (B, L, d_input)
        mask: (B, L) bool, True = real, False = padding
        """
        B, L, _ = x.shape
        x = self.encoder(x) + self.pos_enc[:, :L, :]
        x = self.dropout(x)

        # TransformerEncoder expects src_key_padding_mask: True = ignore
        pad_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Masked mean pool
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.decoder(x)


class TransformerClassifier(TransformerModel):
    """Same architecture as :class:`TransformerModel`, Lightning-friendly constructor.

    PyTorch Lightning YAML and sweep code use ``seq_len``, ``nhead``,
    ``num_layers``, and ``dim_feedforward`` instead of ``max_len``, ``n_heads``,
    ``n_layers``, and ``d_ff``.
    """

    def __init__(
        self,
        seq_len: int = 4096,
        d_input: int = 1,
        d_output: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            n_heads=nhead,
            n_layers=num_layers,
            d_ff=dim_feedforward,
            dropout=dropout,
            max_len=seq_len,
        )
