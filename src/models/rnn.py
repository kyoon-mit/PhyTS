"""Bidirectional GRU for time-series classification.

Interface: forward(x: (B, L, 1), mask=None) -> (B, d_output)
Uses masked mean pooling over GRU hidden states.
"""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        d_input: int = 1,
        d_output: int = 10,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_input,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(2 * d_model)
        self.decoder = nn.Linear(2 * d_model, d_output)

    def forward(self, x, mask=None):
        # x: (B, L, 1)
        h, _ = self.rnn(x)  # (B, L, 2*d_model)
        h = self.norm(h)

        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.decoder(h)  # (B, d_output)
