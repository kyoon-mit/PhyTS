"""Simple MLP for sequence regression.

Flattens the input sequence and passes it through fully-connected layers.
Matches the S4Model interface: forward(x: (B, L, 1)) -> (B, d_output).

Parameter count (seq_len=640, hidden_dims=[256,128,64], d_output=3):
  640*256 + 256*128 + 128*64 + 64*3 ≈ 205K
"""

import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_output: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()
        dims   = [seq_len] + hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], d_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, 1) -> (B, L) -> (B, d_output)
        return self.net(x.squeeze(-1))
