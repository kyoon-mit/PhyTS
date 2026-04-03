"""MLP seq2seq denoiser — maps the full flattened sequence to itself.

Interface: (B, L, 1) -> (B, L, 1)

The entire sequence is flattened and passed through a fully-connected MLP.
This is a strong baseline that ignores locality but can learn global spectral
filtering patterns.
"""

import torch.nn as nn


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        seq_len: int = 640,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 512]
        dims = [seq_len] + hidden_dims + [seq_len]
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout)]
        layers.pop()   # remove final dropout
        layers.pop()   # remove final GELU
        layers.pop()   # remove final LayerNorm
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, 1)
        x = x.squeeze(-1)            # (B, L)
        return self.net(x).unsqueeze(-1)  # (B, L, 1)
