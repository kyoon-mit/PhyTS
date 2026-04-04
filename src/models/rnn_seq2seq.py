"""Bidirectional GRU seq2seq denoiser.

Interface: (B, L, 1) -> (B, L, 1)
"""

import torch.nn as nn


class RNNSeq2Seq(nn.Module):
    def __init__(
        self,
        d_input: int = 1,
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
        self.proj = nn.Linear(2 * d_model, d_input)

    def forward(self, x):
        # x: (B, L, 1) — .contiguous() required by cuDNN GRU kernel
        h, _ = self.rnn(x.contiguous())  # (B, L, 2*d_model)
        return self.proj(h)              # (B, L, 1)
