"""1D CNN for time-series classification.

Interface: forward(x: (B, L, 1), mask=None) -> (B, d_output)
Stacked Conv1d blocks with adaptive average pooling.
"""

import torch
import torch.nn as nn


class ConvClassifier(nn.Module):
    def __init__(
        self,
        d_input: int = 1,
        d_output: int = 10,
        channels: list[int] = [32, 64, 128],
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        pad = kernel_size // 2

        layers = []
        in_ch = d_input
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.decoder = nn.Linear(channels[-1], d_output)

    def forward(self, x, mask=None):
        # x: (B, L, 1)
        x = x.transpose(1, 2)  # (B, 1, L)
        if mask is not None:
            x = x * mask.unsqueeze(1).float()
        x = self.backbone(x)  # (B, C, L')

        # Global average pool
        x = x.mean(dim=2)  # (B, C)
        return self.decoder(x)  # (B, d_output)
