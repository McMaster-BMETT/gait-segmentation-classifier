"""Bidirectional Temporal Convolutional Network for gait phase segmentation."""

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """Residual TCN block with non-causal (bidirectional) dilated convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        # Non-causal: symmetric padding on both sides
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, out_channels, T)
        """
        residual = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        # Trim to match input length if needed (odd kernel edge case)
        if out.shape[2] != residual.shape[2]:
            out = out[:, :, :residual.shape[2]]
        return out + residual


class GaitTCN(nn.Module):
    """Bidirectional TCN for per-timestep gait phase classification.

    Architecture: input projection → N residual TCN blocks → output projection
    """

    def __init__(self, num_channels: int = 8, hidden_channels: int = 64,
                 num_classes: int = 6, num_blocks: int = 4,
                 kernel_size: int = 3, dilations: list[int] | None = None,
                 dropout: float = 0.2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        assert len(dilations) == num_blocks

        self.input_proj = nn.Conv1d(num_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList([
            TCNBlock(hidden_channels, hidden_channels, kernel_size,
                     dilation=dilations[i], dropout=dropout)
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, T) sensor input
        Returns:
            (B, num_classes, T) per-timestep logits
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
