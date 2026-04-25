"""Ascending CNN architecture for Phase 3 bubble classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AscendingCNN(nn.Module):
    """Ascending profile: 3 -> 32 -> 64 -> 128 -> 256."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=dropout),
            ConvBlock(32, 64, dropout=dropout),
            ConvBlock(64, 128, dropout=dropout),
            ConvBlock(128, 256, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
