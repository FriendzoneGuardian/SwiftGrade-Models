"""Diamond CNN architecture for Phase 3 bubble classification."""

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


class DiamondCNN(nn.Module):
    """Diamond profile with optional depth extension and final 1x1 projection.

    Supported block depths:
    - 5 blocks  : 3 -> 32 -> 64 -> 128 -> 64 -> 32
    - 9 blocks  : 3 -> 32 -> 48 -> 64 -> 96 -> 128 -> 96 -> 64 -> 48 -> 32
    - 11 blocks : 3 -> 32 -> 40 -> 56 -> 72 -> 96 -> 128 -> 96 -> 72 -> 56 -> 40 -> 32

    "use_unification" enables a final pointwise (1x1) projection that unifies
    the last feature map channels before global pooling.
    """

    _DEPTH_PROFILES: dict[int, list[int]] = {
        5: [32, 64, 128, 64, 32],
        9: [32, 48, 64, 96, 128, 96, 64, 48, 32],
        11: [32, 40, 56, 72, 96, 128, 96, 72, 56, 40, 32],
    }
    _POOL_SCHEDULES: dict[int, list[bool]] = {
        5: [True, True, True, True, True],
        9: [True, True, False, True, False, True, False, True, False],
        11: [True, True, False, True, False, True, False, True, False, False, False],
    }

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.2,
        depth_blocks: int = 5,
        unify_channels: int = 32,
        use_unification: bool = False,
    ) -> None:
        super().__init__()

        if depth_blocks not in self._DEPTH_PROFILES:
            supported = sorted(self._DEPTH_PROFILES.keys())
            raise ValueError(f"Unsupported depth_blocks={depth_blocks}. Supported values: {supported}")

        channels = self._DEPTH_PROFILES[depth_blocks]
        pool_flags = self._POOL_SCHEDULES[depth_blocks]
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels, pool in zip(channels, pool_flags):
            blocks.append(ConvBlock(in_channels, out_channels, dropout=dropout, pool=pool))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        head_in_channels = channels[-1]
        if use_unification:
            self.unification = nn.Sequential(
                nn.Conv2d(head_in_channels, unify_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(unify_channels),
                nn.ReLU(inplace=True),
            )
            head_in_channels = unify_channels
        else:
            self.unification = nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.unification(x)
        return self.head(x)
