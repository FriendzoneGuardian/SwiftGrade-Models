"""Transfer Learning model for Phase 3 bubble classification.

Backbone: ResNet-18 with a replaced classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TransferLearningCNN(nn.Module):
    """ResNet-18 transfer model with configurable freezing policy.

    The loading strategy mirrors notebook compatibility logic across torchvision
    versions and offline environments.
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = self._build_backbone(num_classes=num_classes, freeze_backbone=freeze_backbone)

    @staticmethod
    def _build_backbone(num_classes: int, freeze_backbone: bool) -> nn.Module:
        # Keep compatibility across torchvision versions.
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            # Fallback for older torchvision or offline weight loading constraints.
            try:
                model = models.resnet18(pretrained=True)
            except Exception:
                model = models.resnet18(weights=None)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
