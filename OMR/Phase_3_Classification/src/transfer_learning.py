"""
transfer_learning.py – MobileNetV2 transfer learning model for Phase 3.3
bubble classification.

Backbone : MobileNetV2 (torchvision, ImageNet pretrained)
Task     : Binary classification — blank / filled
Strategy : Two-phase training
           Phase A – backbone frozen, classifier head only
           Phase B – last 3 InvertedResidual blocks + head unfrozen

Drop-in replacement for the previous ResNet-18 implementation.
All public API is identical: __init__, forward, unfreeze_all, unfreeze_top_blocks.
Compatible with ClassificationTrainer, TrainerConfig, and create_dataloaders
without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class TransferLearningCNN(nn.Module):
    """MobileNetV2 backbone with a task-specific classification head.

    Parameters
    ----------
    num_classes:
        Output classes. Defaults to 2 (blank / filled).
    freeze_backbone:
        If True (default), all backbone parameters are frozen on init.
        Only the classifier head trains in Phase A.
        Call unfreeze_top_blocks() or unfreeze_all() to enter Phase B.
    dropout:
        Dropout probability applied inside the classification head.
        Defaults to 0.3 — conservative given the binary task.
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.backbone, self.classifier = self._build(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
        self._num_classes = num_classes

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def _build(
        num_classes: int,
        freeze_backbone: bool,
        dropout: float,
    ) -> tuple[nn.Module, nn.Module]:
        """Load MobileNetV2 backbone and attach a custom head."""

        # Prefer the new weights API; fall back gracefully for older torchvision.
        try:
            base = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            # torchvision < 0.13 compatibility
            base = models.mobilenet_v2(pretrained=True)  # type: ignore[call-arg]

        # Strip the original classifier — keep only the feature extractor.
        backbone = base.features  # nn.Sequential of InvertedResidual blocks

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # MobileNetV2 feature output: (N, 1280, H, W) after features block.
        # AdaptiveAvgPool collapses spatial dims → (N, 1280).
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        return backbone, classifier

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Float32 tensor of shape (N, 3, H, W), values in [0, 1].
            Input size agnostic — AdaptiveAvgPool handles arbitrary H, W.

        Returns
        -------
        torch.Tensor
            Raw logits of shape (N, num_classes).
        """
        features = self.backbone(x)
        return self.classifier(features)

    # ------------------------------------------------------------------
    # Freezing API  (mirrors the old ResNet-18 interface)
    # ------------------------------------------------------------------

    def unfreeze_top_blocks(self, n_blocks: int = 3) -> None:
        """Unfreeze the last n InvertedResidual blocks for Phase B fine-tuning.

        MobileNetV2 features block has 19 sub-modules (indices 0–18).
        Default unfreezes indices 16, 17, 18 — the final three blocks plus
        the last pointwise conv layer.

        Parameters
        ----------
        n_blocks:
            Number of trailing feature blocks to unfreeze. Defaults to 3.
        """
        feature_blocks = list(self.backbone.children())
        total = len(feature_blocks)
        unfreeze_from = max(0, total - n_blocks)

        for block in feature_blocks[unfreeze_from:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always keep classifier head trainable.
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter — use for full fine-tune runs."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_backbone(self) -> None:
        """Re-freeze the backbone — useful for resetting to Phase A."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def trainable_parameter_count(self) -> int:
        """Return the number of parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        """Return the total parameter count regardless of grad status."""
        return sum(p.numel() for p in self.parameters())