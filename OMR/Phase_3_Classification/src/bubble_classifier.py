"""
bubble_classifier.py – Lightweight CNN for single OMR bubble classification.

Classifies an individual bubble image as *filled* (1) or *empty* (0).
The architecture is intentionally shallow so that it can run efficiently on
CPU during inference while remaining trainable on small labelled datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BubbleClassifier(nn.Module):
    """Lightweight CNN that classifies a single OMR bubble as filled or empty.

    Architecture
    ------------
    Three convolutional blocks, each consisting of::

        Conv2d → BatchNorm2d → ReLU → MaxPool2d(2×2)

    Channel progression: 3 → 32 → 64 → 128.

    Followed by::

        AdaptiveAvgPool2d(1×1) → Flatten → Linear(128, num_classes)

    The adaptive pooling layer makes the model input-size agnostic, which is
    useful because bubble crops may vary slightly in pixel dimensions.

    Parameters
    ----------
    num_classes:
        Number of output classes.  Defaults to ``2`` (empty, filled).
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes

        # --- Convolutional backbone ----------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classification head -------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits for a batch of bubble images.

        Parameters
        ----------
        x:
            Float32 tensor of shape ``(N, 3, H, W)`` with values in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(N, num_classes)``.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Convenience inference methods
    # ------------------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities.

        Parameters
        ----------
        x:
            Float32 tensor of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Probability tensor of shape ``(N, num_classes)`` summing to 1
            along the class dimension.
        """
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted class index for each sample.

        Parameters
        ----------
        x:
            Float32 tensor of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``(N,)`` containing class indices.
        """
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)
