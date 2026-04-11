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
    Four convolutional blocks for improved feature extraction, each consisting of::

        Conv2d → BatchNorm2d → ReLU → MaxPool2d(2×2)

    Channel progression: 3 → 32 → 64 → 128 → 256.

    Followed by::

        AdaptiveAvgPool2d(1×1) → Flatten → Dense layers with dropout → output

    The adaptive pooling makes the model input-size agnostic. Four conv layers
    provide deeper feature extraction to reduce hallucinations, while dropout
    in the classifier head prevents overfitting on small datasets.

    Parameters
    ----------
    num_classes:
        Number of output classes.  Defaults to ``2`` (empty, filled).
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes

        # --- Convolutional backbone (4 blocks) -----------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classification head (with dropout) ----------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

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
        x = self.conv_block4(x)
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
