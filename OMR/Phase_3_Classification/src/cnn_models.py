"""Compatibility facade for Phase 3 model architectures.

Canonical implementations live in dedicated modules:
- diamond.py
- ascending.py
- transfer_learning.py
"""

from __future__ import annotations

from ascending import AscendingCNN
from diamond import DiamondCNN
from transfer_learning import TransferLearningCNN

__all__ = ["DiamondCNN", "AscendingCNN", "TransferLearningCNN"]
