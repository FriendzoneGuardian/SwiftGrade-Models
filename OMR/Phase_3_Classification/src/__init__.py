"""
Phase 3 – Classification package.

Public API
----------
BubbleClassifier  : Lightweight CNN that classifies a single bubble as filled/empty.
RelativeRowScorer : Determines the selected answer per row via relative comparison
                    of fill probabilities (Relative Row Scoring algorithm).
DiamondCNN        : Expand-contract CNN for texture-sensitive bubble classification.
AscendingCNN      : Progressive feature hierarchy for robust bubble classification.
RelativeRowDecisionEngine : Row arbitration with tie and multi-mark review flags.
ClassificationTrainer      : Training wrapper with early stopping and checkpoints.
"""

from .bubble_classifier import BubbleClassifier
from .cnn_models import AscendingCNN, DiamondCNN, TransferLearningCNN
from .dataset import build_dataset_index, create_dataloaders, resolve_phase3_dataset_root
from .relative_row_scorer import RelativeRowScorer
from .scoring import RelativeRowDecisionEngine
from .trainer import ClassificationTrainer, TrainerConfig

__all__ = [
    "BubbleClassifier",
    "RelativeRowScorer",
    "DiamondCNN",
    "AscendingCNN",
    "TransferLearningCNN",
    "RelativeRowDecisionEngine",
    "ClassificationTrainer",
    "TrainerConfig",
    "resolve_phase3_dataset_root",
    "build_dataset_index",
    "create_dataloaders",
]
