"""
Phase 3 – Classification package.

Public API
----------
BubbleClassifier  : Lightweight CNN that classifies a single bubble as filled/empty.
RelativeRowScorer : Determines the selected answer per row via relative comparison
                    of fill probabilities (Relative Row Scoring algorithm).
"""

from .bubble_classifier import BubbleClassifier
from .relative_row_scorer import RelativeRowScorer

__all__ = ["BubbleClassifier", "RelativeRowScorer"]
