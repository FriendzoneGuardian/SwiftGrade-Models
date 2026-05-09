"""
Short_Answer_NLP: Automated Essay Scoring Module

Designed to evaluate essays using the ASAP-AES dataset approach.
Extracts linguistic features, trains regression models per essay set,
and generates feedback for student essays.
"""

__version__ = "0.1.0"
__author__ = "SwiftGrade Team"

from .data_loader import ASAPDataLoader
from .feature_extractor import EssayFeatureExtractor
from .metrics import compute_qwk, compute_essay_metrics
from .essay_evaluator import EssayEvaluator
from .feedback_generator import FeedbackGenerator

__all__ = [
    "ASAPDataLoader",
    "EssayFeatureExtractor",
    "compute_qwk",
    "compute_essay_metrics",
    "EssayEvaluator",
    "FeedbackGenerator",
]
