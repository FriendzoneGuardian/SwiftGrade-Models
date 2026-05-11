"""
Metrics for Essay Scoring

QWK (Quadratic Weighted Kappa) and other evaluation metrics
for automated essay scoring tasks.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = None) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK) using sklearn.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_int = np.round(np.asarray(y_pred)).astype(int)
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred_int.max()) + 1
        
    y_pred_int = np.clip(y_pred_int, 0, n_classes - 1)
    
    # Use industry-standard sklearn calculation
    return float(cohen_kappa_score(y_true, y_pred_int, weights='quadratic'))

def compute_essay_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = None
) -> dict:
    """
    Compute comprehensive essay scoring metrics.
    
    Args:
        y_true: Ground truth scores (denormalized)
        y_pred: Predicted scores (continuous/denormalized)
        n_classes: Number of score classes
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    y_pred_int = np.round(y_pred).astype(int)
    y_true_int = np.round(y_true).astype(int)
    
    if n_classes is None:
        n_classes = max(y_true_int.max(), y_pred_int.max()) + 1
    
    y_pred_int = np.clip(y_pred_int, 0, n_classes - 1)
    
    metrics = {
        'qwk': compute_qwk(y_true_int, y_pred_int, n_classes),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'pearson_r': float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
    }
    
    return metrics


def compute_prediction_confidence(y_pred: np.ndarray, y_pred_std: np.ndarray = None) -> np.ndarray:
    """
    Compute confidence scores for predictions.
    
    If std is provided, uses inverse of standard error.
    Otherwise, uses distance from decision boundaries (0.5 units from integers).
    
    Args:
        y_pred: Predicted scores
        y_pred_std: Standard deviation of predictions (optional)
        
    Returns:
        Confidence scores (0 to 1, where 1 is high confidence)
    """
    if y_pred_std is not None:
        # Confidence = 1 / (1 + std)
        confidence = 1.0 / (1.0 + np.asarray(y_pred_std))
    else:
        # Distance from nearest integer
        fractional = np.asarray(y_pred) % 1
        distance_from_int = np.minimum(fractional, 1 - fractional)
        # Map to confidence: 0.5 distance = 0 confidence, 0 distance = 0.5 confidence
        confidence = 0.5 + distance_from_int
    
    return np.clip(confidence, 0, 1)


def flag_ambiguous_predictions(
    y_pred: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.75
) -> np.ndarray:
    """
    Flag predictions with low confidence for human review.
    
    Args:
        y_pred: Predicted scores
        confidence: Confidence scores
        threshold: Confidence threshold below which to flag
        
    Returns:
        Boolean array, True where flagged for review
    """
    return confidence < threshold
