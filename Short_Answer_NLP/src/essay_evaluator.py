"""
Essay Evaluator

Scores new essays using trained models and provides confidence scores.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path

from .feature_extractor import EssayFeatureExtractor
from .metrics import compute_prediction_confidence

logger = logging.getLogger(__name__)


class EssayEvaluator:
    """Evaluate new essays using trained models."""
    
    def __init__(self, regressor, model_dir: str = None):
        """
        Initialize evaluator.
        
        Args:
            regressor: Trained EssayScoreRegressor instance
            model_dir: Directory containing saved models (for lazy loading)
        """
        self.regressor = regressor
        self.model_dir = model_dir
        self.feature_extractor = EssayFeatureExtractor(use_spacy=True)
        self.essay_set_stats = {}
    
    def set_essay_set_stats(self, essay_set_id: int, stats: Dict):
        """Store statistics for an essay set (for confidence/feedback)."""
        self.essay_set_stats[essay_set_id] = stats
    
    def evaluate(
        self,
        essay_text: str,
        essay_set_id: int,
        return_features: bool = False
    ) -> Dict:
        """
        Evaluate a single essay.
        
        Args:
            essay_text: The essay to evaluate
            essay_set_id: The essay set this belongs to
            return_features: If True, include extracted features in output
            
        Returns:
            Dictionary with:
            - predicted_score: Point estimate
            - confidence: Confidence score (0-1)
            - features: (optional) Extracted features
        """
        # Extract features
        features = self.feature_extractor.extract_all(essay_text)
        features_df = pd.DataFrame([features])
        
        # Predict
        y_pred = self.regressor.predict(features_df, essay_set_id)
        predicted_score = float(y_pred[0])
        
        # Confidence
        score_range = self.regressor.score_ranges[essay_set_id]
        fractional = predicted_score % 1
        distance_from_int = min(fractional, 1 - fractional)
        # Confidence: farther from 0.5 (integer boundary) = more confident
        confidence = 0.5 + min(distance_from_int, 0.5)
        
        result = {
            'predicted_score': predicted_score,
            'confidence': float(confidence),
        }
        
        if return_features:
            result['features'] = features
        
        return result
    
    def evaluate_batch(
        self,
        essay_texts: list,
        essay_set_id: int,
        return_features: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate multiple essays.
        
        Args:
            essay_texts: List of essay texts
            essay_set_id: The essay set
            return_features: If True, include features
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for i, essay_text in enumerate(essay_texts):
            try:
                result = self.evaluate(essay_text, essay_set_id, return_features)
                result['essay_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating essay {i}: {e}")
                result = {
                    'essay_index': i,
                    'predicted_score': np.nan,
                    'confidence': 0.0,
                }
                if return_features:
                    result['features'] = {}
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Flag low-confidence predictions
        df['flagged_for_review'] = df['confidence'] < 0.70
        
        return df
    
    def get_feature_stats(
        self,
        essay_set_id: int,
        percentile: int = 75
    ) -> Dict:
        """
        Get feature statistics for an essay set (for comparison).
        
        Args:
            essay_set_id: Essay set ID
            percentile: Percentile to compute (e.g., 75th percentile = above average)
            
        Returns:
            Dictionary of feature thresholds
        """
        if essay_set_id not in self.essay_set_stats:
            logger.warning(f"No statistics available for essay set {essay_set_id}")
            return {}
        
        return self.essay_set_stats[essay_set_id]
