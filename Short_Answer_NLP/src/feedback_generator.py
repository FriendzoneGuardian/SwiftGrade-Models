"""
Feedback Generator

Generates human-readable feedback based on essay features.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Generate constructive feedback from essay features."""
    
    # Thresholds for identifying strengths/weaknesses
    STRONG_THRESHOLDS = {
        'word_count': 300,  # Strong if > 300 words
        'ttr_ratio': 0.7,   # Strong if TTR > 0.70
        'avg_sentence_length': 15,  # Strong if 15-25 words/sentence
        'advanced_word_ratio': 0.15,  # Strong if > 15% advanced words
        'misspelling_rate': 0.01,  # Strong if < 1% misspellings
    }
    
    WEAK_THRESHOLDS = {
        'word_count': 150,  # Weak if < 150 words
        'ttr_ratio': 0.5,   # Weak if TTR < 0.50
        'avg_sentence_length': 10,  # Weak if < 10 words/sentence
        'advanced_word_ratio': 0.05,  # Weak if < 5% advanced words
        'misspelling_rate': 0.05,  # Weak if > 5% misspellings
    }
    
    FEEDBACK_MESSAGES = {
        'word_count_strong': "Strong essay length ({:.0f} words) shows thorough exploration of the topic.",
        'word_count_weak': "Essay is quite short ({:.0f} words). Consider expanding with more ideas and examples.",
        
        'ttr_ratio_strong': "Excellent vocabulary variety. You use different words effectively without excessive repetition.",
        'ttr_ratio_weak': "Word repetition could be reduced. Try using synonyms to increase vocabulary diversity.",
        
        'avg_sentence_length_strong': "Sentence complexity is well-balanced, making the essay easy to follow.",
        'avg_sentence_length_weak': "Sentences are very short. Consider combining related ideas to improve flow.",
        'avg_sentence_length_too_long': "Some sentences are quite long. Breaking them into shorter ones could improve clarity.",
        
        'advanced_word_ratio_strong': "Sophisticated vocabulary demonstrates subject knowledge and maturity.",
        'advanced_word_ratio_weak': "Consider using more advanced vocabulary to strengthen your argument.",
        
        'misspelling_rate_strong': "Excellent proofreading! Very few spelling or grammar errors.",
        'misspelling_rate_weak': "Multiple spelling errors detected. Careful proofreading would improve the essay.",
        
        'subordinate_clause_ratio_strong': "Good use of complex sentence structures.",
        'subordinate_clause_ratio_weak': "More complex sentences would strengthen the essay.",
    }
    
    def __init__(self, percentile_stats: Dict = None):
        """
        Initialize feedback generator.
        
        Args:
            percentile_stats: Dictionary mapping essay_set_id to feature percentiles
        """
        self.percentile_stats = percentile_stats or {}
    
    def generate_feedback(
        self,
        features: Dict,
        predicted_score: float,
        score_range: tuple = None
    ) -> Dict:
        """
        Generate feedback for an essay.
        
        Args:
            features: Feature dictionary from EssayFeatureExtractor
            predicted_score: The predicted score
            score_range: (min, max) score range
            
        Returns:
            Dictionary with strengths, weaknesses, suggestions
        """
        strengths = []
        weaknesses = []
        suggestions = []
        
        # Analyze each feature
        if features.get('word_count', 0) > self.STRONG_THRESHOLDS['word_count']:
            strengths.append(
                self.FEEDBACK_MESSAGES['word_count_strong'].format(features['word_count'])
            )
        elif features.get('word_count', 0) < self.WEAK_THRESHOLDS['word_count']:
            weaknesses.append(
                self.FEEDBACK_MESSAGES['word_count_weak'].format(features['word_count'])
            )
            suggestions.append("✓ Expand your essay with more details, examples, or explanation.")
        
        # TTR (vocabulary variety)
        if features.get('ttr_ratio', 0) > self.STRONG_THRESHOLDS['ttr_ratio']:
            strengths.append(self.FEEDBACK_MESSAGES['ttr_ratio_strong'])
        elif features.get('ttr_ratio', 0) < self.WEAK_THRESHOLDS['ttr_ratio']:
            weaknesses.append(self.FEEDBACK_MESSAGES['ttr_ratio_weak'])
            suggestions.append("✓ Vary your word choices - use a thesaurus to find synonyms.")
        
        # Sentence length
        avg_sent_len = features.get('avg_sentence_length', 0)
        if self.WEAK_THRESHOLDS['avg_sentence_length'] < avg_sent_len < self.STRONG_THRESHOLDS['avg_sentence_length']:
            strengths.append(self.FEEDBACK_MESSAGES['avg_sentence_length_strong'])
        elif avg_sent_len < self.WEAK_THRESHOLDS['avg_sentence_length']:
            weaknesses.append(self.FEEDBACK_MESSAGES['avg_sentence_length_weak'])
            suggestions.append("✓ Combine short sentences using conjunctions like 'and', 'but', 'because'.")
        elif avg_sent_len > 25:
            weaknesses.append(self.FEEDBACK_MESSAGES['avg_sentence_length_too_long'])
            suggestions.append("✓ Break very long sentences at punctuation marks (commas, semicolons).")
        
        # Advanced vocabulary
        if features.get('advanced_word_ratio', 0) > self.STRONG_THRESHOLDS['advanced_word_ratio']:
            strengths.append(self.FEEDBACK_MESSAGES['advanced_word_ratio_strong'])
        elif features.get('advanced_word_ratio', 0) < self.WEAK_THRESHOLDS['advanced_word_ratio']:
            weaknesses.append(self.FEEDBACK_MESSAGES['advanced_word_ratio_weak'])
            suggestions.append("✓ Replace simple words with more sophisticated alternatives (e.g., 'because' → 'since'; 'show' → 'demonstrate').")
        
        # Spelling
        if features.get('misspelling_rate', 0) > self.WEAK_THRESHOLDS['misspelling_rate']:
            weaknesses.append(self.FEEDBACK_MESSAGES['misspelling_rate_weak'])
            suggestions.append("✓ Use a spell-checker and proofread carefully before submitting.")
        else:
            strengths.append(self.FEEDBACK_MESSAGES['misspelling_rate_strong'])
        
        # Subordinate clauses / syntax
        if features.get('subordinate_clause_ratio', 0) > 0.2:
            strengths.append(self.FEEDBACK_MESSAGES['subordinate_clause_ratio_strong'])
        elif features.get('subordinate_clause_ratio', 0) < 0.05:
            weaknesses.append(self.FEEDBACK_MESSAGES['subordinate_clause_ratio_weak'])
            suggestions.append("✓ Use subordinate clauses (e.g., 'Because...', 'Although...', 'While...').")
        
        # Score context
        if score_range:
            score_range_size = score_range[1] - score_range[0]
            score_percentile = (predicted_score - score_range[0]) / max(score_range_size, 1)
            
            if score_percentile >= 0.75:
                context = "This is a high-quality essay."
            elif score_percentile >= 0.50:
                context = "This is a solid essay with good potential."
            else:
                context = "This essay needs improvement in several areas."
        else:
            context = "See feedback below for improvement areas."
        
        return {
            'summary': context,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'suggestions': suggestions,
            'predicted_score': float(predicted_score),
        }
    
    def format_feedback_text(self, feedback: Dict) -> str:
        """Format feedback as readable text."""
        text = []
        text.append(f"📊 Predicted Score: {feedback['predicted_score']:.1f}")
        text.append(f"\n{feedback['summary']}\n")
        
        if feedback['strengths']:
            text.append("✅ STRENGTHS:")
            for strength in feedback['strengths']:
                text.append(f"  • {strength}")
            text.append("")
        
        if feedback['weaknesses']:
            text.append("⚠️  AREAS FOR IMPROVEMENT:")
            for weakness in feedback['weaknesses']:
                text.append(f"  • {weakness}")
            text.append("")
        
        if feedback['suggestions']:
            text.append("💡 SUGGESTIONS:")
            for suggestion in feedback['suggestions']:
                text.append(f"  {suggestion}")
        
        return "\n".join(text)
    
    def generate_batch_feedback(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        score_range: tuple = None
    ) -> List[Dict]:
        """Generate feedback for multiple essays."""
        feedback_list = []
        
        for idx, row in features_df.iterrows():
            features = row.to_dict()
            score = predictions[idx]
            feedback = self.generate_feedback(features, score, score_range)
            feedback_list.append(feedback)
        
        return feedback_list
