"""
Essay Feature Extractor

Computes linguistic, lexical, and syntactic features from essay text.
Features designed to predict essay quality/scores.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

logger = logging.getLogger(__name__)


class EssayFeatureExtractor:
    """Extract linguistic features from essay text."""
    
    # Common misspellings for validation
    COMMON_MISSPELLINGS = {
        'recieve': 'receive', 'occured': 'occurred', 'definately': 'definitely',
        'neccessary': 'necessary', 'seperate': 'separate', 'thier': 'their',
        'untill': 'until', 'wich': 'which', 'becuase': 'because'
    }
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            use_spacy: If True, use spaCy for advanced NLP features.
        """
        self.use_spacy = use_spacy and HAS_SPACY
        self.use_textblob = False # Disabled for performance in smoke test
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def extract_all(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text.
        
        Args:
            text: Essay text
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Basic length features
        features.update(self._length_features(text))
        
        # Vocabulary features
        features.update(self._vocabulary_features(text))
        
        # Grammar and syntax features
        if self.use_spacy:
            features.update(self._syntax_features(text))
        
        # Spelling and mechanics
        if self.use_textblob:
            features.update(self._spelling_features(text))
        else:
            features['misspelling_rate'] = 0.0
        
        return features
    
    def _length_features(self, text: str) -> Dict[str, float]:
        """Compute length-based features."""
        # Clean text
        text = text.strip()
        
        # Sentence and word counts
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        word_count = len(words)
        sentence_count = len(sentences)
        
        features = {
            'word_count': float(word_count),
            'sentence_count': float(sentence_count),
            'paragraph_count': float(text.count('\n') + 1),
            'avg_sentence_length': word_count / max(sentence_count, 1),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0.0,
        }
        
        return features
    
    def _vocabulary_features(self, text: str) -> Dict[str, float]:
        """Compute vocabulary-based features."""
        words = text.lower().split()
        unique_words = set(words)
        word_count = len(words)
        
        # Lexical diversity (Type-Token Ratio)
        ttr = len(unique_words) / max(word_count, 1)
        
        # Lexical richness approximation (words not in 1000 most common English words)
        common_1k_words = self._get_common_1k_words()
        advanced_words = sum(1 for w in unique_words if w not in common_1k_words)
        advanced_word_ratio = advanced_words / max(len(unique_words), 1)
        
        features = {
            'unique_word_count': float(len(unique_words)),
            'ttr_ratio': float(ttr),  # Type-Token Ratio
            'advanced_word_ratio': float(advanced_word_ratio),
            'repetition_ratio': float(1.0 - ttr),  # Higher = more repetition
        }
        
        return features
    
    def _syntax_features(self, text: str) -> Dict[str, float]:
        """Compute syntax features using spaCy."""
        if not self.use_spacy:
            return {}
        
        try:
            doc = self.nlp(text)
            
            # Parse tree depth (approximation)
            max_depth = 0
            for token in doc:
                depth = 0
                ancestor = token
                while ancestor.head != ancestor:
                    depth += 1
                    ancestor = ancestor.head
                max_depth = max(max_depth, depth)
            
            # Clause and subordination
            subordinate_count = sum(1 for token in doc if token.dep_ in ['advcl', 'relcl'])
            clause_count = sum(1 for token in doc if token.pos_ == 'VERB')
            
            # Pronoun usage
            pronoun_count = sum(1 for token in doc if token.pos_ == 'PRON')
            
            features = {
                'max_parse_depth': float(max_depth),
                'subordinate_clause_ratio': subordinate_count / max(clause_count, 1),
                'pronoun_ratio': pronoun_count / max(len(doc), 1),
            }
            
        except Exception as e:
            logger.warning(f"Error extracting syntax features: {e}")
            features = {
                'max_parse_depth': 0.0,
                'subordinate_clause_ratio': 0.0,
                'pronoun_ratio': 0.0,
            }
        
        return features
    
    def _spelling_features(self, text: str) -> Dict[str, float]:
        """Compute spelling and mechanics features."""
        if not self.use_textblob:
            return {'misspelling_rate': 0.0}
        
        try:
            blob = TextBlob(text)
            misspellings = [word for word in blob.words if word.lower() not in blob.correct().words]
            misspelling_rate = len(misspellings) / max(len(blob.words), 1)
            
            features = {
                'misspelling_rate': float(misspelling_rate),
            }
        except Exception as e:
            logger.warning(f"Error extracting spelling features: {e}")
            features = {'misspelling_rate': 0.0}
        
        return features
    
    @staticmethod
    def _get_common_1k_words() -> set:
        """Return a set of the 1000 most common English words."""
        # Simplified list of most common English words
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'being', 'has',
            'had', 'does', 'did', 'should', 'may', 'might', 'must', 'can', 'could',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'either', 'neither',
            'such', 'same', 'different', 'new', 'old', 'good', 'bad', 'better', 'best',
            'more', 'most', 'very', 'quite', 'rather', 'seem', 'appear', 'look', 'sound',
        }
        return common_words
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from multiple essays.
        
        Args:
            texts: List of essay texts
            
        Returns:
            DataFrame with features as columns
        """
        features_list = []
        for i, text in enumerate(texts):
            try:
                features = self.extract_all(text)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error extracting features from text {i}: {e}")
                # Return zeros for failed extractions
                features_list.append({k: 0.0 for k in self.extract_all("sample text").keys()})
        
        return pd.DataFrame(features_list)
