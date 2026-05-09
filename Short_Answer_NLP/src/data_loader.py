"""
Data Loader for ASAP-AES Dataset

Handles loading and preprocessing essays from Excel files.
Validates data integrity and handles multiple essay sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ASAPDataLoader:
    """Load and preprocess ASAP-AES essay data from Excel files."""
    
    EXPECTED_COLUMNS = ["essay_id", "essay_set", "essay", "score", "score2"]
    
    def __init__(self, validate: bool = True):
        """
        Initialize the data loader.
        
        Args:
            validate: If True, validate data upon loading.
        """
        self.validate = validate
        self.data = None
        self.essay_sets = {}
        self.score_ranges = {}
        
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load ASAP-AES data from Excel or CSV.
        
        Args:
            filepath: Path to Excel (.xlsx) or CSV file
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        try:
            if filepath.suffix.lower() == '.xlsx':
                self.data = pd.read_excel(filepath)
            elif filepath.suffix.lower() == '.csv':
                self.data = pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
        
        # Rename columns to match expected format (case-insensitive)
        column_mapping = {
            col: col.lower().replace(' ', '_') 
            for col in self.data.columns
        }
        self.data.rename(columns=column_mapping, inplace=True)
        
        if self.validate:
            self._validate_data()
        
        self._compute_essay_set_stats()
        
        logger.info(f"Successfully loaded {len(self.data)} essays from {len(self.essay_sets)} sets")
        return self.data
    
    def _validate_data(self):
        """Validate data integrity."""
        logger.info("Validating data...")
        
        # Check for required columns
        required = ['essay_id', 'essay_set', 'essay']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for nulls in critical columns
        null_essay = self.data['essay'].isnull().sum()
        if null_essay > 0:
            logger.warning(f"Found {null_essay} essays with null text")
        
        # Check score columns (score or score2 required)
        if 'score' not in self.data.columns and 'score2' not in self.data.columns:
            raise ValueError("No score columns found (expected 'score' or 'score2')")
        
        logger.info("Data validation passed")
    
    def _compute_essay_set_stats(self):
        """Compute statistics for each essay set."""
        for essay_set_id in self.data['essay_set'].unique():
            set_data = self.data[self.data['essay_set'] == essay_set_id]
            
            # Get score range (use primary score column)
            score_col = 'score' if 'score' in set_data.columns else 'score2'
            valid_scores = set_data[score_col].dropna()
            
            self.essay_sets[essay_set_id] = {
                'count': len(set_data),
                'score_min': float(valid_scores.min()),
                'score_max': float(valid_scores.max()),
                'score_mean': float(valid_scores.mean()),
                'score_std': float(valid_scores.std()),
            }
            
            self.score_ranges[essay_set_id] = (
                self.essay_sets[essay_set_id]['score_min'],
                self.essay_sets[essay_set_id]['score_max']
            )
            
            logger.info(
                f"Essay Set {essay_set_id}: {len(set_data)} essays, "
                f"score range: {self.essay_sets[essay_set_id]['score_min']:.1f}-{self.essay_sets[essay_set_id]['score_max']:.1f}"
            )
    
    def get_essay_set(self, essay_set_id: int) -> pd.DataFrame:
        """Get essays for a specific essay set."""
        if self.data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        return self.data[self.data['essay_set'] == essay_set_id].copy()
    
    def get_stats(self, essay_set_id: Optional[int] = None) -> Dict:
        """Get statistics for essay set(s)."""
        if essay_set_id is None:
            return self.essay_sets
        return self.essay_sets.get(essay_set_id, {})
    
    def split_train_val_test(
        self,
        essay_set_id: int,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split essay set data into train/val/test.
        Stratifies by score quintiles.
        
        Args:
            essay_set_id: ID of essay set to split
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        data = self.get_essay_set(essay_set_id)
        
        # Use primary score column
        score_col = 'score' if 'score' in data.columns else 'score2'
        data = data.dropna(subset=[score_col])
        
        # Stratify by score quintiles
        data['score_quintile'] = pd.qcut(
            data[score_col], 
            q=5, 
            labels=False, 
            duplicates='drop'
        )
        
        # Split
        np.random.seed(random_state)
        indices = np.random.permutation(len(data))
        
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        train_df = data.iloc[train_idx].drop('score_quintile', axis=1)
        val_df = data.iloc[val_idx].drop('score_quintile', axis=1)
        test_df = data.iloc[test_idx].drop('score_quintile', axis=1)
        
        logger.info(
            f"Split essay set {essay_set_id}: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def get_essay_text(self, essay_id: int) -> str:
        """Get essay text by ID."""
        if self.data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        
        essay = self.data[self.data['essay_id'] == essay_id]
        if essay.empty:
            raise ValueError(f"Essay ID {essay_id} not found")
        
        return essay.iloc[0]['essay']
    
    def get_essay_score(self, essay_id: int, score_col: str = 'score') -> Optional[float]:
        """Get essay score by ID."""
        if self.data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        
        essay = self.data[self.data['essay_id'] == essay_id]
        if essay.empty:
            raise ValueError(f"Essay ID {essay_id} not found")
        
        return essay.iloc[0][score_col]
