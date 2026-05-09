"""
Essay Score Regressor

Trains regression models per essay set to predict essay scores.
Uses XGBoost for interpretability and GPU compatibility with GTX 1650.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .metrics import compute_essay_metrics

logger = logging.getLogger(__name__)


class EssayScoreRegressor:
    """Train and evaluate essay scoring models per essay set."""
    
    def __init__(self, use_xgboost: bool = True, random_state: int = 42):
        """
        Initialize regressor.
        
        Args:
            use_xgboost: Use XGBoost if available, else RandomForest
            random_state: Random seed
        """
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.random_state = random_state
        self.models = {}  # essay_set_id -> model
        self.scalers = {}  # essay_set_id -> scaler
        self.feature_names = {}  # essay_set_id -> feature names
        self.score_ranges = {}  # essay_set_id -> (min, max)
        
        if self.use_xgboost:
            logger.info("Using XGBoost for essay scoring")
        else:
            logger.info("Using RandomForest for essay scoring (XGBoost not available)")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        essay_set_id: int,
        score_range: Tuple[float, float] = None,
        **model_kwargs
    ) -> Dict[str, float]:
        """
        Train a model for a specific essay set.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            essay_set_id: Essay set identifier
            score_range: (min, max) score range for this set
            **model_kwargs: Additional model hyperparameters
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Training model for essay set {essay_set_id}")
        
        # Store feature names
        self.feature_names[essay_set_id] = X_train.columns.tolist()
        
        # Store score range
        if score_range is None:
            score_range = (y_train.min(), y_train.max())
        self.score_ranges[essay_set_id] = score_range
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.scalers[essay_set_id] = scaler
        
        # Train model
        if self.use_xgboost:
            model = self._train_xgboost(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                essay_set_id,
                **model_kwargs
            )
        else:
            model = self._train_random_forest(
                X_train_scaled, y_train,
                **model_kwargs
            )
        
        self.models[essay_set_id] = model
        
        # Evaluate
        y_pred_val = model.predict(X_val_scaled)
        y_pred_val = np.clip(y_pred_val, score_range[0], score_range[1])
        
        metrics = compute_essay_metrics(
            y_val, y_pred_val,
            n_classes=int(score_range[1] - score_range[0] + 1)
        )
        
        logger.info(f"Validation QWK: {metrics['qwk']:.3f}, MAE: {metrics['mae']:.3f}")
        
        return metrics
    
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        essay_set_id: int,
        **kwargs
    ) -> xgb.XGBRegressor:
        """Train XGBoost model."""
        # Default hyperparameters optimized for essay scoring
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbose': 0,
        }
        params.update(kwargs)
        
        model = xgb.XGBRegressor(**params)
        
        # Early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        return model
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> RandomForestRegressor:
        """Train RandomForest model."""
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'n_jobs': -1,
        }
        params.update(kwargs)
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def predict(self, X: pd.DataFrame, essay_set_id: int) -> np.ndarray:
        """
        Predict scores for essays in a specific set.
        
        Args:
            X: Feature dataframe
            essay_set_id: Essay set ID
            
        Returns:
            Predicted scores (clipped to valid range)
        """
        if essay_set_id not in self.models:
            raise ValueError(f"No model trained for essay set {essay_set_id}")
        
        X_scaled = self.scalers[essay_set_id].transform(X)
        y_pred = self.models[essay_set_id].predict(X_scaled)
        
        # Clip to valid range
        score_range = self.score_ranges[essay_set_id]
        y_pred = np.clip(y_pred, score_range[0], score_range[1])
        
        return y_pred
    
    def get_feature_importance(self, essay_set_id: int, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance for a model."""
        if essay_set_id not in self.models:
            raise ValueError(f"No model trained for essay set {essay_set_id}")
        
        model = self.models[essay_set_id]
        feature_names = self.feature_names[essay_set_id]
        
        if self.use_xgboost:
            importances = model.feature_importances_
        else:
            importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, essay_set_id: int, save_dir: str):
        """Save trained model and scaler."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"model_set_{essay_set_id}.pkl"
        scaler_path = save_dir / f"scaler_set_{essay_set_id}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[essay_set_id], f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[essay_set_id], f)
        
        logger.info(f"Saved model for essay set {essay_set_id} to {save_dir}")
    
    def load_model(self, essay_set_id: int, load_dir: str):
        """Load trained model and scaler."""
        load_dir = Path(load_dir)
        
        model_path = load_dir / f"model_set_{essay_set_id}.pkl"
        scaler_path = load_dir / f"scaler_set_{essay_set_id}.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model files not found for essay set {essay_set_id}")
        
        with open(model_path, 'rb') as f:
            self.models[essay_set_id] = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scalers[essay_set_id] = pickle.load(f)
        
        logger.info(f"Loaded model for essay set {essay_set_id} from {load_dir}")
