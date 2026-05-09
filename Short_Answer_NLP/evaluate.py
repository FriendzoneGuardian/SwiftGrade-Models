"""
Main Entry Point: Evaluate Essays using Trained Models

Usage:
    python evaluate.py --data essays.xlsx --essay-set 1 --output results.csv
"""

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import ASAPDataLoader
from src.feature_extractor import EssayFeatureExtractor
from src.essay_score_regressor import EssayScoreRegressor
from src.essay_evaluator import EssayEvaluator
from src.feedback_generator import FeedbackGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate essays using trained AES models"
    )
    parser.add_argument('--data', required=True, help='Path to essay data (Excel/CSV)')
    parser.add_argument('--essay-set', type=int, required=True, help='Essay set ID')
    parser.add_argument('--models-dir', default='./models', help='Directory with saved models')
    parser.add_argument('--output', default='./outputs/results.csv', help='Output CSV path')
    parser.add_argument('--with-feedback', action='store_true', help='Generate feedback')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ASAP-AES Essay Scoring Evaluation")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading essays from {args.data}...")
    loader = ASAPDataLoader(validate=True)
    loader.load(args.data)
    
    essay_set_data = loader.get_essay_set(args.essay_set)
    logger.info(f"Loaded {len(essay_set_data)} essays from set {args.essay_set}")
    
    # Load trained model
    logger.info(f"Loading trained model for essay set {args.essay_set}...")
    regressor = EssayScoreRegressor(use_xgboost=True)
    regressor.load_model(args.essay_set, args.models_dir)
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = EssayFeatureExtractor(use_spacy=True)
    features_df = feature_extractor.extract_batch(essay_set_data['essay'].tolist())
    
    # Predict
    logger.info("Predicting scores...")
    evaluator = EssayEvaluator(regressor, args.models_dir)
    results_df = evaluator.evaluate_batch(
        essay_set_data['essay'].tolist(),
        args.essay_set,
        return_features=args.with_feedback
    )
    
    # Add ground truth (if available)
    if 'score' in essay_set_data.columns:
        results_df['human_score'] = essay_set_data['score'].values
        
        from src.metrics import compute_essay_metrics
        metrics = compute_essay_metrics(
            essay_set_data['score'].values,
            results_df['predicted_score'].values,
            n_classes=int(loader.score_ranges[args.essay_set][1] -
                         loader.score_ranges[args.essay_set][0] + 1)
        )
        
        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  QWK: {metrics['qwk']:.3f}")
        logger.info(f"  MAE: {metrics['mae']:.3f}")
        logger.info(f"  RMSE: {metrics['rmse']:.3f}")
    
    # Generate feedback if requested
    if args.with_feedback:
        logger.info("Generating feedback...")
        feedback_gen = FeedbackGenerator()
        feedback_list = []
        
        for idx, row in results_df.iterrows():
            essay_features = {}
            for col in features_df.columns:
                essay_features[col] = features_df.iloc[idx][col]
            
            feedback = feedback_gen.generate_feedback(
                essay_features,
                row['predicted_score'],
                loader.score_ranges[args.essay_set]
            )
            feedback_list.append(feedback)
        
        results_df['feedback'] = [json.dumps(fb) for fb in feedback_list]
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    # Summary
    flagged_count = results_df['flagged_for_review'].sum()
    logger.info(f"\n{flagged_count} essays flagged for human review (confidence < 0.70)")
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
