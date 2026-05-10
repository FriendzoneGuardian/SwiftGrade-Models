import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import ASAPDataLoader
from feature_extractor import EssayFeatureExtractor
from essay_score_regressor import EssayScoreRegressor

def run_smoke_test():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir.parent / "Unified_Datasets" / "ASAP-AES" / "training_set_rel3.xlsx"
    
    print(f"🚀 Starting NLP Smoke Test")
    print(f"📂 Loading data from {data_path}")
    
    # 1. Load Data
    # loader = ASAPDataLoader()
    # Manual load since standard loader expects 'score' column
    df = pd.read_excel(data_path)
    # Map ASAP-AES columns to internal names
    df = df.rename(columns={'domain1_score': 'score', 'essay_text': 'essay'})
    
    # Filter for Essay Set 1
    set1_data = df[df['essay_set'] == 1].head(10) # Smaller sample for smoke test
    print(f"✅ Loaded {len(set1_data)} essays for Set 1")
    
    # 2. Extract Features
    print("🧠 Extracting features (Small sample)...")
    extractor = EssayFeatureExtractor()
    features_df = extractor.extract_batch(set1_data['essay'].tolist())
    
    # 3. Prepare for training
    X = features_df
    y = set1_data['score'].values
    
    # Simple split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📈 Training on {len(X_train)} samples, validating on {len(X_val)}")
    
    # 4. Train
    regressor = EssayScoreRegressor()
    start_time = time.time()
    metrics = regressor.train(
        X_train, y_train, 
        X_val, y_val, 
        essay_set_id=1,
        n_estimators=50, # Fast training
        max_depth=3
    )
    elapsed = time.time() - start_time
    
    print(f"\n✨ Smoke Test Complete!")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"📊 QWK: {metrics['qwk']:.4f}")
    print(f"📊 MAE: {metrics['mae']:.4f}")
    print(f"📊 RMSE: {metrics['rmse']:.4f}")
    
    return metrics, elapsed

if __name__ == "__main__":
    try:
        metrics, elapsed = run_smoke_test()
        # Output results to a temporary file for the agent to read and log to Obsidian
        with open("smoke_test_results.txt", "w") as f:
            f.write(f"QWK: {metrics['qwk']:.4f}\n")
            f.write(f"MAE: {metrics['mae']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"Time: {elapsed:.1f}s\n")
            f.write(f"Status: PASS\n")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        with open("smoke_test_results.txt", "w") as f:
            f.write(f"Status: FAIL\n")
            f.write(f"Error: {str(e)}\n")
