"""
Full NLP Training Run — ASAP-AES Essay Set 1
Architecture: Linguistic Features (spaCy) + XGBoost Regressor
Config: 100 estimators, patience 30, stratified 70/15/15 split
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from feature_extractor import EssayFeatureExtractor
from essay_score_regressor import EssayScoreRegressor
from metrics import compute_essay_metrics

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_NAME      = "XGBoost + Linguistic Features (spaCy)"
ESSAY_SET       = 1
N_ESTIMATORS    = 100
PATIENCE        = 30
MAX_DEPTH       = 6
LEARNING_RATE   = 0.1
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
RANDOM_SEED     = 42

def run_full_training():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    base_dir  = Path(__file__).parent.parent
    data_path = base_dir.parent / "Unified_Datasets" / "ASAP-AES" / "training_set_rel3.xlsx"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  NLP FULL TRAINING — {MODEL_NAME}")
    print(f"  Essay Set {ESSAY_SET} | {N_ESTIMATORS} estimators | patience {PATIENCE}")
    print("=" * 70)
    print(f"\n📂 Loading data from {data_path}")

    df = pd.read_excel(data_path)
    df = df.rename(columns={'domain1_score': 'score'})

    set_data = df[df['essay_set'] == ESSAY_SET].copy()
    set_data = set_data.dropna(subset=['score'])
    total_essays = len(set_data)
    print(f"✅ Loaded {total_essays} essays for Set {ESSAY_SET}")
    print(f"   Score range: {set_data['score'].min():.0f} – {set_data['score'].max():.0f}")

    # ── 2. Stratified Split ──────────────────────────────────────────────────
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(total_essays)
    train_n = int(total_essays * TRAIN_RATIO)
    val_n   = int(total_essays * VAL_RATIO)

    train_data = set_data.iloc[indices[:train_n]]
    val_data   = set_data.iloc[indices[train_n:train_n + val_n]]
    test_data  = set_data.iloc[indices[train_n + val_n:]]

    print(f"📊 Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # ── 3. Feature Extraction ────────────────────────────────────────────────
    print("\n🧠 Extracting linguistic features …")
    extractor = EssayFeatureExtractor()
    t0 = time.time()

    X_train = extractor.extract_batch(train_data['essay'].tolist())
    X_val   = extractor.extract_batch(val_data['essay'].tolist())
    X_test  = extractor.extract_batch(test_data['essay'].tolist())

    y_train = train_data['score'].values
    y_val   = val_data['score'].values
    y_test  = test_data['score'].values

    feat_time = time.time() - t0
    print(f"✅ Features extracted in {feat_time:.1f}s  ({X_train.shape[1]} features)")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print(f"\n🏋️ Training XGBoost  (n_estimators={N_ESTIMATORS}, patience={PATIENCE}) …")
    regressor = EssayScoreRegressor()
    t0 = time.time()

    val_metrics = regressor.train(
        X_train, y_train,
        X_val, y_val,
        essay_set_id=ESSAY_SET,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        early_stopping_rounds=PATIENCE,
    )
    train_time = time.time() - t0
    print(f"✅ Training complete in {train_time:.1f}s")
    print(f"   Val QWK: {val_metrics['qwk']:.4f} | Val MAE: {val_metrics['mae']:.4f}")

    # ── 5. Test Evaluation ───────────────────────────────────────────────────
    print("\n🎯 Evaluating on held-out test set …")
    y_pred_test = regressor.predict(X_test, essay_set_id=ESSAY_SET)
    score_range = regressor.score_ranges[ESSAY_SET]
    test_metrics = compute_essay_metrics(
        y_test, y_pred_test,
        n_classes=int(score_range[1]) + 1
    )

    print(f"   Test QWK:      {test_metrics['qwk']:.4f}")
    print(f"   Test MAE:      {test_metrics['mae']:.4f}")
    print(f"   Test RMSE:     {test_metrics['rmse']:.4f}")
    print(f"   Test Pearson:  {test_metrics['pearson_r']:.4f}")

    # ── 6. Feature Importance ────────────────────────────────────────────────
    importance_df = regressor.get_feature_importance(ESSAY_SET, top_n=10)
    print("\n📈 Top 10 Feature Importances:")
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']:30s}  {row['importance']:.4f}")

    # ── 7. Save Model ────────────────────────────────────────────────────────
    regressor.save_model(ESSAY_SET, str(model_dir))
    print(f"\n💾 Model saved to {model_dir}/")

    # ── 8. Collect results ───────────────────────────────────────────────────
    total_time = feat_time + train_time
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": MODEL_NAME,
        "dataset": f"ASAP-AES Set {ESSAY_SET}",
        "samples": total_essays,
        "split": f"{len(train_data)}/{len(val_data)}/{len(test_data)}",
        "n_estimators": N_ESTIMATORS,
        "patience": PATIENCE,
        "features": int(X_train.shape[1]),
        "val_qwk": float(val_metrics['qwk']),
        "val_mae": float(val_metrics['mae']),
        "test_qwk": float(test_metrics['qwk']),
        "test_mae": float(test_metrics['mae']),
        "test_rmse": float(test_metrics['rmse']),
        "test_pearson": float(test_metrics['pearson_r']),
        "feat_time_s": round(feat_time, 1),
        "train_time_s": round(train_time, 1),
        "total_time_s": round(total_time, 1),
        "status": "PASS",
    }

    # Dump to JSON for the agent to pick up
    results_path = base_dir / "outputs" / "full_train_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  ✨  FULL TRAINING COMPLETE")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    try:
        results = run_full_training()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback; traceback.print_exc()
        results_path = Path(__file__).parent.parent / "outputs" / "full_train_results.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({"status": "FAIL", "error": str(e)}, f, indent=2)
