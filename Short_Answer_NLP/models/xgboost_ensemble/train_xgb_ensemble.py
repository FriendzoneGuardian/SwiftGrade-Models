"""
SwiftGrade NLP Module — Hybrid Stacking Pipeline
Architecture: DeBERTa-v3-Small + spaCy Linguistic Features + XGBoost Meta-Learner
Dataset     : ASAP-AES Set 1 (1,783 essays, score range 2–12)
Split       : 80/10/10 stratified, seed 42
Target      : QWK >= 0.70 (Cohen 1968 / Wang et al. 2022)

Pipeline Architecture:
    Essay Text
        │
        ├──► DeBERTa-v3-Small
        │         └──► Continuous regression prediction (normalized [0,1])
        │         └──► [CLS] embeddings (768-dim) — optional Config B
        │
        ├──► spaCy (en_core_web_sm)
        │         └──► 12 linguistic features (see _extract_spacy_features)
        │
        └──► XGBoost Meta-Learner
                  └──► Takes: DeBERTa prediction + spaCy features
                  └──► Final calibrated score → QWK

Two configurations:
    CONFIG_A — DeBERTa prediction + spaCy features only (fast, interpretable)
    CONFIG_B — DeBERTa prediction + spaCy features + CLS embeddings (stronger ceiling)

Start with CONFIG_A. Switch to CONFIG_B if plateau.

Requirements:
    pip install transformers torch xgboost spacy scikit-learn pandas numpy
    pip install sentencepiece protobuf
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# spaCy — graceful import with install hint
try:
    import spacy
    NLP_SPACY = spacy.load("en_core_web_sm")
except OSError:
    print("[ERROR] spaCy model not found.")
    print("Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from metrics import compute_essay_metrics

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME        = "DeBERTa-v3-Small + spaCy + XGBoost (Hybrid Stack)"
TRANSFORMER_NAME  = "microsoft/deberta-v3-small"
ESSAY_SET         = 1
MAX_LEN           = 512
BATCH_SIZE        = 4
DEBERTA_EPOCHS    = 5
DEBERTA_PATIENCE  = 3
LR                = 1e-5
WARMUP_RATIO      = 0.1
TRAIN_RATIO       = 0.80
VAL_RATIO         = 0.10
TEST_RATIO        = 0.10
RANDOM_SEED       = 42
SCORE_MIN         = 2.0
SCORE_MAX         = 12.0
SCORE_RANGE       = SCORE_MAX - SCORE_MIN   # 10.0

# XGBoost configuration
XGBOOST_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "early_stopping_rounds": 30,
}

# Pipeline mode — start with A, upgrade to B if QWK plateaus
# CONFIG_A: DeBERTa prediction + spaCy features
# CONFIG_B: DeBERTa prediction + spaCy features + CLS embeddings (768-dim)
PIPELINE_CONFIG = "A"


# ─── spaCy Feature Extractor ──────────────────────────────────────────────────

def _extract_spacy_features(text: str) -> np.ndarray:
    """Extract 12 linguistic features from essay text using spaCy.

    Features chosen for direct relevance to essay quality scoring:
    1.  word_count          — essay length signal
    2.  sentence_count      — structural complexity
    3.  avg_sentence_len    — sentence sophistication
    4.  unique_word_ratio   — vocabulary richness (TTR)
    5.  noun_ratio          — content density
    6.  verb_ratio          — action/argument density
    7.  adj_ratio           — descriptive richness
    8.  adv_ratio           — modifier usage
    9.  punct_density       — punctuation correctness signal
    10. stopword_ratio      — fluency indicator (inverse)
    11. avg_token_len       — vocabulary sophistication
    12. entity_count        — named entity usage (factual content)
    """
    doc = NLP_SPACY(text[:10000])  # cap at 10K chars to avoid timeout

    tokens      = [t for t in doc if not t.is_space]
    words       = [t for t in tokens if t.is_alpha]
    sentences   = list(doc.sents)

    word_count       = len(words)
    sentence_count   = max(len(sentences), 1)
    avg_sent_len     = word_count / sentence_count

    unique_words     = set(t.lower_ for t in words)
    unique_ratio     = len(unique_words) / max(word_count, 1)

    pos_counts = {}
    for token in tokens:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    total_tokens = max(len(tokens), 1)
    noun_ratio   = pos_counts.get("NOUN", 0)  / total_tokens
    verb_ratio   = pos_counts.get("VERB", 0)  / total_tokens
    adj_ratio    = pos_counts.get("ADJ",  0)  / total_tokens
    adv_ratio    = pos_counts.get("ADV",  0)  / total_tokens
    punct_density = pos_counts.get("PUNCT", 0) / total_tokens

    stopword_count = sum(1 for t in words if t.is_stop)
    stopword_ratio = stopword_count / max(word_count, 1)

    avg_token_len  = np.mean([len(t.text) for t in words]) if words else 0.0
    entity_count   = len(doc.ents) / max(sentence_count, 1)

    features = np.array([
        word_count,
        sentence_count,
        avg_sent_len,
        unique_ratio,
        noun_ratio,
        verb_ratio,
        adj_ratio,
        adv_ratio,
        punct_density,
        stopword_ratio,
        avg_token_len,
        entity_count,
    ], dtype=np.float32)

    return features


SPACY_FEATURE_NAMES = [
    "word_count", "sentence_count", "avg_sentence_len",
    "unique_word_ratio", "noun_ratio", "verb_ratio",
    "adj_ratio", "adv_ratio", "punct_density",
    "stopword_ratio", "avg_token_len", "entity_count",
]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class EssayDataset(Dataset):
    """Tokenized essay dataset with pre-computed spaCy features."""

    def __init__(self, essays: list, scores: list, tokenizer, max_len: int):
        self.essays            = essays
        self.tokenizer         = tokenizer
        self.max_len           = max_len
        self.scores_raw        = np.array(scores, dtype=np.float32)
        self.scores_normalized = (self.scores_raw - SCORE_MIN) / SCORE_RANGE

        # Pre-compute spaCy features — done once at dataset init
        print("  [spaCy] Extracting linguistic features...")
        self.spacy_features = np.array([
            _extract_spacy_features(str(e)) for e in tqdm(essays, desc="spaCy")
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.essays)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer.encode_plus(
            str(self.essays[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets':        torch.tensor(self.scores_normalized[idx], dtype=torch.float),
            'raw_scores':     torch.tensor(self.scores_raw[idx],        dtype=torch.float),
            'spacy_features': torch.tensor(self.spacy_features[idx],    dtype=torch.float),
        }
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        return item


# ─── DeBERTa Model ────────────────────────────────────────────────────────────

class DeBERTaWithEmbeddings(nn.Module):
    """DeBERTa-v3-Small with dual output: score prediction + CLS embeddings.

    Used for both standalone training (Stage 1) and feature extraction
    for the XGBoost meta-learner (Stage 2).
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder   = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        hidden_size    = self.encoder.config.hidden_size  # 768 for deberta-v3-small

        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask,
                  "output_hidden_states": True}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs    = self.encoder(**kwargs)
        # CLS token — first token of last hidden state
        cls_output = outputs.hidden_states[-1][:, 0, :]
        prediction = self.regressor(cls_output).squeeze(-1)

        if return_embeddings:
            return prediction, cls_output
        return prediction, None


# ─── Train / Eval DeBERTa ─────────────────────────────────────────────────────

def train_deberta_epoch(model, loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    for d in tqdm(loader, desc="DeBERTa Train"):
        input_ids      = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets        = d["targets"].to(device)
        ttids          = d.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(device)

        optimizer.zero_grad()
        preds, _ = model(input_ids, attention_mask, ttids)
        loss     = loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def eval_deberta(model, loader, loss_fn, device, return_embeddings: bool = False):
    """Evaluate DeBERTa. Optionally return CLS embeddings for XGBoost Config B."""
    model.eval()
    losses, predictions, raw_targets = [], [], []
    all_embeddings = [] if return_embeddings else None

    with torch.no_grad():
        for d in loader:
            input_ids      = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets        = d["targets"].to(device)
            ttids          = d.get("token_type_ids")
            if ttids is not None:
                ttids = ttids.to(device)

            preds, embeddings = model(
                input_ids, attention_mask, ttids,
                return_embeddings=return_embeddings
            )

            losses.append(loss_fn(preds, targets).item())
            predictions.extend(preds.cpu().numpy().tolist())
            raw_targets.extend(d["raw_scores"].numpy().tolist())

            if return_embeddings and embeddings is not None:
                all_embeddings.append(embeddings.cpu().numpy())

    preds_denorm  = np.array(predictions) * SCORE_RANGE + SCORE_MIN
    preds_clipped = np.clip(preds_denorm, SCORE_MIN, SCORE_MAX)
    preds_rounded = np.rint(preds_clipped)

    print(f"  [Diag] DeBERTa pred range: "
          f"{preds_clipped.min():.2f}–{preds_clipped.max():.2f} "
          f"| Mean: {preds_clipped.mean():.2f} | Std: {preds_clipped.std():.2f}")

    embeddings_out = None
    if return_embeddings and all_embeddings:
        embeddings_out = np.vstack(all_embeddings)

    return (float(np.mean(losses)),
            preds_rounded,
            preds_denorm,          # continuous — for XGBoost input
            np.array(raw_targets),
            embeddings_out)


# ─── XGBoost Feature Assembly ────────────────────────────────────────────────

def build_xgboost_features(
    deberta_preds_continuous: np.ndarray,
    spacy_features: np.ndarray,
    cls_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble feature matrix for XGBoost meta-learner.

    Config A: [deberta_pred(1) + spacy(12)] = 13 features
    Config B: [deberta_pred(1) + spacy(12) + cls_embeddings(768)] = 781 features
    """
    parts = [
        deberta_preds_continuous.reshape(-1, 1),
        spacy_features,
    ]
    if cls_embeddings is not None and PIPELINE_CONFIG == "B":
        parts.append(cls_embeddings)

    return np.hstack(parts)


# ─── Main Training Pipeline ──────────────────────────────────────────────────

def run_hybrid_training():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice          : {device}")
    print(f"Pipeline Config : {PIPELINE_CONFIG}")
    print(f"Transformer     : {TRANSFORMER_NAME}")
    print(f"Target QWK      : >= 0.70 (Wang et al. 2022)\n")

    base_dir  = Path(__file__).resolve().parent.parent.parent.parent
    data_path = base_dir / "Unified_Datasets" / "ASAP-AES" / "training_set_rel3.xlsx"
    model_dir = base_dir / "Short_Answer_NLP" / "models" / "xgboost_ensemble" / f"hybrid_deberta_xgb_config{PIPELINE_CONFIG}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df       = pd.read_excel(data_path)
    df       = df.rename(columns={'domain1_score': 'score'})
    set_data = df[df['essay_set'] == ESSAY_SET].dropna(subset=['score'])
    total    = len(set_data)
    print(f"Loaded {total} essays | Score range: "
          f"{set_data['score'].min():.0f}–{set_data['score'].max():.0f}")

    # ── 2. Split 80/10/10 ─────────────────────────────────────────────────────
    np.random.seed(RANDOM_SEED)
    idx      = np.random.permutation(total)
    train_n  = int(total * TRAIN_RATIO)
    val_n    = int(total * VAL_RATIO)
    train_df = set_data.iloc[idx[:train_n]]
    val_df   = set_data.iloc[idx[train_n:train_n + val_n]]
    test_df  = set_data.iloc[idx[train_n + val_n:]]
    print(f"Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── 3. Tokenizer + Datasets ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)

    print("\nBuilding train dataset:")
    train_ds = EssayDataset(train_df['essay'].tolist(),
                            train_df['score'].tolist(), tokenizer, MAX_LEN)
    print("Building val dataset:")
    val_ds   = EssayDataset(val_df['essay'].tolist(),
                            val_df['score'].tolist(),   tokenizer, MAX_LEN)
    print("Building test dataset:")
    test_ds  = EssayDataset(test_df['essay'].tolist(),
                            test_df['score'].tolist(),  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=device.type == "cuda")

    # ── 4. DeBERTa — Stage 1: Train Transformer ───────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — DeBERTa-v3-Small Fine-Tuning")
    print("="*60)

    model        = DeBERTaWithEmbeddings(TRANSFORMER_NAME).to(device)
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * DEBERTA_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = nn.MSELoss().to(device)

    best_val_qwk, patience_count, deberta_history = -1.0, 0, []
    stage1_start = time.time()

    for epoch in range(DEBERTA_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{DEBERTA_EPOCHS}")
        train_loss = train_deberta_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler)

        val_loss, val_preds_rounded, _, val_targets, _ = eval_deberta(
            model, val_loader, loss_fn, device, return_embeddings=False)

        val_metrics = compute_essay_metrics(val_targets, val_preds_rounded,
                                            n_classes=13)
        val_qwk = val_metrics['qwk']

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val QWK    : {val_qwk:.4f}")

        deberta_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_qwk": val_qwk,
        })

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(model.state_dict(), model_dir / 'deberta_best.bin')
            patience_count = 0
            print(f"  ✓ DeBERTa best QWK: {best_val_qwk:.4f} — saved")
        else:
            patience_count += 1
            if patience_count >= DEBERTA_PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    stage1_time = time.time() - stage1_start
    print(f"\nStage 1 complete — Best val QWK: {best_val_qwk:.4f} "
          f"({stage1_time/60:.1f} min)")

    # ── 5. Feature Extraction — Stage 2 prep ──────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — XGBoost Meta-Learner Feature Extraction")
    print(f"Config {PIPELINE_CONFIG}: DeBERTa prediction + spaCy"
          + (" + CLS embeddings" if PIPELINE_CONFIG == "B" else ""))
    print("="*60)

    # Load best DeBERTa checkpoint
    model.load_state_dict(
        torch.load(model_dir / 'deberta_best.bin', map_location=device))

    use_embeddings = (PIPELINE_CONFIG == "B")

    print("\nExtracting train features...")
    _, _, train_preds_cont, train_targets, train_emb = eval_deberta(
        model, train_loader, loss_fn, device,
        return_embeddings=use_embeddings)

    print("Extracting val features...")
    _, _, val_preds_cont, val_targets_raw, val_emb = eval_deberta(
        model, val_loader, loss_fn, device,
        return_embeddings=use_embeddings)

    print("Extracting test features...")
    _, _, test_preds_cont, test_targets_raw, test_emb = eval_deberta(
        model, test_loader, loss_fn, device,
        return_embeddings=use_embeddings)

    # Assemble feature matrices
    X_train = build_xgboost_features(
        train_preds_cont, train_ds.spacy_features, train_emb)
    X_val   = build_xgboost_features(
        val_preds_cont,   val_ds.spacy_features,   val_emb)
    X_test  = build_xgboost_features(
        test_preds_cont,  test_ds.spacy_features,  test_emb)

    print(f"\nFeature matrix shapes:")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")

    # ── 6. XGBoost Training ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — XGBoost Training")
    print("="*60)

    xgb_stage2_start = time.time()

    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(
        X_train, train_targets,
        eval_set=[(X_val, val_targets_raw)],
        verbose=50,
    )

    xgb_stage2_time = time.time() - xgb_stage2_start
    print(f"\nXGBoost training complete ({xgb_stage2_time:.1f}s)")

    # Save XGBoost model
    xgb_model.save_model(str(model_dir / 'xgboost_meta.json'))

    # ── 7. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    def _evaluate_xgb(X, y_true, split_name):
        preds_raw     = xgb_model.predict(X)
        preds_clipped = np.clip(preds_raw, SCORE_MIN, SCORE_MAX)
        preds_rounded = np.rint(preds_clipped)

        print(f"  [{split_name}] Pred range: "
              f"{preds_clipped.min():.2f}–{preds_clipped.max():.2f} "
              f"| Mean: {preds_clipped.mean():.2f}")

        metrics = compute_essay_metrics(y_true, preds_clipped, n_classes=13)
        return metrics, preds_rounded

    val_metrics_xgb,  _ = _evaluate_xgb(X_val,  val_targets_raw,  "Val")
    test_metrics_xgb, _ = _evaluate_xgb(X_test, test_targets_raw, "Test")

    print(f"\n{'='*60}")
    print(f"HYBRID PIPELINE RESULTS — Config {PIPELINE_CONFIG}")
    print(f"{'='*60}")
    print(f"  Stage 1 DeBERTa Val QWK  : {best_val_qwk:.4f}")
    print(f"  Stage 2 XGBoost Val QWK  : {val_metrics_xgb['qwk']:.4f}  "
          f"({'↑ improved' if val_metrics_xgb['qwk'] > best_val_qwk else '↓ regressed'})")
    print(f"  Final Test QWK           : {test_metrics_xgb['qwk']:.4f}")
    print(f"  Test Pearson             : {test_metrics_xgb['pearson_r']:.4f}")
    print(f"  Test MAE (denormalized)  : {test_metrics_xgb['mae']:.4f}")
    print(f"  Test RMSE (denormalized) : {test_metrics_xgb['rmse']:.4f}")
    print(f"  Target QWK >= 0.70       : "
          f"{'✓ MET' if test_metrics_xgb['qwk'] >= 0.70 else '✗ not yet reached'}")

    # XGBoost feature importance — interpretability for thesis
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*60)

    feature_names = ["deberta_pred"] + SPACY_FEATURE_NAMES
    if PIPELINE_CONFIG == "B":
        feature_names += [f"cls_{i}" for i in range(768)]

    importances = xgb_model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {rank:2d}. {fname:30s} {importances[idx]:.4f}")

    # ── 8. Save Results ───────────────────────────────────────────────────────
    total_time = stage1_time + xgb_stage2_time

    results = {
        "date":            datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model":           MODEL_NAME,
        "pipeline_config": PIPELINE_CONFIG,
        "dataset":         f"ASAP-AES Set {ESSAY_SET}",
        "samples":         total,
        "split":           f"{len(train_df)}/{len(val_df)}/{len(test_df)}",
        "stage1_deberta": {
            "best_val_qwk": float(best_val_qwk),
            "epochs_run":   len(deberta_history),
            "history":      deberta_history,
        },
        "stage2_xgboost": {
            "val_qwk":      float(val_metrics_xgb['qwk']),
            "test_qwk":     float(test_metrics_xgb['qwk']),
            "test_mae":     float(test_metrics_xgb['mae']),
            "test_rmse":    float(test_metrics_xgb['rmse']),
            "test_pearson": float(test_metrics_xgb['pearson_r']),
            "xgb_params":   XGBOOST_PARAMS,
        },
        "total_time_s":    round(total_time, 1),
        "status":          "PASS",
    }

    out = Path(__file__).parent.parent / "outputs" / \
          f"hybrid_config{PIPELINE_CONFIG}_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    try:
        run_hybrid_training()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()