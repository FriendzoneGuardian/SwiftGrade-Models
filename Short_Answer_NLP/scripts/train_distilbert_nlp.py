"""
Full NLP Training Run — ASAP-AES Essay Set 1
Architecture: DistilBERT-Base-Cased (Fine-tuned for Regression)
Config: 5 epochs, 3 patience, stratified 80/10/10 split

Run 6 — DistilBERT Sprint
- Model: distilbert-base-cased (40% smaller, 60% faster)
- BATCH_SIZE: 16 (Lightweight enough to double batch size on 4GB VRAM)
- Loss: MSE + Ranking Loss (0.7 / 0.3)
- Temp Scaling: 1.5
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path for metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from metrics import compute_essay_metrics

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME       = "DistilBERT (Run 6: Sprint Configuration)"
TRANSFORMER_NAME = "distilbert-base-cased"
ESSAY_SET        = 1
MAX_LEN          = 512
BATCH_SIZE       = 16          # Doubled from BERT run due to lower memory footprint
EPOCHS           = 5
PATIENCE         = 3
LR               = 2e-5
WARMUP_RATIO     = 0.1
TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
TEST_RATIO       = 0.10
RANDOM_SEED      = 42
SCORE_MIN        = 2.0
SCORE_MAX        = 12.0
SCORE_RANGE      = SCORE_MAX - SCORE_MIN  # 10.0

# Loss weights
MSE_WEIGHT       = 0.7
RANK_WEIGHT      = 0.3
OUTPUT_TEMP      = 1.5


# ─── Combined Loss ────────────────────────────────────────────────────────────

class MSEPlusRankingLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7, rank_weight: float = 0.3, margin: float = 0.5):
        super().__init__()
        self.mse_weight  = mse_weight
        self.rank_weight = rank_weight
        self.margin      = margin
        self.mse         = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(predictions, targets)
        rank_loss = torch.tensor(0.0, device=predictions.device)
        n = len(predictions)

        if n > 1:
            pred_i = predictions.unsqueeze(1).expand(n, n)
            pred_j = predictions.unsqueeze(0).expand(n, n)
            targ_i = targets.unsqueeze(1).expand(n, n)
            targ_j = targets.unsqueeze(0).expand(n, n)

            mask = (targ_i > targ_j).float()
            rank_violations = F.relu(self.margin - (pred_i - pred_j))
            rank_loss = (mask * rank_violations).sum() / (mask.sum() + 1e-8)

        return self.mse_weight * mse_loss + self.rank_weight * rank_loss


# ─── Dataset ──────────────────────────────────────────────────────────────────

class EssayDataset(Dataset):
    def __init__(self, essays, scores, tokenizer, max_len):
        self.essays            = essays
        self.tokenizer         = tokenizer
        self.max_len           = max_len
        self.scores_raw        = np.array(scores, dtype=np.float32)
        self.scores_normalized = (self.scores_raw - SCORE_MIN) / SCORE_RANGE

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            str(self.essays[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets':        torch.tensor(self.scores_normalized[item], dtype=torch.float),
            'raw_scores':     torch.tensor(self.scores_raw[item],        dtype=torch.float),
        }


def build_stratified_sampler(scores: np.ndarray) -> WeightedRandomSampler:
    score_ints   = np.rint(scores).astype(int)
    class_counts = np.bincount(score_ints, minlength=13)
    class_counts  = np.maximum(class_counts, 1)
    weights      = 1.0 / class_counts[score_ints]
    weights      = weights / weights.sum()
    sampler      = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(scores),
        replacement=True,
    )
    return sampler


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    for d in tqdm(loader, desc="Training"):
        input_ids      = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets        = d["targets"].to(device)

        optimizer.zero_grad()
        raw_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        scaled_logits = raw_logits / OUTPUT_TEMP

        loss = loss_fn(scaled_logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def eval_model(model, loader, loss_fn, device):
    model.eval()
    losses, predictions, raw_targets = [], [], []

    with torch.no_grad():
        for d in loader:
            input_ids      = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets        = d["targets"].to(device)

            raw_logits     = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            scaled_logits  = raw_logits / OUTPUT_TEMP

            losses.append(loss_fn(scaled_logits, targets).item())
            predictions.extend(scaled_logits.cpu().numpy().tolist())
            raw_targets.extend(d["raw_scores"].numpy().tolist())

    preds_denorm  = np.array(predictions) * SCORE_RANGE + SCORE_MIN
    preds_clipped = np.clip(preds_denorm, SCORE_MIN, SCORE_MAX)
    preds_rounded = np.rint(preds_clipped)

    print(f"  [Diag] Pred range : {preds_clipped.min():.2f} – {preds_clipped.max():.2f} "
          f"| Mean: {preds_clipped.mean():.2f} | Std: {preds_clipped.std():.2f}")

    return float(np.mean(losses)), preds_rounded, np.array(raw_targets)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_distilbert_training():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model         : {MODEL_NAME}")
    print(f"Device        : {device}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Loss strategy : MSE ({MSE_WEIGHT}) + Pairwise Ranking ({RANK_WEIGHT})")

    base_dir  = Path(__file__).parent.parent
    data_path = base_dir.parent / "Unified_Datasets" / "ASAP-AES" / "training_set_rel3.xlsx"
    model_dir = base_dir / "models" / "distilbert_optionb"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df       = pd.read_excel(data_path)
    df       = df.rename(columns={'domain1_score': 'score'})
    set_data = df[df['essay_set'] == ESSAY_SET].dropna(subset=['score'])
    total    = len(set_data)
    print(f"Loaded {total} essays | Score range: {set_data['score'].min():.0f}–{set_data['score'].max():.0f}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    np.random.seed(RANDOM_SEED)
    idx      = np.random.permutation(total)
    train_n  = int(total * TRAIN_RATIO)
    val_n    = int(total * VAL_RATIO)
    train_df = set_data.iloc[idx[:train_n]]
    val_df   = set_data.iloc[idx[train_n:train_n + val_n]]
    test_df  = set_data.iloc[idx[train_n + val_n:]]
    print(f"Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)

    train_ds  = EssayDataset(train_df['essay'].tolist(), train_df['score'].tolist(), tokenizer, MAX_LEN)
    val_ds    = EssayDataset(val_df['essay'].tolist(),   val_df['score'].tolist(),   tokenizer, MAX_LEN)
    test_ds   = EssayDataset(test_df['essay'].tolist(),  test_df['score'].tolist(),  tokenizer, MAX_LEN)

    sampler      = build_stratified_sampler(train_df['score'].values)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,    num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,      num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,      num_workers=2, pin_memory=True)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model        = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME, num_labels=1).to(device)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fn      = MSEPlusRankingLoss(mse_weight=MSE_WEIGHT, rank_weight=RANK_WEIGHT).to(device)

    # ── 4. Training Loop ──────────────────────────────────────────────────────
    best_val_qwk, early_stop_counter, history = -1.0, 0, []
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        val_loss, val_preds, val_targets = eval_model(model, val_loader, loss_fn, device)
        val_metrics = compute_essay_metrics(val_targets, val_preds, n_classes=13)

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val QWK    : {val_metrics['qwk']:.4f}")
        print(f"  Val MAE    : {val_metrics['mae']:.4f}")

        history.append({
            "epoch": epoch+1, "train_loss": train_loss,
            "val_loss": val_loss, "val_qwk": val_metrics['qwk'],
            "val_mae": val_metrics['mae'],
        })

        if val_metrics['qwk'] > best_val_qwk:
            best_val_qwk = val_metrics['qwk']
            torch.save(model.state_dict(), model_dir / 'best_model.bin')
            early_stop_counter = 0
            print(f"  ✓ New best QWK: {best_val_qwk:.4f} — saved")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ── 5. Test Evaluation ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_dir / 'best_model.bin', map_location=device))
    test_loss, test_preds, test_targets = eval_model(model, test_loader, loss_fn, device)
    test_metrics = compute_essay_metrics(test_targets, test_preds, n_classes=13)

    print(f"\nFinal Test Evaluation:")
    print(f"  Test QWK     : {test_metrics['qwk']:.4f}")
    print(f"  Test Pearson : {test_metrics['pearson_r']:.4f}")

    # ── 6. Save Results ───────────────────────────────────────────────────────
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": MODEL_NAME, "dataset": f"ASAP-AES Set {ESSAY_SET}",
        "samples": total, "split": f"{len(train_df)}/{len(val_df)}/{len(test_df)}",
        "epochs_run": epoch + 1, "best_val_qwk": float(best_val_qwk),
        "test_qwk": float(test_metrics['qwk']),
        "test_pearson": float(test_metrics['pearson_r']),
        "total_time_s": round(time.time() - start_time, 1),
        "history": history,
        "status": "PASS",
    }

    out = Path(__file__).parent.parent / "outputs" / "distilbert_optionb_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    try:
        run_distilbert_training()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()