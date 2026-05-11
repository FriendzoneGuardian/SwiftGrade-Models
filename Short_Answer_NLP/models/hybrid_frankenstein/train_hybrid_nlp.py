"""
Run 8 — DeBERTa-v3-Small + spaCy Linguistic Features (Hybrid Frankenstein)
Architecture: DeBERTa (Semantic) + spaCy (Structural) Joint Fusion
Config: 10 epochs, 3 patience, stratified 80/10/10 split
Goal: Break 0.50 QWK for Advisor Review
"""

import os, sys, time, json, logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path for features and metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from feature_extractor import EssayFeatureExtractor
from metrics import compute_essay_metrics

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME       = "DeBERTa-v3-Small + spaCy (Frankenstein)"
TRANSFORMER_NAME = "microsoft/deberta-v3-small"
ESSAY_SET        = 1
MAX_LEN          = 512
BATCH_SIZE       = 8           # Reduced for hybrid memory overhead
EPOCHS           = 10
PATIENCE         = 3
LR               = 2e-5
WARMUP_RATIO     = 0.1
TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
TEST_RATIO       = 0.10
RANDOM_SEED      = 42
SCORE_MIN        = 2.0
SCORE_MAX        = 12.0
SCORE_RANGE      = SCORE_MAX - SCORE_MIN

# Loss weights
MSE_WEIGHT       = 0.7
RANK_WEIGHT      = 0.3

# ─── Hybrid Model ─────────────────────────────────────────────────────────────

class SwiftGradeHybridScorer(nn.Module):
    def __init__(self, transformer_name: str, num_ling_features: int):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.deberta.config.hidden_size # 768 for small
        
        # Joint Regression Head
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size + num_ling_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Normalized [0, 1] output
        )
        
    def forward(self, input_ids, attention_mask, ling_features):
        # 1. Semantic Features (Transformer)
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Handle both BaseModelOutput and plain tuple (from TorchScript)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state
            
        # Use [CLS] token (index 0)
        cls_output = last_hidden_state[:, 0, :]
        
        # 2. Structural Features (spaCy)
        # Already provided in forward call
        
        # 3. Frankenstein Stitching
        combined = torch.cat((cls_output, ling_features), dim=1)
        
        # 4. Predict
        logits = self.fusion_head(combined)
        return torch.sigmoid(logits).squeeze(-1)

# ─── Combined Loss ────────────────────────────────────────────────────────────

class HybridFrankenLoss(nn.Module):
    def __init__(self, mse_weight=0.7, rank_weight=0.3, margin=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.margin = margin
        self.mse = nn.MSELoss()
        
    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)
        
        # Correct Pairwise Ranking for batch size N
        # Preds: (N,) -> (N, N) matrix of all pairs
        n = preds.size(0)
        s1 = preds.repeat(n, 1)
        s2 = s1.t()
        
        t1 = targets.repeat(n, 1)
        t2 = t1.t()
        
        # target_diff is 1 if t1 > t2, -1 if t1 < t2, 0 if equal
        target_diff = (t1 - t2).sign()
        
        # Flatten all pairs for margin_ranking_loss
        # Only rank pairs with different scores to avoid noise
        mask = (target_diff != 0).view(-1)
        
        if mask.any():
            rank_loss = F.margin_ranking_loss(
                s1.reshape(-1)[mask], 
                s2.reshape(-1)[mask], 
                target_diff.reshape(-1)[mask], 
                margin=self.margin
            )
        else:
            rank_loss = torch.tensor(0.0, device=preds.device)
            
        return self.mse_weight * mse_loss + self.rank_weight * rank_loss

# ─── Data & Training ──────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, essays, features, labels, tokenizer, max_len):
        self.essays = essays
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.essays)
        
    def __getitem__(self, idx):
        essay = str(self.essays[idx])
        encoding = self.tokenizer(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'ling_features': self.features[idx],
            'labels': self.labels[idx]
        }

def run_hybrid_training():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Use absolute paths
    data_path = Path("/root/projects/SwiftGrade-Models/Unified_Datasets/ASAP-AES/training_set_rel3.xlsx")
    outputs_dir = Path("/root/projects/SwiftGrade-Models/Short_Answer_NLP/outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    df = pd.read_excel(data_path)
    set1 = df[df['essay_set'] == ESSAY_SET].copy()
    set1 = set1.rename(columns={'domain1_score': 'score'}).dropna(subset=['score'])
    
    # Normalize scores [0, 1]
    set1['norm_score'] = (set1['score'] - SCORE_MIN) / SCORE_RANGE
    
    # 2. Pre-extract Linguistic Features (Fast marathon)
    logger.info("Extracting spaCy linguistic features...")
    extractor = EssayFeatureExtractor()
    feat_df = extractor.extract_batch(set1['essay'].tolist())
    # Normalize features (Crucial for deep learning concatenation)
    feat_norm = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-8)
    ling_features = feat_norm.values
    
    # 3. Splits
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(set1))
    train_idx = indices[:int(len(set1)*TRAIN_RATIO)]
    val_idx = indices[int(len(set1)*TRAIN_RATIO):int(len(set1)*(TRAIN_RATIO+VAL_RATIO))]
    test_idx = indices[int(len(set1)*(TRAIN_RATIO+VAL_RATIO)):]
    
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    
    train_ds = HybridDataset(set1.iloc[train_idx]['essay'].tolist(), ling_features[train_idx], set1.iloc[train_idx]['norm_score'].tolist(), tokenizer, MAX_LEN)
    val_ds = HybridDataset(set1.iloc[val_idx]['essay'].tolist(), ling_features[val_idx], set1.iloc[val_idx]['norm_score'].tolist(), tokenizer, MAX_LEN)
    test_ds = HybridDataset(set1.iloc[test_idx]['essay'].tolist(), ling_features[test_idx], set1.iloc[test_idx]['norm_score'].tolist(), tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 4. Model & Trainer
    model = SwiftGradeHybridScorer(TRANSFORMER_NAME, ling_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = HybridFrankenLoss(mse_weight=MSE_WEIGHT, rank_weight=RANK_WEIGHT)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*WARMUP_RATIO), num_training_steps=total_steps)
    
    # 5. Loop
    best_qwk = -1
    best_model_path = "/root/projects/SwiftGrade-Models/Short_Answer_NLP/models/hybrid_frankenstein/hybrid_champion.pth"
    
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            feats = batch['ling_features'].to(device)
            labels = batch['labels'].to(device)
            
            preds = model(ids, mask, feats)
            loss = criterion(preds, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            
        # Eval
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                feats = batch['ling_features'].to(device)
                preds = model(ids, mask, feats)
                
                # Denormalize
                val_preds.extend((preds * SCORE_RANGE + SCORE_MIN).cpu().numpy())
                val_labels.extend((batch['labels'] * SCORE_RANGE + SCORE_MIN).cpu().numpy())
        
        metrics = compute_essay_metrics(np.array(val_labels), np.array(val_preds), n_classes=int(SCORE_MAX)+1)
        qwk = metrics['qwk']
        logger.info(f"Epoch {epoch+1} | Loss: {np.mean(train_losses):.4f} | QWK: {qwk:.4f}")
        
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🏆 New Best QWK: {best_qwk:.4f}")

    # Final Test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            feats = batch['ling_features'].to(device)
            preds = model(ids, mask, feats)
            test_preds.extend((preds * SCORE_RANGE + SCORE_MIN).cpu().numpy())
            test_labels.extend((batch['labels'] * SCORE_RANGE + SCORE_MIN).cpu().numpy())
            
    final_metrics = compute_essay_metrics(np.array(test_labels), np.array(test_preds), n_classes=int(SCORE_MAX)+1)
    logger.info(f"🏁 FINAL TEST QWK: {final_metrics['qwk']:.4f}")
    
    # Save results
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": MODEL_NAME,
        "val_qwk": best_qwk,
        "test_qwk": final_metrics['qwk'],
        "test_pearson": final_metrics['pearson_r']
    }
    with open(outputs_dir / "hybrid_train_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_hybrid_training()
