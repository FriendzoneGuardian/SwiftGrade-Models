import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fix path for metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Short_Answer_NLP', 'src')))
from metrics import compute_essay_metrics

# Re-use the Run 8 architecture logic for the hybrid one
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Short_Answer_NLP', 'models', 'hybrid_frankenstein')))
from train_hybrid_nlp import SwiftGradeHybridScorer
from feature_extractor import EssayFeatureExtractor

# Constants
SCORE_MIN = 2.0
SCORE_MAX = 12.0
SCORE_RANGE = 10.0
MAX_LEN = 512
BATCH_SIZE = 8
ESSAY_SET = 1
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

class SimpleEssayDataset(Dataset):
    def __init__(self, essays, labels, tokenizer, max_len):
        self.essays = essays
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.essays)
    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.essays[idx]), truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.float32)}

class HybridAuditDataset(Dataset):
    def __init__(self, essays, features, labels, tokenizer, max_len):
        self.essays = essays
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.essays)
    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.essays[idx]), truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'ling_features': self.features[idx], 'labels': self.labels[idx]}

def full_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path("/root/projects/SwiftGrade-Models/Unified_Datasets/ASAP-AES/training_set_rel3.xlsx")
    df = pd.read_excel(data_path)
    set1 = df[df['essay_set'] == ESSAY_SET].copy().rename(columns={'domain1_score': 'score'}).dropna(subset=['score'])
    
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(set1))
    test_idx = indices[int(len(set1)*(TRAIN_RATIO+VAL_RATIO)):]
    test_df = set1.iloc[test_idx]
    
    audit_results = {}

    runs = [
        {"id": 4, "name": "BERT-Base-Cased", "model_path": "Short_Answer_NLP/models/bert_cased/best_model.bin", "hf_name": "bert-base-cased"},
        {"id": 5, "name": "BERT (Hybrid Loss)", "model_path": "Short_Answer_NLP/models/bert_cased_optionb/best_model.bin", "hf_name": "bert-base-cased"},
        {"id": 6, "name": "DistilBERT", "model_path": "Short_Answer_NLP/models/distilbert_optionb/best_model.bin", "hf_name": "distilbert-base-cased"},
        {"id": 7, "name": "DeBERTa-v3-Small", "model_path": "Short_Answer_NLP/models/deberta_v3_small/best_model.bin", "hf_name": "microsoft/deberta-v3-small"},
    ]

    for run in runs:
        print(f"Auditing Run {run['id']} ({run['name']})...")
        tokenizer = AutoTokenizer.from_pretrained(run['hf_name'])
        ds = SimpleEssayDataset(test_df['essay'].tolist(), test_df['score'].tolist(), tokenizer, MAX_LEN)
        loader = DataLoader(ds, batch_size=BATCH_SIZE)
        
        model = AutoModelForSequenceClassification.from_pretrained(run['hf_name'], num_labels=1).to(device)
        model.load_state_dict(torch.load(run['model_path'], map_location=device))
        model.eval()
        
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                out = model(ids, attention_mask=mask).logits.squeeze(-1)
                # Note: These older models might have been trained on normalized [0,1] or raw [2,12].
                # Based on previous logs, most were trained on normalized.
                p = out.cpu().numpy()
                if p.max() <= 1.5: # Likely normalized
                    p = p * SCORE_RANGE + SCORE_MIN
                preds.extend(p)
                labels.extend(batch['labels'].numpy())
        
        metrics = compute_essay_metrics(np.array(labels), np.array(preds), n_classes=13)
        audit_results[run['id']] = metrics['qwk']
        print(f"  True QWK: {metrics['qwk']:.4f}")

    # Run 8 Audit (Hybrid)
    print("Auditing Run 8 (Hybrid Frankenstein)...")
    extractor = EssayFeatureExtractor()
    feat_df = extractor.extract_batch(set1['essay'].tolist())
    feat_norm = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-8)
    ling_features = feat_norm.values
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    ds8 = HybridAuditDataset(test_df['essay'].tolist(), ling_features[test_idx], test_df['score'].tolist(), tokenizer, MAX_LEN)
    loader8 = DataLoader(ds8, batch_size=BATCH_SIZE)
    
    model8 = SwiftGradeHybridScorer("microsoft/deberta-v3-small", ling_features.shape[1]).to(device)
    model8.load_state_dict(torch.load("Short_Answer_NLP/models/hybrid_frankenstein/hybrid_champion.pth", map_location=device))
    model8.eval()
    
    preds8, labels8 = [], []
    with torch.no_grad():
        for batch in loader8:
            out = model8(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['ling_features'].to(device))
            preds8.extend((out * SCORE_RANGE + SCORE_MIN).cpu().numpy())
            labels8.extend(batch['labels'].numpy())
    
    metrics8 = compute_essay_metrics(np.array(labels8), np.array(preds8), n_classes=13)
    audit_results[8] = metrics8['qwk']
    print(f"  True QWK: {metrics8['qwk']:.4f}")

    print("\n" + "="*40)
    print("FINAL AUDIT SUMMARY (TRUE QWK)")
    print("="*40)
    for rid, qwk in audit_results.items():
        print(f"Run {rid}: {qwk:.4f}")

if __name__ == "__main__":
    full_audit()
