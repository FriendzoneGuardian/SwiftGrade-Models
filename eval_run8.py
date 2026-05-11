import os
import sys
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Short_Answer_NLP', 'models', 'hybrid_frankenstein')))
from train_hybrid_nlp import SwiftGradeHybridScorer, HybridDataset, SCORE_MIN, SCORE_RANGE, MAX_LEN, BATCH_SIZE, ESSAY_SET, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, SCORE_MAX

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Short_Answer_NLP', 'src')))
from feature_extractor import EssayFeatureExtractor
from metrics import compute_essay_metrics

def eval_run8():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = Path("/root/projects/SwiftGrade-Models/Unified_Datasets/ASAP-AES/training_set_rel3.xlsx")
    df = pd.read_excel(data_path)
    set1 = df[df['essay_set'] == ESSAY_SET].copy()
    set1 = set1.rename(columns={'domain1_score': 'score'}).dropna(subset=['score'])
    set1['norm_score'] = (set1['score'] - SCORE_MIN) / SCORE_RANGE
    
    print("Extracting features...")
    extractor = EssayFeatureExtractor()
    feat_df = extractor.extract_batch(set1['essay'].tolist())
    feat_norm = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-8)
    ling_features = feat_norm.values
    
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(set1))
    test_idx = indices[int(len(set1)*(TRAIN_RATIO+VAL_RATIO)):]
    val_idx = indices[int(len(set1)*TRAIN_RATIO):int(len(set1)*(TRAIN_RATIO+VAL_RATIO))]
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    
    # We will test on Validation and Test sets
    val_ds = HybridDataset(set1.iloc[val_idx]['essay'].tolist(), ling_features[val_idx], set1.iloc[val_idx]['norm_score'].tolist(), tokenizer, MAX_LEN)
    test_ds = HybridDataset(set1.iloc[test_idx]['essay'].tolist(), ling_features[test_idx], set1.iloc[test_idx]['norm_score'].tolist(), tokenizer, MAX_LEN)
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = SwiftGradeHybridScorer("microsoft/deberta-v3-small", ling_features.shape[1]).to(device)
    model.load_state_dict(torch.load("/root/projects/SwiftGrade-Models/Short_Answer_NLP/models/hybrid_frankenstein/hybrid_champion.pth", map_location=device))
    model.eval()
    
    def evaluate(loader, name):
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                feats = batch['ling_features'].to(device)
                out = model(ids, mask, feats)
                preds.extend((out * SCORE_RANGE + SCORE_MIN).cpu().numpy())
                labels.extend((batch['labels'] * SCORE_RANGE + SCORE_MIN).cpu().numpy())
                
        metrics = compute_essay_metrics(np.array(labels), np.array(preds), n_classes=int(SCORE_MAX)+1)
        print(f"[{name}] TRUE QWK: {metrics['qwk']:.4f}")
        print(f"[{name}] TRUE RMSE: {metrics['rmse']:.4f}")
        return metrics['qwk']

    print("Evaluating True Performance with Fixed Metrics...")
    evaluate(val_loader, "Validation")
    evaluate(test_loader, "Test")

if __name__ == "__main__":
    eval_run8()
