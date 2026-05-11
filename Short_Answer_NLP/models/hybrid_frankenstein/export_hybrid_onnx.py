"""
Export Frankenstein Hybrid Model to ONNX (Attempt 3)
Using torchscript compatibility and forcing legacy export.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from pathlib import Path
import os, sys

# Add path for model definition
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_hybrid_nlp import SwiftGradeHybridScorer

# ─── Configuration ────────────────────────────────────────────────────────────
TRANSFORMER_NAME = "microsoft/deberta-v3-small"
NUM_LING_FEATURES = 13
CHECKPOINT_PATH = "/root/projects/SwiftGrade-Models/Short_Answer_NLP/models/hybrid_frankenstein/hybrid_champion.pth"
ONNX_PATH = "/root/projects/SwiftGrade-Models/Short_Answer_NLP/models/hybrid_frankenstein/swiftgrade_hybrid_champion.onnx"

def export_onnx():
    print(f"🚀 Loading hybrid champion from {CHECKPOINT_PATH}...")
    
    # Initialize model with TorchScript compatibility
    # This is the secret sauce for DeBERTa ONNX exports
    model = SwiftGradeHybridScorer(TRANSFORMER_NAME, NUM_LING_FEATURES)
    
    # Reload with torchscript config
    model.deberta = AutoModel.from_pretrained(TRANSFORMER_NAME, torchscript=True)
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 512
    
    dummy_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_ling_features = torch.randn(batch_size, NUM_LING_FEATURES, dtype=torch.float32)
    
    print(f"📦 Exporting to {ONNX_PATH}...")
    
    # We use the old-school tracing to avoid the Dynamo pruning
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_ling_features),
            ONNX_PATH,
            input_names=['input_ids', 'attention_mask', 'ling_features'],
            output_names=['score'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'ling_features': {0: 'batch_size'},
                'score': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True
        )
    
    if os.path.exists(ONNX_PATH):
        size = os.path.getsize(ONNX_PATH) / 1024 / 1024
        print(f"✅ Export complete: {ONNX_PATH}")
        print(f"   File size: {size:.2f} MB")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    export_onnx()
