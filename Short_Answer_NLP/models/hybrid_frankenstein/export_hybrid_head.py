"""
Export the Frankenstein Hybrid Decision Head to ONNX
This head takes (DeBERTa_CLS + Linguistic_Features) -> Final Grade
"""

import torch
import torch.nn as nn
from pathlib import Path
import os, sys

# The Head definition must match train_hybrid_nlp.py
class SwiftGradeHybridHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fusion_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, combined_features):
        logits = self.fusion_head(combined_features)
        return torch.sigmoid(logits).squeeze(-1)

# ─── Configuration ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/root/projects/SwiftGrade-Models/Short_Answer_NLP/models/hybrid_frankenstein/hybrid_champion.pth"
ONNX_HEAD_PATH = "/root/projects/SwiftGrade-Models/Short_Answer_NLP/models/hybrid_frankenstein/swiftgrade_hybrid_head.onnx"
INPUT_DIM = 768 + 13 # DeBERTa Small CLS + spaCy features

def export_head():
    print(f"🚀 Loading hybrid head from {CHECKPOINT_PATH}...")
    
    # Initialize head and load weights
    # Note: We need to map the keys because the checkpoint has 'fusion_head.X'
    head = SwiftGradeHybridHead(INPUT_DIM)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Filter state dict for just the head
    head_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('fusion_head.'):
            head_state_dict[k.replace('fusion_head.', 'fusion_head.')] = v
            
    head.load_state_dict(head_state_dict)
    head.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, INPUT_DIM, dtype=torch.float32)
    
    # Export
    print(f"📦 Exporting Head to {ONNX_HEAD_PATH}...")
    torch.onnx.export(
        head,
        dummy_input,
        ONNX_HEAD_PATH,
        input_names=['combined_features'],
        output_names=['score'],
        dynamic_axes={'combined_features': {0: 'batch_size'}, 'score': {0: 'batch_size'}},
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"✅ Head Export complete: {ONNX_HEAD_PATH}")

if __name__ == "__main__":
    export_head()
