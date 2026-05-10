import torch
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def export_onnx():
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "bert_cased" / "best_model.bin"
    output_path = base_dir / "models" / "bert_cased" / "bert_essay_scorer.onnx"
    
    if not model_path.exists():
        print(f"❌ Best model not found at {model_path}")
        return

    print(f"📦 Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Dummy input for tracing
    # (batch_size=1, sequence_length=512)
    dummy_input_ids = torch.zeros(1, 512, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 512, dtype=torch.long)
    
    print(f"🚀 Exporting to ONNX at {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    if output_path.exists():
        print(f"✅ Successfully exported to {output_path}")
        print(f"📏 Model Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    export_onnx()
