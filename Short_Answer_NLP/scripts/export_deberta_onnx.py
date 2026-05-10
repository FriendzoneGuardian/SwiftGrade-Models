import os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def export_deberta():
    model_name = "microsoft/deberta-v3-small"
    base_dir = Path(__file__).parent.parent
    weights_path = base_dir / "models" / "deberta_v3_small" / "best_model.bin"
    onnx_path = base_dir / "models" / "deberta_v3_small" / "swiftgrade_nlp_champion.onnx"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    print(f"Loading weights: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    dummy_input = tokenizer("This is a dummy input for ONNX export.", return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # DeBERTa-v3 small specific inputs
    inputs = (dummy_input["input_ids"], dummy_input["attention_mask"])
    
    print(f"Exporting to: {onnx_path}")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )
    print("Export complete!")

if __name__ == "__main__":
    export_deberta()
