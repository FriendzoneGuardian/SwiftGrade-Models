import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def run_optimum_quantization():
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "models" / "bert_cased"
    pytorch_model_path = model_dir  # Should contain config.json and weights
    onnx_output_dir = model_dir / "onnx_optimum"
    onnx_output_dir.mkdir(parents=True, exist_ok=True)
    
    # We need the config.json and tokenizer files in the model_dir to use ORTModel
    # Since we only saved best_model.bin, we should ideally have the full HF model structure.
    # I'll save the config and tokenizer from the base bert-base-cased if needed.
    
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained("bert-base-cased", num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # Save them to model_dir so ORTModel can load it
    config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Rename best_model.bin to pytorch_model.bin for HF compatibility
    bin_path = model_dir / "best_model.bin"
    hf_bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists() and not hf_bin_path.exists():
        import shutil
        shutil.copy(bin_path, hf_bin_path)

    print("🚀 Exporting model to ONNX via Optimum...")
    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    model.save_pretrained(onnx_output_dir)

    print("⚡ Quantizing to INT8 via Optimum...")
    quantizer = ORTQuantizer.from_pretrained(onnx_output_dir)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    quantizer.quantize(
        save_dir=onnx_output_dir,
        quantization_config=dqconfig,
    )
    
    # Move the quantized model to the main folder with the requested name
    quantized_path = onnx_output_dir / "model_quantized.onnx"
    final_path = model_dir / "bert_essay_scorer_int8.onnx"
    
    if quantized_path.exists():
        import shutil
        shutil.copy(quantized_path, final_path)
        # Also copy the data file if it exists (for large models)
        quantized_data = quantized_path.with_suffix(".onnx.data")
        if quantized_data.exists():
            shutil.copy(quantized_data, final_path.with_suffix(".onnx.data"))
            
        print(f"✅ Successfully created: {final_path}")
        print(f"📊 Final Size: {os.path.getsize(final_path) / (1024*1024):.2f} MB")
    else:
        print("❌ Quantization failed.")

if __name__ == "__main__":
    run_bert_training_dir = Path(__file__).parent.parent / "models" / "bert_cased"
    run_optimum_quantization()
