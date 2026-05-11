"""
SwiftGrade Model Quantization Script
Optimizing Transformers for Mobile/Edge Deployment
"""

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path, output_path):
    print(f"📉 Quantizing {input_path}...")
    if not os.path.exists(input_path):
        print(f"❌ Skipping: {input_path} not found.")
        return
        
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    in_size = os.path.getsize(input_path) / 1024 / 1024
    out_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Success! {in_size:.2f}MB -> {out_size:.2f}MB (Reduction: {(1 - out_size/in_size)*100:.1f}%)")

if __name__ == "__main__":
    # Define models to quantize
    models_to_quant = [
        # NLP
        ("Short_Answer_NLP/models/hybrid_frankenstein/deberta_backbone.onnx", 
         "Short_Answer_NLP/models/hybrid_frankenstein/deberta_backbone_quant.onnx"),
        # OCR
        ("OCR/models/trocr_small_onnx/encoder_model.onnx", "OCR/models/trocr_small_onnx/encoder_model_quant.onnx"),
        ("OCR/models/trocr_small_onnx/decoder_model.onnx", "OCR/models/trocr_small_onnx/decoder_model_quant.onnx")
    ]
    
    for input_m, output_m in models_to_quant:
        quantize_model(input_m, output_m)
