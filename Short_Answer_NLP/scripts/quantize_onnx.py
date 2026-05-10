import os
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_bert():
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "models" / "bert_cased"
    input_model = model_dir / "bert_essay_scorer.onnx"
    output_model = model_dir / "bert_essay_scorer_int8.onnx"
    
    if not input_model.exists():
        print(f"❌ Source ONNX model not found at {input_model}")
        return

    print(f"⚡ Starting Dynamic INT8 Quantization (No Shape Inference)...")
    print(f"   Input:  {input_model}")
    print(f"   Output: {output_model}")

    # Trying to bypass shape inference which is failing on the regression head
    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
        extra_options={'DisableShapeInference': True}
    )

    if output_model.exists():
        old_size = os.path.getsize(input_model) / (1024*1024)
        new_size = os.path.getsize(output_model) / (1024*1024)
        
        input_data = input_model.with_suffix(".onnx.data")
        output_data = output_model.with_suffix(".onnx.data")
        
        if input_data.exists():
            old_size += os.path.getsize(input_data) / (1024*1024)
        if output_data.exists():
            new_size += os.path.getsize(output_data) / (1024*1024)

        print(f"✅ Quantization Complete!")
        print(f"📊 Original Size:   {old_size:.2f} MB")
        print(f"📊 Quantized Size:  {new_size:.2f} MB")
        print(f"📉 Reduction:       {(1 - new_size/old_size)*100:.1f}%")
    else:
        print("❌ Quantization failed.")

if __name__ == "__main__":
    quantize_bert()
