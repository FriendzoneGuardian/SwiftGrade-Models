# 📦 SwiftGrade ONNX Export Guide
**For Coding Agents (WSL/Linux Deployment)**

This document details the expected storage footprint for the PyTorch development environment and outlines the precise steps for exporting the winning models to **ONNX INT8** for production deployment.

---

## 1. Total Raw Storage Estimate (Development State / PyTorch)
During the ML development phase, you are downloading multiple candidates across all three modules for evaluation. Ensure your WSL environment has **5 to 10 GB of free space** to comfortably handle weights, datasets, and cache overhead.

* **Module A: OMR (Computer Vision)**
  * YOLOv8 (Extraction): ~10 MB (Nano) to ~30 MB (Small)
  * Bubble Classifier (PyTorch CNN): ~50 MB to ~100 MB
  * **OMR Total: ~150 MB**
* **Module B: OCR (Handwriting)**
  * Tesseract (English LSTM data): ~30 MB
  * PaddleOCR (Det + Rec + Cls): ~150 MB
  * TrOCR (Base + Small variants): ~500 MB
  * **OCR Total: ~700 MB**
* **Module C: NLP (Essay Scoring)**
  * Hugging Face Transformers (BERT, DistilBERT, RoBERTa): ~1.2 GB
  * Sentence-Transformers (Rubric scoring): ~500 MB
  * NLTK / spaCy tokenizers: ~50 MB
  * **NLP Total: ~1.75 GB**

**Total Development Storage Needed:** **~2.6 GB** for the raw model weights.

---

## 2. The ONNX Exporters (Production Deployment)
When SwiftGrade is ready to deploy to production (e.g., C++ backend, Flutter, or Edge devices), the winning models MUST be exported to **ONNX**. 

Standard ONNX export keeps the same file size (FP32). However, by applying **INT8 Quantization** during the export, you can shrink the models by **~75%** with minimal accuracy loss.

### Module A (OMR) → Very Easy
* **YOLOv8:** Has built-in export logic.
  ```bash
  yolo export model=yolov8n.pt format=onnx int8=True
  ```
* **Bubble Classifier:** Standard PyTorch export.
  ```python
  import torch
  # Quantize first, then export
  torch.onnx.export(model, dummy_input, "classifier.onnx")
  ```
* **Size Outcome:** Drops to around **~15 MB total** (INT8). Lightning-fast on mobile/edge.

### Module B (OCR) → Mixed Difficulty
* **Tesseract:** ❌ **Cannot be exported to ONNX.** It is a compiled C++ application, not a deep learning graph. It must be run natively (e.g., via `flutter_tesseract_ocr`).
* **PaddleOCR:** ✅ **Easy.** Use the official `paddle2onnx` toolkit.
  ```bash
  paddle2onnx --model_dir ./inference/ch_PP-OCRv4_det_infer \
              --model_filename inference.pdmodel \
              --params_filename inference.pdiparams \
              --save_file model.onnx
  ```
  * *Size Outcome:* Drops from 150 MB to **~40 MB** (INT8).
* **TrOCR:** ⚠️ **Moderate.** Because it uses a complex Encoder-Decoder structure (ViT + GPT-2), exporting it creates *multiple* ONNX files (encoder, decoder, past_key_values). You use Hugging Face's `optimum` library:
  ```bash
  optimum-cli export onnx --model microsoft/trocr-base-handwritten ./trocr_onnx/
  ```
  * *Size Outcome:* Drops from 330 MB to **~85 MB** (INT8).

### Module C (NLP) → Very Easy
* **Hugging Face Transformers (BERT / DistilBERT / Sentence-Transformers):** Trivial using the Optimum CLI.
  ```bash
  # Example for the primary text classification model
  optimum-cli export onnx --model bert-base-uncased --task text-classification ./nlp_onnx/
  ```
* **Size Outcome (BERT):** Drops from 440 MB to **~110 MB** (INT8).
* **Size Outcome (DistilBERT):** Drops from 268 MB to **~65 MB** (INT8).

---

## 3. Production Summary
While the development environment requires ~2.6 GB to test the entire suite, the final deployed product will be incredibly lightweight. 

If the final architecture is **YOLOv8 + PaddleOCR + DistilBERT** and all are exported to ONNX INT8, the entire SwiftGrade intelligent grading engine will cost roughly **~120 MB of total storage** in production.
