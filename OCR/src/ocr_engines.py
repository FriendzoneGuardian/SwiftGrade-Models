"""
ocr_engines.py — Wrappers for Table 2.3.2 OCR Configurations
============================================================
Implements inference wrappers for:
  A: Tesseract (LSTM mode)
  B: TrOCR (Transformer)
  C: PaddleOCR (CNN+RNN+CTC)
"""

import os
from PIL import Image

# ─── Config A: Tesseract ────────────────────────────────────────────────────────

class TesseractEngine:
    def __init__(self):
        try:
            import pytesseract
            self.pytesseract = pytesseract
            # TESS_CONFIG: --oem 1 (LSTM), --psm 6 (Uniform block of text)
            self.config = "--oem 1 --psm 6"
        except ImportError:
            self.pytesseract = None
            print("⚠️ pytesseract not installed. TesseractEngine will fail.")

    def run(self, pil_image):
        if not self.pytesseract:
            return None
        try:
            text = self.pytesseract.image_to_string(pil_image, config=self.config)
            return text.strip()
        except Exception as e:
            print(f"Tesseract Error: {e}")
            return None

# ─── Config B: TrOCR ────────────────────────────────────────────────────────────

class TrOCREngine:
    def __init__(self, model_size="base"):
        """model_size can be 'base' or 'small'"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = f"microsoft/trocr-{model_size}-handwritten"
            
            print(f"Loading TrOCR ({model_size}) on {self.device}...")
            self.processor = TrOCRProcessor.from_pretrained(model_id)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
            self.model_loaded = True
        except ImportError:
            self.model_loaded = False
            print("⚠️ transformers/torch not installed. TrOCREngine will fail.")
        except Exception as e:
            self.model_loaded = False
            print(f"⚠️ Failed to load TrOCR: {e}")

    def run(self, pil_image):
        if not self.model_loaded:
            return None
        try:
            # TrOCR expects RGB PIL Image
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            print(f"TrOCR Error: {e}")
            return None

# ─── Config C: PaddleOCR ────────────────────────────────────────────────────────

class PaddleEngine:
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            print("Loading PaddleOCR...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.model_loaded = True
        except ImportError:
            self.model_loaded = False
            print("⚠️ paddleocr not installed. PaddleEngine will fail.")

    def run(self, cv2_image):
        """Note: PaddleOCR works best with raw numpy array (cv2 image)"""
        if not self.model_loaded:
            return None
        try:
            result = self.ocr.ocr(cv2_image, cls=True)
            if not result or not result[0]:
                return ""
            
            # Extract text from the result structure
            lines = [line[1][0] for line in result[0]]
            full_text = " ".join(lines)
            return full_text.strip()
        except Exception as e:
            print(f"PaddleOCR Error: {e}")
            return None
