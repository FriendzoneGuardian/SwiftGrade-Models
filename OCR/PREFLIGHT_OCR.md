# ✈️ Module B — OCR Pre-Flight Checklist
**SwiftGrade / Short Answer OCR Pipeline (Handwriting → Text)**
*Table 2.3.2 — Three-Configuration Variable Test*

---

## 1 · System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Any modern quad-core | Intel i5/i7 or Ryzen 5/7 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Optional (CPU fallback OK) | NVIDIA GTX 1650 / RTX 3060 |
| **CUDA** | — (optional) | 12.1 or 12.4 (for TrOCR GPU) |
| **Disk (free)** | 2 GB | 4 GB (all models + datasets) |
| **Python** | 3.10 | 3.10 or 3.11 |
| **OS** | Windows 10/11 or Ubuntu 22.04 | Ubuntu 22.04 LTS |

> **GPU Note:** Tesseract and PaddleOCR (CPU) run fully offline without a GPU.
> TrOCR benefits from CUDA but falls back to CPU cleanly with reduced speed.

---

## 2 · OCR Configuration Overview (Table 2.3.2)

| Config | Model | Architecture | Mode | Primary Benefit |
|--------|-------|-------------|------|-----------------|
| **A** | Tesseract | Traditional OCR + LSTM | `--oem 1 --psm 6` | Lightweight, fully offline, free, best community support |
| **B** | TrOCR Base/Small | Transformer (ViT encoder + GPT-2 decoder) | HuggingFace inference | Strongest handwriting accuracy, open-source |
| **C** | PaddleOCR | Hybrid CNN + RNN + CTC | Handwriting mode | Modular, multi-script, excellent non-Latin support |

---

## 3 · Image Pre-Processing Requirements

All scanned answer sheets **must pass** these gates before entering any OCR config:

| Requirement | Threshold | Tool |
|-------------|-----------|------|
| **Resolution** | ≥ 300 DPI **or** ≥ 480p (short edge) | PIL `.info['dpi']` / `image.size` |
| **Lighting** | Uniform, no shadows | CLAHE (`cv2.createCLAHE`) |
| **Orientation** | Deskewed (< 2° skew) | `cv2.minAreaRect` + affine warp |
| **Binarization** | Otsu / adaptive threshold applied | `cv2.threshold` with `THRESH_OTSU` |
| **Noise** | Gaussian blur + morphological opening | `cv2.GaussianBlur`, `cv2.morphologyEx` |

---

## 4 · Evaluation Metrics (All Three Configs)

### WER — Word Error Rate
```
WER = (Insertions + Deletions + Substitutions) / Total Reference Words
```
- **Lower is better.** `WER = 0.0` → perfect transcription.
- Computed per document, averaged across the evaluation set.
- Implementation: `jiwer.wer(reference, hypothesis)`

### CER — Character Error Rate
```
CER = (Insertions + Deletions + Substitutions at char level) / Total Reference Characters
```
- More granular than WER — essential for messy or joined handwriting.
- Use CER as the **primary metric** when character-level errors matter.
- Implementation: `jiwer.cer(reference, hypothesis)`

### OCR-to-NLP Drop Rate
```
Drop Rate = Documents with null/garbled OCR output / Total Documents × 100%
```
- Measures **pipeline failure** — OCR outputs so bad the NLP stage cannot score them.
- **Target: minimize below 5%.** Any output that triggers `None`, empty string, or
  pure punctuation/symbol noise counts as a drop.
- Track and log per config in `OCR/outputs/ocr_metrics/eval_metrics_scaffold.json`.

---

## 5 · Pre-Flight Sequence

Follow these steps **in order**. Do not skip ahead.

---

### STEP 0 — Virtual Environment Setup

```powershell
# Navigate to project root
cd C:\Users\franc\Documents\SwiftGrade-Models

# Create isolated environment (if not already created)
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Confirm Python version (must be 3.10 or 3.11)
python --version

# Install OCR deps into the active venv
pip install -r OCR\requirements_ocr.txt

# (Recommended) keep model caches inside the repo for reproducibility
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:PADDLE_HOME = "$PWD\.cache\paddle"
```

---

### STEP 1 — Install Config A: Tesseract

Tesseract is a **system binary** — not a pip package. Install it first.

#### 1A · Install Tesseract Binary

**Windows:**
```
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer
3. ✅ Check "Additional language data (English)" during install
4. Add to PATH: C:\Program Files\Tesseract-OCR
5. Restart terminal
```

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**Verify:**
```powershell
tesseract --version
# Expected: tesseract 5.x.x
```

#### 1B · Install Python Wrapper

```powershell
pip install pytesseract Pillow
```

#### 1C · Handwriting Mode Config

```python
import pytesseract
from PIL import Image

# For handwriting, always use LSTM engine (--oem 1) + block mode (--psm 6)
TESS_CONFIG = "--oem 1 --psm 6"

image = Image.open("handwritten_answer.png")
text = pytesseract.image_to_string(image, config=TESS_CONFIG)
print(text)
```

> **Note:** For Windows, you may need to set the binary path explicitly:
> ```python
> pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
> ```

---

### STEP 2 — Install Config B: TrOCR (PyTorch + HuggingFace)

#### 2A · Install PyTorch with CUDA (recommended)

> ⚠️ **Critical:** Do NOT run plain `pip install torch`.
> That installs the CPU-only build. Use the CUDA wheel.

```powershell
# CUDA 12.1 (GTX 1650 / most modern NVIDIA drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce GTX 1650
```

**CPU-only fallback** (if no NVIDIA GPU):
```powershell
pip install torch torchvision
```

#### 2B · Install Transformers

```powershell
pip install transformers>=4.40.0 tokenizers>=0.19.1 huggingface-hub>=0.20.0
```

#### 2C · Download TrOCR Models

```powershell
# From the OCR module directory
cd C:\Users\franc\Documents\SwiftGrade-Models\OCR
python download_ocr_models.py
```

This downloads:

| Model | Size | Use Case |
|-------|------|----------|
| `microsoft/trocr-base-handwritten` | ~330 MB | Higher accuracy |
| `microsoft/trocr-small-handwritten` | ~170 MB | Faster inference |

**Cache location:**
```
C:\Users\<you>\.cache\huggingface\hub\
```

#### 2D · TrOCR Inference Snippet

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)

image = Image.open("handwritten_answer.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

---

### STEP 3 — Install Config C: PaddleOCR

> ⚠️ PaddlePaddle must be installed **before** paddleocr.

#### 3A · Install PaddlePaddle Backend

```powershell
# CPU (recommended for test/dev on Windows):
pip install paddlepaddle

# GPU (CUDA 11.x):
pip install paddlepaddle-gpu==2.6.1.post112 \
  -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# GPU (CUDA 12.x):
pip install paddlepaddle-gpu==2.6.1.post120 \
  -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Verify:
python -c "import paddle; print(paddle.__version__)"
```

#### 3B · Install PaddleOCR

```powershell
pip install paddleocr>=2.7.0
```

#### 3C · First-Run Model Download

PaddleOCR automatically downloads detection + recognition + angle classification
weights on the **first call** to `PaddleOCR()`:

```python
from paddleocr import PaddleOCR

# Handwriting mode: use_angle_cls=True corrects flipped/rotated text
ocr = PaddleOCR(use_angle_cls=True, lang='en')

result = ocr.ocr("handwritten_answer.png", cls=True)
lines = [line[1][0] for line in result[0]]
full_text = " ".join(lines)
print(full_text)
```

> **Estimated download:** ~50–200 MB (detection + recognition + cls models)

---

### STEP 4 — Install Image Pre-Processing Libraries

```powershell
pip install opencv-python Pillow scikit-image numpy
```

**Verify:**
```python
import cv2, numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

# Quick smoke test
img = np.ones((100, 400), dtype=np.uint8) * 255
gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Pre-processing libs OK — binary shape:", binary.shape)
```

---

### STEP 5 — Install Eval Metric Libraries

```powershell
pip install jiwer>=3.0.0 editdistance>=0.6.3
```

**Verify:**
```python
import jiwer

ref = "the quick brown fox"
hyp = "the quick brown fox"

wer_score = jiwer.wer(ref, hyp)
cer_score = jiwer.cer(ref, hyp)

print(f"WER: {wer_score:.4f}")   # 0.0000
print(f"CER: {cer_score:.4f}")   # 0.0000
print("✅ jiwer OK")
```

---

### STEP 6 — Install All OCR Dependencies in One Shot

```powershell
# From the OCR module directory
cd C:\Users\franc\Documents\SwiftGrade-Models\OCR
pip install -r requirements_ocr.txt
```

> ⚠️ **Remember:** Tesseract (system binary) and PaddlePaddle backend must be
> installed **separately** before this command (see Steps 1 and 3A).

---

### STEP 7 — Run the OCR Pre-Flight Script

```powershell
# With .venv active, from the OCR module directory
cd C:\Users\franc\Documents\SwiftGrade-Models\OCR
python download_ocr_models.py
```

**Expected output summary:**
```
✅ READY               A  Tesseract LSTM
✅ READY               B  TrOCR (trocr-base-handwritten)
✅ READY               B  TrOCR (trocr-small-handwritten)
✅ READY               C  PaddleOCR Handwriting
✅ OK                     Image Pre-processing libs
✅ OK                     WER/CER (jiwer)

🎉 All critical OCR components are READY.
```

---

## 6 · Common Failure Points

| Error | Cause | Fix |
|-------|-------|-----|
| `tesseract: command not found` | Binary not on PATH | Install from UB-Mannheim, add to PATH |
| `pytesseract.TesseractNotFoundError` | Binary path not set | Set `pytesseract.pytesseract.tesseract_cmd` |
| `OSError: microsoft/trocr-base-handwritten not found` | No internet / proxy | Ensure HuggingFace access, or use offline cache |
| `torch.cuda.is_available() = False` | CPU-only PyTorch | Reinstall with CUDA wheel (Step 2A) |
| `ImportError: No module named 'paddle'` | PaddlePaddle not installed | `pip install paddlepaddle` (Step 3A) |
| `paddleocr.PaddleOCR` init hangs | Firewall blocking model CDN | Use VPN or pre-download models manually |
| `jiwer.wer` returns `> 0.5` on clean text | Whitespace normalization mismatch | Apply `jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip()])` |
| Low OCR accuracy on scanned images | DPI too low / poor binarization | Re-scan at 300+ DPI; apply CLAHE + Otsu pre-processing |

---

## 7 · Offline / Air-Gapped Setup

If the machine has **no internet**, pre-download TrOCR on a connected machine:

```python
# On internet-connected machine
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

for model_id in [
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-small-handwritten",
]:
    save_path = f"./offline_ocr_models/{model_id.split('/')[-1]}"
    TrOCRProcessor.from_pretrained(model_id).save_pretrained(save_path)
    VisionEncoderDecoderModel.from_pretrained(model_id).save_pretrained(save_path)
    print(f"Saved → {save_path}")
```

```python
# On air-gapped machine — load from local path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model_path = "./offline_ocr_models/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_path, local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
print("Loaded from local cache.")
```

Transfer `./offline_ocr_models/` via USB or network share.

---

## 8 · VRAM Budget (GTX 1650 — 4 GB)

| Config | Model | Inference VRAM | Fits 4 GB? |
|--------|-------|---------------|------------|
| A | Tesseract | ~0 MB (CPU only) | ✅ Yes |
| B | TrOCR Small | ~600 MB | ✅ Yes |
| B | TrOCR Base | ~1.3 GB | ✅ Yes (comfortable) |
| C | PaddleOCR (CPU) | ~0 MB GPU | ✅ Yes (CPU mode) |
| C | PaddleOCR (GPU) | ~500 MB | ✅ Yes |

> All three configs fit comfortably within the GTX 1650's 4 GB VRAM.
> No CPU offloading required for OCR inference.

---

## 9 · Metrics Scaffold Location

After running `download_ocr_models.py`, a blank metrics file is generated:
```
OCR/outputs/ocr_metrics/eval_metrics_scaffold.json
```

Fill in WER, CER, and drop_rate values after running the evaluation notebook.
This file feeds the Module B evaluation report (Table 2.3.2 results).

---

## 10 · Flutter Integration Options

| Option | Library | Mode | Best For |
|--------|---------|------|----------|
| **Primary** | `flutter_tesseract_ocr` | On-device, offline | Mobile deployment, no server needed |
| **Fallback** | `tesseract.js` | JavaScript, browser/hybrid | Web apps or Electron wrappers |
| **API Mode** | TrOCR / PaddleOCR via Python REST | Server-side | High-accuracy scenarios with backend |

For API mode, expose the OCR engine as a FastAPI endpoint and call it from Flutter via HTTP.

---

## 11 · What Comes After Pre-Flight

Once all components verify green (Step 7 passes):

1. **Place test images** → `OCR/test_images/` (300 DPI scans of handwritten answers)
2. **Run evaluation notebook** → `OCR/notebooks/OCR_Eval_Table232.ipynb`
3. **Compute WER/CER** per config using `jiwer` against ground-truth transcriptions
4. **Log Drop Rate** — count failed outputs per config
5. **Select winning config** — lowest WER/CER + lowest drop rate → feeds Module C NLP pipeline
6. **Feed verified text** → `Short_Answer_NLP/src/` (Module C scoring pipeline)

---

*Pre-flight prepared: May 9, 2026 — SwiftGrade Module B OCR*
