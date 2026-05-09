"""
download_ocr_models.py — SwiftGrade OCR Module B Setup Script
=============================================================
Downloads, installs, and verifies all OCR engines required for:
  Module B — Short Answer OCR (Handwriting → Text)

Configurations tested (Table 2.3.2):
  Config A — Tesseract  (LSTM handwriting mode)
  Config B — TrOCR      (microsoft/trocr-base-handwritten  ~330 MB)
                         (microsoft/trocr-small-handwritten ~170 MB)
  Config C — PaddleOCR  (Handwriting mode, CNN+RNN+CTC)

Eval Metrics Initialized:
  WER  — Word Error Rate
  CER  — Character Error Rate
  OCR-to-NLP Drop Rate — pipeline failure tracker

Run once before any OCR evaluation notebooks:
    python download_ocr_models.py

By default this script uses **project-local caches** (recommended for reproducibility):
  - Hugging Face: <repo>/.cache/huggingface/
  - Paddle:       <repo>/.cache/paddle/

You can override these by setting environment variables before running:
  - HF_HOME / TRANSFORMERS_CACHE
  - PADDLE_HOME
"""

import sys
import os
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime


# ─── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HF_HOME = REPO_ROOT / ".cache" / "huggingface"
DEFAULT_PADDLE_HOME = REPO_ROOT / ".cache" / "paddle"


def _ensure_project_caches():
    """
    Make model downloads deterministic by defaulting caches to repo-local folders.
    This avoids "it works on my machine" issues caused by global user caches.
    """
    hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if not hf_home:
        os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)

    if not os.environ.get("PADDLE_HOME"):
        os.environ["PADDLE_HOME"] = str(DEFAULT_PADDLE_HOME)

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PADDLE_HOME"]).mkdir(parents=True, exist_ok=True)


_ensure_project_caches()

TROCR_MODELS = [
    (
        "microsoft/trocr-base-handwritten",
        "~330 MB",
        "TrOCR Base — highest accuracy for handwriting",
    ),
    (
        "microsoft/trocr-small-handwritten",
        "~170 MB",
        "TrOCR Small — faster inference, lower memory",
    ),
]

REQUIRED_DPI = 300          # Minimum DPI for reliable OCR (per spec)
REQUIRED_RESOLUTION = 480   # Minimum short-edge pixel count (480p)

EVAL_METRICS_TEMPLATE = {
    "module": "Module B — Short Answer OCR",
    "generated_at": None,
    "configs": {
        "A_tesseract": {"wer": None, "cer": None, "drop_rate": None, "notes": ""},
        "B_trocr_base":  {"wer": None, "cer": None, "drop_rate": None, "notes": ""},
        "B_trocr_small": {"wer": None, "cer": None, "drop_rate": None, "notes": ""},
        "C_paddleocr":   {"wer": None, "cer": None, "drop_rate": None, "notes": ""},
    },
    "metric_definitions": {
        "WER":  "(Insertions + Deletions + Substitutions) / total reference words — lower is better; 0% = perfect",
        "CER":  "Same calculation at character level — more granular for noisy/complex handwriting",
        "drop_rate": "% of documents where OCR output is garbled/null/unreadable, blocking NLP scoring",
    },
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_disk_space(required_gb: float = 1.5) -> float:
    """Warn if free disk space is below required_gb."""
    home = os.path.expanduser("~")
    total, used, free = shutil.disk_usage(home)
    free_gb = free / (1024 ** 3)
    print(f"  💾 Free disk space : {free_gb:.1f} GB  (need ~{required_gb} GB)")
    if free_gb < required_gb:
        print(f"  ⚠️  WARNING: Low disk space. TrOCR downloads may fail.")
    else:
        print(f"  ✅ Disk space OK")
    return free_gb


# ─── Section A: Tesseract ──────────────────────────────────────────────────────

def check_tesseract() -> dict:
    """
    Verify Tesseract binary is installed and detect its version.
    Tesseract is a system-level binary — not installed via pip.
    """
    section("CONFIG A — Tesseract LSTM (System Binary Check)")

    result = {"installed": False, "version": None, "pytesseract": False}

    # 1. Check Tesseract binary
    print("  📌 Checking Tesseract system binary...")
    try:
        proc = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode == 0:
            version_line = proc.stdout.strip().split("\n")[0]
            result["version"] = version_line
            result["installed"] = True
            print(f"  ✅ Tesseract found: {version_line}")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ❌ Tesseract binary NOT found on PATH.")
        print()
        print("  ── Install Instructions ─────────────────────────────────")
        print("  Windows:")
        print("    1. Download installer from:")
        print("       https://github.com/UB-Mannheim/tesseract/wiki")
        print("    2. Run installer — include 'Additional language data (eng)'")
        print("    3. Add Tesseract to PATH:")
        print("       C:\\Program Files\\Tesseract-OCR")
        print("    4. Restart terminal and re-run this script.")
        print()
        print("  Ubuntu/Debian:")
        print("    sudo apt-get install tesseract-ocr tesseract-ocr-eng")
        print("  ─────────────────────────────────────────────────────────")

    # 2. Check pytesseract Python wrapper
    print("\n  📥 Checking pytesseract Python wrapper...")
    try:
        import pytesseract
        version = getattr(pytesseract, "__version__", "?")
        print(f"  ✅ pytesseract v{version} — installed")
        result["pytesseract"] = True

        # Quick functional test
        if result["installed"]:
            print("  🔬 Running smoke test (basic OCR on white image)...")
            try:
                from PIL import Image
                import numpy as np
                # Create a blank white image — Tesseract should return empty string gracefully
                img = Image.fromarray(
                    np.ones((100, 400), dtype=np.uint8) * 255
                )
                text = pytesseract.image_to_string(img, config="--psm 6")
                print(f"  ✅ Smoke test passed — output: {repr(text.strip()[:50])}")
            except Exception as e:
                print(f"  ⚠️  Smoke test failed: {e}")
    except ImportError:
        print("  ❌ pytesseract not installed — run: pip install pytesseract")

    # 3. LSTM handwriting mode note
    print()
    print("  📝 Handwriting mode config for evaluation notebooks:")
    print("     pytesseract.image_to_string(img, config='--oem 1 --psm 6')")
    print("     --oem 1  → LSTM OCR engine (required for handwriting)")
    print("     --psm 6  → Assume a uniform block of text")

    return result


# ─── Section B: TrOCR ─────────────────────────────────────────────────────────

def download_trocr_models() -> dict:
    """
    Download TrOCR handwriting models from Hugging Face.
    Uses the VisionEncoderDecoderModel architecture (ViT + GPT-2 decoder).
    """
    section("CONFIG B — TrOCR Transformer-Based OCR (HuggingFace Download)")

    results = {}

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        print("  ❌ transformers not installed.")
        print("     Run: pip install transformers>=4.40.0")
        return {m[0]: False for m in TROCR_MODELS}

    for model_id, size, desc in TROCR_MODELS:
        print(f"\n  📥 {desc} ({size})")
        print(f"     Model: {model_id}")
        ok = True

        # Download processor (tokenizer + image processor)
        print(f"     → Downloading Processor...", end=" ", flush=True)
        try:
            TrOCRProcessor.from_pretrained(model_id)
            print("✅")
        except Exception as e:
            print(f"❌\n     Error: {e}")
            ok = False

        # Download model weights
        print(f"     → Downloading Model weights...", end=" ", flush=True)
        try:
            VisionEncoderDecoderModel.from_pretrained(model_id)
            print("✅")
        except Exception as e:
            print(f"❌\n     Error: {e}")
            ok = False

        if ok:
            print(f"  ✔  {model_id} ready.")
        results[model_id] = ok

    # Print usage snippet
    print()
    print("  📝 Inference snippet for evaluation notebooks:")
    print("""
     from transformers import TrOCRProcessor, VisionEncoderDecoderModel
     from PIL import Image

     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

     image = Image.open("handwritten_answer.png").convert("RGB")
     pixel_values = processor(images=image, return_tensors="pt").pixel_values
     generated_ids = model.generate(pixel_values)
     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
     print(text)
  """)

    return results


# ─── Section C: PaddleOCR ─────────────────────────────────────────────────────

def check_paddleocr() -> dict:
    """
    Verify PaddleOCR installation and download handwriting models.
    PaddlePaddle must be installed as a system prerequisite.
    """
    section("CONFIG C — PaddleOCR Handwriting Mode (CNN+RNN+CTC)")

    result = {"paddle_installed": False, "paddleocr_installed": False, "models_ok": False}

    # 1. Check PaddlePaddle backend
    print("  📌 Checking PaddlePaddle backend...")
    try:
        import paddle
        version = getattr(paddle, "__version__", "?")
        print(f"  ✅ PaddlePaddle v{version} found")
        result["paddle_installed"] = True
    except ImportError:
        print("  ❌ PaddlePaddle NOT installed.")
        print()
        print("  ── Install Instructions ─────────────────────────────────")
        print("  CPU (recommended for dev/test):")
        print("    pip install paddlepaddle")
        print()
        print("  GPU (CUDA 11.x):")
        print("    pip install paddlepaddle-gpu==2.6.1.post112 \\")
        print("      -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html")
        print()
        print("  GPU (CUDA 12.x):")
        print("    pip install paddlepaddle-gpu==2.6.1.post120 \\")
        print("      -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html")
        print("  ─────────────────────────────────────────────────────────")

    # 2. Check paddleocr wrapper
    print("\n  📌 Checking paddleocr Python package...")
    try:
        import importlib
        paddleocr_spec = importlib.util.find_spec("paddleocr")
        if paddleocr_spec is not None:
            import paddleocr as poc
            version = getattr(poc, "__version__", "?")
            print(f"  ✅ paddleocr v{version} found")
            result["paddleocr_installed"] = True
        else:
            raise ImportError("paddleocr not found")
    except ImportError:
        print("  ❌ paddleocr NOT installed.")
        print("     Run: pip install paddleocr>=2.7.0")

    # 3. Attempt to initialize with handwriting model (triggers model download)
    if result["paddle_installed"] and result["paddleocr_installed"]:
        print("\n  📥 Initializing PaddleOCR handwriting model (triggers download)...")
        print("     This downloads detection + recognition + classification weights (~50-200 MB)")
        try:
            from paddleocr import PaddleOCR
            # use_angle_cls=True enables text orientation correction
            # lang='en' for English handwriting
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print("  ✅ PaddleOCR initialized — handwriting models downloaded.")
            result["models_ok"] = True
        except Exception as e:
            print(f"  ❌ PaddleOCR init failed: {e}")
    else:
        print("\n  ⚠️  Skipping model download — install PaddlePaddle first (see above).")

    # Print usage snippet
    print()
    print("  📝 Inference snippet for evaluation notebooks:")
    print("""
     from paddleocr import PaddleOCR
     from PIL import Image

     ocr = PaddleOCR(use_angle_cls=True, lang='en')
     result = ocr.ocr("handwritten_answer.png", cls=True)

     # Extract text lines
     lines = [line[1][0] for line in result[0]]
     full_text = " ".join(lines)
     print(full_text)
  """)

    return result


# ─── Section D: Image Preprocessing Check ─────────────────────────────────────

def check_image_preprocessing_libs() -> dict:
    """
    Verify all image preprocessing libraries are available.
    Per spec: 300 DPI min, 480p min, deskew, binarization, noise reduction.
    """
    section("IMAGE PRE-PROCESSING — Library Verification")

    print("  Requirements per Module B spec:")
    print(f"    • Minimum DPI         : {REQUIRED_DPI} DPI")
    print(f"    • Minimum resolution  : {REQUIRED_RESOLUTION}p (short edge)")
    print(f"    • Deskewing           : Required (orientation correction)")
    print(f"    • Binarization        : Required (Otsu / adaptive threshold)")
    print(f"    • Noise reduction     : Required (Gaussian blur / morphological ops)")
    print()

    checks = [
        ("PIL",          "from PIL import Image",                 "Image loading, DPI metadata read"),
        ("cv2",          "import cv2",                            "Deskew, binarization, CLAHE"),
        ("numpy",        "import numpy as np",                    "Array operations"),
        ("skimage",      "from skimage.filters import threshold_otsu", "Otsu binarization"),
    ]

    results = {}
    for name, import_stmt, purpose in checks:
        print(f"  {'  ' + import_stmt:50s}", end=" ")
        try:
            exec(import_stmt)
            print(f"✅  {purpose}")
            results[name] = True
        except ImportError:
            print(f"❌  NOT installed — {purpose}")
            results[name] = False

    return results


# ─── Section E: Eval Metrics Scaffold ─────────────────────────────────────────

def check_eval_metrics_libs() -> dict:
    """
    Verify WER/CER evaluation libraries and scaffold the metrics output file.
    """
    section("EVAL METRICS — WER / CER / Drop Rate Scaffold")

    print("  Metrics per Table 2.3.2:")
    print("    WER  = (Ins + Del + Sub) / total reference words  [lower is better]")
    print("    CER  = same formula at character level            [more granular]")
    print("    Drop = % docs where OCR failure blocks NLP stage  [minimize]")
    print()

    results = {}

    # jiwer — primary WER/CER library
    print("  📌 Checking jiwer (WER/CER computation)...")
    try:
        import jiwer
        version = getattr(jiwer, "__version__", "?")
        print(f"  ✅ jiwer v{version}")

        # Quick functional test
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        print(f"  🔬 Smoke test — WER: {wer:.4f}  CER: {cer:.4f}  (both should be 0.0)")
        results["jiwer"] = True
    except ImportError:
        print("  ❌ jiwer not installed — run: pip install jiwer>=3.0.0")
        results["jiwer"] = False

    # editdistance — fallback for character-level distance
    print("\n  📌 Checking editdistance (CER fallback)...")
    try:
        import editdistance
        dist = editdistance.eval("kitten", "sitting")
        print(f"  ✅ editdistance — smoke test dist('kitten','sitting') = {dist}  (expected 3)")
        results["editdistance"] = True
    except ImportError:
        print("  ❌ editdistance not installed — run: pip install editdistance>=0.6.3")
        results["editdistance"] = False

    # Scaffold metrics output JSON
    metrics_dir = Path(__file__).parent / "outputs" / "ocr_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "eval_metrics_scaffold.json"

    template = dict(EVAL_METRICS_TEMPLATE)
    template["generated_at"] = datetime.now().isoformat()

    with open(metrics_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\n  📄 Metrics scaffold written → {metrics_path}")
    print("     Fill in WER/CER/drop_rate after running evaluation notebooks.")

    return results


# ─── Section F: Full Preflight Summary ────────────────────────────────────────

def print_summary(tesseract: dict, trocr: dict, paddle: dict,
                  preproc: dict, metrics: dict):
    section("MODULE B — OCR PRE-FLIGHT SUMMARY (Table 2.3.2)")

    rows = []

    # Config A
    tess_ok = tesseract.get("installed") and tesseract.get("pytesseract")
    rows.append(("A  Tesseract LSTM", "✅ READY" if tess_ok else "❌ ACTION NEEDED"))

    # Config B — TrOCR
    for model_id, _, _ in TROCR_MODELS:
        short_name = model_id.split("/")[-1]
        ok = trocr.get(model_id, False)
        rows.append((f"B  TrOCR ({short_name})", "✅ READY" if ok else "❌ ACTION NEEDED"))

    # Config C — PaddleOCR
    paddle_ok = paddle.get("paddle_installed") and paddle.get("paddleocr_installed")
    paddle_models_ok = paddle.get("models_ok", False)
    if paddle_ok and paddle_models_ok:
        rows.append(("C  PaddleOCR Handwriting", "✅ READY"))
    elif paddle_ok:
        rows.append(("C  PaddleOCR Handwriting", "⚠️  INSTALLED / models failed"))
    else:
        rows.append(("C  PaddleOCR Handwriting", "❌ ACTION NEEDED"))

    # Image preprocessing
    preproc_ok = all(preproc.values()) if preproc else False
    rows.append(("   Image Pre-processing libs", "✅ OK" if preproc_ok else "⚠️  Some missing"))

    # Eval metrics
    metrics_ok = metrics.get("jiwer", False)
    rows.append(("   WER/CER (jiwer)", "✅ OK" if metrics_ok else "❌ ACTION NEEDED"))

    print()
    for label, status in rows:
        print(f"  {status:20s}  {label}")

    all_critical_ok = tess_ok and all(trocr.values()) and preproc_ok and metrics_ok

    print()
    if all_critical_ok:
        print("  🎉 All critical OCR components are READY.")
        print("     You can now run OCR evaluation notebooks.")
    else:
        print("  ⚠️  Some components need attention. Address ❌ items above.")
        print("     Re-run this script after fixing to verify.")

    print()
    print("  Next steps after pre-flight:")
    print("    1. Add test handwriting images  → OCR/test_images/")
    print("    2. Run evaluation notebook      → OCR/notebooks/OCR_Eval_Table232.ipynb")
    print("    3. Fill in metrics scaffold     → OCR/outputs/ocr_metrics/eval_metrics_scaffold.json")
    print("    4. Feed verified text           → Short_Answer_NLP pipeline (Module C)")
    print()
    print("  Integration options for Flutter app:")
    print("    • flutter_tesseract_ocr  (Tesseract wrapper, offline-capable)")
    print("    • tesseract.js           (JavaScript port for web/hybrid apps)")
    print("    • TrOCR / PaddleOCR via REST API (Python backend → Flutter HTTP client)")
    print()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  SwiftGrade — Module B OCR Setup Script")
    print("  Table 2.3.2: Tesseract | TrOCR | PaddleOCR")
    print("=" * 60)
    print(f"\n  Python      : {sys.version.split()[0]}")
    print(f"  Working dir : {os.getcwd()}")
    print(f"  Script path : {Path(__file__).resolve()}")
    print(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  HF_HOME     : {os.environ.get('HF_HOME')}")
    print(f"  PADDLE_HOME : {os.environ.get('PADDLE_HOME')}")

    # Disk space guard (~1.5 GB for both TrOCR models)
    print("\n  Checking disk space...")
    check_disk_space(required_gb=1.5)

    # GPU awareness (optional — TrOCR can run on CPU too)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  🎮 GPU: {gpu_name}  ({vram:.1f} GB VRAM) — TrOCR will use GPU")
        else:
            print("  ⚠️  No GPU detected — TrOCR will run on CPU (slower, still functional)")
    except ImportError:
        print("  ⚠️  PyTorch not installed — TrOCR (Config B) unavailable until installed.")

    # Run all checks
    tesseract_result  = check_tesseract()
    trocr_result      = download_trocr_models()
    paddle_result     = check_paddleocr()
    preproc_result    = check_image_preprocessing_libs()
    metrics_result    = check_eval_metrics_libs()

    # Print consolidated summary
    print_summary(
        tesseract_result,
        trocr_result,
        paddle_result,
        preproc_result,
        metrics_result,
    )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
