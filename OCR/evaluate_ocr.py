"""
evaluate_ocr.py — Batch Evaluation Script for Module B
======================================================
Runs the Table 2.3.2 Variable Test across a folder of test images
and computes the final metrics (WER, CER, Drop Rate).

Usage:
  python evaluate_ocr.py --data-dir test_images/ --ground-truth gt.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from image_preprocessor import preprocess_for_ocr
from metrics import calculate_wer, calculate_cer, calculate_drop_rate


def _repo_root() -> Path:
    # OCR/evaluate_ocr.py -> repo root is one level up from OCR/
    return Path(__file__).resolve().parents[1]


def _default_unified_ocr_dir() -> Path:
    return _repo_root() / "Unified_Datasets" / "OCR"

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR Models (Module B)")
    unified_dir = _default_unified_ocr_dir()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(unified_dir / "test_images"),
        help="Folder containing handwriting samples (default: Unified_Datasets/OCR/test_images)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=str(unified_dir / "ground_truth.json"),
        help="JSON dict mapping filename -> true text (default: Unified_Datasets/OCR/ground_truth.json)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  OCR Batch Evaluation Pipeline")
    print("="*60)
    
    data_dir = Path(args.data_dir)
    gt_path = Path(args.ground_truth)

    # Backwards-compatible fallback: if user runs from OCR/ and still has OCR/test_images,
    # keep working even if Unified_Datasets isn't populated yet.
    if not data_dir.exists():
        legacy_dir = Path(__file__).resolve().parent / "test_images"
        if legacy_dir.exists():
            data_dir = legacy_dir
        else:
            print(f"⚠️ Data directory '{data_dir}' not found. Please add images to test.")
            data_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists():
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with gt_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"sample_handwriting.png": "The mitochondria is the powerhouse of the cell."},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"📄 Created dummy ground truth file: {gt_path}")

    if not data_dir.exists():
        # Should be unreachable, but keep the old behavior safe.
        return

    print("Pipeline ready. Execute via OCR_Eval_Table232.ipynb for interactive testing,")
    print("or implement the batch loop here when the dataset is finalized.")

if __name__ == "__main__":
    main()
