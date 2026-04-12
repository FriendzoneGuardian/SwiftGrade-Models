"""
purge_data.py - Extreme cleaning utility for Phase 3 dataset preparation.

Reads bubble crops, computes solidity/darkness diagnostics, and writes cleaned
outputs for training into Unified_Datasets/Phase_3_Ready.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PurgeConfig:
    source_root: Path
    output_root: Path
    reject_root: Path
    report_csv: Path
    solidity_threshold: float = 0.72
    darkness_threshold: float = 0.06
    uncertainty_margin: float = 0.05
    dry_run: bool = False


def run_purge(config: PurgeConfig) -> dict[str, int]:
    image_paths = [p for p in config.source_root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS]
    if not image_paths:
        raise ValueError(f"No image files found under '{config.source_root}'.")

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.reject_root.mkdir(parents=True, exist_ok=True)
    config.report_csv.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "total": 0,
        "kept_filled": 0,
        "kept_blank": 0,
        "kept_uncertain": 0,
        "rejected": 0,
    }

    with config.report_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_path",
                "decision",
                "solidity",
                "darkness",
                "target_path",
            ],
        )
        writer.writeheader()

        for src in image_paths:
            image = cv2.imread(str(src))
            if image is None:
                continue

            solidity = compute_solidity(image)
            darkness = compute_darkness_ratio(image)
            decision = classify_mark(
                solidity=solidity,
                darkness=darkness,
                solidity_threshold=config.solidity_threshold,
                darkness_threshold=config.darkness_threshold,
                uncertainty_margin=config.uncertainty_margin,
            )

            rel = src.relative_to(config.source_root)
            file_name = rel.name
            target_base = config.output_root if decision != "reject" else config.reject_root

            if decision == "filled":
                target_path = target_base / "filled" / file_name
                counts["kept_filled"] += 1
            elif decision == "blank":
                target_path = target_base / "blank" / file_name
                counts["kept_blank"] += 1
            elif decision == "uncertain":
                target_path = target_base / "uncertain" / file_name
                counts["kept_uncertain"] += 1
            else:
                target_path = target_base / "rejected" / file_name
                counts["rejected"] += 1

            counts["total"] += 1

            writer.writerow(
                {
                    "source_path": str(src),
                    "decision": decision,
                    "solidity": f"{solidity:.6f}",
                    "darkness": f"{darkness:.6f}",
                    "target_path": str(target_path),
                }
            )

            if not config.dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target_path)

    return counts


def classify_mark(
    solidity: float,
    darkness: float,
    solidity_threshold: float,
    darkness_threshold: float,
    uncertainty_margin: float,
) -> str:
    if solidity < max(0.0, solidity_threshold - uncertainty_margin) and darkness < max(0.0, darkness_threshold - uncertainty_margin):
        return "blank"

    if solidity >= solidity_threshold and darkness >= darkness_threshold:
        return "filled"

    if abs(solidity - solidity_threshold) <= uncertainty_margin or abs(darkness - darkness_threshold) <= uncertainty_margin:
        return "uncertain"

    if solidity < solidity_threshold * 0.5 and darkness < darkness_threshold * 0.5:
        return "blank"

    return "reject"


def compute_solidity(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return 0.0

    return area / hull_area


def compute_darkness_ratio(image_bgr: np.ndarray, threshold: int = 170) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < threshold)
    total_pixels = gray.size
    if total_pixels == 0:
        return 0.0
    return float(dark_pixels / total_pixels)


def _resolve_default_source(project_root: Path) -> Path:
    candidates = [
        project_root / "Unified_Datasets" / "Phase_2_Cropped",
        project_root / "Unified_Datasets" / "manual_labeled",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if any(p.suffix.lower() in _IMAGE_EXTENSIONS for p in candidate.rglob("*")):
            return candidate

    return candidates[0]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_source = _resolve_default_source(project_root)
    default_output = project_root / "Unified_Datasets" / "Phase_3_Ready"
    default_reject = project_root / "Unified_Datasets" / "Phase_3_Rejects"
    default_report = default_output / "purge_report.csv"

    parser = argparse.ArgumentParser(description="Phase 3 extreme data cleaning via solidity/darkness gating.")
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--reject-root", type=Path, default=default_reject)
    parser.add_argument("--report-csv", type=Path, default=default_report)
    parser.add_argument("--solidity-threshold", type=float, default=0.72)
    parser.add_argument("--darkness-threshold", type=float, default=0.06)
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PurgeConfig(
        source_root=args.source_root,
        output_root=args.output_root,
        reject_root=args.reject_root,
        report_csv=args.report_csv,
        solidity_threshold=args.solidity_threshold,
        darkness_threshold=args.darkness_threshold,
        uncertainty_margin=args.uncertainty_margin,
        dry_run=args.dry_run,
    )

    result = run_purge(config)
    print("[purge_data] completed")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"  report_csv: {config.report_csv}")


if __name__ == "__main__":
    main()
