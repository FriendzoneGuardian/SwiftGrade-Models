#!/usr/bin/env python3
"""Phase A preflight for SwiftGrade-Models.

Checks data handoff readiness, notebook entrypoints, dependency availability,
and model-weight availability before running phase notebooks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

REQUIRED_NOTEBOOKS = [
    "OMR/notebooks/Phase1_Extraction.ipynb",
    "OMR/notebooks/Phase2_Preprocessing.ipynb",
    "OMR/notebooks/Phase3_Classification.ipynb",
]

REQUIRED_PHASE_PATHS = [
    "OMR/Phase_1_Extraction/src",
    "OMR/Phase_2_Preprocessing/src",
    "OMR/Phase_3_Classification/src",
    "Unified_Datasets/Phase_1_Raw",
    "Unified_Datasets/Phase_2_Cropped",
    "Unified_Datasets/Phase_3_Ready",
]

PACKAGE_TO_MODULE = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "albumentations": "albumentations",
    "mlflow": "mlflow",
    "pyyaml": "yaml",
    "pandas": "pandas",
    "numpy": "numpy",
}


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _parse_requirements(requirements_path: Path) -> list[str]:
    if not requirements_path.exists():
        return []

    requirements: list[str] = []
    line_pattern = re.compile(r"^([A-Za-z0-9_.\-]+)")
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_pattern.match(line)
        if match:
            requirements.append(match.group(1))
    return requirements


def _check_dependencies(project_root: Path) -> tuple[list[CheckItem], list[str]]:
    req_path = project_root / "requirements.txt"
    requirements = _parse_requirements(req_path)

    checks: list[CheckItem] = []
    missing: list[str] = []

    for package in requirements:
        module_name = PACKAGE_TO_MODULE.get(package, package)
        spec = find_spec(module_name)
        if spec is None:
            checks.append(CheckItem(name=package, ok=False, detail=f"module '{module_name}' not importable"))
            missing.append(package)
            continue

        version = "unknown"
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass

        checks.append(CheckItem(name=package, ok=True, detail=f"module '{module_name}' importable (version={version})"))

    return checks, missing


def _check_notebook_execution(notebook_path: Path) -> CheckItem:
    if not notebook_path.exists():
        return CheckItem(name=str(notebook_path), ok=False, detail="not found")

    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckItem(name=str(notebook_path), ok=False, detail=f"invalid JSON: {exc}")

    cells = payload.get("cells", [])
    executed = 0
    code_cells = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        if cell.get("execution_count") is not None:
            executed += 1

    return CheckItem(
        name=str(notebook_path),
        ok=True,
        detail=f"code_cells={code_cells}, executed_code_cells={executed}",
    )


def _find_weight_files(project_root: Path, pattern: str) -> list[str]:
    matches: list[str] = []
    for path in project_root.rglob(pattern):
        if path.is_file() and ".venv" not in path.parts:
            matches.append(str(path.relative_to(project_root)))
    matches.sort()
    return matches


def _sample_image_paths(raw_root: Path, limit: int = 5) -> list[str]:
    if not raw_root.exists():
        return []

    found: list[str] = []
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            found.append(str(path))
        if len(found) >= limit:
            break
    return found


def _phase_path_checks(project_root: Path, paths: Iterable[str]) -> list[CheckItem]:
    checks: list[CheckItem] = []
    for rel_path in paths:
        full_path = project_root / rel_path
        checks.append(
            CheckItem(
                name=rel_path,
                ok=full_path.exists(),
                detail="exists" if full_path.exists() else "missing",
            )
        )
    return checks


def build_report(project_root: Path) -> dict:
    phase_checks = _phase_path_checks(project_root, REQUIRED_PHASE_PATHS)
    notebook_checks = [_check_notebook_execution(project_root / nb) for nb in REQUIRED_NOTEBOOKS]
    dependency_checks, missing_packages = _check_dependencies(project_root)

    raw_root = project_root / "Unified_Datasets" / "Phase_1_Raw"
    cropped_root = project_root / "Unified_Datasets" / "Phase_2_Cropped"
    ready_root = project_root / "Unified_Datasets" / "Phase_3_Ready"

    raw_images = _count_images(raw_root)
    cropped_images = _count_images(cropped_root)
    ready_images = _count_images(ready_root)

    yolo_weights = _find_weight_files(project_root, "*.pt")
    classifier_weights = _find_weight_files(project_root, "*.pth")
    sample_images = _sample_image_paths(raw_root)

    blockers: list[str] = []
    warnings: list[str] = []

    if raw_images == 0:
        blockers.append("No raw Phase 1 images found under Unified_Datasets/Phase_1_Raw.")

    if not yolo_weights:
        blockers.append("No YOLO detector weights (.pt) found in repository for Phase 1 extraction.")

    if not classifier_weights:
        warnings.append("No classifier checkpoint (.pth) found yet (expected before first Phase 3 training run).")

    if missing_packages:
        blockers.append("Missing required Python packages: " + ", ".join(sorted(missing_packages)))

    if cropped_images == 0:
        warnings.append("Phase_2_Cropped is currently empty (expected before first run).")
    if ready_images == 0:
        warnings.append("Phase_3_Ready is currently empty (expected before first run).")

    all_required_paths_ok = all(check.ok for check in phase_checks)
    all_notebooks_ok = all(check.ok for check in notebook_checks)

    if not all_required_paths_ok:
        blockers.append("One or more required phase paths are missing.")
    if not all_notebooks_ok:
        blockers.append("One or more required notebooks are missing or invalid.")

    status = "ready" if not blockers else "blocked"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "phase_1_raw_images": raw_images,
            "phase_2_cropped_images": cropped_images,
            "phase_3_ready_images": ready_images,
            "phase_1_sample_images": sample_images,
        },
        "assets": {
            "yolo_weights_pt": yolo_weights,
            "classifier_weights_pth": classifier_weights,
        },
        "checks": {
            "required_paths": [asdict(item) for item in phase_checks],
            "notebooks": [asdict(item) for item in notebook_checks],
            "dependencies": [asdict(item) for item in dependency_checks],
        },
    }


def resolve_project_root(given_root: str | None) -> Path:
    if given_root:
        return Path(given_root).resolve()
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SwiftGrade Phase A preflight checks.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Optional project root path. Defaults to repository root based on script location.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional output JSON path. Defaults to OMR/runs/preflight/preflight_report_<timestamp>.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)

    report = build_report(project_root)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_report_path = project_root / "OMR" / "runs" / "preflight" / f"preflight_report_{stamp}.json"
    report_path = Path(args.report_path).resolve() if args.report_path else default_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[preflight] status={report['status']}")
    print(f"[preflight] report={report_path}")
    print(f"[preflight] phase_1_raw_images={report['dataset']['phase_1_raw_images']}")
    print(f"[preflight] phase_2_cropped_images={report['dataset']['phase_2_cropped_images']}")
    print(f"[preflight] phase_3_ready_images={report['dataset']['phase_3_ready_images']}")

    if report["warnings"]:
        print("[preflight] warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["blockers"]:
        print("[preflight] blockers:")
        for blocker in report["blockers"]:
            print(f"  - {blocker}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
