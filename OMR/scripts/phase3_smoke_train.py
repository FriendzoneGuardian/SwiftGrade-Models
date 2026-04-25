#!/usr/bin/env python3
"""Phase 3 smoke training runner.

This script mirrors the key cells in OMR/notebooks/Phase3_Classification.ipynb:
- runtime setup
- dataset root resolution and indexing
- sampled dataloaders
- model/trainer initialization
- short training run
- row-level decision engine sanity demo
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


def resolve_project_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).resolve()

    script_path = Path(__file__).resolve()
    return script_path.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 smoke training.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--epochs", type=int, default=1, help="Smoke training epochs.")
    parser.add_argument("--sample-limit", type=int, default=2000, help="Maximum samples for smoke training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--model-method",
        type=str,
        default="auto",
        choices=["auto", "ascending", "diamond", "transfer"],
        help="Model methodology selection. 'auto' keeps legacy class-count heuristic.",
    )
    parser.add_argument(
        "--transfer-unfreeze",
        action="store_true",
        help="When using transfer model, unfreeze all backbone layers before training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)

    phase3_src = project_root / "OMR" / "Phase_3_Classification" / "src"
    if str(phase3_src) not in sys.path:
        sys.path.insert(0, str(phase3_src))

    cnn_models = importlib.import_module("cnn_models")
    dataset_module = importlib.import_module("dataset")
    scoring_module = importlib.import_module("scoring")
    trainer_module = importlib.import_module("trainer")

    AscendingCNN = cnn_models.AscendingCNN
    DiamondCNN = cnn_models.DiamondCNN
    TransferLearningCNN = cnn_models.TransferLearningCNN
    build_dataset_index = dataset_module.build_dataset_index
    create_dataloaders = dataset_module.create_dataloaders
    resolve_phase3_dataset_root = dataset_module.resolve_phase3_dataset_root
    RelativeRowDecisionEngine = scoring_module.RelativeRowDecisionEngine
    ClassificationTrainer = trainer_module.ClassificationTrainer
    TrainerConfig = trainer_module.TrainerConfig

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = project_root / "OMR" / "runs" / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    resolution = resolve_phase3_dataset_root(project_root)
    dataset_root = resolution.dataset_root
    index_path = dataset_root / "dataset_index.csv"

    def resize_transform(image):
        return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    index_df = build_dataset_index(dataset_root=dataset_root, index_output_path=index_path)

    bundle = create_dataloaders(
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split_ratio=0.2,
        random_seed=seed,
        sample_limit=args.sample_limit,
        auto_sample_threshold=5000,
        transform_train=resize_transform,
        transform_valid=resize_transform,
        rebuild_index=False,
    )

    class_to_idx = bundle.class_to_idx
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    num_classes = len(class_names)
    if args.model_method == "ascending":
        model = AscendingCNN(num_classes=num_classes)
    elif args.model_method == "diamond":
        model = DiamondCNN(num_classes=num_classes)
    elif args.model_method == "transfer":
        model = TransferLearningCNN(num_classes=num_classes, freeze_backbone=not args.transfer_unfreeze)
        if args.transfer_unfreeze:
            model.unfreeze_all()
    else:
        model = AscendingCNN(num_classes=num_classes) if num_classes <= 3 else DiamondCNN(num_classes=num_classes)

    trainer_config = TrainerConfig(
        epochs=max(1, args.epochs),
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        output_dir=str(run_dir / "models"),
        monitor_metric="val_f1",
        device="auto",
    )

    trainer = ClassificationTrainer(model=model, class_names=class_names, config=trainer_config)
    summary = trainer.train(train_loader=bundle.train_loader, valid_loader=bundle.valid_loader)

    row_demo = {}
    if "filled" in class_to_idx and "blank" in class_to_idx:
        engine = RelativeRowDecisionEngine()
        # Demo row of 5 options with one dominant mark.
        probs = [0.02, 0.06, 0.84, 0.04, 0.04]
        labels = ["A", "B", "C", "D", "E"]
        row_decision = engine.decide_row(probs, labels)
        row_demo = {
            "status": row_decision.status,
            "selected": row_decision.selected,
            "confidence": row_decision.confidence,
            "margin_top2": row_decision.margin_top2,
        }

    run_summary = {
        "project_root": str(project_root),
        "phase3_src": str(phase3_src),
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "dataset_source": resolution.source_name,
        "index_path": str(index_path),
        "indexed_rows": int(len(index_df)),
        "sample_mode": bundle.sample_mode,
        "class_to_idx": class_to_idx,
        "train_batches": int(len(bundle.train_loader)),
        "valid_batches": int(len(bundle.valid_loader)),
        "model": model.__class__.__name__,
        "model_method": args.model_method,
        "transfer_unfreeze": bool(args.transfer_unfreeze),
        "trainer_summary": summary,
        "row_demo": row_demo,
    }

    run_summary_path = run_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("[phase3-smoke] run complete")
    print(f"[phase3-smoke] run_dir={run_dir}")
    print(f"[phase3-smoke] dataset_root={dataset_root} (source={resolution.source_name})")
    print(f"[phase3-smoke] indexed_rows={len(index_df):,}")
    print(f"[phase3-smoke] class_to_idx={class_to_idx}")
    print(f"[phase3-smoke] model={model.__class__.__name__}")
    print(f"[phase3-smoke] best_checkpoint={summary['best_checkpoint']}")
    print(f"[phase3-smoke] summary={run_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
