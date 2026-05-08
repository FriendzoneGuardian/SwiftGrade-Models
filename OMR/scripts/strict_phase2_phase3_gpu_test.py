#!/usr/bin/env python3
"""Strict Phase 2 -> Phase 3 GPU test pipeline.

Flow:
1) Clear and rebuild Phase_2_Cropped with 3x augmentation per source sample.
2) Purge Phase_2_Cropped into Phase_3_Ready using solidity/darkness gating.
3) Train selected Phase 3 model (GPU-first) on Phase_3_Ready.
4) Save threshold sweep and a 200-sample visual proof grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict GPU pipeline for Phase 2/3 test.")
    parser.add_argument("--project-root", type=str, default="/root/projects/SwiftGrade-Models")
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-count", type=int, default=200)
    parser.add_argument("--rotate-limit", type=int, default=20)
    parser.add_argument("--decision-floor", type=float, default=0.80)
    parser.add_argument("--max-source-images", type=int, default=None)
    parser.add_argument("--ground-truth-basis", action="store_true")
    parser.add_argument("--target-augmented", type=int, default=None)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--diamond-depth", type=int, choices=[5, 9, 11], default=5)
    parser.add_argument("--diamond-unify-channels", type=int, default=32)
    parser.add_argument("--diamond-final-projection", action="store_true")
    parser.add_argument("--model-method", type=str, choices=["diamond", "ascending", "transfer"], default="diamond")
    parser.add_argument("--transfer-unfreeze", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_augmenters(rotate_limit: int) -> list[A.Compose]:
    return [
        A.Compose(
            [
                A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REPLICATE, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.CLAHE(clip_limit=(1.5, 3.0), tile_grid_size=(8, 8), p=1.0),
            ]
        ),
        A.Compose(
            [
                A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REPLICATE, p=1.0),
                A.RandomGamma(gamma_limit=(75, 140), p=1.0),
                A.Sharpen(alpha=(0.15, 0.35), lightness=(0.8, 1.2), p=1.0),
            ]
        ),
        A.Compose(
            [
                A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REPLICATE, p=1.0),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=20, val_shift_limit=12, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=0.8),
            ]
        ),
    ]


def augment_phase2(
    source_images: list[Path],
    source_root: Path,
    phase2_out: Path,
    rotate_limit: int,
    preserve_labels: bool = False,
    target_augmented: int | None = None,
) -> dict[str, int]:
    augmenters = build_augmenters(rotate_limit)
    out_count = 0
    labeled_out_count = 0
    known_labels = {"blank", "filled", "uncertain", "crossed", "invalid"}

    for idx, src in enumerate(source_images, start=1):
        image = cv2.imread(str(src))
        if image is None:
            continue

        rel = src.relative_to(source_root)
        label_name = rel.parts[0].lower() if rel.parts else None
        if preserve_labels and label_name not in known_labels:
            continue
        digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
        stem = src.stem

        for aug_idx, augmenter in enumerate(augmenters, start=1):
            aug_image = augmenter(image=image)["image"]
            file_name = f"{stem}__{digest}__aug{aug_idx}.png"
            target_dir = (phase2_out / label_name) if preserve_labels else phase2_out
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / file_name
            cv2.imwrite(str(out_path), aug_image)
            out_count += 1
            if preserve_labels:
                labeled_out_count += 1
            if target_augmented is not None and out_count >= target_augmented:
                break

        if target_augmented is not None and out_count >= target_augmented:
            break

        if idx % 5000 == 0:
            print(f"[augment] processed={idx} / {len(source_images)}  output={out_count}")

    result = {"source_count": len(source_images), "augmented_count": out_count}
    if preserve_labels:
        result["labeled_augmented_count"] = labeled_out_count
    return result


def import_phase3_modules(project_root: Path):
    phase3_src = project_root / "OMR" / "Phase_3_Classification" / "src"
    if str(phase3_src) not in sys.path:
        sys.path.insert(0, str(phase3_src))

    from cnn_models import AscendingCNN, DiamondCNN, TransferLearningCNN
    from dataset import create_dataloaders
    from purge_data import PurgeConfig, run_purge
    from trainer import ClassificationTrainer, TrainerConfig

    return AscendingCNN, DiamondCNN, TransferLearningCNN, create_dataloaders, PurgeConfig, run_purge, ClassificationTrainer, TrainerConfig


def evaluate_and_sweep(
    model: torch.nn.Module,
    valid_loader,
    filled_idx: int,
    decision_floor: float,
) -> tuple[list[dict], dict, dict]:
    all_probs: list[float] = []
    all_targets: list[int] = []

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            logits = model(x.to(next(model.parameters()).device))
            probs = torch.softmax(logits, dim=1)[:, filled_idx]
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_targets.extend(y.detach().cpu().numpy().tolist())

    probs_np = np.array(all_probs)
    targets_np = np.array(all_targets)

    sweep: list[dict] = []
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        pred = (probs_np >= thr).astype(int)
        tp = int(((pred == 1) & (targets_np == 1)).sum())
        fp = int(((pred == 1) & (targets_np == 0)).sum())
        tn = int(((pred == 0) & (targets_np == 0)).sum())
        fn = int(((pred == 0) & (targets_np == 1)).sum())
        n = int(len(targets_np))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / n if n else 0.0

        sweep.append(
            {
                "threshold": thr,
                "samples": n,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision_filled": precision,
                "recall_filled": recall,
                "accuracy": accuracy,
            }
        )

    by_threshold = {row["threshold"]: row for row in sweep}
    requested_floor = max(0.0, min(1.0, float(decision_floor)))
    start_row = by_threshold.get(0.80)

    # User policy: try 0.80 first, then dynamically adjust if floor is not met.
    if start_row and start_row["accuracy"] >= requested_floor:
        best = start_row
        policy = {
            "requested_floor": requested_floor,
            "start_threshold": 0.80,
            "selected_threshold": start_row["threshold"],
            "start_accuracy": start_row["accuracy"],
            "floor_met_at_start": True,
            "adjusted": False,
            "reason": "start-threshold-meets-floor",
        }
        return sweep, best, policy

    eligible = [row for row in sweep if row["accuracy"] >= requested_floor]
    if eligible:
        zero_fp_eligible = [row for row in eligible if row["fp"] == 0]
        if zero_fp_eligible:
            best = max(zero_fp_eligible, key=lambda row: (row["recall_filled"], row["accuracy"], -row["threshold"]))
        else:
            best = max(eligible, key=lambda row: (row["precision_filled"], row["recall_filled"], row["accuracy"]))
        policy = {
            "requested_floor": requested_floor,
            "start_threshold": 0.80,
            "selected_threshold": best["threshold"],
            "start_accuracy": start_row["accuracy"] if start_row else None,
            "floor_met_at_start": False,
            "adjusted": True,
            "reason": "dynamic-adjustment-floor-met",
        }
        return sweep, best, policy

    best = max(sweep, key=lambda row: (row["accuracy"], row["precision_filled"], row["recall_filled"]))
    policy = {
        "requested_floor": requested_floor,
        "start_threshold": 0.80,
        "selected_threshold": best["threshold"],
        "start_accuracy": start_row["accuracy"] if start_row else None,
        "floor_met_at_start": False,
        "adjusted": True,
        "reason": "dynamic-adjustment-floor-unmet-best-possible",
    }
    return sweep, best, policy


def assess_standard(train_summary: dict, recommended_threshold: dict, threshold_policy: dict, floor: float) -> dict:
    history = train_summary.get("history", [])
    last = history[-1] if history else {}
    val_acc = float(last.get("val_acc", 0.0))
    val_f1 = float(last.get("val_f1", 0.0))
    floor_value = max(0.0, min(1.0, float(floor)))

    pass_gate = val_acc >= floor_value and val_f1 >= floor_value
    return {
        "floor": floor_value,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "pass": pass_gate,
        "summary": "PASS" if pass_gate else "FAIL",
        "selected_threshold": recommended_threshold.get("threshold"),
        "threshold_policy": threshold_policy,
    }


def generate_grid_200(
    model: torch.nn.Module,
    valid_loader,
    idx_to_class: dict[int, str],
    filled_idx: int,
    threshold: float,
    out_path: Path,
    grid_count: int,
    model_label: str,
) -> dict:
    model.eval()

    tiles: list[dict] = []
    with torch.no_grad():
        for x, y in valid_loader:
            logits = model(x.to(next(model.parameters()).device))
            probs = torch.softmax(logits, dim=1)
            pf = probs[:, filled_idx].detach().cpu().numpy()
            pred = (pf >= threshold).astype(int)

            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for i in range(len(y_np)):
                if len(tiles) >= grid_count:
                    break
                img = np.transpose(x_np[i], (1, 2, 0))
                img = np.clip(img, 0.0, 1.0)
                truth = idx_to_class[int(y_np[i])]
                pred_name = idx_to_class[int(pred[i])]
                ok = truth == pred_name
                tiles.append(
                    {
                        "img": img,
                        "truth": truth,
                        "pred": pred_name,
                        "pf": float(pf[i]),
                        "ok": ok,
                    }
                )
            if len(tiles) >= grid_count:
                break

    cols = 20
    rows = max(1, int(np.ceil(len(tiles) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= len(tiles):
            continue
        tile = tiles[i]
        ax.imshow(tile["img"])
        prefix = "OK" if tile["ok"] else "ERR"
        ax.set_title(f"{prefix}\nT:{tile['truth'][0]} P:{tile['pred'][0]}\nPf={tile['pf']:.2f}", fontsize=6)

    fig.suptitle(f"Phase 3 {model_label} Validation Grid ({len(tiles)} samples, thr={threshold:.2f})", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    false_count = sum(1 for tile in tiles if not tile["ok"])
    return {"grid_count": len(tiles), "false_count": false_count, "image_path": str(out_path)}


def generate_bucketed_grid_200(
    model: torch.nn.Module,
    valid_loader,
    idx_to_class: dict[int, str],
    filled_idx: int,
    threshold: float,
    out_path: Path,
    grid_count: int,
    seed: int,
) -> dict:
    model.eval()
    records: list[dict] = []

    with torch.no_grad():
        for x, y in valid_loader:
            logits = model(x.to(next(model.parameters()).device))
            probs = torch.softmax(logits, dim=1)
            pf = probs[:, filled_idx].detach().cpu().numpy()
            pred = (pf >= threshold).astype(int)

            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for i in range(len(y_np)):
                img = np.transpose(x_np[i], (1, 2, 0))
                img = np.clip(img, 0.0, 1.0)
                truth = idx_to_class[int(y_np[i])]
                gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                dark_ratio = float(np.mean(gray < (170.0 / 255.0)))
                records.append(
                    {
                        "img": img,
                        "truth": truth,
                        "pf": float(pf[i]),
                        "pred_filled": int(pred[i]),
                        "dark_ratio": dark_ratio,
                    }
                )

    filled_darkness = [r["dark_ratio"] for r in records if r["truth"] == "filled"]
    q40 = float(np.quantile(filled_darkness, 0.40)) if filled_darkness else None
    q60 = float(np.quantile(filled_darkness, 0.60)) if filled_darkness else None

    for r in records:
        if r["truth"] == "filled":
            if q40 is not None and r["dark_ratio"] <= q40:
                r["bucket"] = "light_filled"
            elif q60 is not None and r["dark_ratio"] >= q60:
                r["bucket"] = "dark_filled"
            else:
                r["bucket"] = "filled_mid"
        elif r["truth"] == "blank":
            r["bucket"] = "blank"
        elif r["truth"] == "invalid":
            r["bucket"] = "invalid"
        else:
            r["bucket"] = "other"

    rng = random.Random(seed)
    required = ["light_filled", "dark_filled", "blank", "invalid"]
    target_each = max(1, grid_count // len(required))
    selected: list[dict] = []
    bucket_available: dict[str, int] = {}

    for b in required:
        items = [r for r in records if r["bucket"] == b]
        bucket_available[b] = len(items)
        rng.shuffle(items)
        selected.extend(items[:target_each])

    if len(selected) < grid_count:
        selected_ids = {id(r) for r in selected}
        fill_pool = [r for r in records if id(r) not in selected_ids]
        fill_pool.sort(key=lambda r: 0 if r["bucket"] in required else 1)
        selected.extend(fill_pool[: max(0, grid_count - len(selected))])

    selected = selected[:grid_count]

    cols = 20
    rows = max(1, int(np.ceil(len(selected) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)

    bucket_counts: dict[str, int] = {k: 0 for k in required}
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= len(selected):
            continue
        tile = selected[i]
        ax.imshow(tile["img"])
        b = tile["bucket"]
        if b in bucket_counts:
            bucket_counts[b] += 1
        pred_name = "filled" if tile["pred_filled"] == 1 else "blank"
        title = f"{b}\nT:{tile['truth'][:1]} P:{pred_name[:1]}\nPf={tile['pf']:.2f}"
        ax.set_title(title, fontsize=6)

    fig.suptitle("Phase 3 Sorter Grid (Light Filled / Dark Filled / Blank / Invalid)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "grid_count": len(selected),
        "bucket_counts": bucket_counts,
        "bucket_available": bucket_available,
        "darkness_q40": q40,
        "darkness_q60": q60,
        "image_path": str(out_path),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    set_seed(args.seed)

    source_root = Path(args.source_root).resolve() if args.source_root else (project_root / "Unified_Datasets" / "manual_labeled")
    phase2_out = project_root / "Unified_Datasets" / "Phase_2_Cropped"
    phase3_ready = project_root / "Unified_Datasets" / "Phase_3_Ready"
    phase3_reject = project_root / "Unified_Datasets" / "Phase_3_Rejects"

    print(f"[strict] project_root={project_root}")
    print(f"[strict] source_root={source_root}")
    print(f"[strict] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[strict] cuda_device={torch.cuda.get_device_name(0)}")

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    source_images = list_images(source_root)
    random.Random(args.seed).shuffle(source_images)
    if args.max_source_images is not None and args.max_source_images > 0:
        source_images = source_images[: args.max_source_images]
    if not source_images:
        raise ValueError(f"No source images found under: {source_root}")

    expected_augmented = args.target_augmented if args.target_augmented is not None else (len(source_images) * 3)
    phase2_existing = count_images(phase2_out)
    phase3_existing = count_images(phase3_ready)

    if args.ground_truth_basis:
        reuse_phase3 = phase3_existing >= expected_augmented
        if reuse_phase3:
            print("[strict] ground-truth basis mode: detected existing augmented Phase_3_Ready; starting from Phase 3")
            aug_counts = {
                "source_count": len(source_images),
                "augmented_count": phase3_existing,
                "labeled_augmented_count": phase3_existing,
                "reused_existing": True,
                "expected_augmented": expected_augmented,
            }
        else:
            print("[strict] ground-truth basis mode: no full augmentation found; starting from Phase 2-style augmentation")
            clear_directory(phase3_ready)
            clear_directory(phase3_reject)
            aug_counts = augment_phase2(
                source_images,
                source_root,
                phase3_ready,
                args.rotate_limit,
                preserve_labels=True,
                target_augmented=expected_augmented,
            )
            aug_counts["reused_existing"] = False
            aug_counts["expected_augmented"] = expected_augmented
    else:
        reuse_phase2 = phase2_existing >= expected_augmented
        if reuse_phase2:
            print("[strict] detected existing augmented Phase_2_Cropped; starting from Phase 3 purge/training")
            aug_counts = {
                "source_count": len(source_images),
                "augmented_count": phase2_existing,
                "reused_existing": True,
                "expected_augmented": expected_augmented,
            }
        else:
            print("[strict] clearing Phase_2_Cropped and rebuilding with 3x augmentation per sample")
            clear_directory(phase2_out)
            aug_counts = augment_phase2(
                source_images,
                source_root,
                phase2_out,
                args.rotate_limit,
                target_augmented=expected_augmented,
            )
            aug_counts["reused_existing"] = False
            aug_counts["expected_augmented"] = expected_augmented

        print("[strict] clearing Phase_3_Ready / Phase_3_Rejects before purge")
        clear_directory(phase3_ready)
        clear_directory(phase3_reject)

    AscendingCNN, DiamondCNN, TransferLearningCNN, create_dataloaders, PurgeConfig, run_purge, ClassificationTrainer, TrainerConfig = import_phase3_modules(project_root)

    report_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = project_root / "OMR" / "runs" / f"strict_{report_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.ground_truth_basis:
        purge_counts = {
            "mode": "ground_truth_basis",
            "bypassed": True,
            "total": 0,
            "kept_filled": 0,
            "kept_blank": 0,
            "kept_uncertain": 0,
            "rejected": 0,
        }
    else:
        purge_config = PurgeConfig(
            source_root=phase2_out,
            output_root=phase3_ready,
            reject_root=phase3_reject,
            report_csv=run_dir / "purge_report.csv",
            solidity_threshold=0.74,
            darkness_threshold=0.08,
            uncertainty_margin=0.04,
            dry_run=False,
        )
        purge_counts = run_purge(purge_config)

    def resize_transform(image):
        return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    bundle = create_dataloaders(
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers),
        val_split_ratio=0.2,
        random_seed=args.seed,
        sample_limit=args.target_augmented,
        auto_sample_threshold=10**9,
        transform_train=resize_transform,
        transform_valid=resize_transform,
        rebuild_index=not args.reuse_index,
    )

    class_to_idx = bundle.class_to_idx
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    filled_idx = class_to_idx.get("filled")
    if filled_idx is None:
        raise ValueError("'filled' class is required for strict false-positive control.")

    if args.model_method == "diamond":
        model = DiamondCNN(
            num_classes=len(class_names),
            depth_blocks=args.diamond_depth,
            unify_channels=max(8, args.diamond_unify_channels),
            use_unification=bool(args.diamond_final_projection),
        )
    elif args.model_method == "ascending":
        model = AscendingCNN(num_classes=len(class_names))
    else:
        model = TransferLearningCNN(
            num_classes=len(class_names),
            freeze_backbone=not bool(args.transfer_unfreeze),
        )
        if args.transfer_unfreeze:
            model.unfreeze_all()
    trainer_config = TrainerConfig(
        epochs=max(1, args.epochs),
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=max(1, args.patience),
        early_stopping_min_delta=1e-4,
        output_dir=str(run_dir / "models"),
        monitor_metric="val_f1",
        device="cuda" if torch.cuda.is_available() else "auto",
    )
    trainer = ClassificationTrainer(model=model, class_names=class_names, config=trainer_config)
    train_summary = trainer.train(train_loader=bundle.train_loader, valid_loader=bundle.valid_loader)

    sweep, best, threshold_policy = evaluate_and_sweep(
        trainer.model,
        bundle.valid_loader,
        filled_idx,
        decision_floor=args.decision_floor,
    )
    standard_assessment = assess_standard(
        train_summary=train_summary,
        recommended_threshold=best,
        threshold_policy=threshold_policy,
        floor=args.decision_floor,
    )

    grid_path = run_dir / "proof_grid_200.png"
    grid_info = generate_grid_200(
        trainer.model,
        bundle.valid_loader,
        idx_to_class,
        filled_idx,
        best["threshold"],
        grid_path,
        args.grid_count,
        model.__class__.__name__,
    )

    bucket_grid_path = run_dir / "proof_grid_200_sorter_buckets.png"
    bucket_grid_info = generate_bucketed_grid_200(
        trainer.model,
        bundle.valid_loader,
        idx_to_class,
        filled_idx,
        best["threshold"],
        bucket_grid_path,
        args.grid_count,
        seed=args.seed,
    )

    report = {
        "project_root": str(project_root),
        "source_root": str(source_root),
        "phase2_output": str(phase2_out),
        "phase3_ready": str(phase3_ready),
        "phase3_reject": str(phase3_reject),
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "ground_truth_basis": bool(args.ground_truth_basis),
        "augmentation": aug_counts,
        "purge": purge_counts,
        "class_to_idx": class_to_idx,
        "model_config": {
            "name": model.__class__.__name__,
            "method": args.model_method,
            "diamond_depth": args.diamond_depth,
            "diamond_unify_channels": max(8, args.diamond_unify_channels),
            "diamond_final_projection": bool(args.diamond_final_projection),
            "transfer_unfreeze": bool(args.transfer_unfreeze),
        },
        "train_batches": len(bundle.train_loader),
        "valid_batches": len(bundle.valid_loader),
        "train_summary": train_summary,
        "threshold_sweep": sweep,
        "recommended_threshold": best,
        "threshold_policy": threshold_policy,
        "standard_assessment": standard_assessment,
        "grid_200": grid_info,
        "grid_200_sorter_buckets": bucket_grid_info,
    }

    report_path = run_dir / "strict_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[strict] run_dir={run_dir}")
    print(f"[strict] augmented_count={aug_counts['augmented_count']}")
    print(f"[strict] purge_counts={purge_counts}")
    print(f"[strict] recommended_threshold={best['threshold']}")
    print(f"[strict] standard={standard_assessment['summary']} floor={standard_assessment['floor']}")
    print(f"[strict] grid_200={grid_path}")
    print(f"[strict] grid_200_sorter_buckets={bucket_grid_path}")
    print(f"[strict] report={report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
