"""
train_phase3_3.py – Phase 3.3 MobileNetV2 training launcher.

Two-phase strategy
------------------
Phase A : Backbone frozen. Only classifier head trains.
          Runs for PHASE_A_EPOCHS or until early stopping.
          High LR acceptable — only 256+2 head parameters update.

Phase B : Last 3 InvertedResidual blocks + head unfrozen.
          LR drops to PHASE_B_LR to avoid destroying pretrained features.
          Continues until MAX_EPOCHS total or patience exhausted.

Usage
-----
    python train_phase3_3.py
    python train_phase3_3.py --project-root /root/projects/SwiftGrade-Models
    python train_phase3_3.py --batch-size 64 --phase-a-epochs 15
    python train_phase3_3.py --full-finetune   # skip Phase A, unfreeze all from epoch 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path resolution — allow running from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT_DEFAULT = _SCRIPT_DIR.parents[2]  # SwiftGrade-Models root
# Ensure the package search path includes this module's `src` directory
# (previous code added a non-existent nested `src/src` path).
sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import create_dataloaders
from trainer import ClassificationTrainer, TrainerConfig
from transfer_learning import TransferLearningCNN

# ---------------------------------------------------------------------------
# Hardcoded defaults (override via CLI)
# ---------------------------------------------------------------------------
PHASE_A_EPOCHS: int = 20       # frozen backbone warmup
PHASE_B_EPOCHS: int = 80       # unfrozen fine-tune
MAX_EPOCHS: int = 100          # hard ceiling
PATIENCE: int = 30             # early stopping patience (both phases)
BATCH_SIZE: int = 32
PHASE_A_LR: float = 1e-3
PHASE_B_LR: float = 1e-5       # must be low — pretrained weights are sensitive
WEIGHT_DECAY: float = 1e-4
RANDOM_SEED: int = 42
NUM_WORKERS: int = 4
DROPOUT: float = 0.3
N_UNFREEZE_BLOCKS: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir(project_root: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    # Place runs inside OMR/runs/ directory per workspace layout
    run_dir = project_root / "OMR" / "runs" / f"mobilenetv2_{timestamp}" / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _print_gpu_info() -> None:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[GPU] {name} — {mem:.1f} GB VRAM")
    else:
        print("[GPU] CUDA not available — training on CPU (expect very slow runtime)")


def _build_transforms():
    """
    Returns train and validation transforms using torchvision.

    Train : resize → random horizontal flip → color jitter → normalize
    Valid : resize → normalize only (no augmentation)

    ImageNet mean/std used because MobileNetV2 backbone was pretrained on ImageNet.
    """
    try:
        from torchvision import transforms as T

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        transform_train = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        transform_valid = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        return transform_train, transform_valid

    except Exception as exc:
        print(f"[WARN] Could not build torchvision transforms: {exc}")
        print("[WARN] Falling back to no-transform mode. Ensure images are already 224x224.")
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    _seed_everything(RANDOM_SEED)
    _print_gpu_info()

    project_root = Path(args.project_root)
    run_dir = _make_run_dir(project_root)
    print(f"[Run] Output directory: {run_dir}")

    # -- Transforms ----------------------------------------------------------
    transform_train, transform_valid = _build_transforms()

    # -- DataLoaders — support optional downsample for smoke tests ---------
    print(f"[Data] Building dataloaders (sample_limit={args.sample_limit})...")
    bundle = create_dataloaders(
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        val_split_ratio=0.2,
        random_seed=RANDOM_SEED,
        sample_limit=args.sample_limit,          # allow CLI override for smoke tests
        auto_sample_threshold=999_999_999,  # safety: disable auto-downsampling
        transform_train=transform_train,
        transform_valid=transform_valid,
        rebuild_index=False,
    )

    print(f"[Data] Classes       : {bundle.class_to_idx}")
    print(f"[Data] Sample mode   : {bundle.sample_mode}")
    print(f"[Data] Train batches : {len(bundle.train_loader)}")
    print(f"[Data] Valid batches : {len(bundle.valid_loader)}")

    class_names = sorted(bundle.class_to_idx, key=bundle.class_to_idx.get)
    num_classes = len(class_names)

    # -- Model ---------------------------------------------------------------
    freeze_on_init = not args.full_finetune
    model = TransferLearningCNN(
        num_classes=num_classes,
        freeze_backbone=freeze_on_init,
        dropout=DROPOUT,
    )

    print(f"[Model] Total params     : {model.total_parameter_count():,}")
    print(f"[Model] Trainable params : {model.trainable_parameter_count():,}")

    # =========================================================================
    # PHASE A — frozen backbone, classifier head only
    # =========================================================================
    phase_a_summary = None

    if not args.full_finetune:
        print("\n" + "=" * 60)
        print("PHASE A — Backbone frozen, training classifier head only")
        print("=" * 60)

        phase_a_dir = run_dir / "phase_a"
        phase_a_dir.mkdir(parents=True, exist_ok=True)

        config_a = TrainerConfig(
            epochs=args.phase_a_epochs,
            learning_rate=PHASE_A_LR,
            weight_decay=WEIGHT_DECAY,
            early_stopping_patience=args.patience,
            early_stopping_min_delta=1e-4,
            output_dir=str(phase_a_dir),
            monitor_metric="val_f1",
            device="auto",
        )

        trainer_a = ClassificationTrainer(
            model=model,
            class_names=class_names,
            config=config_a,
        )

        optimizer_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE_A_LR,
            weight_decay=WEIGHT_DECAY,
        )

        phase_a_summary = trainer_a.train(
            train_loader=bundle.train_loader,
            valid_loader=bundle.valid_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer_a,
        )

        print(f"[Phase A] Best epoch : {phase_a_summary['best_epoch']}")
        print(f"[Phase A] Best val_f1: {phase_a_summary['best_metric']:.6f}")

        # Unfreeze top blocks for Phase B
        model.unfreeze_top_blocks(n_blocks=N_UNFREEZE_BLOCKS)
        print(f"\n[Phase B prep] Unfroze top {N_UNFREEZE_BLOCKS} blocks")
        print(f"[Phase B prep] Trainable params now: {model.trainable_parameter_count():,}")

    else:
        # Full fine-tune from epoch 1
        model.unfreeze_all()
        print("[Mode] Full fine-tune — all parameters trainable from epoch 1")
        print(f"[Model] Trainable params : {model.trainable_parameter_count():,}")

    # =========================================================================
    # PHASE B — partial or full unfreeze, low LR
    # =========================================================================
    print("\n" + "=" * 60)
    if args.full_finetune:
        print("PHASE B — Full fine-tune (all layers)")
    else:
        print(f"PHASE B — Top {N_UNFREEZE_BLOCKS} blocks + head, LR={PHASE_B_LR}")
    print("=" * 60)

    phase_b_dir = run_dir / "phase_b"
    phase_b_dir.mkdir(parents=True, exist_ok=True)

    phase_b_epochs = MAX_EPOCHS if args.full_finetune else args.phase_b_epochs

    config_b = TrainerConfig(
        epochs=phase_b_epochs,
        learning_rate=PHASE_B_LR,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=1e-4,
        output_dir=str(phase_b_dir),
        monitor_metric="val_f1",
        device="auto",
    )

    trainer_b = ClassificationTrainer(
        model=model,
        class_names=class_names,
        config=config_b,
    )

    optimizer_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE_B_LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Create scheduler with graceful fallback for older PyTorch versions
    try:
        scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_b,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True,
        )
    except TypeError:
        scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_b,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
        )

    phase_b_summary = trainer_b.train(
        train_loader=bundle.train_loader,
        valid_loader=bundle.valid_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer_b,
        scheduler=scheduler_b,
    )

    print(f"[Phase B] Best epoch : {phase_b_summary['best_epoch']}")
    print(f"[Phase B] Best val_f1: {phase_b_summary['best_metric']:.6f}")

    # =========================================================================
    # Combined summary
    # =========================================================================
    combined_summary = {
        "model": "MobileNetV2",
        "task": "binary_bubble_classification",
        "class_names": class_names,
        "training_mode": "full_finetune" if args.full_finetune else "two_phase",
        "phase_a": phase_a_summary,
        "phase_b": phase_b_summary,
        "best_val_f1_overall": phase_b_summary["best_metric"],
        "run_dir": str(run_dir),
        "dataset_root": str(bundle.dataset_root),
        "sample_mode": bundle.sample_mode,
    }

    summary_path = run_dir / "combined_summary.json"
    summary_path.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print(f"\n[Done] Combined summary saved to: {summary_path}")

    # Compare against priors
    prior_diamond   = 0.9723
    prior_ascending = 0.9757
    final_f1 = phase_b_summary["best_metric"]
    print("\n[Comparison vs Priors]")
    print(f"  Diamond CNN (3.1)   : {prior_diamond:.4f}")
    print(f"  Ascending CNN (3.2) : {prior_ascending:.4f}")
    print(f"  MobileNetV2 (3.3)   : {final_f1:.4f}  {'✓ beats ascending' if final_f1 > prior_ascending else '✗ below ascending — valid finding'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.3 — MobileNetV2 transfer learning trainer")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(_PROJECT_ROOT_DEFAULT),
        help="Root of the SwiftGrade-Models repository",
    )
    parser.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--phase-a-epochs",  type=int,   default=PHASE_A_EPOCHS)
    parser.add_argument("--phase-b-epochs",  type=int,   default=PHASE_B_EPOCHS)
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Early stopping patience for both phases",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit dataset samples for quick smoke tests (default: full dataset)",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Skip Phase A and fine-tune all layers from epoch 1 at PHASE_B_LR",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())