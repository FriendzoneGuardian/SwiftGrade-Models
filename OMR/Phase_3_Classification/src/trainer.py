"""
trainer.py - Modular training loop for Phase 3 classification.

Includes:
- deterministic training behavior,
- early stopping,
- checkpointing,
- JSONL metric logs for papertrail workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    output_dir: str = "OMR/Phase_3_Classification/models"
    monitor_metric: str = "val_f1"
    device: str = "auto"


class ClassificationTrainer:
    """Training wrapper for binary or multi-class bubble classification."""

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.config = config or TrainerConfig()
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_jsonl_path = self.output_dir / "training_epoch_metrics.jsonl"
        self.summary_path = self.output_dir / "training_summary.json"
        self.best_checkpoint_path = self.output_dir / "best_model.pth"

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        criterion = criterion or nn.CrossEntropyLoss()
        optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_metric = -float("inf")
        best_epoch = -1
        patience_counter = 0
        history: list[dict[str, Any]] = []

        if self.metrics_jsonl_path.exists():
            self.metrics_jsonl_path.unlink()

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                train_mode=True,
            )
            valid_metrics = self._run_epoch(
                loader=valid_loader,
                criterion=criterion,
                optimizer=None,
                train_mode=False,
            )

            

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": valid_metrics["loss"],
                "val_acc": valid_metrics["accuracy"],
                "val_f1": valid_metrics["f1"],
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            }
            history.append(epoch_metrics)
            self._append_jsonl(self.metrics_jsonl_path, epoch_metrics)

            monitor_value = float(epoch_metrics.get(self.config.monitor_metric, valid_metrics["f1"]))
            # Step scheduler with metric if required by its API (e.g., ReduceLROnPlateau)
            if scheduler is not None:
                try:
                    scheduler.step(monitor_value)
                except TypeError:
                    # Older/newer schedulers may accept no-arg step()
                    scheduler.step()

            improved = monitor_value > (best_metric + self.config.early_stopping_min_delta)
            if improved:
                best_metric = monitor_value
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(
                    path=self.best_checkpoint_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    metric_value=monitor_value,
                )
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        summary = {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "monitor_metric": self.config.monitor_metric,
            "epochs_ran": len(history),
            "class_names": self.class_names,
            "best_checkpoint": str(self.best_checkpoint_path),
            "metrics_jsonl": str(self.metrics_jsonl_path),
            "history": history,
        }

        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        train_mode: bool,
    ) -> dict[str, float]:
        self.model.train(mode=train_mode)

        losses: list[float] = []
        all_targets: list[int] = []
        all_preds: list[int] = []

        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train_mode):
                logits = self.model(inputs)
                loss = criterion(logits, targets)

            if train_mode and optimizer is not None:
                loss.backward()
                optimizer.step()

            losses.append(float(loss.detach().item()))
            preds = torch.argmax(logits, dim=1)
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

        if not losses:
            raise ValueError("Encountered an empty dataloader while training/evaluating.")

        accuracy = float(np.mean(np.asarray(all_preds) == np.asarray(all_targets)))
        f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        return {"loss": float(np.mean(losses)), "accuracy": accuracy, "f1": f1}

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                "Batch format is invalid. Expected tuple/list like (inputs, targets). "
                "Check dataset __getitem__ return values."
            )
        inputs = batch[0]
        targets = batch[1]
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError("Batch items must be torch tensors.")
        return inputs, targets.long()

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metric_value: float,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "metric_value": metric_value,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": self.class_names,
            },
            path,
        )
