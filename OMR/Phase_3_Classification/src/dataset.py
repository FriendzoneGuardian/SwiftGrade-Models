"""
dataset.py - Dataset indexing and loading utilities for Phase 3 classification.

This module addresses three recurring pipeline failures:
1) brittle dataset paths,
2) ad-hoc train/val splits,
3) non-reproducible sampling behavior on large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_LABEL_ALIASES = {
    "filled": "filled",
    "fill": "filled",
    "marked": "filled",
    "blank": "blank",
    "empty": "blank",
    "uncertain": "uncertain",
    "review": "uncertain",
    "crossed": "crossed",
    "invalid": "invalid",
}
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}


@dataclass(frozen=True)
class DatasetRootResolution:
    dataset_root: Path
    source_name: str


@dataclass(frozen=True)
class DataLoadersBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    class_to_idx: dict[str, int]
    index_path: Path
    dataset_root: Path
    sample_mode: str


class IndexedOMRDataset(Dataset):
    """PyTorch dataset backed by a dataframe index."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self._index = index_df.reset_index(drop=True)
        self._class_to_idx = class_to_idx
        self._transform = transform

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._index.iloc[idx]
        image_path = Path(row["path"])
        label_name = str(row["label"])

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at '{image_path}'.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self._transform is not None:
            try:
                transformed = self._transform(image=image_rgb)
            except TypeError:
                transformed = self._transform(image_rgb)
            image = transformed["image"] if isinstance(transformed, dict) and "image" in transformed else transformed
        else:
            image = image_rgb

        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
            if image_tensor.ndim == 3 and image_tensor.shape[0] not in (1, 3):
                image_tensor = image_tensor.permute(2, 0, 1)
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
        else:
            image_np = np.asarray(image)
            if image_np.ndim != 3:
                raise ValueError("Transformed image must be HxWxC or CxHxW tensor.")
            if image_np.shape[0] in (1, 3):
                chw = image_np.astype(np.float32)
            else:
                chw = np.transpose(image_np, (2, 0, 1)).astype(np.float32)
            if chw.max() > 1.0:
                chw = chw / 255.0
            image_tensor = torch.from_numpy(chw)

        label_idx = self._class_to_idx[label_name]
        return image_tensor, label_idx


def resolve_phase3_dataset_root(project_root: str | Path | None = None) -> DatasetRootResolution:
    """Resolve canonical Phase 3 dataset root with stable fallback behavior."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    candidates: list[tuple[str, Path]] = [
        ("phase3_ready", root / "Unified_Datasets" / "Phase_3_Ready"),
        ("manual_labeled", root / "Unified_Datasets" / "manual_labeled"),
    ]

    for source_name, path in candidates:
        if not path.exists():
            continue
        if any(_is_image_file(p) for p in path.rglob("*")):
            return DatasetRootResolution(dataset_root=path, source_name=source_name)

    raise FileNotFoundError(
        "No dataset root resolved. Expected images in either "
        "'Unified_Datasets/Phase_3_Ready' or 'Unified_Datasets/manual_labeled'."
    )


def build_dataset_index(
    dataset_root: str | Path,
    index_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and optionally persist a metadata-only dataset index."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: '{root}'")

    rows: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not _is_image_file(path):
            continue

        split, label = _extract_split_and_label(path, root)
        if label is None:
            continue

        stat = path.stat()
        rows.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "split": split,
                "label": label,
                "file_size": int(stat.st_size),
                "modified_time": float(stat.st_mtime),
            }
        )

    if not rows:
        raise ValueError(
            "No labeled images found while indexing dataset. "
            "Expected class folders such as filled/blank/uncertain."
        )

    df = pd.DataFrame(rows).sort_values(by=["split", "label", "relative_path"]).reset_index(drop=True)

    if index_output_path is not None:
        output_path = Path(index_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def create_dataloaders(
    project_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split_ratio: float = 0.2,
    random_seed: int = 42,
    sample_limit: int | None = None,
    auto_sample_threshold: int = 5000,
    transform_train: Callable[[Any], Any] | None = None,
    transform_valid: Callable[[Any], Any] | None = None,
    rebuild_index: bool = False,
) -> DataLoadersBundle:
    """Create train/valid dataloaders with deterministic split and sampling."""
    resolution = resolve_phase3_dataset_root(project_root)
    index_path = resolution.dataset_root / "dataset_index.csv"

    if index_path.exists() and not rebuild_index:
        index_df = pd.read_csv(index_path)
    else:
        index_df = build_dataset_index(resolution.dataset_root, index_output_path=index_path)

    for col in ("path", "split", "label"):
        if col not in index_df.columns:
            raise ValueError(f"Dataset index missing required column '{col}'.")

    normalized = index_df.copy()
    normalized["label"] = normalized["label"].astype(str).str.lower()

    total_count = len(normalized)
    sample_mode = "full"
    effective_sample_limit = sample_limit
    if effective_sample_limit is None and total_count > auto_sample_threshold:
        effective_sample_limit = auto_sample_threshold

    if effective_sample_limit is not None and total_count > effective_sample_limit:
        normalized = _stratified_downsample(normalized, limit=effective_sample_limit, seed=random_seed)
        sample_mode = "stratified"

    if {"train", "valid"}.issubset(set(normalized["split"].unique())):
        train_df = normalized[normalized["split"] == "train"].copy()
        valid_df = normalized[normalized["split"] == "valid"].copy()
    else:
        train_df, valid_df = _stratified_train_valid_split(
            normalized,
            val_ratio=val_split_ratio,
            seed=random_seed,
        )

    class_names = sorted(train_df["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_dataset = IndexedOMRDataset(train_df, class_to_idx=class_to_idx, transform=transform_train)
    valid_dataset = IndexedOMRDataset(valid_df, class_to_idx=class_to_idx, transform=transform_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataLoadersBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        class_to_idx=class_to_idx,
        index_path=index_path,
        dataset_root=resolution.dataset_root,
        sample_mode=sample_mode,
    )


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS


def _normalize_label(name: str) -> str | None:
    key = name.strip().lower()
    return _LABEL_ALIASES.get(key)


def _normalize_split(name: str) -> str | None:
    key = name.strip().lower()
    return _SPLIT_ALIASES.get(key)


def _extract_split_and_label(path: Path, root: Path) -> tuple[str, str | None]:
    rel_parts = list(path.relative_to(root).parts)
    if len(rel_parts) < 2:
        return "unspecified", None

    split = _normalize_split(rel_parts[0])
    if split is not None and len(rel_parts) >= 3:
        label = _normalize_label(rel_parts[1])
        return split, label

    label = _normalize_label(rel_parts[0])
    return "unspecified", label


def _stratified_downsample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()

    fractions = df["label"].value_counts(normalize=True)
    sampled_parts: list[pd.DataFrame] = []

    for label, frac in fractions.items():
        target = max(1, int(round(frac * limit)))
        group = df[df["label"] == label]
        sampled_parts.append(group.sample(n=min(target, len(group)), random_state=seed))

    sampled = pd.concat(sampled_parts, axis=0)
    if len(sampled) > limit:
        sampled = sampled.sample(n=limit, random_state=seed)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _stratified_train_valid_split(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_split_ratio must be between 0 and 1.")

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []

    for label, group in df.groupby("label"):
        shuffled = group.sample(frac=1.0, random_state=seed)
        valid_count = max(1, int(round(len(group) * val_ratio)))
        if valid_count >= len(group):
            valid_count = max(1, len(group) - 1)

        valid_parts.append(shuffled.iloc[:valid_count])
        train_parts.append(shuffled.iloc[valid_count:])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_df = pd.concat(valid_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if train_df.empty or valid_df.empty:
        raise ValueError("Train/valid split produced an empty partition.")

    return train_df, valid_df
