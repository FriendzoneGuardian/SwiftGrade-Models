"""
omr_dataset.py – PyTorch Dataset for OMR sheet images.

Provides a standard ``torch.utils.data.Dataset`` interface over a collection
of OMR image files, with optional label support and transform pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OMRDataset(Dataset):
    """Dataset of OMR sheet images loaded from disk.

    Images are loaded as BGR ``numpy.ndarray`` objects via OpenCV so that they
    are immediately compatible with the Phase 2 :class:`CLAHEPreprocessor`.
    An optional ``transform`` callable is applied after loading, enabling
    on-the-fly augmentation or tensor conversion.

    Parameters
    ----------
    image_paths:
        Ordered list of paths to image files.
    labels:
        Optional list of labels (one per image).  May be any type – integers
        for classification, dicts for structured annotations, etc.  When
        ``None``, ``__getitem__`` returns only the image array.
    transform:
        Optional callable ``(image: np.ndarray) -> Any`` applied to each
        loaded image.  Receives a BGR ``numpy.ndarray`` and should return
        whatever format the downstream model expects.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: Optional[list] = None,
        transform=None,
    ) -> None:
        if labels is not None and len(labels) != len(image_paths):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) must match "
                f"labels length ({len(labels)})."
            )
        self._image_paths = [Path(p) for p in image_paths]
        self._labels = labels
        self.transform = transform

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Any:
        """Load and return the sample at *index*.

        Parameters
        ----------
        index:
            Integer index in ``[0, len(self))``.

        Returns
        -------
        image : np.ndarray or transformed type
            BGR image (or the result of applying ``self.transform``).
        label : optional
            Present only when labels were provided at construction time.
        """
        path = self._image_paths[index]
        image: np.ndarray = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at '{path}'.")

        if self.transform is not None:
            image = self.transform(image)

        if self._labels is not None:
            return image, self._labels[index]
        return image

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def image_paths(self) -> list[Path]:
        """List of image paths as :class:`pathlib.Path` objects."""
        return list(self._image_paths)

    @property
    def labels(self) -> Optional[list]:
        """Labels list, or ``None`` if no labels were provided."""
        return self._labels
