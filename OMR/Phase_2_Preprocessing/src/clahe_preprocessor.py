"""
clahe_preprocessor.py – Multi-channel CLAHE preprocessing for OMR images.

Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
luminance channel in LAB colour space, avoiding hue distortion while
normalising uneven illumination.  See ``LEGACY_REFERENCE.md`` for the
mathematical background.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class CLAHEPreprocessor:
    """Applies multi-channel CLAHE in LAB colour space to BGR images.

    Processing pipeline for :meth:`process`:

    1. Optional Gaussian blur (3 × 3 kernel) to suppress JPEG block artefacts.
    2. Convert BGR → LAB.
    3. Apply CLAHE to the L (luminance) channel only.
    4. Merge channels and convert LAB → BGR.

    Parameters
    ----------
    clip_limit:
        CLAHE clip limit.  Controls the maximum slope of the CDF mapping and
        thus the degree of noise amplification.  Defaults to ``2.0``.
    tile_grid_size:
        ``(cols, rows)`` tile grid for adaptive histogram computation.
        Defaults to ``(8, 8)``.
    denoise:
        When ``True`` (default), apply a 3 × 3 Gaussian blur before CLAHE to
        attenuate JPEG compression artefacts.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        denoise: bool = True,
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise = denoise
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-channel CLAHE to a single BGR image.

        Parameters
        ----------
        image:
            Input BGR image as a ``numpy.ndarray`` of shape ``(H, W, 3)``
            and dtype ``uint8``.

        Returns
        -------
        numpy.ndarray
            Preprocessed BGR image, same shape and dtype as input.
        """
        if self.denoise:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_equalized = self._clahe.apply(l_channel)

        lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
        result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        return result

    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply :meth:`process` to every image in *images*.

        Parameters
        ----------
        images:
            List of BGR ``numpy.ndarray`` images.

        Returns
        -------
        list[numpy.ndarray]
            List of preprocessed BGR images in the same order.
        """
        return [self.process(img) for img in images]

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert a preprocessed BGR image to a normalised PyTorch tensor.

        Conversion steps:

        1. BGR → RGB channel reorder.
        2. HWC → CHW axis transposition.
        3. Cast to ``float32`` and normalise to ``[0, 1]``.

        Parameters
        ----------
        image:
            Preprocessed BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(3, H, W)`` with values in ``[0, 1]``.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.from_numpy(chw)
