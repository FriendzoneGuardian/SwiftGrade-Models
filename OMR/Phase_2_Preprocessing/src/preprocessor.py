"""
preprocessor.py - OMR Phase 2 preprocessing.

This module centralizes the paper-aligned Phase 2 path:

* CLAHE normalization
* optional blue-aware compensation
* significance-gated template subtraction
* inner masking
* variance and solidity diagnostics

The goal is to expose one reusable preprocessor for both notebook and runtime
use, while keeping the math aligned with the Phase 2 documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessorDiagnostics:
    used_template_subtraction: bool
    variance: float
    mean_gray: float
    darkness_score: float
    solidity: float
    inner_mask_ratio: float
    flat_region: bool
    likely_mark: bool
    subtraction_applied: bool
    blue_ratio: float


class OMRPreprocessor:
    """Phase 2 preprocessing for OMR bubble crops.

    Parameters
    ----------
    clip_limit:
        CLAHE clip limit.
    tile_grid_size:
        CLAHE tile grid.
    denoise:
        Apply a small Gaussian blur before CLAHE.
    inner_mask_ratio:
        Fraction of the crop width/height used for the circular inner mask.
    variance_threshold:
        Below this masked grayscale variance, the crop is treated as flat.
    solidity_threshold:
        Minimum contour solidity expected for a true filled mark.
    subtraction_threshold:
        Mean absolute difference required before template subtraction is used.
    subtraction_blend:
        Blend factor applied when significance-gated subtraction is active.
    blue_boost:
        Apply a conservative blue-sensitive pass after CLAHE.
    blue_boost_strength:
        Strength of the blue-sensitive pass.
    template_path:
        Optional path to a master blank template.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        denoise: bool = True,
        inner_mask_ratio: float = 0.60,
        variance_threshold: float = 120.0,
        darkness_threshold: float = 28.0,
        blue_ratio_threshold: float = 0.08,
        solidity_threshold: float = 0.68,
        subtraction_threshold: float = 24.0,
        subtraction_blend: float = 0.40,
        blue_boost: bool = True,
        blue_boost_strength: float = 0.22,
        template_path: str | Path | None = None,
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise = denoise
        self.inner_mask_ratio = inner_mask_ratio
        self.variance_threshold = variance_threshold
        self.darkness_threshold = darkness_threshold
        self.blue_ratio_threshold = blue_ratio_threshold
        self.solidity_threshold = solidity_threshold
        self.subtraction_threshold = subtraction_threshold
        self.subtraction_blend = subtraction_blend
        self.blue_boost = blue_boost
        self.blue_boost_strength = blue_boost_strength
        self.template_path = Path(template_path) if template_path is not None else None
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self._template_bgr = self._load_template(self.template_path)

    def process(
        self,
        image: np.ndarray,
        template: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process a BGR image and return a normalized BGR crop."""
        processed, _diagnostics = self.process_with_diagnostics(image, template=template)
        return processed

    def process_with_diagnostics(
        self,
        image: np.ndarray,
        template: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process a crop and return diagnostics for calibration.

        The diagnostics are meant for notebook-level calibration and papertrail
        inspection.
        """
        source = self._ensure_bgr_uint8(image)

        if self.denoise:
            source = cv2.GaussianBlur(source, (3, 3), sigmaX=1.0)

        clahe_bgr = self._apply_clahe(source)
        boosted_bgr = self._boost_blue_signal(clahe_bgr) if self.blue_boost else clahe_bgr

        template_bgr = self._resolve_template(template, boosted_bgr.shape)
        subtraction_applied = False
        used_template_subtraction = template_bgr is not None
        if template_bgr is not None:
            boosted_bgr, subtraction_applied = self._significance_gated_subtraction(
                boosted_bgr,
                template_bgr,
            )

        masked_gray, mask = self._masked_gray(boosted_bgr)
        variance = float(masked_gray.var()) if masked_gray.size else 0.0
        mean_gray = float(masked_gray.mean()) if masked_gray.size else 255.0
        darkness_score = 255.0 - mean_gray
        solidity = self._estimate_solidity(masked_gray, mask)
        blue_ratio = self._blue_ratio(boosted_bgr)
        likely_mark = (
            variance >= self.variance_threshold
            or darkness_score >= self.darkness_threshold
            or blue_ratio >= self.blue_ratio_threshold
        )
        flat_region = not likely_mark

        diagnostics = PreprocessorDiagnostics(
            used_template_subtraction=used_template_subtraction,
            variance=variance,
            mean_gray=mean_gray,
            darkness_score=darkness_score,
            solidity=solidity,
            inner_mask_ratio=self.inner_mask_ratio,
            flat_region=flat_region,
            likely_mark=likely_mark,
            subtraction_applied=subtraction_applied,
            blue_ratio=blue_ratio,
        )

        return boosted_bgr, diagnostics.__dict__

    def process_batch(
        self,
        images: list[np.ndarray],
        template: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        return [self.process(image, template=template) for image in images]

    def calibrate(
        self,
        images: list[np.ndarray],
        template: np.ndarray | None = None,
    ) -> dict[str, float]:
        if not images:
            raise ValueError("images must not be empty.")

        metrics = [self.process_with_diagnostics(image, template=template)[1] for image in images]
        return {
            "count": float(len(metrics)),
            "mean_variance": float(np.mean([item["variance"] for item in metrics])),
            "mean_darkness_score": float(np.mean([item["darkness_score"] for item in metrics])),
            "mean_solidity": float(np.mean([item["solidity"] for item in metrics])),
            "mean_blue_ratio": float(np.mean([item["blue_ratio"] for item in metrics])),
            "flat_rate": float(np.mean([1.0 if item["flat_region"] else 0.0 for item in metrics])),
            "likely_mark_rate": float(np.mean([1.0 if item["likely_mark"] else 0.0 for item in metrics])),
            "subtraction_rate": float(np.mean([1.0 if item["subtraction_applied"] else 0.0 for item in metrics])),
        }

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert a processed BGR image into a normalized tensor."""
        rgb = cv2.cvtColor(self._ensure_bgr_uint8(image), cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.from_numpy(chw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_equalized = self._clahe.apply(l_channel)
        lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    def _boost_blue_signal(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        blue_mask = ((hue >= 90.0) & (hue <= 150.0)).astype(np.float32)
        if not np.any(blue_mask):
            return image

        boost = 1.0 + (self.blue_boost_strength * blue_mask)
        hsv[:, :, 1] = np.clip(saturation * boost, 0.0, 255.0)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _significance_gated_subtraction(
        self,
        image: np.ndarray,
        template: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(image_gray, template_gray)
        mean_diff = float(diff.mean())
        if mean_diff < self.subtraction_threshold:
            return image, False

        gated = np.where(diff > self.subtraction_threshold, diff, 0).astype(np.float32)
        # Keep subtraction conservative for subtle fills: strong template matches are
        # suppressed, but low-level ink is allowed to survive.
        blend_scale = min(1.0, mean_diff / max(self.subtraction_threshold * 2.0, 1.0))
        effective_blend = self.subtraction_blend * blend_scale
        adjusted_gray = np.clip(image_gray.astype(np.float32) - gated * effective_blend, 0, 255)
        adjusted_bgr = cv2.cvtColor(adjusted_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return adjusted_bgr, True

    def _masked_gray(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = self._inner_mask(gray.shape)
        masked = gray[mask > 0]
        return masked, mask

    def _inner_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = int(min(height, width) * self.inner_mask_ratio * 0.5)
        radius = max(radius, 1)
        cv2.circle(mask, center, radius, 255, -1)
        return mask

    def _estimate_solidity(self, masked_gray: np.ndarray, mask: np.ndarray) -> float:
        if masked_gray.size == 0:
            return 0.0

        full_gray = np.zeros_like(mask, dtype=np.uint8)
        full_gray[mask > 0] = masked_gray.astype(np.uint8)
        _, thresh = cv2.threshold(full_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0.0:
            return 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0.0:
            return 0.0

        return contour_area / hull_area

    def _resolve_template(
        self,
        template: np.ndarray | None,
        target_shape: tuple[int, int, int],
    ) -> np.ndarray | None:
        resolved = template if template is not None else self._template_bgr
        if resolved is None:
            return None

        resolved = self._ensure_bgr_uint8(resolved)
        if resolved.shape[:2] != target_shape[:2]:
            resolved = cv2.resize(resolved, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return resolved

    @staticmethod
    def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] != 3:
            raise ValueError("image must have 3 channels in BGR format.")
        return image

    @staticmethod
    def _load_template(template_path: Path | None) -> np.ndarray | None:
        if template_path is None:
            return None
        if not template_path.exists():
            return None
        template = cv2.imread(str(template_path))
        return template

    @staticmethod
    def _blue_ratio(image: np.ndarray) -> float:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        return float(np.mean((b > g) & (b > r)))


# Backwards compatibility for existing imports.
CLAHEPreprocessor = OMRPreprocessor
