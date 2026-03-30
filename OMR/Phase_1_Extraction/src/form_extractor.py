"""
form_extractor.py – Crops and perspective-warps a detected OMR form to a
standardised top-down view.

Mathematical details of the four-point transform and homography are documented
in ``LEGACY_REFERENCE.md``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .yolo_detector import YOLODetector


class FormExtractor:
    """Detects an OMR form and returns a rectified, top-down crop.

    Uses :class:`YOLODetector` (via dependency injection) so that the
    underlying detection model can be swapped without touching this class.

    Parameters
    ----------
    detector:
        A configured :class:`YOLODetector` instance.
    target_size:
        ``(width, height)`` of the output image in pixels.  Defaults to
        ``(800, 1000)``.
    """

    def __init__(
        self,
        detector: YOLODetector,
        target_size: tuple[int, int] = (800, 1000),
    ) -> None:
        self.detector = detector
        self.target_size = target_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect the form in *image* and return a perspective-corrected crop.

        Steps
        -----
        1. Run YOLO detection to find the largest form bounding box.
        2. Treat the four corners of the axis-aligned bounding box as the
           initial quadrilateral (a full perspective warp becomes possible
           once a segmentation or keypoint model provides non-axis-aligned
           corners; the bounding-box approximation is intentionally simple and
           consistent here).
        3. Apply :meth:`_four_point_transform` to produce the rectified image.

        Parameters
        ----------
        image:
            BGR image as a ``numpy.ndarray`` with shape ``(H, W, 3)``.

        Returns
        -------
        numpy.ndarray or None
            Rectified form image at ``self.target_size``, or ``None`` when
            no form is detected.
        """
        detection = self.detector.detect_largest(image)
        if detection is None:
            return None

        x1, y1, x2, y2 = detection["bbox"]
        # Build the four corners from the axis-aligned bounding box.
        pts = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        return self._four_point_transform(image, pts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _four_point_transform(
        self,
        image: np.ndarray,
        pts: np.ndarray,
    ) -> np.ndarray:
        """Apply a perspective warp that maps *pts* to a rectangle.

        Corner ordering convention (see ``LEGACY_REFERENCE.md``):

        * **Top-left**     – smallest ``x + y`` sum
        * **Top-right**    – smallest ``x - y`` difference
        * **Bottom-right** – largest  ``x + y`` sum
        * **Bottom-left**  – largest  ``x - y`` difference

        The destination size is determined by the max-width / max-height
        formulas, then scaled to ``self.target_size``.

        Parameters
        ----------
        image:
            Full source image.
        pts:
            Array of shape ``(4, 2)`` containing the four corner coordinates
            in *any* order.

        Returns
        -------
        numpy.ndarray
            Warped image of size ``self.target_size`` (width × height).
        """
        rect = self._order_points(pts)
        tl, tr, br, bl = rect

        # Compute the width of the new image.
        width_bottom = float(np.linalg.norm(br - bl))
        width_top = float(np.linalg.norm(tr - tl))
        max_width = max(int(width_bottom), int(width_top))

        # Compute the height of the new image.
        height_right = float(np.linalg.norm(tr - br))
        height_left = float(np.linalg.norm(tl - bl))
        max_height = max(int(height_right), int(height_left))

        # Ensure non-zero dimensions.
        max_width = max(max_width, 1)
        max_height = max(max_height, 1)

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        # Resize to the requested target size.
        warped = cv2.resize(warped, self.target_size, interpolation=cv2.INTER_LINEAR)
        return warped

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Sort four points into (top-left, top-right, bottom-right, bottom-left).

        Parameters
        ----------
        pts:
            Array of shape ``(4, 2)``.

        Returns
        -------
        numpy.ndarray
            Sorted array of shape ``(4, 2)``, dtype ``float32``.
        """
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        coord_sum = pts.sum(axis=1)
        rect[0] = pts[np.argmin(coord_sum)]   # top-left:     smallest x+y
        rect[2] = pts[np.argmax(coord_sum)]   # bottom-right: largest  x+y

        coord_diff = np.diff(pts, axis=1).ravel()   # x - y for each point
        rect[1] = pts[np.argmin(coord_diff)]  # top-right:    smallest x-y
        rect[3] = pts[np.argmax(coord_diff)]  # bottom-left:  largest  x-y

        return rect
