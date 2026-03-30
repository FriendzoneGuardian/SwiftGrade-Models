"""
yolo_detector.py – YOLO-based object detector wrapper for OMR form extraction.

Wraps ``ultralytics.YOLO`` to provide a clean, typed interface that returns
structured detection results without exposing Ultralytics internals to callers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Wraps an Ultralytics YOLO model for object detection on OMR images.

    Parameters
    ----------
    model_path:
        Path to the YOLO model weights file (e.g. ``"best.pt"``).
    conf_threshold:
        Minimum confidence score for a detection to be returned.  Defaults to
        ``0.5`` – see ``LEGACY_REFERENCE.md`` for the rationale.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).  Defaults to
        ``"cpu"`` for broad compatibility.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run inference on *image* and return all detections above threshold.

        Parameters
        ----------
        image:
            BGR image as a ``numpy.ndarray`` with shape ``(H, W, 3)``.

        Returns
        -------
        list[dict]
            Each dict contains:

            * ``label``      – class name string
            * ``confidence`` – float in ``[0, 1]``
            * ``bbox``       – ``[x1, y1, x2, y2]`` in pixel coordinates
            * ``bbox_xyxy``  – alias of ``bbox`` (same list, retained for
                               downstream compatibility)
        """
        results = self._model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names: dict[int, str] = result.names
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                bbox = [x1, y1, x2, y2]
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": bbox,
                        "bbox_xyxy": bbox,
                    }
                )

        return detections

    def detect_largest(
        self,
        image: np.ndarray,
        label: Optional[str] = None,
    ) -> Optional[dict]:
        """Return the detection with the largest bounding-box area.

        Parameters
        ----------
        image:
            BGR image as a ``numpy.ndarray``.
        label:
            If provided, only detections whose ``label`` matches this string
            are considered.

        Returns
        -------
        dict or None
            The detection dict (same schema as :meth:`detect`) with the
            greatest bounding-box area, or ``None`` if no detections exist.
        """
        detections = self.detect(image)
        if label is not None:
            detections = [d for d in detections if d["label"] == label]

        if not detections:
            return None

        def _area(det: dict) -> float:
            x1, y1, x2, y2 = det["bbox"]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return max(detections, key=_area)
