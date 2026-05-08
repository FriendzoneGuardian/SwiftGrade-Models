"""
Phase 1 – Form Extraction package.

Public API
----------
YOLODetector  : Wraps an Ultralytics YOLO model for form/region detection.
FormExtractor : Detects a form bounding box and applies a perspective warp.
OMRDataset    : PyTorch Dataset for OMR sheet images.
"""

from .yolo_detector import YOLODetector
from .form_extractor import FormExtractor
from .omr_dataset import OMRDataset

__all__ = ["YOLODetector", "FormExtractor", "OMRDataset"]
