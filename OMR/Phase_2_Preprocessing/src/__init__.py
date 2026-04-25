"""
Phase 2 – Preprocessing package.

Public API
----------
OMRPreprocessor : Paper-aligned Phase 2 preprocessing pipeline.
CLAHEPreprocessor : Backwards-compatible alias for OMRPreprocessor.
"""

from .preprocessor import CLAHEPreprocessor, OMRPreprocessor

__all__ = ["OMRPreprocessor", "CLAHEPreprocessor"]
