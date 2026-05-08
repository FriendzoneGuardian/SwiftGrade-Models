# 📌 Legacy Extractor Reference

To the future agent working on **Phase 1 (Extraction)**:

The user has skipped physically copying the old extraction code into this new clean architecture for now. However, you can find the proven bounding-box mathematical models, fiducial finders, and YOLO wrapper scripts in the legacy repository folder.

**📍 Legacy Location to Reference:**
`Project-SwiftGrade-Models/SwiftGradeOMRv2 - Trial2/scripts/`

*Files to study and refactor into PyTorch/Ultralytics:*
- `extract_bubbles.sh`
- `organize_bubbles.py`
- `config.py` & `paths.py` (for bounding box offsets)

**Do NOT just copy-paste them.** Study their logic and rewrite them using our clean `yolo_detector.py` object-oriented structure.
