# 📌 Legacy Preprocessor Reference

To the future agent working on **Phase 2 (Preprocessing)**:

We did not copy the old standalone scripts into this structure yet. We need you to refactor the old logic into the new unified `OMRPreprocessor` class.

**📍 Legacy Location to Reference:**
`Project-SwiftGrade-Models/SwiftGradeOMRv2 - Trial2/Phase2/scripts/`

*Files to study and refactor:*
- `classify_bubbles_rules.py` (Contains the "Secret Sauce" logic: significance-gated subtraction, multi-channel CLAHE, solidity analysis).
- `train_classifier.py` 

**Do NOT just copy-paste them.** Study the augmentation and normalization math, then cleanly encapsulate it into the `preprocessor.py` for this phase.
