# 🚀 Next Steps: Phase 2 Reboot Roadmap

## ✅ Canonical Paths
- Canonical root: `SwiftGradeOMRv2 - Trial3` (Test v3).
- All commands and references should use `SwiftGradeOMRv2 - Trial3` (avoid `/root/...` paths).
- Phase 2 assets live under `SwiftGradeOMRv2 - Trial3/Phase2/`.

## ⚡ Immediate Actions (Week 1)
- [ ] **Fix Training Pipeline:** Resolve critical bugs in `model_training.py` (F1-score metric and unpacking errors).
- [ ] **Centralized Config:** Implement `config.yaml` to unify hyperparameters across YOLO and CNN stages.
- [ ] **Data Purge:** Execute the "Extreme Cleaning" script to generate the 114K sample Gold Standard dataset.

## 🏗️ Structural Overhaul (Week 2-3)
- [ ] **Unified Pipeline:** Create `train_pipeline.py` to handle end-to-end training and inference paths.
- [ ] **Preprocessing Integration:** Ensure all existing scripts import from the centralized `preprocessing.py`.
- [ ] **TensorBoard Setup:** Enable real-time visualization of training metrics and confusion matrices.

## 🎯 Validation & Deployment (Week 4+)
- [ ] **Cross-Team Validation:** Test the trained model against datasets from multiple team members.
- [ ] **Inference Optimization:** Benchmark the `inference.py` script for real-world document speeds.
- [ ] **Review Interface:** Implement the confidence-gated review interface for teachers.

---
> [!IMPORTANT]
> All new code should adhere to the **Localized Architecture** (no hardcoded absolute paths) as established in `SwiftGradeOMRv2 - Trial3`.

---

# ✅ Concrete Phase 2 Reboot Plan (Trial3 Canonical, Hybrid Classifier)

**Summary**  
Stabilize Phase 2 end-to-end using the current Trial3 codebase, explicitly document the binary-now / 4-class-later strategy, and align documentation so contributors can run the pipeline without path/structure ambiguity.

## 📌 Documentation Alignment (Trial3 Canonical)
- Update runbooks and references to use `SwiftGradeOMRv2 - Trial3` as the canonical root and remove `/root/...` paths.
- Align “Trial2” labels in docs to “Trial3/Test v3” consistently.
- Keep all Phase 2 references under `SwiftGradeOMRv2 - Trial3/Phase2/`.

## 🧪 Phase 2 Reboot Execution (Binary Now)
- Preprocessor validation: ensure `SwiftGradeOMRv2 - Trial3/Phase2/master_blank.png` exists. If missing, generate using `Phase2/scripts/generate_template.py` with a blank-bubble input directory.
- Rule-based auto-labeling: run `Phase2/scripts/classify_bubbles_rules.py` to populate `Phase2/auto_labeled` with `filled/light`, `filled/dark`, `blank`, and `uncertain`.
- Manual review: run `Phase2/scripts/review_classifications.py` and move corrections into `Phase2/manual_labeled/{filled,blank}`.
- Train classifier: run `Phase2/scripts/train_classifier.py` (YOLOv8-cls) to produce `Phase2/models/cls_train/weights/best.pt`.
- ML inference: run `Phase2/scripts/classify_bubbles_ml.py` to generate `Phase2/ml_classified` and `predictions.csv`.
- Row scoring: run `Phase2/scripts/score_rows.py` to output `Phase2/row_scores/summary.csv`.

## 🔁 Hybrid Upgrade Path (4-Class Later)
- Add label schema for `crossed` and `invalid`.
- Extend `Phase2/config/classification.yaml` to include 4 classes.
- Update `classify_bubbles_rules.py` to emit `crossed/invalid` placeholders.
- Update `train_classifier.py` and `classify_bubbles_ml.py` for 4-class training and inference.
- Gate the 4-class upgrade on two successful binary cycles with acceptable uncertainty rate and row-score accuracy.

## ✅ Test Plan
- Paths/config sanity: run `python scripts/paths.py` and `python scripts/config.py`.
- Preprocessor check: run rule-based classification on 100–200 bubbles and confirm expected `blank/filled/uncertain` distribution.
- Training smoke test: verify `Phase2/models/cls_train/weights/best.pt` and `Phase2/models/cls_train/results.png`.
- Inference sanity: confirm `Phase2/ml_classified/predictions.csv` is created and track uncertainty rate.
- Row scoring sanity: run `Phase2/scripts/score_rows.py` on a labeled set and verify row accuracy and ambiguity counts.

## 📝 Assumptions
- Canonical codebase is `SwiftGradeOMRv2 - Trial3`.
- Phase 2 scope only (preprocessing, auto-labeling, training, inference, row scoring).
- Binary classifier ships first; 4-class upgrade is explicitly planned but gated.
