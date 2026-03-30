# OMRChecker + OMR-Basis Adoption Assessment for SwiftGrade Overhaul

Date: 2026-03-27
Branch assessed: Hot-and-Unstable
Assessed against:
- OMR findings handoff (OMRChecker reverse-engineering notes)
- Current Trial3 Phase2 code
- Newly attached OMR-Basis repository

## Executive Answer

No, everything was not hit.

Trial3 has a strong local bubble preprocessor, but it still lacks the page-level architecture needed for robust real-world grading. With OMR-Basis available, the right move is a staged architecture overhaul instead of incremental threshold tuning.

## Scope Checked

SwiftGrade current path:
- SwiftGradeOMRv2 - Trial3/Phase2/scripts/butler_preprocessor.py
- SwiftGradeOMRv2 - Trial3/Phase2/scripts/run_classifier.py
- SwiftGradeOMRv2 - Trial3/Phase2/config/classification.yaml
- SwiftGradeOMRv2 - Trial3/Phase3_Classification_Masterclass.ipynb

OMR-Basis reference stack:
- OMR-Basis/src/processors/manager.py
- OMR-Basis/src/processors/CropOnMarkers.py
- OMR-Basis/src/processors/CropPage.py
- OMR-Basis/src/processors/FeatureBasedAlignment.py
- OMR-Basis/src/core.py
- OMR-Basis/src/template.py
- OMR-Basis/src/processors/builtins.py
- OMR-Basis/src/tests/test_all_samples.py
- OMR-Basis/src/tests/test_edge_cases.py

## Current Trial3 Gaps (Pre-Overhaul)

1. Static threshold routing is brittle under lighting/ink variance.
2. No geometry stage (marker-first / contour fallback / ORB rescue) before scoring.
3. No row-level arbitration for multi-mark and erasure ambiguity.
4. Limited pipeline observability for FP root-cause tracing.

## Salvage Matrix

### Adopt Directly

1. Processor/plugin orchestration pattern
- Source: OMR-Basis/src/processors/manager.py

2. Marker-based perspective correction
- Source: OMR-Basis/src/processors/CropOnMarkers.py

3. Contour-based page fallback
- Source: OMR-Basis/src/processors/CropPage.py

4. ORB-based rescue alignment
- Source: OMR-Basis/src/processors/FeatureBasedAlignment.py

5. Dynamic threshold algorithms (largest-gap global + local)
- Source: OMR-Basis/src/core.py

6. Snapshot and edge-case testing discipline
- Source: OMR-Basis/src/tests/test_all_samples.py
- Source: OMR-Basis/src/tests/test_edge_cases.py

### Adapt Before Adoption

1. FieldBlock/Bubble abstractions for section-wise traversal
- Source: OMR-Basis/src/template.py

2. Evaluation/reporting flow
- Source: OMR-Basis/src/evaluation.py

3. Optional LUT Levels preprocessor
- Source: OMR-Basis/src/processors/builtins.py

### Do Not Copy As-Is

1. Monolithic end-to-end response function style in core.py
2. Simple mean-intensity final bubble marking logic in core.py
3. Hardcoded default threshold constants as a primary policy

## What to Keep from Butler

1. Circular masked scoring
2. Variance gating
3. Significance-gated master-blank subtraction
4. Solidity-aware filtering
5. Uncertainty bucket routing

These remain the local quality layer after geometry + dynamic threshold overhaul.

## Implemented Overhaul Foundation (Papertrail Enabled)

The following foundational pieces are now implemented in Trial3 Phase2:

1. Config expansion for dynamic thresholds, strict filled confirmation, and papertrail outputs
- File: SwiftGradeOMRv2 - Trial3/Phase2/config/classification.yaml

2. Butler preprocessor diagnostics and optional Levels support
- File: SwiftGradeOMRv2 - Trial3/Phase2/scripts/butler_preprocessor.py

3. Dynamic largest-gap threshold module
- File: SwiftGradeOMRv2 - Trial3/Phase2/scripts/dynamic_thresholds.py

4. Two-pass classifier with decision audit CSV and threshold reports
- File: SwiftGradeOMRv2 - Trial3/Phase2/scripts/run_classifier.py

5. Evaluation script for confusion/precision/FPR analysis using audit logs + optional ground truth
- File: SwiftGradeOMRv2 - Trial3/Phase2/scripts/evaluate_classifier.py

6. Preflight checker for dependency/path breakpoints
- File: SwiftGradeOMRv2 - Trial3/Phase2/scripts/preflight_check.py

## Papertrail Artifacts

When enabled in config, each run now emits:

1. audit_metrics.csv
- Per-image diagnostics, threshold source, strict-gate reasons, and final decision.

2. threshold_report.json
- Global and local dynamic threshold stats and confidence.

3. run_summary.json
- Run metadata, aggregate counts, and artifact paths.

4. preflight_report.json
- Dependency/path readiness report (cv2 availability, dataset presence, master blank status).

## Pending Overhaul Stages (Not Yet Wired)

1. Geometry runtime integration
- Marker-first correction + contour fallback + ORB rescue are assessed and planned, but not yet connected to the current crop-level classifier runtime.

2. Row-level arbitration
- Multi-mark and erasure conflict-resolution scoring module is still pending.

3. Processor manager migration
- Full plugin-based processor orchestration is a planned migration stage.

## Revised Integration Sequence

1. Freeze baseline on tagged ground-truth subset.
2. Roll out dynamic threshold + strict decision policy (done foundation).
3. Add geometry layer with feature flags.
4. Add row-level arbitration and conflict handling.
5. Harden notebook metrics and confidence calibration.
6. Promote only when gates pass on unseen profiles.

## Acceptance Criteria

1. FILLED precision improves materially with reduced blank leakage.
2. BLANK TN rate remains stable or improves.
3. FILLED recall does not collapse beyond agreed guardrail.
4. Dynamic thresholds beat static thresholds across multiple lighting profiles.
5. Row-level scoring accuracy improves after arbitration stage.

## Licensing Note

OMR-Basis is MIT licensed. Preserve attribution and license text when reusing substantial portions.

## Final Assessment

The original OMRChecker assessment was directionally correct.

With OMR-Basis now available and the new Phase2 papertrail foundation in place, SwiftGrade is positioned for a controlled, measurable overhaul rather than ad-hoc tuning.
