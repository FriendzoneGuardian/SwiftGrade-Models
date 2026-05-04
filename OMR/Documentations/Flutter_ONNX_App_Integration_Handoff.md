# Flutter ONNX Integration Handoff

Date: 2026-05-04  
Owner: Models Team  
Target: Flutter App Team

## Goal
Integrate the OMR classifier into the Flutter app using ONNX Runtime with inference behavior matching this models repository.

## Non-Goals
1. No TensorFlow or TFLite integration path.
2. No model retraining inside the app repository.
3. No Phase 1 detector training changes from app code.

## Required Artifacts From Models Repo
Ship these together as one model bundle:

1. ONNX model file.
2. `manifest.json`.
3. `checksum.sha256`.
4. strict report reference metadata.

Suggested bundle layout:

```text
model_bundle/
  model.onnx
  manifest.json
  checksum.sha256
  strict_report_ref.json
```

## Runtime Contract (Must Match)
Enforce this contract during app startup before first inference:

1. Input tensor shape: `[1, 3, 64, 64]`
2. Input dtype: `float32`
3. Input color space: `RGB`
4. Input normalization: `pixel / 255.0`
5. Output tensor shape: `[1, 2]` logits
6. Class order: `blank=0`, `filled=1`
7. Default operating threshold: `0.8`

## Row Decision Contract
Match model-side row scoring logic:

1. `min_fill_prob = 0.30`
2. `tie_threshold = 0.15`
3. `multi_mark_threshold = 0.50`
4. `review_band = 0.06`

Allowed statuses:

1. `blank`
2. `selected`
3. `tie_review`
4. `multi_mark_review`

## App Integration Sequence
1. Add ONNX Runtime package and native setup in Flutter app.
2. Add model bundle under Flutter assets and register it.
3. Build singleton model session loader.
4. Implement preprocessing parity pipeline.
5. Implement inference adapter (`input -> logits`).
6. Implement postprocessing (`logits -> probabilities -> threshold decision`).
7. Implement row decision engine parity.
8. Add startup contract validator.
9. Add calibration tests and smoke tests.

## Preprocessing Parity
For each bubble crop:

1. Decode image.
2. Resize to `64x64`.
3. Convert to RGB.
4. Convert to float32.
5. Normalize each channel by dividing by `255.0`.
6. Reorder `HWC -> CHW`.
7. Add batch dimension to form `[1, 3, 64, 64]`.

## Postprocessing Parity
Given output logits `[blank_logit, filled_logit]`:

1. Apply softmax to obtain probabilities.
2. Extract `p_filled`.
3. Compute `is_filled = p_filled >= operating_threshold`.
4. Feed per-bubble `p_filled` into row decision engine.

## Suggested App Service Interfaces
```text
ModelSessionService
  init()
  runLogits(Float32List inputTensor) -> Float32List logits

BubbleScoringService
  scoreBubble(Uint8List cropBytes) -> BubbleScore

RowDecisionService
  decideRow(List<double> bubbleProbs, List<String> labels) -> RowDecision

SheetScoringService
  scoreSheet(List<RowInput> rows) -> List<RowDecision>
```

## Dart Starter Files In This Repo
Copy these files into the Flutter app repo and wire the ONNX plugin in the runner stub:

1. `OMR/Documentations/flutter_dart_sample/omr_types.dart`
2. `OMR/Documentations/flutter_dart_sample/omr_preprocessing.dart`
3. `OMR/Documentations/flutter_dart_sample/omr_postprocessing.dart`
4. `OMR/Documentations/flutter_dart_sample/omr_inference_service.dart`
5. `OMR/Documentations/flutter_dart_sample/ort_runner_stub.dart`

## Startup Validation Rules
Fail fast if any check fails:

1. Model file is missing.
2. Manifest file is missing or invalid.
3. Input name mismatch.
4. Output name mismatch.
5. Input or output shape mismatch.
6. Missing class mapping for `blank` and `filled`.
7. Checksum mismatch.

## Calibration Gate (Before Merge)
1. Probability parity on fixed calibration samples versus Python reference.
2. Row decision parity on expected statuses.
3. No crash on malformed image payloads.
4. Human-readable startup errors for contract mismatches.

## Release Checklist
1. Freeze model bundle and increment semantic version.
2. Commit manifest and checksum.
3. Record strict run reference.
4. Run smoke tests on target devices.
5. Publish release note with model version and threshold policy.

## Rollback Checklist
1. Keep previous model bundle available.
2. Switch active model version to previous known-good.
3. Re-run smoke set and verify behavior restoration.
4. Log rollback reason and failing model version.

## Ownership Split
Models Team:
1. Publish immutable model bundles.
2. Publish threshold policy and strict report references.

App Team:
1. Integrate bundle into runtime.
2. Enforce startup contract checks.
3. Ship release and monitor production behavior.

## Source References In This Repository
1. Strict run summary and ONNX notes: `results.md`
2. Strict threshold policy and resize behavior: `OMR/scripts/strict_phase2_phase3_gpu_test.py`
3. Input normalization and CHW conversion: `OMR/Phase_3_Classification/src/dataset.py`
4. Row decision thresholds and statuses: `OMR/Phase_3_Classification/src/scoring.py`
5. Latest strict report: `OMR/runs/strict_2026-04-19_134913/strict_pipeline_report.json`
