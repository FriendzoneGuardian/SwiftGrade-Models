# Flutter ONNX Integration Kit

This folder is the single handoff package for integrating the OMR model into the Flutter app.

## Folder Contents
1. `Flutter_ONNX_App_Integration_Handoff.md` - full integration playbook.
2. `model_manifest.template.json` - model contract template.
3. `strict_report_ref.template.json` - strict run reference template.
4. `omr_types.dart` - shared models and manifest parser.
5. `omr_preprocessing.dart` - image preprocessing to NCHW tensor.
6. `omr_postprocessing.dart` - softmax and row decision logic.
7. `omr_inference_service.dart` - end-to-end scoring service.
8. `ort_runner_stub.dart` - runtime adapter to replace with ONNX package calls.

## Quick Integration Steps
1. Copy all Dart files into your Flutter app module.
2. Add ONNX runtime package in the app project.
3. Replace `OrtRunnerStub` with real plugin code.
4. Add model assets:
   - `model.onnx`
   - `manifest.json`
   - `checksum.sha256`
   - `strict_report_ref.json`
5. Register assets in `pubspec.yaml`.
6. Initialize `OmrInferenceService` once at app startup.
7. Run startup contract validation from `OmrModelManifest.validateContract()`.
8. Run calibration tests before merge.

## Runtime Contract (Must Match)
1. Input shape: `[1, 3, 64, 64]`
2. Input dtype: `float32`
3. Color space: `RGB`
4. Normalization: `pixel / 255.0`
5. Output shape: `[1, 2]` logits
6. Class order: `blank=0`, `filled=1`
7. Operating threshold: `0.8` (or manifest override)

## Row Decision Defaults
1. `min_fill_prob = 0.30`
2. `tie_threshold = 0.15`
3. `multi_mark_threshold = 0.50`
4. `review_band = 0.06`

## Validation Gate
Before production handoff:
1. Verify model session init on target devices.
2. Verify probability parity against Python reference samples.
3. Verify row decision status parity (`blank`, `selected`, `tie_review`, `multi_mark_review`).
4. Verify checksum and manifest checks fail fast on mismatch.
