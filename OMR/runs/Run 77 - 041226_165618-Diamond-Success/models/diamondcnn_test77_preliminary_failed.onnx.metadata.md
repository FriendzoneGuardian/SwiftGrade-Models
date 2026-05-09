# ONNX Export Metadata - Test #77 (Preliminary, Failed)

## Release Label
- PRELIMINARY RESEARCH ARTIFACT - NOT PRODUCTION RELEASE

## Purpose
- OMR module checkpoint for thesis submission, not final deployment.

## Source Run
- Run 77 - 041226_165618-Diamond-Failed

## Export Output
- ONNX: /root/projects/SwiftGrade-Models/OMR/runs/Run 77 - 041226_165618-Diamond-Failed/models/diamondcnn_test77_preliminary_failed.onnx
- Metadata JSON: /root/projects/SwiftGrade-Models/OMR/runs/Run 77 - 041226_165618-Diamond-Failed/models/diamondcnn_test77_preliminary_failed.onnx.metadata.json

## Declared Metrics (for thesis record)
- val F1: 0.972
- val accuracy: 0.978
- operating threshold: 0.80

## Confirmed Risk Notes
- confidence tail fragility on hard blank and light filled edge cases: confirmed
- invalid class evaluation: pending

## Status
- FAILED (kept as research artifact for traceability)
