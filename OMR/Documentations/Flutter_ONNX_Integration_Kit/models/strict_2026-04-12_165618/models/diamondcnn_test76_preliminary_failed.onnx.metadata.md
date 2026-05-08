# ONNX Export Metadata - Test #76 (Preliminary, Failed)

## Release Label
- PRELIMINARY RESEARCH ARTIFACT - NOT PRODUCTION RELEASE

## Purpose
- OMR module checkpoint for thesis submission, not final deployment.

## Source Run
- strict_2026-04-12_165618

## Export Output
- ONNX: /root/projects/SwiftGrade-Models/OMR/runs/strict_2026-04-12_165618/models/diamondcnn_test76_preliminary_failed.onnx
- Metadata JSON: /root/projects/SwiftGrade-Models/OMR/runs/strict_2026-04-12_165618/models/diamondcnn_test76_preliminary_failed.onnx.metadata.json

## Declared Metrics (for thesis record)
- val F1: 0.972
- val accuracy: 0.978
- operating threshold: 0.80

## Confirmed Risk Notes
- confidence tail fragility on hard blank and light filled edge cases: confirmed
- invalid class evaluation: pending

## Status
- FAILED (kept as research artifact for traceability)
