# SwiftGrade OMR Development Log
Date: 2026-04-12

## Executive Summary
This entry documents a validated breakthrough in the Phase 3 OMR classification pipeline under a ground-truth constrained protocol. The system now supports cache-aware phase routing, ground-truth label preservation, bucketed visual verification, and reproducible GPU execution using the Diamond CNN architecture.

## Development Progress
### 1. Pipeline Control and Reliability
- Added strict runner controls for `patience`, decision floor, source capping, index reuse, and target augmented scale.
- Added cache-aware phase routing:
  - If sufficient augmented cache exists, start from Phase 3 directly.
  - If cache is insufficient, rebuild from augmentation stage.

### 2. Ground-Truth Integrity
- Added ground-truth basis mode (`--ground-truth-basis`) that preserves labels from `manual_labeled` and bypasses purge relabeling.
- This removes a key validity threat in which heuristic relabeling can drift from human-annotated truth.

### 3. Verification Artifacts
- Standard grid export retained: `proof_grid_200.png`.
- Added bucketed sorter grid export: `proof_grid_200_sorter_buckets.png`.
- Bucket accounting includes: light filled, dark filled, blank, invalid (when available in labels).

## Breakthrough Run Snapshot
Run: `OMR/runs/strict_2026-04-12_152601`

- Ground-truth mode: enabled
- Standard assessment: PASS
- Validation accuracy: 0.9675
- Validation F1: 0.9519
- Selected threshold: 0.80

At threshold 0.80:
- Filled precision: 0.9749
- Filled recall: 0.7984
- Accuracy: 0.9526

Grid outputs:
- `OMR/runs/strict_2026-04-12_152601/proof_grid_200.png`
- `OMR/runs/strict_2026-04-12_152601/proof_grid_200_sorter_buckets.png`

## ONNX Export Readiness Assessment
Status: Verified ready.

A direct export from the best checkpoint was executed and validated.

Generated artifact:
- `OMR/runs/strict_2026-04-12_152601/models/best_model.onnx`

Validation result:
- ONNX checker: PASS
- Opset imports: default domain opset 18
- Input tensor name: `input`
- Output tensor name: `logits`

## Technical Note
The environment attempted opset 13 conversion but retained opset 18 due converter constraints; the final exported graph is valid and checkable, and is suitable for inference deployment workflows that support opset 18.

## Conclusion
The current pipeline is demonstrably stable at scale with academically defensible label handling, reproducible artifact generation, and validated ONNX export capability.