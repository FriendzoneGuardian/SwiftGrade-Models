# Test 77 - Phase 3 Ascending Model (Strict Protocol)

Date: 2026-04-19
Status: COMPLETED

## Objective
Run Phase 3 strict benchmark using the Ascending model with the same unified evaluation protocol used for the prior Diamond benchmark, then export the best checkpoint to ONNX.

## Executed Command
```bash
source .venv/bin/activate && python OMR/scripts/strict_phase2_phase3_gpu_test.py --model-method ascending --epochs 100 --patience 20 --ground-truth-basis --target-augmented 169125 --decision-floor 0.75
```

## Final Run Summary
- Run directory: OMR/runs/strict_2026-04-19_134913
- Model: AscendingCNN
- Epochs requested: 100
- Epochs ran: 100
- Early stopping patience: 20
- Ground-truth basis mode: enabled
- Augmented samples: 169125
- Standard assessment: PASS

## Core Metrics
- Best epoch: 88
- Best validation F1 (monitor metric): 0.9757324313057165
- Final validation accuracy: 0.9833259423503325
- Final validation F1: 0.9750703174044788

## Unified Decision Metrics
- Recommended threshold: 0.8
- Threshold samples: 33825
- TP: 6653
- FP: 125
- TN: 26480
- FN: 567
- Filled precision: 0.9815579817055179
- Filled recall: 0.9214681440443213
- Threshold accuracy: 0.9795417590539541

## Proof Artifacts
- Strict report: OMR/runs/strict_2026-04-19_134913/strict_pipeline_report.json
- Training summary: OMR/runs/strict_2026-04-19_134913/models/training_summary.json
- Training history: OMR/runs/strict_2026-04-19_134913/models/training_epoch_metrics.jsonl
- Proof grid (200): OMR/runs/strict_2026-04-19_134913/proof_grid_200.png
- Grid false count: 2
- Bucketed proof grid (200): OMR/runs/strict_2026-04-19_134913/proof_grid_200_sorter_buckets.png

## ONNX Export
- ONNX file: OMR/runs/strict_2026-04-19_134913/models/ascendingcnn_test77.onnx
- Export source checkpoint: OMR/runs/strict_2026-04-19_134913/models/best_model.pth
- ONNX size (bytes): 35207
- ONNX checker: PASS
- Opset: ai.onnx v18
- Input name: input
- Output name: logits

## Notes
- Strict runner now supports explicit model methods: diamond, ascending, transfer.
- This run used the same unified evaluation path and artifact outputs as the prior strict Diamond workflow.
