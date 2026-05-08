# Runs (Official Sequence, Latest-First)

Date: 2026-05-08
Status: BACKPROPAGATED

## Official Runs (Backwards Count)
- #78: Ascending Success (without ONNX)
  - Run directory: OMR/runs/Run 78 - 041926_134913-Ascending-Success
  - Model: AscendingCNN
  - Basis: strict pipeline `standard_assessment.summary = PASS`

- #77: Diamond Preliminary Failed
  - Run directory: OMR/runs/Run 77 - 041226_165618-Diamond-Failed
  - Model: DiamondCNN
  - Artifact: OMR/runs/Run 77 - 041226_165618-Diamond-Failed/models/diamondcnn_test77_preliminary_failed.onnx
  - Basis: preliminary failed research artifact

- #76: Diamond Success (with ONNX)
  - Run directory: OMR/runs/Run 76 - 041226_152601-Diamond-Success
  - Model: DiamondCNN
  - ONNX file: OMR/runs/Run 76 - 041226_152601-Diamond-Success/models/best_model.onnx
  - Basis: strict pipeline `standard_assessment.summary = PASS`

## MobileNet Smoke Tests (79+)
- #85: Run 85 - 050826_102802-MobileNetV2-Failed
- #84: Run 84 - 050826_102709-MobileNetV2-Failed
- #83: Run 83 - 050826_102448-MobileNetV2-Failed
- #82: Run 82 - 050826_102108-MobileNetV2-Failed
- #81: Run 81 - 050826_101929-MobileNetV2-Failed
- #80: Run 80 - 050826_101848-MobileNetV2-Failed
- #79: Run 79 - 050826_101534-MobileNetV2-Failed

## Auxiliary Runs (Labelled, Failed)
- #75: Run 75 - 041926_133036-TransferLearning-Failed
- #74: Run 74 - 041926_133010-Diamond-Failed
- #73: Run 73 - 041926_132939-Ascending-Failed
- #72: Run 72 - 041926_132923-Preflight-Failed
- #71: Run 71 - 041226_164034-Diamond-Failed
- #70: Run 70 - 041226_163905-Diamond-Failed
- #69: Run 69 - 041226_162804-Diamond-Failed

## Notes
- Official strict lineage remains `#76`, `#77`, `#78`.
- MobileNet smoke tests are explicitly labelled `#79` and above.
- Auxiliary runs are now explicitly labelled as `#75` and below.
- Display is reverse count order to match backward tracking.
