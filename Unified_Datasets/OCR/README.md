## Unified OCR datasets

This folder is the **default dataset location for Module B (OCR)**.

### Expected layout

- **`test_images/`**: handwriting crops to evaluate OCR on
- **`ground_truth.json`**: mapping of `filename -> true text` used to compute WER/CER

Example `ground_truth.json`:

```json
{
  "sample_handwriting.png": "The mitochondria is the powerhouse of the cell."
}
```

### Notes

- `OCR/evaluate_ocr.py` defaults to `Unified_Datasets/OCR/test_images` and
  `Unified_Datasets/OCR/ground_truth.json`.
- If you pass `--data-dir` / `--ground-truth`, those explicit paths win.
