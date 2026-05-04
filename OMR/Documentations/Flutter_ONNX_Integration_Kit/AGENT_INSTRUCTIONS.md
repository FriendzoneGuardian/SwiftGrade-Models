# Agent Instructions — Flutter ONNX Integration Kit

Purpose
- Short, machine-friendly instructions to maintain and extend the kit in-repo.

Assume kit root
- Treat this folder as the kit root. All paths and commands below are relative to the kit root unless otherwise stated.

Folder contents (current)
- `Flutter_ONNX_App_Integration_Handoff.md`: teammate-facing handoff and integration notes.
- `model_manifest.template.json`: template for model bundle manifest.
- `strict_report_ref.template.json`: template referencing strict run report metadata.
- `README.md`: kit overview and quick integration steps.
- `flutter_dart_sample/` (source samples): contains Dart example files mirrored here:
  - `omr_types.dart`
  - `omr_preprocessing.dart`
  - `omr_postprocessing.dart`
  - `omr_inference_service.dart`
  - `ort_runner_stub.dart`
- `models/` (optional): target location for `*.onnx`, `manifest.json`, `checksum.sha256`, and `validation.json`.

Primary tasks for an automation agent or maintainer
1. Ensure a distributable ONNX model is present at `models/model.onnx` inside the kit.
2. Produce `models/manifest.json` and `models/checksum.sha256` for each model bundle.
3. Run the parity validation script (see Validation) and record results in `models/validation.json`.
4. Keep `model_manifest.template.json` and `strict_report_ref.template.json` up to date when model metadata changes.
5. When new Dart examples are added, copy into `flutter_dart_sample/` and mirror into the kit folder.

ONNX Export Recipe (how an agent should generate `model.onnx`)
- Export from a PyTorch checkpoint (example). Run this in the Python environment that has PyTorch matching the training env.

Notes
- This kit is the working root. Output `onnx` files should be written into `./models/`.
- The checkpoint path is likely outside the kit; set the `CKPT` variable to point to the checkpoint used for export.

Example `export_to_onnx.py` (kit-root relative)

```python
# export_to_onnx.py (example)
import torch
from Phase_3_Classification.src.ascending import AscendingCNN

# Set this to your training checkpoint (outside this kit)
CKPT = '../path/to/best_model.pth'
OUT = 'models/ascending_model.onnx'

# instantiate model with the same args used at training
model = AscendingCNN(num_classes=2)
state = torch.load(CKPT, map_location='cpu')
model.load_state_dict(state.get('model_state_dict', state))
model.eval()

# dummy input matches training contract: [1,3,64,64]
dummy = torch.randn(1,3,64,64)

torch.onnx.export(model, dummy, OUT,
                  input_names=['input'], output_names=['logits'],
                  opset_version=18, do_constant_folding=True)
print('Wrote', OUT)
```

Packaging manifest & checksum
- Create `models/manifest.json` with fields: `name`, `version`, `onnx_filename`, `input_shape`, `output_shape`, `opset`, `sha256`.
- Compute checksum and save to `models/checksum.sha256` (run from the kit root):

```bash
# from kit root
python -c "import hashlib;print(hashlib.sha256(open('models/ascending_model.onnx','rb').read()).hexdigest())" > models/checksum.sha256
```

Validation (parity check)
- Quick Python test using `onnxruntime` to confirm model runs and outputs sane logits (kit-root relative):

```python
import onnxruntime as ort
import numpy as np
m = ort.InferenceSession('models/ascending_model.onnx')
# random input matching [1,3,64,64]
x = np.random.rand(1,3,64,64).astype('float32')
res = m.run(None, {'input': x})
print('logits shape', [r.shape for r in res])
```

- Record the output softmax probabilities and save `models/validation.json` with `sample_seed`, `p_filled`, and `logits_shape`.

Flutter integration pointers for an agent
- The kit contains `ort_runner_stub.dart` and service stubs. An agent should:
  - Replace the stub with calls for the chosen Flutter ONNX runtime (e.g., `onnxruntime_mobile`, `onnx_flutter`, or platform channels to native ORT libs).
  - Ensure the Flutter asset path matches `models/ascending_model.onnx` (relative to the app assets) and that the app loads it with the same input dtype/order (float32, RGB, HWC→CHW, /255.0 normalization).
  - Confirm output ordering: class 0 = blank, class 1 = filled (matches repo). Use recommended operating threshold 0.8 by default.

Maintenance notes
- If this kit becomes a repository root, ensure `models/` is tracked or the release packaging step publishes model artifacts. Some upstream repos exclude `*.onnx`; prefer storing the ONNX in the kit `models/` folder or as release assets.
- When updating the kit, bump `models/manifest.json` `version` and record the export provenance (training run id and checkpoint path).

Troubleshooting
- If the exported ONNX fails shape checks, verify the model class initialization args match the training code in `Phase_3_Classification/src/ascending.py`.
- If ONNXRuntime throws unsupported op errors, try exporting with a lower `opset_version` (e.g., 17) or use `torch.jit.trace` then export.

Agent action checklist (idempotent, quick)
- [ ] Place `models/*.onnx` in the kit.
- [ ] Generate `models/checksum.sha256`.
- [ ] Fill `models/manifest.json` with metadata.
- [ ] Run `validation.py` (or snippet) and save `validation.json`.
- [ ] Commit changes under `OMR/Documentations/Flutter_ONNX_Integration_Kit/models/` or publish as a release asset.

Contact / provenance
- If a run folder was copied into this kit, see `models/strict_2026-04-12_165618/strict_pipeline_report.json` for the included Diamond run metadata and `models/strict_2026-04-12_165618/models/diamondcnn_test76_preliminary_failed.onnx.metadata.json` for ONNX metadata.
- For training/export provenance outside this kit, refer to upstream repo run artifacts and to the training code (e.g., `Phase_3_Classification/src/ascending.py`) if available in your monorepo.

---
Generated by assistant for maintainers and automated agents. Update as needed when export or packaging policies change.
