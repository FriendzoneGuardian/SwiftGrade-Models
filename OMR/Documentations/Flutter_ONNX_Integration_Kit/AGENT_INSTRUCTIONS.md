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
1. Use the ONNX model(s) already included in `./models/` — do NOT create or publish new model artifacts from training inside this kit unless you have explicit permission and provenance documentation.
2. If a `manifest.json` and `checksum.sha256` are not present for an included model, generate them from the included model file (do not re-export or re-train a model to produce these).
3. Run the parity validation script (see Validation) against the included model and record results in `models/validation.json`.
4. Keep `model_manifest.template.json` and `strict_report_ref.template.json` up to date when included model metadata or provenance changes.
5. When new Dart examples are added, copy into `flutter_dart_sample/` and mirror into the kit folder.

ONNX Export (ADVANCED, OPTIONAL — DO NOT RUN BY DEFAULT)
- This kit is intended to distribute and consume pre-built ONNX artifacts that are included under `./models/`.
- Do NOT generate or publish new model ONNX files from training unless you are the model owner, have explicit permission, and will record provenance in `models/manifest.json`.
- If you are explicitly instructed and authorized to export a model from a checkpoint, perform that work outside this kit, then copy the resulting `*.onnx` into `./models/` and update the manifest and checksum. Treat export workflows as an out-of-band operation.

Packaging manifest & checksum (for included models)
- If the kit already contains `models/<model>.onnx`, produce `models/manifest.json` and `models/checksum.sha256` for that included file (do not re-export a model to create these files).
- Example (run from kit root):

```bash
# replace <model.onnx> with the included file name
python -c "import hashlib;print(hashlib.sha256(open('models/<model.onnx>','rb').read()).hexdigest())" > models/checksum.sha256
```

Validation (parity check)
- Run validation only against models that are included in `./models/`. The purpose is to confirm the included artifact runs under ONNXRuntime and produces the expected output shape and reasonable probabilities.

```python
import onnxruntime as ort
import numpy as np
import json

MODEL = 'models/<model.onnx>'
sess = ort.InferenceSession(MODEL)
seed = 42
rng = np.random.RandomState(seed)
x = rng.rand(1,3,64,64).astype('float32')
res = sess.run(None, {'input': x})
logits_shape = [r.shape for r in res]
probs = np.exp(res[0]) / np.exp(res[0]).sum(axis=1, keepdims=True)
p_filled = float(probs[0,1])
out = {'sample_seed': seed, 'p_filled': p_filled, 'logits_shape': logits_shape}
open('models/validation.json','w').write(json.dumps(out,indent=2))
print('Wrote models/validation.json')
```

Flutter integration pointers for an agent
- The kit contains `ort_runner_stub.dart` and service stubs. An agent should:
  - Replace the stub with calls for the chosen Flutter ONNX runtime (e.g., `onnxruntime_mobile`, `onnx_flutter`, or platform channels to native ORT libs).
  - Ensure the Flutter asset path points to the included model under `assets/models/` or similar, and that the app loads it with the same input dtype/order (float32, RGB, HWC→CHW, /255.0 normalization).
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
