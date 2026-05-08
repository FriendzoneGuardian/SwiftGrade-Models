# 🧠 Phase 3: Classification & Decision Engine (Instructions)

This folder contains the final piece of the pipeline: taking perfectly cropped, normalized bubbles and mathematically proving which ones are intended answers. 

Our prior efforts in `Trial 3` established the foundation for this logic. We must *salvage* those concepts and code them into clean Object-Oriented PyTorch classes inside the `src/` directory.

## 📌 1. Progress to Port (What we already know)
From our previous deep-dive experiments (documented in `Phase3_Classification_Masterclass.ipynb` and our `omr_secret_sauce.md`), we locked in the following architecture:
- **We abandoned TensorFlow/Keras** due to crashing metrics (like `f1_score`) and bloated training loops. 
- **The "Extreme Clean" Dataset Strategy:** We realized that training on raw crops confused the CNN. We must purge the dataset using a geometric **Solidity Score** (Area / Convex Area). Shaded marks have high solidity; empty 'A-E' rings have low solidity.
- **The Architectures:** We conceptualized two custom PyTorch models:
  - `Diamond CNN`: Expands channels in the middle layers to capture complex shading textures before contracting for binary decisions.
  - `Ascending Layer CNN`: A more standard, robust hierarchical feature extractor.
- **Relative Row Scoring:** Asking "Is this bubble filled or empty?" fails when evaluating erasures. The new engine must evaluate rows (e.g., A, B, C, D, E) probabilistically, identifying the *strongest* signal and flagging 50/50 ties for a Teacher Override.

---

## 🛠️ 2. Lacking Infrastructure (What needs to be Coded here)

To the agent working on Phase 3, you must create the following files inside `src/`. **Do not write batch executors in these files; they must be importable classes for our Jupyter Notebooks.**

### A) `diamond.py`
- Define the PyTorch `nn.Module` class for the **Diamond CNN** only.
- Preserve depth profiles and pooling schedules used by production smoke runs.
- Keep constructor parameters stable for trainer/script compatibility.

### B) `ascending.py`
- Define the PyTorch `nn.Module` class for the **Ascending CNN** only.
- Ensure it accepts the exact tensor dimensions produced by Phase 2.
- Keep a standard forward pass and regularization behavior.

### C) `transfer_learning.py`
- Define the PyTorch `nn.Module` class for the **Transfer Learning** methodology.
- Use ResNet-18 with compatibility logic for torchvision weight APIs.
- Provide a frozen-backbone default, with explicit unfreeze control for fine-tuning cycles.

### D) `cnn_models.py`
- Keep this file as a compatibility facade/registry that re-exports Diamond, Ascending, and Transfer Learning classes.
- Do not place architecture internals here after separation.

### E) `dataset.py`
- Build a custom PyTorch `Dataset` class (`OMRDataset`) and dataloaders.
- It must ingest imagery from the `Unified_Datasets/Phase_3_Ready/` folder.
- Ensure the dataloader integrates seamlessly with `albumentations` for on-the-fly, real-time dataset jitter, rotation (±20° limits), and noise injection.

### F) `purge_data.py` (Script)
- We need a one-off utility script that reads `Unified_Datasets/Phase_2_Cropped/` and mathematically calculates the **Solidity** of every bubble.
- If the solidity gating confirms it is just a printed ring or a "ghost erasure", it should forcefully relabel it as `Blank` or move it to a rejection folder before creating the final `Phase_3_Ready` dataset. (This is the "Extreme Cleaning" step).

### G) `scoring.py` (The Decision Engine)
- Build the `RelativeRowScorer` class.
- It takes a tensor of predictions (5 probabilities for a 5-bubble row).
- **Logic:** Sort the predictions. If the highest probability is `> Threshold`, the bubble is marked as the answer. 
- **Erasure Checking:** If highest probability is e.g. 60% and second-highest is 50%, flag this specific row uniquely as `Tie / Ambiguous (Requires Human Review)`. 

### H) `trainer.py`
- A clean, modular PyTorch training loop wrapper.
- Must include built-in `MLflow` or `TensorBoard` integration for loss/accuracy curve plotting.
- Must implement Early Stopping and Model Checkpoint saving (to `Phase_3_Classification/models/`).
