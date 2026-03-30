# 🗺️ SwiftGrade-Vision Development Roadmap & Agent Checklist

Hello Future Agent! You are initializing the development phase of the brand new `SwiftGrade-Vision` repository. All the chaotic bloat from the old legacy models has been purged. 

Your objective is to execute the following phases of development in **Strict PyTorch** and **YOLO (Ultralytics)**. Do not drift back to TensorFlow or Keras.

---

### 🚨 Prerequisites
- [ ] Ensure the User has extracted their dataset ZIP into the `Unified_Datasets/` folder.
- [ ] Verify you have read the `OMR/Documentations/NOTEBOOK_WORKFLOW.md` to understand that we write classes in `/src/` but execute them exclusively via Jupyter Notebooks in `OMR/notebooks/`.

---

### Phase 1: YOLO Extraction Engine (The Foundation)
**Goal:** Mathematically identify the edges of the form and zero in on the exact grid layout, regardless of smartphone photo skew or rotation constraints.
- [ ] **Construct:** `OMR/Phase_1_Extraction/src/yolo_detector.py`. 
    - *What it needs:* Object-oriented wrapper around `ultralytics` YOLOv8. Focus on finding document corner fiducials and cropping down strictly to the bubble grid area.
- [ ] **Test:** Create an `.ipynb` in `OMR/notebooks/` to feed images from `Unified_Datasets/Phase_1_Raw/` through your detector and visualize the bounding boxes.

### Phase 2: Preprocessing Engine (The "Secret Sauce")
**Goal:** Digitize physical *Scantron Drop-Out Ink* to remove printed geometric lines so the neural net only has to care about graphite.
- [ ] **Construct:** `OMR/Phase_2_Preprocessing/src/preprocessor.py`.
    - *What it needs:* Port the math for **Multi-Channel CLAHE** and **Significance-Gated Subtraction**. It must perfectly normalize varied lighting conditions and mathematically "erase" the printed A/B/C letter rings.
- [ ] **Construct:** `OMR/Phase_2_Preprocessing/src/crop_engine.py`.
    - *What it needs:* Taking the Phase 1 grid boundaries and Phase 2's normalized sheets to slice out the raw 224x224 (approximate) bubble squares.
- [ ] **Test:** Save these crops into `Unified_Datasets/Phase_2_Cropped/` via an `.ipynb` execution script in `OMR/notebooks/`.

### Phase 3: Classification & Decision Engine (The Brain)
**Goal:** Outperform standard baseline OMRs by judging erasures and marks *probabilistically* rather than using absolute thresholding.
- [ ] **Construct:** `OMR/Phase_3_Classification/src/dataset.py` & `purge_data.py`.
    - *What it needs:* The dataloader. Before training, run the "Extreme Cleaning" script (Solidity = Area/Convex Area) to flag and discard ambiguous erasures from the dataset. Move the pristine crops to `Unified_Datasets/Phase_3_Ready/`. Inject ±20° rotation via Albumentations into the dataloader.
- [ ] **Construct:** `OMR/Phase_3_Classification/src/cnn_models.py`.
    - *What it needs:* The **Diamond CNN** or **Ascending Layer CNN** PyTorch architecture declarations.
- [ ] **Construct:** `OMR/Phase_3_Classification/src/scoring.py` (The Decision Engine).
    - *What it needs:* **Relative Row Scoring.** Compare probabilities of bubbles (A, B, C, D, E) against each other. It finds the "Dominant" bubble, flags 50/50 ties for Teacher Override, and mathematically eliminates the problem of faint erasures.
- [ ] **Construct:** `OMR/Phase_3_Classification/src/trainer.py`
    - *What it needs:* MLflow setup, Early Stopping, and Model Checkpoints.
- [ ] **Test:** Open `OMR/notebooks/Phase3_Classification_Masterclass.ipynb` to execute the full PyTorch training loop and validate inference accuracy on test sets.

---
**Next Step for the Agent reading this:** Start with `Phase 1` and notify the User you are ready to architect the YOLO pipeline.
