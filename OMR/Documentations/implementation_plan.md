# 📝 SwiftGrade OMR: Core Migration & Refactoring Plan

## 1. Context: The Pivot
The current `Project-SwiftGrade-Models` repository has become an experimental dumping ground with mixed workflows, scattered trials (`Trial2`, `Trial3`), and conflicting frameworks (TensorFlow vs. PyTorch). To achieve deterministic, production-ready 99.9% accuracy, we are **salvaging the proven math/logic** and creating a highly disciplined, phased project structure. 

Crucially, **TensorFlow and Keras are officially retired**. The new codebase will be powered exclusively by the **YOLO (Ultralytics)** and **PyTorch** stacks.

---

## 🏗️ 2. The Professional Directory Structure
The new project will inherently enforce best practices by strictly grouping code into defined phases for the OMR pipeline.

```text
Project-SwiftGrade-Vision/
├── OMR/
│   ├── Documentations/          # Agent Artifacts & Vibe Coder Collaborations
│   │   ├── Butler_Phase2_Report.md
│   │   ├── OMRChecker_Adoption_Assessment.md
│   │   └── (Other relevant markdown files from Collaboration_Docs)
│   │
│   ├── OMR-Basis/               # Open Source Reference Project
│   │   └── (Will copy existing contents from the root OMR-Basis directory here)
│   │
│   ├── Phase_1_Extraction/      # YOLO Detection & Localization
│   │   └── src/
│   │       ├── yolo_detector.py # Finds fiducials and isolates bubbles
│   │       └── utils.py
│   │
│   ├── Phase_2_Preprocessing/   # Normalization & Augmentation
│   │   └── src/
│   │       ├── preprocessor.py  # OMRPreprocessor (CLAHE, Subtraction)
│   │       └── crop_engine.py   # Slices bubbles from Phase 1 masks
│   │
│   └── Phase_3_Classification/  # PyTorch CNN & Decision Engine
│       └── src/
│           ├── cnn_models.py    # Diamond/Ascending CNN (PyTorch)
│           ├── dataset.py       # PyTorch DataLoaders
│           ├── trainer.py       # MLflow/TensorBoard training loop
│           └── scoring.py       # Relative Row Scoring Rules
│
├── Unified_Datasets/            # (Gitignored - User will extract zip here)
│   ├── README.md                # Notes to place Phase 1/2/3 zip contents here
│   ├── Phase_1_Raw/             # Full scanned sheets
│   ├── Phase_2_Cropped/         # Preprocessed bubble crops
│   └── Phase_3_Ready/           # Fully annotated, purged standard dataset
│
├── notebooks/                   # STRICTLY for viewing datasets & prototyping
└── requirements.txt             # Trimmed dependencies (PyTorch focused, TF dropped)
```

---

## 🚀 3. Migration Roadmap (Action Items)

### Step 1: Initialize the Skeleton & Transfer Context
- **Create internal structure:** Scaffold `OMR/Phase_{1,2,3}`, `OMR/Documentations`, `OMR/OMR-Basis`, and `Unified_Datasets`.
- **Migrate Docs:** Move relevant, current files from the old `Collaboration_Docs` directly into `OMR/Documentations/`.
- **Migrate Reference:** Copy the entire contents of the existing root `OMR-Basis` folder directly into `OMR/OMR-Basis/`. (I will handle this moving automatically!)
- **Dataset Setup:** Write a brief `.md` in `Unified_Datasets` as a reminder to extract the dataset ZIP there.

### Step 2: Establish the Golden Environment
- **Distill `requirements.txt`:** Create a single, clean `.txt` using only the non-deprecated dependencies needed across Phases 1-3. 
- *Included:* `torch`, `torchvision`, `ultralytics`, `opencv-python`, `scikit-learn`, `albumentations`, `mlflow`.
- *Excluded:* `tensorflow`, `keras`, and unused visualizers from the old trials.

### Step 3: Architect Phase 1 (Extraction)
- Port over the YOLO bounding box and fiducial extraction logic.
- Verify the output correctly isolates the grid locations to hand off cleanly into Phase 2.

### Step 4: Architect Phase 2 (Preprocessing & Augmentation)
- Centralize the "Secret Sauce" into a robust `OMRPreprocessor` class.
- Incorporate Multi-channel CLAHE and Significance-Gated Subtraction.
- Validate that the logic effectively bridges Phase 1's crops with Phase 3's required tensor dimensions.

### Step 5: Architect Phase 3 (Classification)
- Build out the dataset purger logic (calculating Solidity to remove noise).
- Implement the baseline PyTorch models (e.g., Ascending/Diamond CNN).
- Wire in the Decision Engine to execute **Relative Row Scoring** across the extracted row inputs.

---

> [!IMPORTANT]  
> **Awaiting Final Green Light**  
> Is this structure absolutely dialed in for you now? Once approved, we can spin up the scripts to physically create the project layout, copy your docs and reference codebase, and generate the tailored requirements file!
