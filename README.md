# 🚀 SwiftGrade OMR Pipeline

Welcome to the newly overhauled SwiftGrade OMR workspace! This repository houses the entire YOLO and PyTorch logic pipeline for detecting, preprocessing, and classifying OMR answer sheets with 99.9% targeted accuracy.

## 📓 General Pipeline Instructions: The Notebook Workflow
To maintain reproducibility, visualize outputs quickly, and allow for clean execution logic, **all execution of these phases should be performed via Jupyter Notebooks (`.ipynb`)**. 

1. **Do not run the `src/` modules directly from the terminal for your data processing workflows.** 
2. The core mathematics and classes exist in `OMR/Phase_.../src/`, but you interact with them and execute the data loading, training loops, and validation exclusively through the notebooks housed in the `OMR/notebooks/` directory.
3. Example: `OMR/notebooks/Phase3_Classification_Masterclass.ipynb` acts as the master trigger for Phase 3. 
4. If you need to write a new execution step (e.g. running Phase 1 Extraction across a batch), construct a new `.ipynb` file in the `OMR/notebooks/` folder.

This approach guarantees that visual data representations (like augmented crops, CLAHE outputs, and bounding boxes) are immediately visible next to the code that ran them, simplifying debugging and data tracking!

## Directory Structure
- `OMR/`: Houses the 3 distinct phases (Extraction, Preprocessing, Classification) and its core scripts.
    - `OMR/notebooks/`: The execution ground layer! Run all your pipeline tests from here.
    - `OMR/Documentations/`: Where we store all our Vibe Coding design patterns and Agent logic.
- `Unified_Datasets/`: The centralized storage for Phase 1 Raw, Phase 2 Cropped, and Phase 3 Ready dataset iterations.
