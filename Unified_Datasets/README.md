# 🗃️ Unified Datasets

This folder is designed to store all iterations and transformations of the dataset across the OMR Pipeline, as well as consolidated training data for the NLP module.

**Instructions:**
Extract the contents of your existing datasets zip straight into this root `Unified_Datasets` directory.

### Folder Usage

#### OMR Pipeline (Phases 1-3)
*   **`Phase_1_Raw/`**: Contains the un-edited, full answer sheet scans or images.
*   **`Phase_2_Cropped/`**: Will contain the isolated, normalized bubble crops extracted by the preprocessing engine.
*   **`Phase_3_Ready/`**: Will house the extreme-cleaned, purged (solidity analyzed), and strictly labeled dataset ready for PyTorch dataloaders.

#### Short_Answer_NLP Module (Free-Text Essay Scoring)
*   **ASAP-AES Excel Files**: Place ASAP-AES dataset files (.xlsx) here for essay scoring model training
    - Download from Kaggle: https://www.kaggle.com/c/asap-aes
    - Expected files: Multiple Excel files (one per essay set, typically 8 sets)
    - Notebooks reference: `../../Unified_Datasets/` (from `Short_Answer_NLP/notebooks/`)
    - Required columns in Excel: `essay_set`, `essay`, `score` (and optionally `score2` for inter-rater agreement)
