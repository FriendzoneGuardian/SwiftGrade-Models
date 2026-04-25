# 🗃️ Unified Datasets

This folder is designed to store all iterations and transformations of the dataset across the OMR Pipeline.

**Instructions:**
Extract the contents of your existing datasets zip straight into this root `Unified_Datasets` directory.

### Folder Usage

*   **`Phase_1_Raw/`**: Contains the un-edited, full answer sheet scans or images.
*   **`Phase_2_Cropped/`**: Will contain the isolated, normalized bubble crops extracted by the preprocessing engine.
*   **`Phase_3_Ready/`**: Will house the extreme-cleaned, purged (solidity analyzed), and strictly labeled dataset ready for PyTorch dataloaders.
