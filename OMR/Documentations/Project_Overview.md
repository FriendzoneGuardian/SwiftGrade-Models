# 🌟 Project SwiftGrade: OMR Overhaul & Reboot

## 📊 Overview
Project **SwiftGrade** is a high-precision Optical Mark Recognition (OMR) system designed to achieve **99.9% classification accuracy**. The project is currently undergoing a **Phase 2 Reboot** to address noise, tilt, and lighting variations that previously hindered performance.

The project leverages a hybrid architecture combining **YOLO-based object detection** for bubble extraction and a **CNN** for classification. The current reboot ships **binary classification (filled/blank) with an uncertain review bucket**, with a planned upgrade path to 4-class (`crossed`, `invalid`) once binary performance is stable.

## 🎯 Mission Statement
To provide a robust, industrial-grade grading engine that is agnostic to ink/pencil types, resilient to 20° tilt/warping, and provides human-in-the-loop confidence scoring for absolute grading integrity.

## 🏗️ Core Components
1. **YOLO Detection:** Identifies fiducials and extracts individual bubble crops from raw answer sheets.
2. **OMR Preprocessor (Reboot):** A centralized pipeline for normalization (CLAHE), template subtraction, and solidity gating to eliminate "hollow letter" noise.
3. **CNN Classifier:** A deep learning model optimized for texture recognition. Current target is binary (filled/blank + uncertain), with a planned 4-class expansion.
4. **Decision Engine:** Performs relative row-level scoring to resolve ambiguities and erasures.

## 📜 Strategic Context
This overhaul is informed by the **Full Paper Revisions After Proposal (v6-1)**, which defines the "Secret Sauce" methodology:
- **Relative Scoring:** Comparing bubbles within a row rather than absolute thresholds.
- **Aggressive Masking:** 78% inner masking to bypass ring borders.
- **Gold Standard Dataset:** A purged dataset of 114K+ samples for training.
