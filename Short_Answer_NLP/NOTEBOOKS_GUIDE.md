# Jupyter Notebooks Guide

This document explains the three Jupyter notebooks for the Short_Answer_NLP module.

## 📊 1. ASAP-AES_EDA.ipynb (Exploratory Data Analysis)

**Purpose:** Understand the structure and characteristics of the ASAP-AES dataset

**Cells:**
1. **Markdown Header** - Overview and objectives
2. **Imports** - Load libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
3. **Configuration** - Specify data directory and verify ASAP-AES files exist
4. **Load Data** - Read Excel files for all essay sets
5. **Inspect Structure** - Display columns, data types, missing values, sample essays
6. **Score Statistics** - Compute mean, median, std dev, min/max for each essay set
7. **Score Distributions** - Visualize histograms and statistics for each set
8. **Inter-Rater Agreement** - Compute QWK (Quadratic Weighted Kappa) between raters
9. **Essay Length Analysis** - Analyze word counts and essay length statistics
10. **Next Steps** - Recommendations for training phase

**Key Outputs:**
- Dataset summary statistics (essay counts, score ranges)
- Inter-rater agreement scores (QWK values)
- Score distribution visualizations
- Essay length statistics

**Runtime:** ~2-3 minutes

**Prerequisites:**
- ASAP-AES Excel files in `Unified_Datasets/` folder (consolidated data location)

---

## 🏋️ 2. Train_Essay_Scoring_Model.ipynb (Model Training)

**Purpose:** Train XGBoost regressors for each essay set with QWK optimization

**Cells:**
1. **Markdown Header** - Overview of training workflow
2. **Setup & Imports** - Load modules, set configurations, create directories
3. **Load Data** - Use ASAPDataLoader to read all essay sets
4. **Extract Features** - Compute 15+ linguistic features (word count, TTR, grammar depth, etc.)
5. **Train Models** - Train independent XGBoost models per essay set with Optuna hyperparameter tuning optimized for QWK
6. **Feature Importance** - Analyze which features are most influential for predictions
7. **Summary & Next Steps** - Review training results and deployment options

**Key Outputs:**
- Trained models saved to `../models/` (one per essay set)
- Training results table (QWK, MAE, RMSE per set)
- Feature importance rankings
- `training_results.csv` with performance metrics

**Runtime:** 30-60 minutes (depending on dataset size and Optuna trials)

**Prerequisites:**
- ASAP-AES Excel files in `Unified_Datasets/` folder
- spaCy English model installed: `python -m spacy download en_core_web_sm`

---

## 🎯 3. Evaluate_Essays.ipynb (Interactive Scoring & Feedback)

**Purpose:** Score new essays and generate personalized improvement feedback

**Cells:**
1. **Markdown Header** - Overview and capabilities
2. **Setup & Imports** - Load modules, initialize feature extractor
3. **Load Models** - Read trained XGBoost models from `../models/` directory
4. **Interactive Scoring** - Helper function `score_essay()` for real-time essay evaluation
   - Extracts features from essay text
   - Runs inference with trained model
   - Returns predicted score and confidence
5. **Generate Feedback** - Helper function `generate_essay_feedback()` 
   - Analyzes feature values (word count, vocabulary, spelling, sentence structure)
   - Generates strengths, weaknesses, and improvement suggestions
6. **Batch Evaluation** - Function `batch_evaluate_essays()` for scoring multiple essays from CSV
   - Loads CSV file
   - Scores each essay
   - Saves results with predictions and confidence scores
7. **Summary & API Integration** - Deployment options and next steps

**Key Usage:**

**Interactive Scoring:**
```python
# Edit essay_text and essay_set_id, then run:
essay_text = "Your essay here..."
essay_set_id = 1  # 1-8
result = score_essay(essay_text, essay_set_id)
# Returns: {predicted_score, rounded_score, confidence, features}
```

**Batch Evaluation:**
```python
results_df = batch_evaluate_essays(
    csv_file=Path('../data/essays_to_score.csv'),
    essay_col='essay_text',
    set_col='essay_set',
    output_file=Path('../outputs/batch_scores.csv')
)
```

**Key Outputs:**
- Predicted scores for individual essays
- Personalized feedback (strengths, weaknesses, suggestions)
- Feature analysis (which features drove the score)
- Batch evaluation CSV with results
- Confidence scores and flags

**Runtime:** <1 second per essay

**Prerequisites:**
- Trained models from Train_Essay_Scoring_Model.ipynb
- Feature extractor initialized and spaCy model available

---

## 🔄 Typical Workflow

1. **Start with ASAP-AES_EDA.ipynb**
   - Ensure ASAP-AES Excel files are in `../data/`
   - Run all cells to inspect dataset structure
   - Verify data quality and score distributions

2. **Run Train_Essay_Scoring_Model.ipynb**
   - Takes 30-60 minutes
   - Trains models for each essay set
   - Reviews feature importance
   - Saves trained models to `../models/`

3. **Use Evaluate_Essays.ipynb**
   - Load trained models
   - Score individual essays interactively
   - Generate feedback for students
   - Batch evaluate large datasets

---

## 📁 Data & Output Directories

**Expected directory structure:**
```
SwiftGrade-Models/
├── Unified_Datasets/                  # Consolidated training data (all ASAP-AES Excel files)
│   ├── essays_set_1.xlsx
│   ├── essays_set_2.xlsx
│   └── ...
├── Short_Answer_NLP/
│   ├── models/                        # Trained XGBoost models (auto-generated)
│   │   ├── essay_set_1_model.pkl
│   │   ├── essay_set_2_model.pkl
│   │   └── ...
│   ├── outputs/                       # Results and reports
│   │   ├── training_results.csv       # From Train notebook
│   │   └── batch_scores.csv           # From Evaluate notebook
│   └── notebooks/
│       ├── ASAP-AES_EDA.ipynb
│       ├── Train_Essay_Scoring_Model.ipynb
│       └── Evaluate_Essays.ipynb
```

---

## ⚙️ Configuration

Edit `../configs/essay_scoring_config.json` to customize:
- Feature list to use
- XGBoost hyperparameters per essay set
- QWK thresholds for confidence classification
- Feedback rules and suggestions

---

## 🚀 Deployment (Phase 2)

The evaluation functions can be wrapped in a Flask API for Flutter app integration:

```python
@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    essay_text = data['essay_text']
    essay_set = data['essay_set']
    result = score_essay(essay_text, essay_set)
    return jsonify(result)
```

---

## 📌 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `../data` | Location of ASAP-AES Excel files |
| `MODEL_DIR` | `../models` | Where trained models are saved |
| `OUTPUT_DIR` | `../outputs` | Results and evaluation outputs |
| `n_trials` (Optuna) | 50 | Hyperparameter tuning iterations |
| `train_size` | 0.70 | Fraction for training split |
| `val_size` | 0.15 | Fraction for validation split |
| `test_size` | 0.15 | Fraction for test split |

---

## 🔧 Troubleshooting

**Issue:** "No models found in models directory"
- **Solution:** Run Train_Essay_Scoring_Model.ipynb first

**Issue:** "spaCy model not found"
- **Solution:** `python -m spacy download en_core_web_sm`

**Issue:** "ASAP-AES Excel files not found"
- **Solution:** Copy files to `Unified_Datasets/` folder at the workspace root

**Issue:** Low QWK scores (< 0.5)
- **Solution:** 
  - Increase `n_trials` for better hyperparameter tuning
  - Add more linguistic features
  - Check data quality and outliers

---

**Last Updated:** 2024
**Version:** 1.0
