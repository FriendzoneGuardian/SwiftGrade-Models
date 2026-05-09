"""
README: Short_Answer_NLP Module

Automated Essay Scoring using ASAP-AES Dataset Approach
"""

# Short_Answer_NLP: Automated Essay Scoring Module

## Overview

This module implements Automated Essay Scoring (AES) using the ASAP-AES dataset methodology. It extracts linguistic features from student essays, trains regression models, and generates human-readable feedback.

**Key Features:**
- ✅ **Feature Extraction**: 13+ linguistic features (length, vocabulary, grammar, spelling)
- ✅ **Per-Essay-Set Models**: Separate XGBoost models for each essay set (different rubrics)
- ✅ **QWK Evaluation**: Uses Quadratic Weighted Kappa (standard for essay scoring)
- ✅ **Confidence Scoring**: Flags low-confidence predictions for human review
- ✅ **Feedback Generation**: Constructive feedback with strengths, weaknesses, suggestions
- ✅ **GPU Optimized**: Works with GTX 1650 (4GB VRAM)

## Directory Structure

```
Short_Answer_NLP/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load ASAP-AES Excel data
│   ├── feature_extractor.py     # Extract 13+ linguistic features
│   ├── metrics.py               # QWK and evaluation metrics
│   ├── essay_score_regressor.py # Train XGBoost models
│   ├── essay_evaluator.py       # Score new essays
│   └── feedback_generator.py    # Generate feedback
├── notebooks/
│   ├── ASAP-AES_EDA.ipynb       # Exploratory data analysis
│   ├── Train_Essay_Scoring_Model.ipynb
│   └── Evaluate_Essays.ipynb
├── configs/
│   └── essay_scoring_config.json
├── models/
│   └── [trained models saved here]
├── outputs/
│   └── [evaluation results saved here]
├── evaluate.py                  # Batch evaluation entry point
├── requirements_aes.txt
└── README.md (this file)
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_aes.txt
   python -m spacy download en_core_web_sm
   ```

2. **Prepare ASAP-AES data:**
   - Download from Kaggle: https://www.kaggle.com/c/asap-aes
   - Place Excel files in `../Unified_Datasets/` folder (consolidated data location)

## Usage

### 1. Exploratory Data Analysis

Open `notebooks/ASAP-AES_EDA.ipynb` to:
- Inspect essay sets and score distributions
- Understand inter-rater agreement (QWK)
- Identify data quality issues

### 2. Train Models

Open `notebooks/Train_Essay_Scoring_Model.ipynb` to:
- Load ASAP-AES data
- Extract features for all essays
- Train one XGBoost model per essay set
- Evaluate on test set and save models

### 3. Evaluate New Essays

**Interactive (Jupyter):**
```python
# In notebook
from src.essay_evaluator import EssayEvaluator

evaluator = EssayEvaluator(regressor, model_dir='./models')

result = evaluator.evaluate(
    essay_text="...",
    essay_set_id=1
)

print(f"Predicted Score: {result['predicted_score']:.1f}")
print(f"Confidence: {result['confidence']:.2f}")
```

**Batch (Command line):**
```bash
python evaluate.py \
    --data essays.xlsx \
    --essay-set 1 \
    --models-dir ./models \
    --output results.csv \
    --with-feedback
```

## Features Extracted

The module extracts 13 linguistic features:

| Category | Features |
|----------|----------|
| **Length** | word_count, sentence_count, avg_sentence_length |
| **Vocabulary** | ttr_ratio (type-token ratio), advanced_word_ratio, repetition_ratio |
| **Grammar** | subordinate_clause_ratio, pronoun_ratio, max_parse_depth |
| **Spelling** | misspelling_rate |

## Model Architecture

- **Framework**: XGBoost Regressor (RandomForest fallback)
- **Features**: StandardScaler normalized
- **Optimization**: Hyperparameters tuned for QWK metric
- **Training**: Per-essay-set (never mix essay sets)
- **Evaluation**: Quadratic Weighted Kappa (QWK), MAE, RMSE

## Output Format

### Prediction Output (CSV)
```
essay_index,predicted_score,confidence,flagged_for_review,feedback
0,7.5,0.82,False,"{...}"
```

### Feedback JSON
```json
{
  "summary": "This is a high-quality essay.",
  "strengths": [
    "Strong essay length shows thorough exploration.",
    "Excellent vocabulary variety."
  ],
  "weaknesses": [],
  "suggestions": [
    "✓ Consider using more advanced vocabulary."
  ],
  "predicted_score": 7.5
}
```

## Configuration

Edit `configs/essay_scoring_config.json` to:
- Change model hyperparameters
- Adjust confidence thresholds
- Enable/disable feature types
- Specify output paths

## Metrics Explained

### QWK (Quadratic Weighted Kappa)
- **Range**: -1 to 1 (1 = perfect agreement)
- **Target**: 0.70+ indicates good model
- **Why**: Standard metric for essay scoring (mimics human inter-rater agreement)

### MAE (Mean Absolute Error)
- Average absolute difference between predicted and human scores
- Lower is better

### Confidence Score
- **0.0**: Very uncertain (close to decision boundary)
- **1.0**: Very confident (far from boundary)
- Predictions with confidence < 0.70 flagged for human review

## Troubleshooting

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Memory issues with large datasets
- Reduce batch size in feature extraction
- Use CPU-only mode if GPU memory is limited
- Process essays in smaller chunks

### XGBoost not installed
- Falls back to RandomForest automatically
- Install XGBoost: `pip install xgboost`

## Future Enhancements

- [ ] Fine-tune BERT/RoBERTa on ASAP-AES (if classical ML underperforms)
- [ ] Incremental model retraining with new essays
- [ ] Multi-dimensional rubric evaluation (separate scores for Content, Grammar, etc.)
- [ ] Integration with Flutter app via API
- [ ] Database persistence for scoring history

## References

- ASAP-AES Kaggle Competition: https://www.kaggle.com/c/asap-aes
- Quadratic Weighted Kappa: https://en.wikipedia.org/wiki/Cohen%27s_kappa
- XGBoost Documentation: https://xgboost.readthedocs.io/

## License

Part of SwiftGrade Project

---

**Version**: 0.1.0  
**Last Updated**: May 9, 2026
