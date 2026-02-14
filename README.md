# ML Assignment 2 — Classification Models + Streamlit Deployment (BITS WILP)

## a) Problem Statement
Implement six classification models on a single dataset, evaluate them using:
Accuracy, AUC, Precision, Recall, F1, MCC — and deploy an interactive Streamlit app with:
- Dataset upload (CSV)
- Model selection dropdown
- Metrics display
- Confusion matrix / classification report

## b) Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI (loaded via `sklearn.datasets.load_breast_cancer`)
- **Task:** Binary classification
- **Instances (rows):** 569 (≥ 500)
- **Features:** 30 (≥ 12)
- **Target column:** `target`

## c) Models Used
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Metrics Comparison Table (auto-generated)
The table is automatically generated in the app from `model/metrics.json`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9944 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9649 | 0.9954 | 0.9595 | 0.9861 | 0.9726 | 0.9245 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with very high Accuracy/AUC and best MCC; scaling helps it perform extremely well on this dataset. |
| Decision Tree | Lowest performance among all models; prone to overfitting and weaker generalization compared to ensemble methods. |
| kNN | Excellent performance with perfect Recall (1.0); works well after scaling but is sensitive to the choice of k. |
| Naive Bayes | Fast and reasonably strong; slightly lower than kNN/LR/ensembles but still good overall. |
| Random Forest (Ensemble) | Consistently strong and stable; improves generalization compared to a single decision tree. |
| XGBoost (Ensemble) | Best overall Accuracy/F1 among ensembles with very high AUC; strong generalization on this dataset. |

## Project Structure
```
ml_assignment2_project/
│-- app.py
│-- requirements.txt
│-- README.md
│-- data/
│   ├─ dataset.csv
│   └─ test_sample.csv
│-- model/
│   ├─ train_models.py
│   ├─ metrics.json
│   └─ artifacts/
│       ├─ logistic.joblib
│       ├─ dt.joblib
│       ├─ knn.joblib
│       ├─ nb.joblib
│       ├─ rf.joblib
│       └─ xgb.joblib
```

## How to Run on BITS Virtual Lab
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Streamlit Cloud Deployment
1) Push this folder to GitHub  
2) Streamlit Community Cloud → New app → select repo/branch → choose `app.py` → Deploy

## Streamlit App Features Implemented
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics (comparison table)
- Confusion matrix and classification report

## Quick Test
Upload: `data/test_sample.csv` (includes target column, so confusion matrix + report will show).
