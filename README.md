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
3. K-Nearest Neighbors  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Metrics Comparison Table (auto-generated)
The table is automatically generated in the app from `model/metrics.json`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| kNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |

### Observations (fill after your run)
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| XGBoost (Ensemble) | |

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
│   └─ artifacts/   (generated on first run)
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

## Quick Test
Upload: `data/test_sample.csv` (includes target column, so confusion matrix + report will show).
