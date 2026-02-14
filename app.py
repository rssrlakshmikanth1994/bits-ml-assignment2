import os
import json
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import confusion_matrix, classification_report

from model.train_models import train_and_save

MODEL_FILES = {
    "Logistic Regression": "logistic.joblib",
    "Decision Tree": "dt.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "nb.joblib",
    "Random Forest (Ensemble)": "rf.joblib",
    "XGBoost (Ensemble)": "xgb.joblib",
}

METRICS_PATH = os.path.join("model", "metrics.json")
ART_DIR = os.path.join("model", "artifacts")

st.set_page_config(page_title="BITS WILP ML Assignment 2", layout="wide")
st.title("BITS WILP — ML Assignment 2 (Classification Models + Streamlit)")

st.markdown("""
This app demonstrates 6 classification models on the **Breast Cancer Wisconsin (Diagnostic)** dataset (UCI).
It supports:
- CSV upload (recommended: test data)
- Model selection dropdown
- Metrics comparison table
- Confusion matrix / classification report
""")

def ensure_trained():
    # If any required model file missing, train once and save artifacts + metrics.json
    needed = [os.path.join(ART_DIR, f) for f in MODEL_FILES.values()]
    if not os.path.exists(METRICS_PATH) or any(not os.path.exists(p) for p in needed):
        with st.spinner("Training models (one-time setup)..."):
            os.makedirs(ART_DIR, exist_ok=True)
            train_and_save(random_state=42)

@st.cache_resource
def load_models():
    models = {}
    for ui_name, fname in MODEL_FILES.items():
        models[ui_name] = joblib.load(os.path.join(ART_DIR, fname))
    return models

def load_metrics_blob():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

ensure_trained()
models = load_models()
metrics_blob = load_metrics_blob()

with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox("Model Selection", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Dataset Upload (CSV)", type=["csv"])
    st.caption("Use `data/test_sample.csv` for quick testing (includes target column).")

st.subheader("Evaluation Metrics (Comparison Table)")

key_map = {
    "Logistic Regression": "logistic",
    "Decision Tree": "dt",
    "kNN": "knn",
    "Naive Bayes": "nb",
    "Random Forest (Ensemble)": "rf",
    "XGBoost (Ensemble)": "xgb",
}

rows = []
m = metrics_blob["metrics"]
for ui_name in MODEL_FILES.keys():
    k = key_map[ui_name]
    rows.append({
        "ML Model Name": ui_name,
        "Accuracy": m[k]["accuracy"],
        "AUC Score": m[k]["auc"],
        "Precision": m[k]["precision"],
        "Recall": m[k]["recall"],
        "F1 Score": m[k]["f1"],
        "MCC Score": m[k]["mcc"],
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.subheader("Confusion Matrix / Classification Report")

target_col = metrics_blob.get("target_col", "target")
model = models[selected_model]

if uploaded is None:
    st.info("Upload a CSV file from the sidebar to run evaluation/predictions.")
else:
    df = pd.read_csv(uploaded)
    st.write("Uploaded data preview:")
    st.dataframe(df.head(10), use_container_width=True)

    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y_true = df[target_col]
        y_pred = model.predict(X)

        st.write("### Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

        st.write("### Classification Report")
        st.code(classification_report(y_true, y_pred, zero_division=0))
    else:
        st.warning(f"Target column `{target_col}` not found. Showing predictions only.")
        preds = model.predict(df)
        out = df.copy()
        out["prediction"] = preds
        st.dataframe(out.head(30), use_container_width=True)
