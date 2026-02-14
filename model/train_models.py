import json
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
TARGET_COL = "target"

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
    }
    return metrics


def train_and_save(random_state: int = 42):
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset. Found: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = {
        "logistic": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))
        ]),
        "dt": DecisionTreeClassifier(random_state=random_state),
        "knn": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=9))
        ]),
        "nb": GaussianNB(),
        "rf": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state
        ),
    }

    metrics_all = {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(ART_DIR, f"{key}.joblib"))
        metrics_all[key] = evaluate_model(model, X_test, y_test)

    out = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI (via sklearn.datasets.load_breast_cancer)",
        "target_col": TARGET_COL,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "metrics": metrics_all
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    blob = train_and_save()
    print("Training complete. Saved:")
    print("-", METRICS_PATH)
    print("-", ART_DIR)
