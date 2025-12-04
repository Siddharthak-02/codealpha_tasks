"""
CodeAlpha - Disease Prediction from Medical Data

Task 4:
Objective:
    Predict the possibility of diseases based on patient data.

Approach:
    Apply classification algorithms (SVM, Logistic Regression, Random Forest, XGBoost)
    to structured medical datasets.

Key Features:
    - Uses features like symptoms, age, blood test results, etc. (simulated here)
    - Works on three disease datasets: breast cancer, diabetes, heart disease
    - Evaluates models using Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")  # keep terminal output clean

# Try to import XGBoost, but if anything fails, just skip using it
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print("XGBoost not available, will skip it. Reason:", e)
    XGBOOST_AVAILABLE = False


# --------- DATASET LOADERS ---------

def load_breast_cancer_dataset():
    """Load real breast cancer dataset from sklearn (based on UCI)."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return "Breast Cancer (sklearn)", X, y


def load_synthetic_diabetes_dataset(n_samples=1000, random_state=42):
    """
    Create a synthetic diabetes-like dataset.
    Feature names look like real medical features.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        random_state=random_state,
        weights=[0.55, 0.45],
    )
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return "Diabetes (synthetic)", X, y


def load_synthetic_heart_dataset(n_samples=1000, random_state=24):
    """
    Create a synthetic heart-disease-like dataset.
    Feature names are medically meaningful.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        random_state=random_state,
        weights=[0.6, 0.4],
    )
    feature_names = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
    ]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return "Heart Disease (synthetic)", X, y


# --------- MODEL BUILDING ---------

def build_models():
    """Return a dictionary of ML models to train."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "SVM (RBF kernel)": SVC(
            kernel="rbf", probability=True, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    return models


# --------- EVALUATION ---------

def evaluate_model(name, model, X_test, y_test):
    """Calculate metrics for a trained model and print them."""
    y_pred = model.predict(X_test)

    # probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        try:
            decision_fn = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-decision_fn))
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = float("nan")

    print(f"\nModel: {name}")
    print("-" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    return metrics


def run_experiment(dataset_name, X, y):
    """
    Train and evaluate all models on a single dataset.
    Returns a DataFrame with metrics for each model.
    """
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    models = build_models()
    results = []

    for model_name, clf in models.items():
        pipe = Pipeline([
            ("scaler", scaler),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(model_name, pipe, X_test, y_test)
        metrics["model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("model")
    print("\nSummary metrics for dataset:", dataset_name)
    print(results_df)
    return results_df


# --------- MAIN PIPELINE ---------

def main():
    datasets = [
        load_breast_cancer_dataset(),
        load_synthetic_diabetes_dataset(),
        load_synthetic_heart_dataset(),
    ]

    overall_results = {}

    for ds_name, X, y in datasets:
        df_metrics = run_experiment(ds_name, X, y)
        overall_results[ds_name] = df_metrics

    # Save summary metrics for each dataset
    for ds_name, metrics_df in overall_results.items():
        safe_name = ds_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        csv_name = f"metrics_{safe_name}.csv"
        metrics_df.to_csv(csv_name)
        print(f"\nSaved metrics for {ds_name} to {csv_name}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
