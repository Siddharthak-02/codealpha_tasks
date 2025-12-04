# credit_pipeline.py
# Improved version: auto-handle class imbalance (simple upsampling),
# safe metrics, and clear prints. Ready to run in your project folder.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import joblib
from sklearn.datasets import make_classification
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")  # keep output clean for now

# -----------------------
# 1) Load or create data
# -----------------------
data_path_candidates = [
    "credit_data.csv",
    "/mnt/data/credit_data.csv",
]

df = None
for p in data_path_candidates:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            print(f"Loaded dataset from {p}")
            break
        except Exception as e:
            print(f"Found file {p} but failed to read it: {e}")

if df is None:
    print("No CSV found, generating synthetic dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        weights=[0.7, 0.3],
        random_state=42,
    )
    rng = np.random.RandomState(42)
    df = pd.DataFrame(X, columns=[f"feat{i}" for i in range(1, 9)])
    df["income"] = (np.abs(df["feat1"]) * 20000 + 15000).astype(int)
    df["total_debt"] = (np.abs(df["feat2"]) * 15000 + 1000).astype(int)
    df["num_late_payments"] = np.clip((np.abs(df["feat3"]) * 3).round().astype(int), 0, 12)
    df["payment_history_score"] = np.clip((50 + df["feat4"] * 15).round().astype(int), 300, 850)
    df["credit_utilization"] = np.clip((np.abs(df["feat5"]) * 80).round().astype(int), 0, 200)
    df["age"] = np.clip((np.abs(df["feat6"]) * 15 + 25).round().astype(int), 18, 80)
    df["employment_years"] = np.clip((np.abs(df["feat7"]) * 6).round().astype(int), 0, 40)
    purposes = ["home", "car", "education", "business", "personal"]
    df["loan_purpose"] = rng.choice(purposes, size=len(df), p=[0.25, 0.2, 0.15, 0.1, 0.3])

    base_prob = 0.3 + (df["payment_history_score"] - 300) / 1000 - df["credit_utilization"]/300 - df["num_late_payments"]*0.03
    base_prob = np.clip(base_prob, 0.01, 0.99)
    target = (rng.rand(len(df)) < base_prob).astype(int)
    df["target"] = target
    df = df.drop(columns=[c for c in df.columns if c.startswith("feat")])

# -----------------------
# 2) Feature engineering
# -----------------------
df["debt_to_income"] = df["total_debt"] / (df["income"] + 1)
df["avg_payment_delay"] = df["num_late_payments"] / (df["employment_years"] + 1)

def score_cat(s):
    if s >= 750:
        return "high"
    elif s >= 650:
        return "medium"
    return "low"
df["credit_score_cat"] = df["payment_history_score"].apply(score_cat)

# Save processed snapshot
df.to_csv("processed_credit_data.csv", index=False)
print("Saved processed dataset as processed_credit_data.csv")

# -----------------------
# 3) Prepare features
# -----------------------
target_col = "target"
y = df[target_col]
X = df.drop(columns=[target_col])

numeric_features = [
    "income", "total_debt", "num_late_payments", "payment_history_score",
    "credit_utilization", "age", "employment_years", "debt_to_income", "avg_payment_delay"
]
categorical_features = ["loan_purpose", "credit_score_cat"]

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# -----------------------
# 4) Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nClass distribution (train):")
print(y_train.value_counts(normalize=False))
print("\nClass distribution (test):")
print(y_test.value_counts(normalize=False))

# If minority in training set is < 20%, do simple upsampling
minority_frac = y_train.mean()
if minority_frac < 0.20:
    print(f"\nMinority fraction in train is {minority_frac:.3f} — applying simple upsampling to training set.")
    train = X_train.copy()
    train["target"] = y_train.values
    majority = train[train["target"] == 0]
    minority = train[train["target"] == 1]
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=42)
    train_upsampled = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
    y_train = train_upsampled["target"]
    X_train = train_upsampled.drop(columns=["target"])
    print("Upsampling done. New train distribution:")
    print(y_train.value_counts())
else:
    print("\nNo upsampling required.")

# -----------------------
# 5) Models to try
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
}

trained_pipelines = {}
reports = {}

for name, clf in models.items():
    print(f"\nTraining {name} ...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    # get probabilities if possible
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)
    else:
        # fallback to decision function converted via sigmoid
        try:
            dfcn = pipe.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-dfcn))
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)

    # metrics (use zero_division to avoid warnings)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        rocauc = roc_auc_score(y_test, y_proba)
    except ValueError:
        rocauc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    reports[name] = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": rocauc,
        "confusion_matrix": cm,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

    print(f"{name} classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(cm)
    print(f"ROC-AUC: {rocauc:.4f}")

# -----------------------
# 6) Save best model
# -----------------------
# choose best by ROC-AUC (ignore NaNs)
valid_scores = {k: v for k, v in reports.items() if not (pd.isna(v["roc_auc"]))}
if valid_scores:
    best_name = max(valid_scores.keys(), key=lambda k: valid_scores[k]["roc_auc"])
else:
    best_name = max(reports.keys(), key=lambda k: reports[k]["f1"])

best_pipeline = trained_pipelines[best_name]
joblib.dump(best_pipeline, "best_model.pkl")
print(f"\nBest model saved as best_model.pkl ({best_name})")

print("\nDone.")
