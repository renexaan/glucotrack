"""Inference-time preprocessing that mirrors the notebooks step for step.

Each function takes RAW patient inputs (the columns a user can type in) and
reproduces exactly what the notebook did before the model saw the data. The
export cell at the end of every notebook asserts that these functions give the
same matrix as the notebook's own in-memory pipeline, so the app cannot drift
away from the experiments.

Nothing here is fitted. Every learned parameter (imputer medians, IQR fences,
frequency map, encoder categories, scaler mean/std, centroids) is read from the
artifact bundle that the notebook saved after fitting on the training split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Classification (notebooks/classification.ipynb, sections 3.1 - 3.7)
# --------------------------------------------------------------------------- #

CLASSIFICATION_RAW_COLUMNS = [
    "gender", "age", "location", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level", "race",
]


def engineer_classification_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The three engineered features from notebook section 3.5 (row-wise only)."""
    out = frame.copy()
    out["bmi_category"] = pd.cut(
        out["bmi"],
        bins=[-np.inf, 18.5, 25.0, 30.0, 35.0, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese_I", "Obese_II+"],
    ).astype(str)
    out["age_risk_interaction"] = (out["age"] * out["hbA1c_level"]) / 100.0
    out["metabolic_risk_score"] = (
        (out["hbA1c_level"] >= 5.7).astype(int)
        + (out["blood_glucose_level"] >= 140).astype(int)
        + (out["bmi"] >= 30).astype(int)
        + out["hypertension"].astype(int)
        + out["heart_disease"].astype(int)
    )
    return out


def preprocess_classification(raw: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Raw patient rows -> the exact feature matrix the classifiers were trained on."""
    X = engineer_classification_features(raw)[b["x_columns"]].copy()

    # step 1: imputation (a no-op on complete inputs, kept for fidelity)
    num_cols = b["numeric_imputed_cols"]
    X[num_cols] = b["num_imputer"].transform(X[num_cols])
    cat_all = b["onehot_cols"] + b["freq_cols"]
    if cat_all:
        X[cat_all] = b["cat_imputer"].transform(X[cat_all])

    # step 2: winsorise with the fences learned on the training split
    for col, (lo, hi) in b["iqr_bounds"].items():
        X[col] = X[col].clip(lo, hi)

    # step 3: frequency encoding, unseen level -> 0.0 exactly as in the notebook
    for col in b["freq_cols"]:
        X[col + "_freq"] = X[col].map(b["freq_map"]).fillna(0.0).astype(float)
        X = X.drop(columns=[col])

    # steps 4-6: one-hot, standardise, assemble in training column order
    ohe = b["ohe"].transform(X[b["onehot_cols"]])
    num = b["scaler"].transform(X[b["numeric_cols"]])
    return pd.DataFrame(np.hstack([num, ohe]), columns=b["feature_names"], index=X.index)


# --------------------------------------------------------------------------- #
# Regression (notebooks/regression.ipynb, sections 3.3 - 3.5)
# --------------------------------------------------------------------------- #

REGRESSION_RAW_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "diabetes",
]


def preprocess_regression(raw: pd.DataFrame, b: dict) -> np.ndarray:
    X = raw.copy()
    X["age_bmi_interaction"] = X["age"] * X["bmi"]
    return b["preprocessor"].transform(X[b["x_columns"]])


# --------------------------------------------------------------------------- #
# Clustering (notebooks/clustering.ipynb, sections 3 - 4)
# --------------------------------------------------------------------------- #

CLUSTERING_RAW_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level",
]


def preprocess_clustering(raw: pd.DataFrame, b: dict) -> np.ndarray:
    return np.asarray(b["preprocessor"].transform(raw[b["x_columns"]]), dtype=np.float32)
