"""Loading of the artifacts exported by the notebooks (section 8 / 12 of each notebook)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

# file -> notebook that produces it, used for a precise error message when one is missing
REQUIRED = {
    "classification": ("notebooks/classification.ipynb (section 8)",
                       ["bundle.joblib", "metrics.csv", "tuning.csv", "cv_top2.csv", "confusion.json",
                        "roc.json", "stats.json", "selection_scorecard.csv", "feature_importance.csv"]),
    "regression": ("notebooks/regression.ipynb (final export cell)",
                   ["bundle.joblib", "metrics.csv", "cv_top2.csv", "tuning.csv", "stats.json",
                    "pred_vs_actual_sample.csv", "feature_importance_rf.csv", "coefficients.csv",
                    "polynomial_degrees.csv"]),
    "clustering": ("notebooks/clustering.ipynb (section 12)",
                   ["bundle.joblib", "metrics.csv", "k_selection.csv", "figures.npz", "profiles.json",
                    "stats.json", "posthoc_agglomerative.csv"]),
}


def missing_artifacts() -> list[str]:
    """Human-readable list of absent artifact files (empty when everything is present)."""
    out = []
    for track, (notebook, files) in REQUIRED.items():
        absent = [f for f in files if not (ART / track / f).exists()]
        if absent:
            out.append(f"{track}: {', '.join(absent)}  ->  run {notebook}")
    return out


@st.cache_resource(show_spinner=False)
def bundle(track: str) -> dict:
    return joblib.load(ART / track / "bundle.joblib")


@st.cache_data(show_spinner=False)
def table(track: str, name: str) -> pd.DataFrame:
    return pd.read_csv(ART / track / name)


@st.cache_data(show_spinner=False)
def jsonf(track: str, name: str) -> dict:
    with open(ART / track / name) as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def figures() -> dict[str, np.ndarray]:
    with np.load(ART / "clustering" / "figures.npz") as z:
        return {k: z[k] for k in z.files}
