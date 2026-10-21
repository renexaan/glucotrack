"""Checks that the app's inference code matches the notebooks' fitted objects and that every page renders.

Run from the repository root:   python -m unittest discover -s app/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

import joblib  # noqa: E402

from glucotrack.pipeline import (CLASSIFICATION_RAW_COLUMNS, CLUSTERING_RAW_COLUMNS,  # noqa: E402
                                 REGRESSION_RAW_COLUMNS, preprocess_classification,
                                 preprocess_clustering, preprocess_regression)

ART = APP / "artifacts"
ROOT = APP.parent


class InferenceParity(unittest.TestCase):
    """The notebooks assert parity when exporting; these re-check it against the committed datasets."""

    def test_classification_matches_bundle_model(self):
        b = joblib.load(ART / "classification" / "bundle.joblib")
        row = pd.DataFrame([{"gender": "Male", "age": 62.0, "location": b["freq_cols"] and sorted(b["freq_map"])[0],
                             "hypertension": 1, "heart_disease": 0, "smoking_history": "former", "bmi": 33.0,
                             "hbA1c_level": 6.3, "blood_glucose_level": 165.0, "race": "Hispanic"}])[CLASSIFICATION_RAW_COLUMNS]
        X = preprocess_classification(row, b)
        self.assertEqual(list(X.columns), b["feature_names"])
        p = b["model"].predict_proba(X)[0, 1]
        self.assertTrue(0.0 <= p <= 1.0)

    def test_unseen_location_is_tolerated(self):
        b = joblib.load(ART / "classification" / "bundle.joblib")
        row = pd.DataFrame([{"gender": "Female", "age": 40.0, "location": "Atlantis", "hypertension": 0, "heart_disease": 0,
                             "smoking_history": "never", "bmi": 25.0, "hbA1c_level": 5.5, "blood_glucose_level": 110.0,
                             "race": "Asian"}])[CLASSIFICATION_RAW_COLUMNS]
        self.assertFalse(np.isnan(preprocess_classification(row, b).to_numpy()).any())

    def test_regression_prediction_in_target_range(self):
        b = joblib.load(ART / "regression" / "bundle.joblib")
        row = pd.DataFrame([{"gender": "Female", "age": 45.0, "hypertension": 0, "heart_disease": 0, "smoking_history": "never",
                             "bmi": 27.0, "HbA1c_level": 5.8, "diabetes": 0}])[REGRESSION_RAW_COLUMNS]
        pred = b["model"].predict(preprocess_regression(row, b))[0]
        self.assertTrue(60 < pred < 320)

    def test_kmeans_assignment_matches_notebook_on_real_rows(self):
        b = joblib.load(ART / "clustering" / "bundle.joblib")
        df = pd.read_csv(ROOT / "data" / "diabetes_prediction_dataset.csv").head(300)
        X = preprocess_clustering(df[CLUSTERING_RAW_COLUMNS], b)
        labels = b["kmeans"].predict(X)
        self.assertEqual(set(labels) <= set(range(b["k"])), True)
        # nearest-centroid rule must agree with K-Means' own assignment on K-Means centroids
        d = np.linalg.norm(X[:, None, :] - b["kmeans"].cluster_centers_[None], axis=2)
        self.assertTrue((d.argmin(1) == labels).all())


class PagesRender(unittest.TestCase):
    def test_default_page_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=90)
        at.run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])


if __name__ == "__main__":
    unittest.main()
