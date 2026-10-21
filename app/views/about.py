"""Page 5: About (methodology, limitations and how to read the results)."""
from __future__ import annotations

import streamlit as st

from glucotrack import data, ui

SECTIONS = [
    "1. Project overview", "2. Problem statement and objectives", "3. Dataset and provenance",
    "4. Cleaning and preprocessing", "5. Feature engineering", "6. Regression methodology",
    "7. Classification methodology", "8. Clustering methodology", "9. Model comparison and selection",
    "10. Limitations, leakage and ethics", "11. How to interpret the results",
]


def render() -> None:
    cs = data.jsonf("classification", "stats.json")
    rs = data.jsonf("regression", "stats.json")
    ks = data.jsonf("clustering", "stats.json")
    cm = data.table("classification", "metrics.csv")
    final = data.bundle("classification")["model_name"]
    f = cm[cm["Model"] == final].iloc[0]
    rule = cs["label_rule"]
    ab = rs["ablation_r2_linear"]

    ui.page_header("About GlucoTrack",
                   "What was built, how, and where its limits are. Choose a section on the left.")
    nav, content = st.columns([2, 7], gap="large")
    with nav:
        with ui.card("about-nav"):
            ui.section("Contents")
            choice = st.radio("Section", SECTIONS, label_visibility="collapsed", key="about_section")
    with content:
        with ui.card("about-body", choice):
            i = SECTIONS.index(choice)
            [_overview, _problem, _dataset, _clean, _features, _reg, _cls, _clu, _compare, _limits, _interpret][i](
                cs, rs, ks, f, final, rule, ab)


def _overview(cs, rs, ks, f, final, rule, ab):
    ui.body("GlucoTrack is a machine-learning capstone for <b>23CSE301 Machine Learning</b> (academic year 2026-27). It covers three "
            "problem types on diabetes-related records: <b>regression</b>, <b>classification</b> and <b>clustering</b>.")
    ui.body("This application is a window onto the notebooks. It loads the exact fitted models, preprocessing objects and result tables "
            "that the notebooks export, and nothing is retrained in the browser. An assertion at the end of each notebook checks that the "
            "app's inference code reproduces the notebook's own processed data and predictions.")
    ui.tiles([("Regression", "10 algorithms", "blood glucose"), ("Classification", "10 algorithms", "diabetes label"),
              ("Clustering", "2 algorithms", "patient segments")])
    ui.callout("neutral", "Educational use only",
               "GlucoTrack is a research and teaching artefact. It is not a medical device and its outputs must not be used to make health decisions.")


def _problem(cs, rs, ks, f, final, rule, ab):
    ui.body("<b>Classification.</b> Predict whether a record carries the diabetes label from routinely collected clinical and demographic "
            "measurements. <b>Regression.</b> Estimate blood glucose from demographic and clinical inputs. <b>Clustering.</b> Group patients "
            "by risk-factor profile without using the label, then interpret the groups afterwards.")
    ui.body("<b>Objectives.</b>")
    st.markdown("""
- Build complete, leakage-aware pipelines: audit, exploratory analysis, cleaning, feature engineering, modelling, tuning and evaluation.
- Compare ten regression algorithms, ten classification algorithms and two clustering algorithms on consistent splits.
- Report results honestly, including where the data limits what any model can achieve.
- Make the fitted models explorable through this application.""")


def _dataset(cs, rs, ks, f, final, rule, ab):
    ui.body(f"<b>Classification dataset.</b> {ui.esc(cs['dataset'])}: {cs['rows_raw']:,} records and {cs['columns_raw']} columns "
            f"(demographics, five race indicators, location, hypertension, heart disease, smoking history, BMI, HbA1c, blood glucose and the "
            f"diabetes label). The label is imbalanced: {cs['positive_rate']:.1%} diabetic, a ratio of {cs['imbalance_ratio']:.2f} to 1.")
    ui.body(f"<b>Regression and clustering dataset.</b> The Diabetes Prediction Dataset supplied for the project "
            f"(<code>data/diabetes_prediction_dataset.csv</code>): 100,000 records and 9 columns (gender, age, hypertension, heart disease, "
            f"smoking history, BMI, HbA1c, blood glucose, diabetes). The repository does not record an external source link for this file.")
    ui.callout("warn", "Units and synthetic structure",
               f"The sources do not state units. This app labels HbA1c in % and glucose in mg/dL, inferred from the value ranges (3.5 to 9.0 and "
               f"80 to 300). The data also shows signs of being synthetic: in the classification file each glycaemic marker takes only 18 distinct "
               f"values and 25,495 records share a BMI of exactly 27.32.")


def _clean(cs, rs, ks, f, final, rule, ab):
    st.markdown(f"""
**Classification**
- The five race indicator columns were verified to be mutually exclusive and collapsed into one `race` column.
- {cs['duplicates_removed']} exact duplicate rows were removed; no values were missing. The administrative `year` column was dropped.
- IQR outliers in BMI, HbA1c and glucose were **winsorised, not deleted**, because 100% of the flagged HbA1c and glucose outliers are diabetic. Deleting them would have removed about 39% of the positive class.
- The data was split 80/20 with stratification (`random_state=42`). Imputers, the winsorising fences, the `location` frequency map, the one-hot encoder and the scaler were all **fitted on the training split only**, and a notebook assertion proves the scaler never saw test data.

**Regression**
- {3854:,} duplicate rows were removed, leaving {rs['rows_after_dedup']:,}. No outliers were clipped or deleted.
- An 80/20 random split, then a `StandardScaler` for numeric columns and one-hot encoding for categorical columns, fitted on the training split only.

**Clustering**
- All {ks['rows']:,} rows are kept, including {ks['duplicates_kept']:,} duplicates, because the population being segmented should not change. Median/most-frequent imputation, standardisation and one-hot encoding are fitted on the full feature matrix (clustering has no train/test split) and the diabetes label is excluded.""")


def _features(cs, rs, ks, f, final, rule, ab):
    st.markdown("""
**Classification: three engineered features**

| Feature | Rationale |
|---|---|
| `bmi_category` | Risk steps up at obesity rather than rising linearly; WHO bands let a linear model fit that. |
| `age_risk_interaction` | Age multiplied by HbA1c; a linear model cannot represent a product of two inputs unless it is supplied. |
| `metabolic_risk_score` | Counts five risk flags (HbA1c ≥ 5.7, glucose ≥ 140, BMI ≥ 30, hypertension, heart disease) using fixed thresholds, so it is leakage-free. |

**Regression: `age_bmi_interaction`** (age × BMI) lets linear models represent the combined effect of the two inputs.

**Clustering:** no engineered features; the eight original features are used.""")
    ui.caption("The notebook validates each classification feature against the target: diabetes prevalence rises across BMI bands and across risk-score values.")


def _reg(cs, rs, ks, f, final, rule, ab):
    bm = rs["best_metrics"]
    st.markdown(f"""
- Ten algorithms on the same preprocessed split: Linear, Ridge, Lasso, ElasticNet, Polynomial (degrees 2 and 3 compared), Decision Tree, Random Forest, Gradient Boosting, SVR and KNN.
- Metrics: R², RMSE and MAE on the hold-out test set; 5-fold cross-validated R² for the two best models; `GridSearchCV` on the Decision Tree and Random Forest.
- **Result:** {rs['best_model']} has the highest test R² ({bm['R² Score']:.4f}, RMSE {bm['RMSE']:.2f}, MAE {bm['MAE']:.2f} mg/dL), but several linear models tie to four decimals. Tuning lifted the Decision Tree from a negative R² to 0.1823.
- **Ceiling:** the target takes only {len(rs['target_distinct_values'])} distinct values and the inputs explain little of it. R² of about 0.18 across nine different algorithms is evidence of a data limit, not a modelling failure.
- **Ablation:** removing the diabetes label cuts linear-model R² from {ab['with_diabetes_label']:.3f} to {ab['without_diabetes_label']:.3f}.""")


def _cls(cs, rs, ks, f, final, rule, ab):
    st.markdown(f"""
- **Part A:** Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC. **Part B:** Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), MLP.
- Every model was scored on the same {cs['test_rows']:,}-record stratified test set, with accuracy, weighted precision / recall / F1, ROC-AUC and a confusion matrix. Class-specific metrics are shown alongside because weighted averages are dominated by the majority class.
- Grid searches use 3-fold stratified cross-validation and score on **diabetic-class F1**, not accuracy: a model that never predicts diabetes already scores {cs['majority_baseline_accuracy']:.1%} accuracy.
- Part B models were fitted with library defaults first and again after tuning, so the effect of tuning is measurable.
- KNN and SVC were tuned and refitted on stratified subsamples for tractability; their scores are conservative lower bounds.
- The two best models were cross-validated with 5 folds on the training split.""")


def _clu(cs, rs, ks, f, final, rule, ab):
    st.markdown(f"""
- **K-Means** was run for k = 2 to 10. The elbow curve has no sharp bend, so k was chosen with silhouette, Davies-Bouldin and Calinski-Harabasz together: k = {ks['k']}.
- **Agglomerative clustering** was run with Ward, Complete, Average and Single linkage on a reproducible {ks['hier_rows']:,}-row subset (the dendrogram uses {ks['dendrogram_rows']} rows) because linkage is quadratic in memory.
- Visualisations: elbow curve, dendrogram, PCA for both algorithms (two components explain {sum(ks['pca_explained_variance']):.1%} of variance for K-Means) and t-SNE for K-Means.
- The **diabetes label is never used for fitting**. It appears only in post-hoc interpretation, where agreement is low (K-Means adjusted Rand index {ks['kmeans_posthoc']['ARI']:.3f}).
- **Assigning new patients:** K-Means has a native `predict`. Agglomerative clustering does not, so the app assigns new points to the nearest Ward cluster centroid. This is a documented extension, not a feature of the estimator.""")


def _compare(cs, rs, ks, f, final, rule, ab):
    ui.body(f"<b>Classification: {ui.esc(final)}</b> was selected. It ranks first on weighted F1 ({f['F1 (weighted)']:.4f}), diabetic-class F1 "
            f"({f['F1 (diabetic)']:.4f}) and ROC-AUC ({f['ROC-AUC']:.4f}). The five leading models are within about 0.001 of each other in "
            f"weighted F1, so the choice rests on ROC-AUC and recall rather than a large gap.")
    ui.body(f"<b>Regression: {ui.esc(rs['best_model'])}</b> has the highest test R², but it is effectively tied with Ridge, Linear and ElasticNet regression.")
    ui.body("<b>Clustering:</b> K-Means (k=4) is preferred because on the same 2,000 records it beats Ward on all three metrics, and it scales to all rows.")
    pg = ui.PAGES.get("compare")
    if pg is not None:
        st.page_link(pg, label="See the full tables and charts on Model Comparison")


def _limits(cs, rs, ks, f, final, rule, ab):
    st.markdown(f"""
**Target leakage in the classifier.** The diabetes label looks rule-assigned: the highest glucose among non-diabetic records is exactly {rule['max_glucose_nondiabetic']:.0f}
and the highest HbA1c is {rule['max_hba1c_nondiabetic']:.1f}. All {rule['n_glucose_above_200']:,} records above glucose 200 and all {rule['n_hba1c_above_6_6']:,} above HbA1c 6.6 are diabetic.
Glucose and HbA1c are also model inputs, so high test scores partly measure how well a model recovers that rule. This explains the Decision Tree's precision of exactly 1.0000, and it
means the reported scores will not transfer to real patients. The {ui.esc(final)} model is **not clinically validated**.

**Leakage in the regression inputs.** The diabetes label is an input feature, although it was derived from the same measurements as the target.

**Other limitations**
- The data is substantially synthetic, so metrics describe the dataset, not clinical performance.
- Probabilities from tree ensembles are not guaranteed to be calibrated. No decision-threshold tuning or class-imbalance correction was applied.
- The regression target has {len(rs['target_distinct_values'])} distinct values, so R² is capped regardless of algorithm. No prediction intervals are available.
- Clustering structure is weak (silhouette below 0.25 for balanced solutions). Hierarchical results come from a subset.
- Gender has an `Other` category with very few records, and the data contains no information on real-world demographics, so nothing here should be read as a statement about any group.

**Ethical considerations.** Do not use this application to screen, triage or advise real people. Predictions are not diagnoses and no treatment guidance is given or implied.""")


def _interpret(cs, rs, ks, f, final, rule, ab):
    st.markdown("""
- **Probability (Diabetes Risk)** is the model's score for the diabetic class, compared with a 50% decision threshold. It is not a clinical risk percentage.
- **Predicted class and probability are different things.** A record can be assigned to the diabetic class with a probability of 55%.
- **Glucose estimate** is a regression output. R², RMSE and MAE describe performance across the test set, not the error of one estimate.
- **Cluster numbers** are arbitrary labels. A cluster is described only by how its members differ from the dataset average.
- **Inputs outside the training range** are limited by the form. Inputs beyond a labelling threshold (glucose above 200, HbA1c above 6.6) are flagged because the classifier mostly follows the dataset's rule there.
- **Test vs cross-validation:** test-set and cross-validation numbers are reported separately and are not interchangeable.
- **Reproducing everything:** run the three notebooks in `notebooks/` top to bottom; the last section of each rewrites the app's artifacts.""")
