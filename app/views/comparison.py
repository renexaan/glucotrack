"""Page 4: Model Comparison (results of the three capstone tracks)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from glucotrack import charts, data, ui

CLS_FMT = {"Accuracy": "{:.4f}", "Precision (weighted)": "{:.4f}", "Recall (weighted)": "{:.4f}", "F1 (weighted)": "{:.4f}",
           "ROC-AUC": "{:.4f}", "Precision (diabetic)": "{:.4f}", "Recall (diabetic)": "{:.4f}", "F1 (diabetic)": "{:.4f}"}


def render() -> None:
    ui.page_header("Model Comparison",
                   "How every algorithm performed, how the final models were chosen, and what the numbers do and do not show.")
    t1, t2, t3 = st.tabs(["Classification", "Regression", "Clustering"])
    with t1:
        _classification()
    with t2:
        _regression()
    with t3:
        _clustering()


# --------------------------------------------------------------------------- classification
def _classification() -> None:
    stats = data.jsonf("classification", "stats.json")
    metrics = data.table("classification", "metrics.csv")
    tuning = data.table("classification", "tuning.csv")
    cv = data.table("classification", "cv_top2.csv")
    score = data.table("classification", "selection_scorecard.csv")
    cms = data.jsonf("classification", "confusion.json")
    roc = data.jsonf("classification", "roc.json")
    imp = data.table("classification", "feature_importance.csv")
    final_name = data.bundle("classification")["model_name"]
    f = metrics[metrics["Model"] == final_name].iloc[0]

    ui.tiles([("Algorithms compared", str(stats["n_algorithms"]), "5 in Part A, 5 in Part B"),
              ("Test records", f"{stats['test_rows']:,}", f"{stats['test_positives']:,} diabetic"),
              ("Majority-class baseline", f"{stats['majority_baseline_accuracy']:.1%}", "accuracy of always predicting non-diabetic"),
              ("Final model", final_name, f"weighted F1 {f['F1 (weighted)']:.4f}, ROC-AUC {f['ROC-AUC']:.4f}")])
    ui.callout("info", "Evaluation method",
               f"All ten models were scored on the same {stats['evaluation_split']} (n = {stats['test_rows']:,}). "
               f"This is a binary problem, so ROC-AUC is the ordinary binary AUC; one-vs-rest averaging would only be needed for "
               f"more than two classes. Cross-validation results are shown separately below and are not mixed into the test table.")

    with ui.card("cmp-cls-table", "Consolidated comparison: all ten classification algorithms",
                 "Ranked by weighted F1 on the test set. Part A = classical models, Part B = ensembles and neural network."):
        cols = ["Rank", "Model", "Part", "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)", "ROC-AUC"]
        ui.show_table(metrics[cols], CLS_FMT, highlight_max=("Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)", "ROC-AUC"))
        with st.expander("Show diabetic-class metrics (positive class only)"):
            ui.show_table(metrics[["Model", "Precision (diabetic)", "Recall (diabetic)", "F1 (diabetic)"]], CLS_FMT,
                          highlight_max=("Precision (diabetic)", "Recall (diabetic)", "F1 (diabetic)"))
            ui.caption("Weighted averages are dominated by the 91.5% majority class. Class-specific metrics show how well "
                       "diabetic patients are actually found.")
        top5 = metrics.head(5)
        best_rec = metrics.loc[metrics["Recall (diabetic)"].idxmax()]
        fp = cms[best_rec["Model"]][0][1]
        cv_std = cv["F1 (weighted) std"].max()
        ui.callout("warn", "Read the ranking with care",
                   f"The top five models span only {top5['F1 (weighted)'].max() - top5['F1 (weighted)'].min():.4f} in weighted F1 "
                   f"(accuracy {top5['Accuracy'].min():.4f} to {top5['Accuracy'].max():.4f}); the cross-validation standard deviation of the two best "
                   f"is up to {cv_std:.4f}. {ui.esc(best_rec['Model'])} ranks #{int(best_rec['Rank'])} on weighted F1 yet has the highest "
                   f"recall on the diabetic class ({best_rec['Recall (diabetic)']:.4f}), at the cost of {fp:,} false alarms.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with ui.card("cmp-cls-bars", "Metric comparison"):
            metric = st.segmented_control("Metric", ["F1 (weighted)", "ROC-AUC", "Accuracy", "Recall (diabetic)", "Precision (diabetic)"],
                                          default="F1 (weighted)", selection_mode="single", key="cmp_cls_metric") or "F1 (weighted)"
            st.plotly_chart(charts.metric_bars(metrics, metric, "Model", final_name), width="stretch", config=charts.CONFIG)
    with c2:
        with ui.card("cmp-cls-cm", "Confusion matrix"):
            chosen = st.selectbox("Model", list(metrics["Model"]), index=list(metrics["Model"]).index(final_name), key="cmp_cls_cm")
            cm = cms[chosen]
            st.plotly_chart(charts.confusion_fig(cm, f"{chosen}: test set"), width="stretch", config=charts.CONFIG)
            ui.caption(f"{cm[1][0]:,} diabetic patients missed (false negatives) and {cm[0][1]:,} false alarms (false positives). "
                       "Each cell also shows its share of the true class.")

    with ui.card("cmp-cls-roc", "ROC curves"):
        st.plotly_chart(charts.roc_fig(roc, metrics, final_name), width="stretch", config=charts.CONFIG)
        ui.caption("Curves come from the predicted scores on the test set (probabilities where the model provides them, "
                   "decision-function scores for the SVM). The selected model is drawn thicker.")

    with ui.card("cmp-cls-select", f"Final model selection: {final_name}"):
        top = score.sort_values("Mean rank").head(5)
        ui.show_table(top.drop(columns=["Rank"]).rename(columns={"Mean rank": "Mean of 3 ranks"}),
                      {"F1 (weighted)": "{:.4f}", "F1 (diabetic)": "{:.4f}", "Recall (diabetic)": "{:.4f}", "Precision (diabetic)": "{:.4f}",
                       "ROC-AUC": "{:.4f}", "Mean of 3 ranks": "{:.2f}"})
        ui.body(f"<b>{ui.esc(final_name)}</b> ranks first on weighted F1, diabetic-class F1 and ROC-AUC, so the choice does not depend on "
                f"which of the three is preferred. The weighted-F1 margin over Random Forest is only "
                f"{score.iloc[0]['F1 (weighted)'] - score.iloc[1]['F1 (weighted)']:.4f}, inside one cross-validation standard deviation. "
                f"The decisive evidence is ranking quality (ROC-AUC) and recall. The Decision Tree's precision of 1.0000 is an artefact of "
                f"rule-assigned labels, not a reason to prefer it (see About).")
        ui.section("5-fold cross-validation of the two best models (training split)")
        cvf = cv.rename(columns={"F1 (weighted) mean": "CV F1 (weighted)", "F1 (weighted) std": "± std", "ROC-AUC mean": "CV ROC-AUC",
                                 "ROC-AUC std": "± std ", "F1 (diabetic) mean": "CV F1 (diabetic)", "F1 (diabetic) std": "± std  "})
        ui.show_table(cvf[["Model", "CV data", "CV F1 (weighted)", "± std", "CV F1 (diabetic)", "± std  ", "CV ROC-AUC", "± std ", "Test F1 (weighted)"]],
                      {c: "{:.4f}" for c in ["CV F1 (weighted)", "± std", "CV F1 (diabetic)", "± std  ", "CV ROC-AUC", "± std ", "Test F1 (weighted)"]})

    with ui.card("cmp-cls-tune", "Hyperparameter tuning (Part B)",
                 "GridSearchCV, 3-fold stratified, scored on diabetic-class F1. Default and tuned models are both scored on the test set."):
        t = tuning[["Model", "Configs searched", "Default F1 (diabetic)", "Tuned F1 (diabetic)", "Gain F1 (diabetic)", "Gain ROC-AUC"]]
        ui.show_table(t, {"Default F1 (diabetic)": "{:.4f}", "Tuned F1 (diabetic)": "{:.4f}", "Gain F1 (diabetic)": "{:+.4f}", "Gain ROC-AUC": "{:+.4f}"})
        top_gain = tuning.loc[tuning["Gain F1 (diabetic)"].idxmax()]
        fin = tuning[tuning["Model"] == final_name]
        extra = ""
        if len(fin):
            g = fin.iloc[0]
            extra = (f" The selected model's test diabetic F1 changed by {g['Gain F1 (diabetic)']:+.4f} while its ROC-AUC changed by "
                     f"{g['Gain ROC-AUC']:+.4f}.")
        ui.callout("neutral", "What tuning achieved",
                   f"Gains are small. {ui.esc(top_gain['Model'])} improved most ({top_gain['Gain F1 (diabetic)']:+.4f} diabetic F1, "
                   f"{top_gain['Gain ROC-AUC']:+.4f} ROC-AUC).{extra} Models this close to each other suggest they sit near the information "
                   f"ceiling of this dataset, so the search space barely matters.")
        with st.expander("Selected hyperparameters"):
            ui.kv_table([(r["Model"], r["Best parameters"]) for _, r in tuning.iterrows()])

    with ui.card("cmp-cls-imp", f"What the {final_name} relies on"):
        st.plotly_chart(charts.importance_fig(imp, f"{final_name}: top feature importances"), width="stretch", config=charts.CONFIG)
        ui.caption("Importance measures how much each feature reduced impurity across the ensemble. It describes the model, not causation.")


# --------------------------------------------------------------------------- regression
def _regression() -> None:
    stats = data.jsonf("regression", "stats.json")
    metrics = data.table("regression", "metrics.csv")
    cv = data.table("regression", "cv_top2.csv")
    tuning = data.table("regression", "tuning.csv")
    poly = data.table("regression", "polynomial_degrees.csv")
    sample = data.table("regression", "pred_vs_actual_sample.csv")
    imp = data.table("regression", "feature_importance_rf.csv")
    coef = data.table("regression", "coefficients.csv")
    best = stats["best_model"]
    bm = stats["best_metrics"]

    ui.tiles([("Algorithms compared", "10", "all trained on the same preprocessed split"),
              ("Test records", f"{stats['test_rows']:,}", f"{stats['train_rows']:,} for training"),
              ("Best test R²", f"{bm['R² Score']:.4f}", best),
              ("Test RMSE / MAE", f"{bm['RMSE']:.1f} / {bm['MAE']:.1f}", "mg/dL")])
    ui.callout("info", "Evaluation method",
               f"Results below come from the {stats['evaluation_split']} (n = {stats['test_rows']:,}). 5-fold cross-validated R² for the "
               f"two best models is computed on the training split and shown separately. Blood glucose takes only "
               f"{len(stats['target_distinct_values'])} distinct values, which places a structural ceiling on R².")

    with ui.card("cmp-reg-table", "Consolidated comparison: all ten regression algorithms", "Ranked by test-set R²."):
        ui.show_table(metrics, {"R² Score": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"},
                      highlight_max=("R² Score",), highlight_min=("RMSE", "MAE"))
        ui.callout("warn", "A tie at the top",
                   "Lasso, ElasticNet, Linear and Ridge regression all reach R² = 0.1825 to four decimals. The ranking among them is "
                   "decided in the fifth decimal and carries no practical meaning. The tuned Decision Tree (0.1823) and Random Forest "
                   "(0.1812) reach the same plateau.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with ui.card("cmp-reg-r2", "R² by algorithm"):
            st.plotly_chart(charts.r2_bars(metrics, best), width="stretch", config=charts.CONFIG)
    with c2:
        with ui.card("cmp-reg-cv", "5-fold cross-validation (two best models)", "Cross-validated R² on the training split, not the test set."):
            ui.show_table(cv, {"Mean CV R²": "{:.4f}", "Std CV R²": "{:.4f}"})
            ui.section("Polynomial regression: degree comparison (test set)")
            ui.show_table(poly, {"R²": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"})
            ui.section("Hyperparameter tuning (GridSearchCV, 5-fold)")
            ui.show_table(tuning.drop(columns=["Best Parameters"]), {"Baseline R²": "{:.4f}", "Tuned R²": "{:.4f}", "Improvement": "{:+.4f}"})
            with st.expander("Selected hyperparameters"):
                ui.kv_table([(r["Model"], r["Best Parameters"]) for _, r in tuning.iterrows()])

    c3, c4 = st.columns(2, gap="large")
    with c3:
        with ui.card("cmp-reg-pva", f"Predicted vs actual: {best}"):
            st.plotly_chart(charts.pred_vs_actual_fig(sample, model=best), width="stretch", config=charts.CONFIG)
    with c4:
        with ui.card("cmp-reg-res", f"Residuals: {best}"):
            st.plotly_chart(charts.residual_fig(sample, model=best), width="stretch", config=charts.CONFIG)
    c5, c6 = st.columns(2, gap="large")
    with c5:
        with ui.card("cmp-reg-imp", "Feature importance: Random Forest (untuned)"):
            st.plotly_chart(charts.importance_fig(imp.rename(columns={"Feature": "feature", "Importance": "importance"}),
                                                  "Random Forest regressor: top 12 importances"), width="stretch", config=charts.CONFIG)
    with c6:
        with ui.card("cmp-reg-coef", f"{best} coefficients"):
            st.plotly_chart(charts.coef_fig(coef), width="stretch", config=charts.CONFIG)
    ab = stats["ablation_r2_linear"]
    ui.callout("alert", "The diabetes label is an input to the regression model",
               f"The notebook's ablation shows linear-model R² falls from {ab['with_diabetes_label']:.4f} to {ab['without_diabetes_label']:.4f} "
               f"when the dataset's diabetes label is removed, and to {ab['without_diabetes_and_hba1c']:.4f} without the label and HbA1c. "
               f"The label was itself derived from the target's variable, so most of the explanatory power comes from it. "
               f"Both the figure and the Glucose Estimate page keep it in, as submitted.")


# --------------------------------------------------------------------------- clustering
def _chaining_note() -> str:
    figs = data.figures()
    shares = {m: np.bincount(figs[f"hier_labels_{m}"]).max() / len(figs[f"hier_labels_{m}"]) for m in ("complete", "average", "single")}
    lo, hi = min(shares.values()), max(shares.values())
    return (f"Complete, Average and Single linkage put {lo:.0%} to {hi:.0%} of the records in one cluster (chaining), "
            "which inflates their silhouette and Davies-Bouldin scores.")


def _clustering() -> None:
    stats = data.jsonf("clustering", "stats.json")
    metrics = data.table("clustering", "metrics.csv")
    posthoc = data.table("clustering", "posthoc_agglomerative.csv")

    ui.tiles([("Algorithms", "2", "K-Means and Agglomerative"), ("Chosen k", str(stats["k"]), "elbow plus three metrics"),
              ("K-Means rows", f"{stats['rows']:,}", "all records"), ("Agglomerative rows", f"{stats['hier_rows']:,}", "reproducible subset")])
    ui.callout("info", "Evaluation method",
               "Clustering is unsupervised, so there is no test set. The three internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) "
               "are computed on the fitted clusters. Silhouette uses a fixed 10,000-row sample for K-Means; the others use every row.")
    with ui.card("cmp-clu-table", "K-Means vs Agglomerative clustering"):
        main = metrics[metrics["Algorithm"].isin(["K-Means (k=4)", "K-Means (k=4, same 2,000-row subset)", "Agglomerative (ward)"])]
        ui.show_table(main, {"Silhouette Score": "{:.4f}", "Davies-Bouldin Index": "{:.4f}", "Calinski-Harabasz Index": "{:,.1f}"})
        km = metrics[metrics["Algorithm"].str.contains("same 2,000")].iloc[0]
        wd = metrics[metrics["Algorithm"] == "Agglomerative (ward)"].iloc[0]
        wins = sum(int(x) for x in (km["Silhouette Score"] > wd["Silhouette Score"],
                                    km["Davies-Bouldin Index"] < wd["Davies-Bouldin Index"],
                                    km["Calinski-Harabasz Index"] > wd["Calinski-Harabasz Index"]))
        ui.callout("warn", "Like-for-like comparison",
                   f"Calinski-Harabasz grows with the number of rows, so compare the 2,000-row K-Means row with Ward. On the same records "
                   f"K-Means is better on {wins} of 3 metrics (silhouette {km['Silhouette Score']:.4f} vs {wd['Silhouette Score']:.4f}; "
                   f"Davies-Bouldin {km['Davies-Bouldin Index']:.4f} vs {wd['Davies-Bouldin Index']:.4f}; "
                   f"Calinski-Harabasz {km['Calinski-Harabasz Index']:.1f} vs {wd['Calinski-Harabasz Index']:.1f}).")
        ui.section("All results, including every linkage")
        ui.show_table(metrics, {"Silhouette Score": "{:.4f}", "Davies-Bouldin Index": "{:.4f}", "Calinski-Harabasz Index": "{:,.1f}"},
                      highlight_max=("Silhouette Score", "Calinski-Harabasz Index"), highlight_min=("Davies-Bouldin Index",))
        ui.caption(_chaining_note())
        ui.section("Post-hoc agreement with the diabetes label (never used for fitting)")
        ui.show_table(posthoc, {"Adjusted Rand Index (post-hoc)": "{:.4f}", "Normalized Mutual Information (post-hoc)": "{:.4f}"})
        ui.caption(f"K-Means: adjusted Rand index {stats['kmeans_posthoc']['ARI']:.4f}, normalised mutual information {stats['kmeans_posthoc']['NMI']:.4f}. "
                   "Low agreement is expected: the clusters describe risk-factor profiles, not diabetes status.")
    pg = ui.PAGES.get("segment")
    if pg is not None:
        st.page_link(pg, label="Open the Elbow curve, dendrogram, PCA and t-SNE figures on the Patient Segment page")
