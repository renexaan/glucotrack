"""Page 3: Patient Segment (clustering track)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from glucotrack import charts, data, ui
from glucotrack.pipeline import CLUSTERING_RAW_COLUMNS, preprocess_clustering

P = "seg_"
SMOKING_ORDER = ["never", "former", "current", "not current", "ever", "No Info"]
KM, WARD = "K-Means (k=4)", "Agglomerative (Ward)"
FEATURES = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
NICE = {"age": "Age", "hypertension": "Hypertension", "heart_disease": "Heart disease", "bmi": "BMI",
        "HbA1c_level": "HbA1c", "blood_glucose_level": "Blood glucose"}
UNIT = {"age": " yrs", "bmi": " kg/m²", "HbA1c_level": " %", "blood_glucose_level": " mg/dL"}
EXAMPLES = {
    "Example A: younger, no comorbidities": dict(age=24.0, gender="Female", smoking="never", hypertension=False,
                                                  heart=False, bmi=22.5, hba1c=5.1, glucose=105),
    "Example B: older, with heart disease": dict(age=70.0, gender="Male", smoking="former", hypertension=True,
                                                 heart=True, bmi=29.0, hba1c=5.9, glucose=150),
}


def _defaults(stats: dict) -> dict:
    r = stats["ranges"]
    return {"age": float(round(r["age"]["median"])), "gender": "Female", "smoking": "never", "hypertension": False,
            "heart": False, "bmi": round(r["bmi"]["median"], 1), "hba1c": round(r["HbA1c_level"]["median"], 1),
            "glucose": int(r["blood_glucose_level"]["median"])}


def _init(stats: dict) -> None:
    for k, v in _defaults(stats).items():
        st.session_state.setdefault(P + k, v)
    st.session_state.setdefault(P + "method", KM)


def _load_example(name: str, stats: dict) -> None:
    ex = _defaults(stats); ex.update(EXAMPLES[name])
    for k, v in ex.items():
        st.session_state[P + k] = v
    st.session_state.pop(P + "result", None)


def _reset(stats: dict) -> None:
    for k, v in _defaults(stats).items():
        st.session_state[P + k] = v
    st.session_state.pop(P + "result", None)
    st.session_state[P + "example"] = "Choose…"


def _inputs() -> dict:
    s = st.session_state
    return {"gender": s[P + "gender"], "age": float(s[P + "age"]), "hypertension": int(s[P + "hypertension"]),
            "heart_disease": int(s[P + "heart"]), "smoking_history": s[P + "smoking"], "bmi": float(s[P + "bmi"]),
            "HbA1c_level": float(s[P + "hba1c"]), "blood_glucose_level": float(s[P + "glucose"])}


def _assign() -> None:
    inputs, method = _inputs(), st.session_state.get(P + "method") or KM
    try:
        b = data.bundle("clustering")
        raw = pd.DataFrame([inputs])[CLUSTERING_RAW_COLUMNS]
        x = preprocess_clustering(raw, b)
        if method == KM:
            cluster = int(b["kmeans"].predict(x)[0])
            xy = b["pca"].transform(x)[0]
        else:
            dist = np.linalg.norm(b["ward_centroids"] - x, axis=1)
            cluster = int(np.argmin(dist))
            xy = b["hier_pca"].transform(x)[0]
        st.session_state[P + "result"] = {"inputs": inputs, "method": method, "cluster": cluster, "xy": xy}
        st.session_state.pop(P + "error", None)
    except Exception as exc:
        st.session_state[P + "error"] = f"{type(exc).__name__}: {exc}"
        st.session_state.pop(P + "result", None)


def _fmt(feature: str, value: float) -> str:
    if feature in ("hypertension", "heart_disease"):
        return f"{value:.1%}"
    return f"{value:.1f}{UNIT.get(feature, '')}"


def describe(profile: dict, cluster: int, top: int = 3) -> str:
    """Evidence-based summary: the features whose cluster mean is furthest from the dataset mean, in SD units."""
    c = profile["clusters"][str(cluster)]
    z = {f: (c["mean"][f] - profile["overall_mean"][f]) / profile["overall_std"][f] for f in FEATURES}
    parts = []
    for f in sorted(z, key=lambda k: abs(z[k]), reverse=True)[:top]:
        direction = "higher" if z[f] > 0 else "lower"
        parts.append(f"<b>{NICE[f].lower()}</b> is {direction} than average "
                     f"({_fmt(f, c['mean'][f])} vs {_fmt(f, profile['overall_mean'][f])} overall)")
    return (f"Cluster {cluster} holds {c['size']:,} records ({c['share']:.1%} of the {'dataset' if profile['n'] > 5000 else 'sample'}). "
            "Compared with the whole dataset, " + "; ".join(parts) + ".")


def render() -> None:
    stats = data.jsonf("clustering", "stats.json")
    profiles = data.jsonf("clustering", "profiles.json")
    r = stats["ranges"]
    _init(stats)

    ui.page_header("Patient Segment",
                   "Find which cluster a patient profile falls into, then explore the clustering experiments behind it. "
                   "This page demonstrates the unsupervised-learning track.")

    left, right = st.columns([5, 6], gap="large")
    with left:
        with ui.card("seg-form", "Patient profile", "Clustering uses eight features and never sees the diabetes label."):
            st.segmented_control("Clustering method", [KM, WARD], key=P + "method", selection_mode="single",
                                 help="K-Means has a native predict step. Agglomerative clustering does not, so new points are "
                                      "assigned to the nearest Ward cluster centroid (an extension described below).")
            ui.section("Demographics")
            c1, c2 = st.columns(2)
            c1.number_input("Age (years)", min_value=float(r["age"]["min"]), max_value=float(r["age"]["max"]), step=1.0, key=P + "age")
            c2.selectbox("Gender", stats["categories"]["gender"], key=P + "gender")
            ui.section("Clinical measurements")
            c3, c4 = st.columns(2)
            c3.number_input("BMI (kg/m²)", min_value=float(r["bmi"]["min"]), max_value=float(r["bmi"]["max"]), step=0.1, key=P + "bmi")
            c4.selectbox("Smoking history", [s for s in SMOKING_ORDER if s in stats["categories"]["smoking_history"]],
                         key=P + "smoking", format_func=ui.label)
            st.slider("HbA1c (%)", min_value=float(r["HbA1c_level"]["min"]), max_value=float(r["HbA1c_level"]["max"]), step=0.1, key=P + "hba1c")
            st.slider("Blood glucose (mg/dL)", min_value=int(r["blood_glucose_level"]["min"]),
                      max_value=int(r["blood_glucose_level"]["max"]), step=1, key=P + "glucose")
            ui.section("Comorbidities")
            c5, c6 = st.columns(2)
            c5.toggle("Hypertension", key=P + "hypertension")
            c6.toggle("Heart disease", key=P + "heart")
            st.button("Find Segment", type="primary", width="stretch", on_click=_assign, key=P + "go")
            e1, e2 = st.columns(2)
            e1.button("Reset", width="stretch", on_click=_reset, args=(stats,), key=P + "reset")
            e2.selectbox("Load an illustrative example", ["Choose…"] + list(EXAMPLES), key=P + "example",
                         label_visibility="collapsed",
                         on_change=lambda: _load_example(st.session_state[P + "example"], stats)
                         if st.session_state[P + "example"] in EXAMPLES else None)
            ui.caption("Examples are illustrative value sets, not real patients.")

    with right:
        res, err = st.session_state.get(P + "result"), st.session_state.get(P + "error")
        if err:
            with ui.card("seg-error"):
                ui.callout("alert", "The assignment could not be computed", f"{ui.esc(err)}.")
        elif res is None:
            with ui.card("seg-empty", "Assigned segment"):
                ui.empty_state("No segment yet", "Fill in the profile and press Find Segment, or load an illustrative example.")
                ui.callout("info", "How segments are named",
                           "Clusters are numbered 0 to 3 and described only by how their members differ from the dataset average. "
                           "No clinical meaning is attached to them.")
        else:
            _result(res, stats, profiles)

    _experiments(stats, profiles)


def _result(res: dict, stats: dict, profiles: dict) -> None:
    method, cluster, inputs = res["method"], res["cluster"], res["inputs"]
    stale = inputs != _inputs() or method != st.session_state.get(P + "method")
    prof = profiles["kmeans" if method == KM else "ward"]
    figs = data.figures()
    colour = charts.cluster_color(cluster)

    with ui.card("seg-result", "Assigned segment"):
        st.markdown(ui.badge(method, "teal") + " " + ui.badge(f"{stats['rows']:,} records, k = {stats['k']}"),
                    unsafe_allow_html=True)
        if stale:
            ui.callout("warn", "Inputs changed", "The values on the left differ from the ones analysed below. Press Find Segment to update.")
        ui.big_result("Assigned cluster", f"Cluster {cluster}")
        ui.body(describe(prof, cluster))
        if method == WARD:
            ui.callout("info", "How this assignment works",
                       f"scikit-learn's AgglomerativeClustering has no predict method. The patient is assigned to the nearest "
                       f"Ward cluster centroid, computed from the {stats['hier_rows']:,}-row subset used in the notebook. "
                       f"This is an extension for the demo, not a native estimator feature.")

        # profile versus patient, in standard-deviation units
        c = prof["clusters"][str(cluster)]
        z_cluster = [(c["mean"][f] - prof["overall_mean"][f]) / prof["overall_std"][f] for f in FEATURES]
        z_patient = [(inputs[f] - prof["overall_mean"][f]) / prof["overall_std"][f] for f in FEATURES]
        st.plotly_chart(charts.profile_fig([NICE[f] for f in FEATURES], z_cluster, z_patient, cluster),
                        width="stretch", config=charts.CONFIG)

        rows = []
        for f in FEATURES:
            binary = f in ("hypertension", "heart_disease")
            rows.append({"Feature": NICE[f],
                         "This patient": ("Yes" if inputs[f] else "No") if binary else f"{inputs[f]:.1f}{UNIT.get(f, '')}",
                         f"Cluster {cluster} mean": _fmt(f, c["mean"][f]), "Dataset mean": _fmt(f, prof["overall_mean"][f])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ui.caption("For Hypertension and Heart disease the cluster and dataset columns show the share of records with the condition.")

        if method == KM:
            xy, labels, evr = figs["pca_kmeans_xy"], figs["pca_kmeans_labels"], stats["pca_explained_variance"]
            title = "Where this patient sits among 5,000 sampled records (PCA, K-Means colouring)"
        else:
            xy, labels, evr = figs["hier_pca_xy"], figs["hier_labels_ward"], None
            title = "Where this patient sits among the 2,000-row subset (PCA, Ward colouring)"
        st.plotly_chart(charts.pca_fig(xy, labels, title, evr=evr, point=res["xy"]), width="stretch", config=charts.CONFIG)
        ui.caption("PCA keeps only part of the variance, so overlap in two dimensions does not imply overlap in the full feature space.")

        sh = c["diabetic_share"]
        ui.callout("neutral", "Post-hoc check, not used for fitting",
                   f"{sh:.1%} of records in this cluster carry the diabetes label (dataset overall {prof['overall_diabetic_share']:.1%}). "
                   f"The label was never used to build the clusters, and the clusters describe risk-factor profiles, not diabetes status "
                   f"(K-Means agreement with the label: adjusted Rand index {stats['kmeans_posthoc']['ARI']:.3f}).")
        out = pd.DataFrame([{**inputs, "method": method, "cluster": cluster}])
        ui.csv_download("Download result (CSV)", out, "glucotrack_patient_segment.csv", P + "dl")


def _experiments(stats: dict, profiles: dict) -> None:
    figs = data.figures()
    kt = data.table("clustering", "k_selection.csv")
    metrics = data.table("clustering", "metrics.csv")
    k = stats["k"]

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    with ui.card("seg-exp", "Clustering experiments", "The results behind the segments above. Figures are drawn from arrays exported by notebooks/clustering.ipynb."):
        tabs = st.tabs(["Elbow and k choice", "Dendrogram", "PCA views", "t-SNE", "Metrics", "Cluster profiles"])

        with tabs[0]:
            st.plotly_chart(charts.elbow_fig(kt, k), width="stretch", config=charts.CONFIG)
            st.plotly_chart(charts.kmetric_fig(kt, k), width="stretch", config=charts.CONFIG)
            ge3 = kt[kt["k"] >= 3]
            k_sil = int(ge3.loc[ge3["Silhouette"].idxmax(), "k"]); k_ch = int(ge3.loc[ge3["Calinski-Harabasz"].idxmax(), "k"])
            k2 = kt[kt["k"] == 2].iloc[0]
            ui.callout("info", f"Why k = {k}",
                       f"The elbow has no sharp bend, so k was chosen with the three metrics. k = 2 has the highest silhouette overall "
                       f"({k2['Silhouette']:.4f}) but its smallest cluster holds only {int(k2['Smallest cluster']):,} records, a coarse split. "
                       f"For k of 3 or more, k = {k_sil} has the best silhouette and k = {k_ch} the highest Calinski-Harabasz score. "
                       f"Values come from all {stats['rows']:,} records.")
            ui.show_table(kt.round(4), {"Inertia": "{:,.0f}", "Calinski-Harabasz": "{:,.0f}", "Smallest cluster": "{:,}"})

        with tabs[1]:
            st.plotly_chart(charts.dendrogram_fig(figs["dendrogram_Z"]), width="stretch", config=charts.CONFIG)
            ui.caption(f"Computed on a reproducible {stats['dendrogram_rows']}-row sample because linkage needs all pairwise distances. "
                       "The plot is truncated to the last 30 merges; leaves show how many samples each merged group holds.")

        with tabs[2]:
            ev = stats["pca_explained_variance"]
            st.plotly_chart(charts.pca_fig(figs["pca_kmeans_xy"], figs["pca_kmeans_labels"],
                                           f"PCA 2D, K-Means (k={k}), 5,000-row sample", evr=ev), width="stretch", config=charts.CONFIG)
            link = st.segmented_control("Agglomerative linkage", ["ward", "complete", "average", "single"], default="ward",
                                        key=P + "pca_link", selection_mode="single") or "ward"
            st.plotly_chart(charts.pca_fig(figs["hier_pca_xy"], figs[f"hier_labels_{link}"],
                                           f"PCA 2D, Agglomerative ({link} linkage, k={k}), 2,000-row subset"), width="stretch", config=charts.CONFIG)
            ui.caption("Each panel is coloured by the labels of the model that produced it. The two PCA projections are fitted separately.")

        with tabs[3]:
            st.plotly_chart(charts.tsne_fig(figs["tsne_xy"], figs["tsne_labels"]), width="stretch", config=charts.CONFIG)
            ui.caption("t-SNE is shown for K-Means only (the assessment requires it for at least one algorithm). Distances between islands "
                       "and cluster sizes are not meaningful in t-SNE; use it for local neighbourhood structure.")

        with tabs[4]:
            ui.section("Required comparison: K-Means and Agglomerative")
            focus = metrics[metrics["Algorithm"].isin(["K-Means (k=4)", "K-Means (k=4, same 2,000-row subset)", "Agglomerative (ward)"])]
            ui.show_table(focus.rename(columns={"Algorithm": "Algorithm"}),
                          {"Silhouette Score": "{:.4f}", "Davies-Bouldin Index": "{:.4f}", "Calinski-Harabasz Index": "{:,.1f}"})
            ui.section("All linkages")
            ui.show_table(metrics, {"Silhouette Score": "{:.4f}", "Davies-Bouldin Index": "{:.4f}", "Calinski-Harabasz Index": "{:,.1f}"},
                          highlight_max=("Silhouette Score", "Calinski-Harabasz Index"), highlight_min=("Davies-Bouldin Index",))
            shares = [np.bincount(figs[f"hier_labels_{m}"]).max() / len(figs[f"hier_labels_{m}"]) for m in ("complete", "average", "single")]
            ui.callout("warn", "Read the table carefully",
                       f"Calinski-Harabasz grows with the number of rows, so the full-data K-Means row cannot be compared with 2,000-row rows. "
                       f"Compare K-Means on the same subset with Ward instead. Complete, Average and Single linkage put {min(shares):.0%} to "
                       f"{max(shares):.0%} of records in one cluster (chaining), which inflates their silhouette and Davies-Bouldin scores.")

        with tabs[5]:
            which = st.segmented_control("Clustering", [KM, WARD], default=KM, key=P + "prof_method", selection_mode="single") or KM
            prof = profiles["kmeans" if which == KM else "ward"]
            rows = []
            for cid, c in prof["clusters"].items():
                rows.append({"Cluster": int(cid), "Records": c["size"], "Share": c["share"],
                             **{NICE[f]: c["mean"][f] for f in FEATURES}, "Diabetes label share (post-hoc)": c["diabetic_share"]})
            overall = {"Cluster": "All", "Records": prof["n"], "Share": 1.0, **{NICE[f]: prof["overall_mean"][f] for f in FEATURES},
                       "Diabetes label share (post-hoc)": prof["overall_diabetic_share"]}
            frame = pd.concat([pd.DataFrame(rows), pd.DataFrame([overall])], ignore_index=True)
            frame["Cluster"] = frame["Cluster"].astype(str)
            ui.show_table(frame, {"Share": "{:.1%}", "Age": "{:.1f}", "Hypertension": "{:.1%}", "Heart disease": "{:.1%}", "BMI": "{:.1f}",
                                  "HbA1c": "{:.2f}", "Blood glucose": "{:.1f}", "Diabetes label share (post-hoc)": "{:.1%}"})
            ui.caption("Means per cluster computed from the data; hypertension and heart disease are shares. "
                       + ("" if which == KM else "Ward profiles come from the 2,000-row subset, so they are noisier. ")
                       + "The diabetes label is shown only to interpret clusters after fitting.")
