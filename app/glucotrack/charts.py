"""Plotly figure builders. Every figure has a title, labelled axes and a legend where several series appear."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from . import theme

CONFIG = {"displayModeBar": False, "responsive": True}


def _base(fig: go.Figure, title: str, height: int = 380, legend: bool = True, x: str | None = None,
          y: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=theme.INK)),
        height=height, margin=dict(l=56, r=16, t=52, b=84),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family=theme.FONT_STACK, size=12, color=theme.INK_2),
        showlegend=legend, hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.26, x=0, font=dict(size=11)),
        colorway=theme.CATEGORICAL,
    )
    fig.update_xaxes(title_text=x, showgrid=True, gridcolor="#EEEBE3", zeroline=False, linecolor=theme.BORDER_STRONG,
                     ticks="outside", tickcolor=theme.BORDER_STRONG, title_font=dict(size=12))
    fig.update_yaxes(title_text=y, showgrid=True, gridcolor="#EEEBE3", zeroline=False, linecolor=theme.BORDER_STRONG,
                     ticks="outside", tickcolor=theme.BORDER_STRONG, title_font=dict(size=12))
    return fig


# ----------------------------------------------------------------- classification
def metric_bars(df: pd.DataFrame, metric: str, name_col: str, highlight: str, height: int = 400) -> go.Figure:
    d = df.sort_values(metric, ascending=True)
    colours = [theme.TEAL if n == highlight else theme.SLATE for n in d[name_col]]
    fig = go.Figure(go.Bar(x=d[metric], y=d[name_col], orientation="h", marker_color=colours,
                           text=[f"{v:.4f}" for v in d[metric]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>" + metric + ": %{x:.4f}<extra></extra>",
                           name=metric, showlegend=False))
    lo = min(0.0, d[metric].min() * 1.1)
    fig.update_xaxes(range=[lo, max(1.0, d[metric].max()) * 1.12])
    return _base(fig, f"{metric}: all algorithms (highlighted = selected model)", height, legend=False,
                 x=metric, y="Algorithm")


def confusion_fig(cm: list[list[int]], title: str) -> go.Figure:
    """Row-normalised heatmap; counts and the share of each true class are printed in every cell."""
    cm = np.array(cm)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    text = [[f"{cm[i, j]:,}<br>({pct[i, j]:.1f}% of row)" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=pct, x=["Predicted: non-diabetic", "Predicted: diabetic"],
        y=["Actual: non-diabetic", "Actual: diabetic"], text=text, texttemplate="%{text}",
        textfont=dict(size=14), colorscale=[[0, "#FFFFFF"], [1, theme.TEAL]], showscale=False, zmin=0, zmax=100,
        hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>", name="Share of true class"))
    fig = _base(fig, title, 340, legend=False, x="Predicted label", y="True label")
    fig.update_yaxes(autorange="reversed")
    return fig


def roc_fig(roc: dict, metrics: pd.DataFrame, highlight: str) -> go.Figure:
    fig = go.Figure()
    auc = dict(zip(metrics["Model"], metrics["ROC-AUC"]))
    order = [m for m in metrics["Model"] if m != highlight] + [highlight]
    palette = {m: theme.CATEGORICAL[i % len(theme.CATEGORICAL)] for i, m in enumerate(metrics["Model"])}
    dash = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    for i, m in enumerate(order):
        is_final = m == highlight
        fig.add_trace(go.Scatter(
            x=roc[m]["fpr"], y=roc[m]["tpr"], mode="lines", name=f"{m} (AUC {auc[m]:.4f})",
            line=dict(color=theme.TEAL if is_final else palette[m], width=3.4 if is_final else 1.6,
                      dash="solid" if is_final else dash[i % len(dash)]),
            opacity=1 if is_final else 0.85,
            hovertemplate=f"{m}<br>FPR %{{x:.3f}}, TPR %{{y:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier (AUC 0.5000)",
                             line=dict(color=theme.INK, width=1, dash="dot"), hoverinfo="skip"))
    fig = _base(fig, "ROC curves, test set, all ten algorithms", 460, x="False positive rate",
                y="True positive rate (recall on diabetic class)")
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, yanchor="top", font=dict(size=11)),
                      margin=dict(l=56, r=210, t=52, b=52))
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.02])
    return fig


def importance_fig(df: pd.DataFrame, title: str, name_col: str = "feature", val_col: str = "importance",
                   n: int = 12) -> go.Figure:
    d = df.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=d[val_col], y=d[name_col], orientation="h", marker_color=theme.TEAL,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>", showlegend=False, name="Importance"))
    return _base(fig, title, 380, legend=False, x="Importance (mean decrease in impurity)", y="Feature")


# ----------------------------------------------------------------- regression
def pred_vs_actual_fig(sample: pd.DataFrame, marker: float | None = None, model: str = "") -> go.Figure:
    jitter = np.random.default_rng(42).normal(0, 1.2, len(sample))
    fig = go.Figure(go.Scattergl(
        x=sample["actual"] + jitter, y=sample["predicted"], mode="markers", name="Test-set patients (sample)",
        marker=dict(size=5, color=theme.CATEGORICAL[0], opacity=0.28), hovertemplate="actual %{x:.0f}, predicted %{y:.1f}<extra></extra>"))
    lo, hi = float(sample["actual"].min()) - 5, float(sample["actual"].max()) + 5
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (y = x)",
                             line=dict(color=theme.RED, dash="dash", width=1.6)))
    if marker is not None:
        fig.add_hline(y=marker, line=dict(color=theme.TEAL, width=2.2),
                      annotation=dict(text=f"This prediction: {marker:.1f}", font=dict(color=theme.TEAL_DARK, size=12),
                                      xanchor="left", x=0.01, yanchor="bottom"))
    return _base(fig, f"Predicted vs actual blood glucose{f' ({model})' if model else ''}, 6,000-row test sample", 400,
                 x="Actual blood glucose (mg/dL, jittered for visibility)", y="Predicted blood glucose (mg/dL)")


def residual_fig(sample: pd.DataFrame, model: str = "") -> go.Figure:
    resid = sample["actual"] - sample["predicted"]
    fig = go.Figure(go.Scattergl(x=sample["predicted"], y=resid, mode="markers", name="Test-set patients (sample)",
                                 marker=dict(size=5, color=theme.CATEGORICAL[0], opacity=0.28),
                                 hovertemplate="predicted %{x:.1f}, residual %{y:.1f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=theme.RED, dash="dash", width=1.6), annotation_text="zero error",
                  annotation_position="top left")
    return _base(fig, f"Residuals{f' ({model})' if model else ''}", 400, x="Predicted blood glucose (mg/dL)",
                 y="Residual = actual - predicted (mg/dL)")


def coef_fig(df: pd.DataFrame) -> go.Figure:
    d = df.reindex(df["coefficient"].abs().sort_values().index)
    d = d[d["coefficient"] != 0].tail(12)
    colours = [theme.CATEGORICAL[0] if v > 0 else theme.CATEGORICAL[3] for v in d["coefficient"]]
    fig = go.Figure(go.Bar(x=d["coefficient"], y=d["feature"], orientation="h", marker_color=colours, showlegend=False,
                           name="Coefficient", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    return _base(fig, "Lasso coefficients, standardised inputs (blue raises the estimate, orange lowers it; non-zero only)", 400, legend=False,
                 x="Coefficient (mg/dL per 1 SD)", y="Feature")


def r2_bars(df: pd.DataFrame, highlight: str) -> go.Figure:
    d = df.sort_values("R² Score")
    colours = [theme.TEAL if n == highlight else theme.SLATE for n in d["Algorithm"]]
    fig = go.Figure(go.Bar(x=d["R² Score"], y=d["Algorithm"], orientation="h", marker_color=colours,
                           text=[f"{v:.4f}" for v in d["R² Score"]], textposition="outside", cliponaxis=False,
                           showlegend=False, name="R²", hovertemplate="%{y}: R² %{x:.4f}<extra></extra>"))
    fig.update_xaxes(range=[min(0, d["R² Score"].min()) * 1.25, 0.3])
    return _base(fig, "Test-set R²: all ten regressors (highlighted = highest R²)", 400, legend=False,
                 x="R² on the 20% hold-out test set", y="Algorithm")


# ----------------------------------------------------------------- clustering
def cluster_color(i: int) -> str:
    return theme.CATEGORICAL[i % len(theme.CATEGORICAL)]


def pca_fig(xy: np.ndarray, labels: np.ndarray, title: str, evr=None, point: np.ndarray | None = None,
            point_label: str = "Your patient", height: int = 430) -> go.Figure:
    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        m = labels == c
        # SVG scatter (not WebGL) so the patient marker below is always drawn on top
        fig.add_trace(go.Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"Cluster {int(c)} (n={int(m.sum()):,})",
            marker=dict(size=6, color=cluster_color(int(c)), symbol=theme.CLUSTER_SYMBOLS[int(c) % 6], opacity=0.55,
                        line=dict(width=0)),
            hovertemplate=f"Cluster {int(c)}<br>PC1 %{{x:.2f}}, PC2 %{{y:.2f}}<extra></extra>"))
    if point is not None:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers+text", name=point_label, text=[point_label],
                                 textposition="top center", textfont=dict(color=theme.INK, size=12),
                                 marker=dict(size=17, color=theme.INK, symbol="star", line=dict(width=1.5, color="#FFF"))))
    xl = f"Principal component 1{f' ({evr[0]:.1%} of variance)' if evr is not None else ''}"
    yl = f"Principal component 2{f' ({evr[1]:.1%} of variance)' if evr is not None else ''}"
    return _base(fig, title, height, x=xl, y=yl)


def tsne_fig(xy: np.ndarray, labels: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        m = labels == c
        fig.add_trace(go.Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"Cluster {int(c)} (n={int(m.sum()):,})",
            marker=dict(size=6, color=cluster_color(int(c)), symbol=theme.CLUSTER_SYMBOLS[int(c) % 6], opacity=0.7)))
    return _base(fig, "t-SNE (2,000-row sample), coloured by K-Means cluster", 430, x="t-SNE dimension 1", y="t-SNE dimension 2")


def elbow_fig(k_table: pd.DataFrame, k: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=k_table["k"], y=k_table["Inertia"], mode="lines+markers", name="Inertia",
                               line=dict(color=theme.TEAL, width=2.4), marker=dict(size=8),
                               hovertemplate="k=%{x}<br>inertia %{y:,.0f}<extra></extra>"))
    row = k_table[k_table["k"] == k].iloc[0]
    fig.add_trace(go.Scatter(x=[k], y=[row["Inertia"]], mode="markers", name=f"Chosen k = {k}",
                             marker=dict(size=15, color=theme.CATEGORICAL[1], symbol="diamond", line=dict(width=2, color="#FFF"))))
    return _base(fig, "K-Means elbow curve (inertia vs k)", 380, x="Number of clusters (k)", y="Inertia (within-cluster sum of squares)")


def kmetric_fig(k_table: pd.DataFrame, k: int) -> go.Figure:
    specs = [("Silhouette", "higher is better"), ("Davies-Bouldin", "lower is better"),
             ("Calinski-Harabasz", "higher is better")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{n} ({d})" for n, d in specs], horizontal_spacing=0.08)
    for i, (n, _) in enumerate(specs, start=1):
        fig.add_trace(go.Scatter(x=k_table["k"], y=k_table[n], mode="lines+markers", name=n,
                                 line=dict(color=theme.CATEGORICAL[i - 1], width=2), marker=dict(size=7),
                                 showlegend=False), row=1, col=i)
        row = k_table[k_table["k"] == k].iloc[0]
        fig.add_trace(go.Scatter(x=[k], y=[row[n]], mode="markers", name=f"Chosen k = {k}",
                                 marker=dict(size=13, color=theme.INK, symbol="diamond"), showlegend=(i == 1)), row=1, col=i)
        fig.update_xaxes(title_text="k", row=1, col=i)
    fig = _base(fig, "Internal validation metrics for each candidate k", 340, y=None)
    fig.update_layout(margin=dict(l=48, r=16, t=76, b=64))
    for a in fig.layout.annotations:
        a.font.size = 11
    return fig


def dendrogram_fig(Z: np.ndarray, p: int = 30) -> go.Figure:
    d = scipy_dendrogram(Z, truncate_mode="lastp", p=p, no_plot=True)
    fig = go.Figure()
    palette = {"C0": theme.CATEGORICAL[0], "C1": theme.CATEGORICAL[1], "C2": theme.CATEGORICAL[2], "C3": theme.CATEGORICAL[3]}
    for xs, ys, col in zip(d["icoord"], d["dcoord"], d["color_list"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=palette.get(col, theme.INK_2), width=1.6),
                                 hoverinfo="skip", showlegend=False))
    n_leaf = len(d["ivl"])
    fig = _base(fig, "Agglomerative clustering dendrogram, Ward linkage (500-row sample, last 30 merges)", 420, legend=False,
                x="Merged groups (number of samples in parentheses)", y="Ward distance (Euclidean)")
    fig.update_xaxes(tickmode="array", tickvals=[5 + 10 * i for i in range(n_leaf)], ticktext=d["ivl"],
                     tickangle=-60, showgrid=False)
    return fig


def profile_fig(features: list[str], z_cluster: list[float], z_patient: list[float], cluster: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=features, x=z_cluster, orientation="h", name=f"Cluster {cluster} average",
                         marker_color=cluster_color(cluster), opacity=0.9,
                         hovertemplate="%{y}<br>cluster mean = %{x:+.2f} SD from dataset mean<extra></extra>"))
    fig.add_trace(go.Scatter(y=features, x=z_patient, mode="markers", name="This patient",
                             marker=dict(size=12, color=theme.INK, symbol="diamond", line=dict(width=1.5, color="#FFF")),
                             hovertemplate="%{y}<br>patient = %{x:+.2f} SD from dataset mean<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=theme.INK_2, width=1))
    return _base(fig, "Standardised difference from the dataset average", 380,
                 x="Standard deviations above (+) or below (-) the dataset mean", y="Feature")
