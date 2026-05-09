from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


CN_FONTS = [
    "SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS",
]


def setup_chinese_font():
    if not HAS_MATPLOTLIB:
        return
    for font in CN_FONTS:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def _setup_cn_font():
    setup_chinese_font()


def plot_cluster_sizes(sizes: list[int], output_path: Path | None = None):
    if not HAS_MATPLOTLIB:
        return
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sizes, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Size Distribution")
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_degree_histogram(degrees: np.ndarray, output_path: Path | None = None):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    deg_nonzero = degrees[degrees > 0]
    ax.hist(deg_nonzero, bins=40, color="coral", edgecolor="white")
    ax.set_xlabel("Degree (non-zero)")
    ax.set_ylabel("Count")
    ax.set_title("Graph Degree Distribution")
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_industry_stacked(
    labels: np.ndarray,
    industries: np.ndarray,
    n_top: int = 10,
    output_path: Path | None = None,
):
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return
    _setup_cn_font()

    df = pd.DataFrame({"cluster_id": labels, "industry": industries})
    n_clusters = int(labels.max()) + 1

    top_industries = df["industry"].value_counts().head(n_top).index.tolist()
    df["industry_top"] = df["industry"].apply(lambda x: x if x in top_industries else "其他")

    stacked = (
        df.groupby(["cluster_id", "industry_top"])
        .size()
        .unstack(fill_value=0)
    )
    stacked = stacked.loc[stacked.sum(axis=1).sort_values(ascending=False).index]
    stacked = stacked.head(30)

    fig, ax = plt.subplots(figsize=(14, 6))
    stacked.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Industry Composition (Top Industries)")
    ax.legend(title="Industry", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_barra_radar(
    cluster_profile_df: pd.DataFrame,
    cluster_id: int,
    style_cols: list[str],
    output_path: Path | None = None,
):
    if not HAS_MATPLOTLIB or len(style_cols) < 3:
        return
    _setup_cn_font()

    row = cluster_profile_df[cluster_profile_df["cluster_id"] == cluster_id]
    if row.empty:
        return

    values = []
    labels_radar = []
    for sc in style_cols:
        col = f"avg_{sc}" if f"avg_{sc}" in cluster_profile_df.columns else sc
        if col in cluster_profile_df.columns:
            val = row[col].values[0]
            if not pd.isna(val):
                values.append(float(val))
                labels_radar.append(sc)

    if len(values) < 3:
        return

    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values = values + [values[0]]
    angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="steelblue")
    ax.fill(angles, values, alpha=0.25, color="steelblue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=8)
    ax.set_title(f"Cluster {cluster_id} Barra Style Profile", pad=20)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factor_correlation_heatmap(
    style_df: pd.DataFrame,
    output_path: Path | None = None,
):
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return
    _setup_cn_font()
    style_cols = [c for c in style_df.columns if c not in ["ts_code", "trade_date", "cluster_id"]]
    avail = [c for c in style_cols if c in style_df.columns and style_df[c].dtype in (np.float32, np.float64)]
    if len(avail) < 3:
        return

    corr = style_df[avail].astype(float).corr()
    fig, ax = plt.subplots(figsize=(max(8, len(avail) * 0.7), max(6, len(avail) * 0.6)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"fontsize": 7})
    ax.set_title("Style Exposure Correlation Heatmap")
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_graph_quality_summary(
    graph_quality_report: dict,
    output_path: Path | None = None,
):
    if not HAS_MATPLOTLIB:
        return
    _setup_cn_font()
    graphs = graph_quality_report.get("graphs", {})
    if not graphs:
        return

    names = list(graphs.keys())
    edge_counts = [g.get("edge_count", 0) for g in graphs.values()]
    avg_degrees = [g.get("avg_degree", 0) for g in graphs.values()]
    isolated = [float(g.get("isolated_ratio", "0%").rstrip("%")) for g in graphs.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(names, edge_counts, color="steelblue")
    axes[0].set_title("Edge Count by Graph Type")
    axes[0].set_ylabel("Edges")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, avg_degrees, color="coral")
    axes[1].set_title("Avg Degree by Graph Type")
    axes[1].set_ylabel("Avg Degree")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, isolated, color="seagreen")
    axes[2].set_title("Isolated Node Ratio (%)")
    axes[2].set_ylabel("% Isolated")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_diagnostics_summary(
    diagnostics: dict,
    output_path: Path | None = None,
):
    if not HAS_MATPLOTLIB:
        return
    _setup_cn_font()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = {
        "silhouette_score": "Silhouette Score",
        "davies_bouldin_score": "Davies-Bouldin Score",
        "calinski_harabasz_score": "Calinski-Harabasz Score",
    }

    for ax, (key, label) in zip(axes, metrics.items()):
        val = diagnostics.get(key, float("nan"))
        if not pd.isna(val):
            ax.bar([label], [val], color="steelblue")
            ax.text(0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label)
        ax.set_ylabel("Value")

    fig.suptitle("Clustering Quality Diagnostics", fontsize=12)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pagerank_top(
    graph_stats_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Path | None = None,
):
    if not HAS_MATPLOTLIB or "pagerank" not in graph_stats_df.columns:
        return
    _setup_cn_font()
    top = graph_stats_df.nlargest(top_n, "pagerank")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["ts_code"].astype(str), top["pagerank"].astype(float), color="teal")
    ax.set_xlabel("PageRank")
    ax.set_title(f"Top {top_n} Stocks by PageRank")
    ax.invert_yaxis()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_figures(
    cluster_labels: np.ndarray,
    degrees: np.ndarray,
    style_df: pd.DataFrame | None,
    industry_arr: np.ndarray,
    output_dir: Path,
    cluster_profile_df: pd.DataFrame | None = None,
    graph_quality_report: dict | None = None,
    diagnostics: dict | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    sizes = []
    for cid in range(int(cluster_labels.max()) + 1):
        sizes.append(int((cluster_labels == cid).sum()))

    p = output_dir / "cluster_sizes.png"
    plot_cluster_sizes(sizes, p)
    figures["cluster_sizes"] = p

    p2 = output_dir / "degree_hist.png"
    plot_degree_histogram(degrees, p2)
    figures["degree_hist"] = p2

    if len(industry_arr) == len(cluster_labels):
        p3 = output_dir / "cluster_industry_stacked.png"
        plot_cluster_industry_stacked(cluster_labels, industry_arr, n_top=10, output_path=p3)
        figures["cluster_industry_stacked"] = p3

    if style_df is not None and len(style_df) > 0:
        p4 = output_dir / "style_exposure_corr.png"
        plot_factor_correlation_heatmap(style_df, p4)
        figures["style_corr"] = p4

        if cluster_profile_df is not None:
            for cid in cluster_profile_df["cluster_id"].unique()[:3]:
                style_cols = [c for c in style_df.columns if c not in ["ts_code", "trade_date", "cluster_id"]]
                p_radar = output_dir / f"cluster_{cid}_barra_radar.png"
                plot_barra_radar(cluster_profile_df, int(cid), style_cols, p_radar)
                figures[f"cluster_{cid}_radar"] = p_radar

    if graph_quality_report:
        p5 = output_dir / "graph_quality_summary.png"
        plot_graph_quality_summary(graph_quality_report, p5)
        figures["graph_quality"] = p5

    if diagnostics:
        p6 = output_dir / "cluster_diagnostics.png"
        plot_cluster_diagnostics_summary(diagnostics, p6)
        figures["cluster_diagnostics"] = p6

    if style_df is not None and "pagerank" not in style_df.columns:
        pass

    return figures
