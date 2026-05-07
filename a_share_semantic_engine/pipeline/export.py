from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse


def save_table(df: pd.DataFrame, path_base: Path, write_parquet: bool = True, write_csv: bool = True) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    if write_parquet:
        try:
            df.to_parquet(path_base.with_suffix(".parquet"), index=False)
        except Exception:
            pass
    if write_csv:
        df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def save_sparse_graph(adj: sparse.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, adj)


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def plot_cluster_sizes(cluster_df: pd.DataFrame, path: Path) -> None:
    counts = cluster_df["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_degree_hist(graph_stats: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(graph_stats["degree"], bins=min(20, max(5, graph_stats["degree"].nunique())))
    ax.set_title("Graph Degree Histogram")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
