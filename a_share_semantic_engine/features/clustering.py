from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False

logger = logging.getLogger(__name__)


def run_clustering(
    features: np.ndarray,
    cfg: dict,
    record_ids: list[str],
) -> tuple[pd.DataFrame, Any]:
    """
    Backward-compatible wrapper that calls run_kmeans.
    cfg keys: cluster.n_clusters, cluster.batch_size, cluster.max_iter, project.random_state
    """
    n_clusters = int(cfg.get("cluster", {}).get("n_clusters", 20))
    seed = int(cfg.get("project", {}).get("random_state", 42))
    batch_size = int(cfg.get("cluster", {}).get("batch_size", 256))
    max_iter = int(cfg.get("cluster", {}).get("max_iter", 100))
    labels, dists = run_kmeans(features, n_clusters=n_clusters, seed=seed, batch_size=batch_size, max_iter=max_iter)
    return pd.DataFrame({"record_id": record_ids, "cluster_id": labels, "cluster_distance": dists}), None


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    batch_size: int = 256,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MiniBatchKMeans clustering. CPU fallback.
    """
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        n_init="auto",
    )
    labels = model.fit_predict(features)
    centers = model.cluster_centers_[labels]
    dists = np.linalg.norm(features - centers, axis=1).astype(np.float32)
    return labels.astype(np.int32), dists


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    HDBSCAN clustering for non-spherical clusters.
    Returns labels and probabilities.
    """
    try:
        import hdbscan
    except ImportError:
        logger.warning("hdbscan not installed, falling back to kmeans")
        labels = run_kmeans(features, n_clusters=max(2, len(features) // 50))[0]
        return labels, np.ones(len(features), dtype=np.float32)

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = model.fit_predict(features).astype(np.int32)
    probs = model.probabilities_.astype(np.float32)
    return labels, probs


def run_leiden(
    adj: sparse.csr_matrix,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Leiden community detection on sparse adjacency matrix.
    Falls back to Louvain if igraph unavailable.
    """
    try:
        import igraph as ig
        import leidenalg as la

        N = adj.shape[0]
        sources, targets = adj.nonzero()
        weights = adj[sources, targets].A1 if hasattr(adj, "A") else np.ones(len(sources))

        g = ig.Graph(N, edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
        if len(weights) == g.ecount():
            g.es["weight"] = weights.tolist()

        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"] if g.ecount() > 0 else None,
            resolution_parameter=resolution,
            seed=seed,
        )
        labels = np.array(part.membership, dtype=np.int32)
        logger.info("Leiden: %d communities found", len(set(labels)))
        return labels

    except ImportError:
        logger.warning("leidenalg/igraph not available, falling back to Louvain")
        return run_louvain(adj, resolution, seed)


def run_louvain(
    adj: sparse.csr_matrix,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Louvain community detection via python-louvain / networkx.
    """
    try:
        import networkx as nx

        N = adj.shape[0]
        sources, targets = adj.nonzero()
        data = np.asarray(adj[sources, targets]).flatten() if hasattr(adj, "A") else np.ones(len(sources))

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for s, t, w in zip(sources, targets, data):
            G.add_edge(s, t, weight=w)

        import community.community_louvain as comm_louvain
        parts = comm_louvain.best_partition(G, resolution=resolution, random_state=seed)
        labels = np.array([parts.get(i, 0) for i in range(N)], dtype=np.int32)
        logger.info("Louvain: %d communities found", len(set(labels)))
        return labels

    except ImportError:
        logger.warning("python-louvain not available, using spectral clustering fallback")
        return _spectral_fallback(adj, n_clusters=max(2, adj.shape[0] // 100))


def _spectral_fallback(adj: sparse.csr_matrix, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    N = adj.shape[0]
    if N < n_clusters:
        n_clusters = max(2, N)

    try:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = model.fit_predict(adj.toarray())
    except Exception:
        labels = np.zeros(N, dtype=np.int32)

    return labels.astype(np.int32)


def compute_modularity(
    adj: sparse.csr_matrix,
    labels: np.ndarray,
) -> float:
    """
    Compute graph modularity Q.
    """
    try:
        import networkx as nx
        import community.community_louvain as comm_louvain

        N = adj.shape[0]
        sources, targets = adj.nonzero()
        data = np.asarray(adj[sources, targets]).flatten()
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for s, t, w in zip(sources, targets, data):
            G.add_edge(s, t, weight=w)

        parts = {i: int(labels[i]) for i in range(N)}
        Q = comm_louvain.modularity(parts, G)
        return float(Q)
    except Exception:
        return 0.0


def compute_nmi(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """
    Normalized Mutual Information between two label arrays.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(labels_a, labels_b)
    except Exception:
        return 0.0


def compute_purity(
    labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Cluster purity: fraction of clusters where majority class dominates.
    """
    try:
        from sklearn.metrics import cluster
        contingency = cluster.contingency_matrix(true_labels, labels)
        return float(np.sum(np.max(contingency, axis=0)) / len(labels))
    except Exception:
        return 0.0


def cluster_summary(
    labels: np.ndarray,
    ts_codes: list[str],
    feature_df: pd.DataFrame | None = None,
    industry_col: str = "l1_name",
) -> pd.DataFrame:
    """
    Produce a per-cluster summary DataFrame.

    Args:
        labels: cluster/community labels per stock.
        ts_codes: list of stock codes aligned with labels.
        feature_df: optional DataFrame with stock features and industry.
        industry_col: column name for industry grouping.

    Returns:
        DataFrame with cluster_id, count, top_industries, avg_size特征.
    """
    n_clusters = int(labels.max()) + 1
    records = []
    for cid in range(n_clusters):
        mask = labels == cid
        count = int(mask.sum())
        rec = {
            "cluster_id": cid,
            "count": count,
        }
        if feature_df is not None and industry_col in feature_df.columns:
            inds = feature_df.loc[mask, industry_col].value_counts()
            rec["top_industry"] = inds.index[0] if len(inds) > 0 else ""
            rec["top_industry_pct"] = float(inds.iloc[0] / count) if count > 0 else 0.0
        records.append(rec)

    summary = pd.DataFrame(records)
    return summary
