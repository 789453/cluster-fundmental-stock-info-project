from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def compute_graph_metrics(
    adj: sparse.csr_matrix,
    ts_codes: list[str],
    labels: np.ndarray | None = None,
    style_df: pd.DataFrame | None = None,
    snap_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute comprehensive node-level graph metrics.

    Args:
        adj: CSR adjacency matrix (N, N).
        ts_codes: list of stock codes aligned with rows/cols.
        labels: optional community/cluster labels for bridge_score and community features.
        style_df: optional DataFrame with style exposures for neighbor_style_dispersion.
        snap_df: optional DataFrame with return/ROE/momentum for neighbor aggregation.

    Returns:
        DataFrame with per-stock graph metrics.
    """
    N = adj.shape[0]
    metrics = pd.DataFrame({"ts_code": ts_codes})

    sources, targets = adj.nonzero()
    data = np.asarray(adj[sources, targets]).flatten()

    weighted_degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    metrics["degree"] = weighted_degree
    metrics["unweighted_degree"] = np.diff(adj.indptr).astype(np.int32)
    metrics["weighted_degree"] = weighted_degree

    logger.info("computing pagerank...")
    pr = _pagerank(adj)
    metrics["pagerank"] = pr.astype(np.float32)

    logger.info("computing eigenvector centrality...")
    ev = _eigenvector_centrality(adj)
    metrics["eigenvector_centrality"] = ev.astype(np.float32)

    logger.info("computing betweenness (sampled)...")
    bet = _betweenness_approx(adj, n_samples=min(200, N))
    metrics["betweenness_approx"] = bet.astype(np.float32)

    logger.info("computing clustering coefficient...")
    cc = _clustering_coefficient(adj)
    metrics["clustering_coefficient"] = cc.astype(np.float32)

    logger.info("computing kcore...")
    kc = _kcore(adj)
    metrics["kcore_number"] = kc.astype(np.int16)

    if labels is not None:
        metrics["community_id"] = labels.astype(np.int16)
        comm_sizes = pd.Series(labels).value_counts().to_dict()
        metrics["community_size"] = metrics["community_id"].map(comm_sizes).astype(np.int16)

        if snap_df is not None and len(snap_df) == N:
            logger.info("computing bridge_score...")
            bridge = _bridge_score(adj, labels)
            metrics["bridge_score"] = bridge.astype(np.float32)

        if style_df is not None and len(style_df) == N:
            logger.info("computing neighbor_style_dispersion...")
            disp = _neighbor_style_dispersion(adj, style_df)
            metrics["neighbor_style_dispersion"] = disp.astype(np.float32)

    if snap_df is not None and len(snap_df) == N:
        logger.info("computing neighbor aggregates...")
        for ret_col in ["pct_chg", "roe", "ret_20d"]:
            if ret_col in snap_df.columns:
                col_name = f"neighbor_avg_{ret_col}"
                metrics[col_name] = _neighbor_aggregate(adj, snap_df[ret_col].values.astype(np.float32)).astype(np.float32)

        logger.info("computing semantic_industry_consistency...")
        sw_col = "l1_name" if "l1_name" in snap_df.columns else None
        if sw_col is not None:
            cons = _semantic_industry_consistency(adj, labels, snap_df[sw_col].values)
            metrics["semantic_industry_consistency"] = cons.astype(np.float32)

    metrics["ts_code"] = ts_codes
    logger.info("graph_metrics: %d rows, %s", len(metrics), metrics.columns.tolist())
    return metrics


def _pagerank(adj: sparse.csr_matrix, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    N = adj.shape[0]
    out_deg = np.asarray(adj.sum(axis=1)).flatten()
    out_deg = np.where(out_deg == 0, 1, out_deg)
    P = adj.astype(np.float64).copy().tocsr()
    for i in range(N):
        P.data[P.indptr[i]:P.indptr[i+1]] /= out_deg[i]
    P = P.T.tocsr()

    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr_new = (1 - alpha) / N + alpha * (P @ pr)
        if np.abs(pr_new - pr).sum() < tol:
            break
        pr = pr_new
    return pr / pr.sum()


def _eigenvector_centrality(adj: sparse.csr_matrix, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    N = adj.shape[0]
    w, v = sparse.linalg.eigs(adj.astype(np.float64), k=1, which="LM", maxiter=max_iter)
    ev = np.abs(v[:, 0])
    ev /= ev.sum()
    return ev.astype(np.float32)


def _betweenness_approx(adj: sparse.csr_matrix, n_samples: int = 100) -> np.ndarray:
    try:
        import networkx as nx
        G = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
        bc = nx.betweenness_centrality(G, k=min(n_samples, len(G.nodes)), weight=None, seed=42)
        return np.array([bc.get(i, 0.0) for i in range(adj.shape[0])], dtype=np.float32)
    except Exception:
        N = adj.shape[0]
        bet = np.zeros(N, dtype=np.float32)
        rows, cols = adj.nonzero()
        edges = list(zip(rows, cols))

        if len(edges) == 0:
            return bet

        np.random.seed(42)
        for _ in range(n_samples):
            src = np.random.randint(0, N)
            bfs_dist, bfs_parent = _bfs(adj, src)
            for node in range(N):
                if node != src and bfs_dist[node] >= 0:
                    paths = _count_paths_to_root(node, bfs_parent)
                    bet[node] += paths / n_samples
        return bet


def _bfs(adj: sparse.csr_matrix, start: int):
    N = adj.shape[0]
    dist = np.full(N, -1, dtype=np.int32)
    parent = np.full(N, -1, dtype=np.int32)
    queue = [start]
    dist[start] = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u].indices:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent


def _count_paths_to_root(node: int, parent: np.ndarray) -> float:
    count = 0.0
    while node != -1:
        count += 1
        node = parent[node]
    return 1.0 / max(count - 1, 1)


def _clustering_coefficient(adj: sparse.csr_matrix) -> np.ndarray:
    N = adj.shape[0]
    cc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nei = adj[i].indices
        k = len(nei)
        if k < 2:
            cc[i] = 0.0
            continue
        tri = 0
        for j in nei:
            for k2 in adj[j].indices:
                if k2 != i and k2 in nei:
                    tri += 1
        cc[i] = tri / (k * (k - 1)) if k > 1 else 0.0
    return cc


def _kcore(adj: sparse.csr_matrix) -> np.ndarray:
    import networkx as nx
    A_bin = adj.copy()
    A_bin.data[:] = 1
    G = nx.from_scipy_sparse_array(A_bin)
    core = nx.core_number(G)
    kcore = np.array([core.get(i, 0) for i in range(adj.shape[0])], dtype=np.int32)
    return kcore


def _neighbor_aggregate(adj: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    N = adj.shape[0]
    result = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nei = adj[i].indices
        if len(nei) == 0:
            result[i] = 0.0
        else:
            result[i] = np.nanmean(values[nei])
    return result


def _bridge_score(adj: sparse.csr_matrix, labels: np.ndarray) -> np.ndarray:
    N = adj.shape[0]
    bridge = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nei = adj[i].indices
        if len(nei) == 0:
            bridge[i] = 0.0
            continue
        label_i = labels[i]
        cross_comm = np.sum(labels[nei] != label_i)
        total_w = adj[i].sum()
        bridge[i] = cross_comm / max(len(nei), 1)
    return bridge


def _neighbor_style_dispersion(adj: sparse.csr_matrix, style_df: pd.DataFrame) -> np.ndarray:
    style_cols = [c for c in style_df.columns if c not in ["ts_code", "trade_date", "cluster_id"]]
    if not style_cols:
        return np.zeros(adj.shape[0], dtype=np.float32)

    style_mat = style_df[style_cols].values.astype(np.float32)
    N = adj.shape[0]
    disp = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nei = adj[i].indices
        if len(nei) < 2:
            disp[i] = 0.0
        else:
            nei_styles = style_mat[nei]
            center = nei_styles.mean(axis=0)
            diffs = np.linalg.norm(nei_styles - center, axis=1)
            disp[i] = np.nanmean(diffs) if len(diffs) > 0 and not np.all(np.isnan(diffs)) else 0.0
    return disp


def _semantic_industry_consistency(
    adj: sparse.csr_matrix,
    labels: np.ndarray | None,
    industries: np.ndarray,
) -> np.ndarray:
    N = adj.shape[0]
    cons = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nei = adj[i].indices
        if len(nei) == 0:
            cons[i] = 0.0
            continue
        label_i = labels[i] if labels is not None else industries[i]
        if labels is not None:
            same_cluster = np.sum(labels[nei] == label_i) / max(len(nei), 1)
            same_industry = np.sum(industries[nei] == industries[i]) / max(len(nei), 1)
            cons[i] = same_cluster if labels is not None else same_industry
        else:
            cons[i] = np.sum(industries[nei] == industries[i]) / max(len(nei), 1)
    return cons


def compute_conductance(adj: sparse.csr_matrix, labels: np.ndarray) -> float:
    """Compute graph conductance for a given clustering."""
    N = adj.shape[0]
    unique_labels = np.unique(labels)
    conductances = []

    for label in unique_labels:
        mask = labels == label
        if mask.sum() == 0:
            continue
        internal = 0.0
        external = 0.0
        for i in np.where(mask)[0]:
            for j in adj[i].indices:
                if mask[j]:
                    internal += adj[i, j]
                else:
                    external += adj[i, j]
        denom = internal + external
        if denom > 0:
            conductances.append(external / denom)

    return float(np.mean(conductances)) if conductances else 1.0
