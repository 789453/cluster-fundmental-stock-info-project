from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph


def pagerank_power_iteration(adj: sparse.csr_matrix, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_deg = np.divide(1.0, out_deg, where=out_deg > 0, out=np.zeros_like(out_deg, dtype=float))
    P = sparse.diags(inv_deg) @ adj
    pr = np.full(n, 1.0 / n, dtype=float)
    base = np.full(n, (1.0 - alpha) / n, dtype=float)

    for _ in range(max_iter):
        prev = pr.copy()
        pr = alpha * (P.T @ pr) + base
        if np.abs(pr - prev).sum() < tol:
            break
    return pr.astype(np.float32)


def compute_graph_stats(adj: sparse.csr_matrix, record_ids: list[str]) -> pd.DataFrame:
    degree = np.diff(adj.indptr).astype(int)
    weighted_degree = np.asarray(adj.sum(axis=1)).ravel().astype(float)
    _, labels = csgraph.connected_components(adj, directed=False, return_labels=True)
    comp_sizes = pd.Series(labels).value_counts().to_dict()
    pagerank = pagerank_power_iteration(adj)

    out = pd.DataFrame(
        {
            "record_id": record_ids,
            "degree": degree,
            "weighted_degree": weighted_degree,
            "component_id": labels.astype(int),
            "component_size": [int(comp_sizes[int(x)]) for x in labels],
            "pagerank": pagerank,
        }
    )
    return out
