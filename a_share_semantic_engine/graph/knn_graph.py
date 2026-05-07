from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(features: np.ndarray, k: int, min_sim: float, mutual: bool) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    n = features.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(features)
    dist, idx = nn.kneighbors(features)
    rows = []
    cols = []
    vals = []
    for i in range(n):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            sim = float(1.0 - d)
            if sim < min_sim:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(sim)

    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    if mutual:
        mat = mat.minimum(mat.T)
    else:
        mat = mat.maximum(mat.T)
    mat.setdiag(0)
    mat.eliminate_zeros()

    coo = mat.tocoo()
    edges = pd.DataFrame({"src": coo.row.astype(int), "dst": coo.col.astype(int), "weight": coo.data.astype(float)})
    return mat, edges
