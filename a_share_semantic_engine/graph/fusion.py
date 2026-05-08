from __future__ import annotations
from typing import Literal

from scipy import sparse
import numpy as np


def fuse_graphs(graphs: dict[str, sparse.csr_matrix], weights: dict[str, float]) -> sparse.csr_matrix:
    fused = None
    total = 0.0
    for name, graph in graphs.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        fused = graph.multiply(w) if fused is None else fused + graph.multiply(w)
        total += w
    if fused is None:
        raise ValueError("no graphs to fuse")
    if total > 0:
        fused = fused.multiply(1.0 / total)
    fused = fused.tocsr()
    fused.setdiag(0)
    fused.eliminate_zeros()
    return fused


def fuse_sparse_graphs(
    graphs: dict[str, sparse.csr_matrix],
    weights: dict[str, float],
    normalize: Literal["max", "row_sum", "none"] = "max",
    topk: int = 50,
    min_weight: float = 1e-6,
) -> sparse.csr_matrix:
    """
    Sparse weighted sum of graphs with per-row topk pruning.

    Args:
        graphs: {name: csr_matrix} for each graph type.
        weights: {name: fusion_weight}.
        normalize: normalization method per graph before fusion.
        topk: keep only topk neighbors per row after fusion.
        min_weight: drop edges below this weight.

    Returns:
        fused CSR matrix.
    """
    N = next(iter(graphs.values())).shape[0]
    fused_data: dict[tuple[int, int], float] = {}

    for name, adj in graphs.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue

        adj_norm = adj.tocsr()
        if normalize == "max":
            max_val = adj_norm.data.max() if adj_norm.nnz > 0 else 1.0
            if max_val > 0:
                adj_norm = adj_norm.multiply(1.0 / max_val)
        elif normalize == "row_sum":
            row_sums = np.array(adj_norm.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv = sparse.diags(1.0 / row_sums)
            adj_norm = (inv @ adj_norm).tocsr()

        for i in range(N):
            row = adj_norm.getrow(i)
            for j_idx in range(row.nnz):
                j = row.indices[j_idx]
                val = row.data[j_idx] * w
                key = (min(i, j), max(i, j))
                fused_data[key] = fused_data.get(key, 0.0) + val

    rows, cols, data = [], [], []
    for (i, j), w in fused_data.items():
        if w >= min_weight:
            rows.append(i)
            cols.append(j)
            data.append(w)

    if not rows:
        return sparse.csr_matrix((N, N), dtype=np.float32)

    fused = sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    fused = fused.maximum(fused.T)

    if topk > 0:
        rows_out, cols_out, data_out = [], [], []
        for i in range(N):
            row = fused.getrow(i)
            nnz = row.nnz
            if nnz <= topk:
                for j_idx in range(nnz):
                    rows_out.append(i)
                    cols_out.append(row.indices[j_idx])
                    data_out.append(row.data[j_idx])
            else:
                top_indices = np.argpartition(row.data, -topk)[-topk:]
                top_indices = top_indices[np.argsort(row.data[top_indices])[::-1]]
                for j_idx in top_indices:
                    rows_out.append(i)
                    cols_out.append(row.indices[j_idx])
                    data_out.append(row.data[j_idx])
        fused = sparse.csr_matrix((data_out, (rows_out, cols_out)), shape=(N, N), dtype=np.float32)
        fused = fused.maximum(fused.T)

    fused.setdiag(0)
    fused.eliminate_zeros()

    return fused
