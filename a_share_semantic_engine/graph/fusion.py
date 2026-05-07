from __future__ import annotations

from scipy import sparse


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
