from __future__ import annotations
import logging
from typing import Any

import numpy as np
import scipy.sparse as sparse

logger = logging.getLogger(__name__)


def build_multiplex_graph(
    graphs: dict[str, sparse.csr_matrix],
    weights: dict[str, float] | None = None,
) -> sparse.csr_matrix:
    """
    Fuse multiple graphs into one.
    
    Args:
        graphs: Dictionary of {name: adjacency_matrix}.
        weights: Dictionary of {name: weight}.
    """
    if not graphs:
        raise ValueError("No graphs provided for fusion.")
        
    if weights is None:
        weights = {name: 1.0 / len(graphs) for name in graphs}
        
    # Ensure all graphs have the same shape
    first_name = next(iter(graphs))
    shape = graphs[first_name].shape
    N = shape[0]
    
    fused = sparse.csr_matrix((N, N), dtype=np.float32)
    
    total_weight = sum(weights.values())
    
    for name, adj in graphs.items():
        if adj.shape != shape:
            logger.warning(f"Skipping graph {name} due to shape mismatch: {adj.shape} vs {shape}")
            continue
            
        w = weights.get(name, 0.0) / total_weight
        if w > 0:
            fused += adj * w
            
    logger.info(f"Fused {len(graphs)} graphs with weights: {weights}")
    return fused
