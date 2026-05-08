from __future__ import annotations
import logging
from typing import Any

import numpy as np
import faiss
import scipy.sparse as sparse

logger = logging.getLogger(__name__)


def faiss_knn_search(
    features: np.ndarray,
    k: int = 30,
    use_gpu: bool = True,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform kNN search using FAISS.
    
    Args:
        features: (N, D) feature matrix.
        k: Number of neighbors.
        use_gpu: Whether to use GPU.
        metric: 'cosine' or 'l2'.
    
    Returns:
        distances: (N, k) distance matrix.
        indices: (N, k) index matrix.
    """
    N, D = features.shape
    X = np.ascontiguousarray(features.astype(np.float32))
    
    if metric == "cosine":
        # Cosine similarity is inner product of normalized vectors
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(D)
    elif metric == "l2":
        index = faiss.IndexFlatL2(D)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if use_gpu:
        try:
            # Check for GPU resources
            if hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using FAISS GPU for kNN")
            elif hasattr(faiss, "gpu") and hasattr(faiss.gpu, "StandardGpuResources"):
                res = faiss.gpu.StandardGpuResources()
                index = faiss.gpu.index_cpu_to_gpu(res, 0, index)
                logger.info("Using FAISS GPU for kNN")
            else:
                logger.warning("FAISS GPU resources not found. Falling back to CPU.")
        except Exception as e:
            logger.warning(f"Failed to use FAISS GPU: {e}. Falling back to CPU.")

    index.add(X)
    distances, indices = index.search(X, k + 1)  # k+1 because the first neighbor is the node itself
    
    # Remove self-loops (the first column)
    return distances[:, 1:], indices[:, 1:]


def build_faiss_knn_graph(
    features: np.ndarray,
    k: int = 30,
    min_sim: float = 0.0,
    mutual: bool = True,
    use_gpu: bool = True,
    metric: str = "cosine",
) -> sparse.csr_matrix:
    """
    Build a kNN graph using FAISS.
    """
    N = features.shape[0]
    distances, indices = faiss_knn_search(features, k=k, use_gpu=use_gpu, metric=metric)
    
    rows = np.repeat(np.arange(N), k)
    cols = indices.flatten()
    data = distances.flatten()
    
    # Filter by minimum similarity
    if min_sim > 0:
        mask = data >= min_sim
        rows = rows[mask]
        cols = cols[mask]
        data = data[mask]
        
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    if mutual:
        # Mutual kNN: edge exists only if both nodes are in each other's kNN
        adj = adj.minimum(adj.transpose())
        
    return adj
