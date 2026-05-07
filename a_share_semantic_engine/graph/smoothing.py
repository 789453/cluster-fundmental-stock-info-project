from __future__ import annotations

import numpy as np
from scipy import sparse

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False


def graph_smooth(adj: sparse.csr_matrix, features: np.ndarray, alpha: float = 0.7, steps: int = 1) -> np.ndarray:
    x = features.astype(np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    inv_deg = np.divide(1.0, deg, where=deg > 0, out=np.zeros_like(deg))
    norm_adj = sparse.diags(inv_deg) @ adj
    
    if HAS_TORCH_CUDA:
        # Move to GPU
        x_t = torch.from_numpy(x).cuda()
        adj_t = torch.from_numpy(norm_adj.toarray()).cuda() # Note: dense on GPU for small graphs
        out_t = x_t.clone()
        for _ in range(steps):
            out_t = alpha * x_t + (1.0 - alpha) * (adj_t @ out_t)
        return out_t.cpu().numpy()
    
    out = x.copy()
    for _ in range(steps):
        out = alpha * x + (1.0 - alpha) * (norm_adj @ out)
    return np.asarray(out, dtype=np.float32)
