from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False


def run_clustering(features: np.ndarray, cfg: dict, record_ids: list[str]) -> tuple[pd.DataFrame, Any]:
    n = len(features)
    n_clusters = min(int(cfg["cluster"]["n_clusters"]), max(2, n))
    
    # Check for GPU clustering if enabled in config or requested
    # For now we use MiniBatchKMeans as fallback, but we could use torch-based k-means
    # if the dataset is large. For 5500 rows, CPU is fine.
    
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=int(cfg["cluster"]["batch_size"]),
        max_iter=int(cfg["cluster"]["max_iter"]),
        random_state=int(cfg["project"]["random_state"]),
        n_init="auto",
    )
    labels = model.fit_predict(features)
    centers = model.cluster_centers_[labels]
    
    if HAS_TORCH_CUDA:
        # Example of how we could use GPU for distance calculation
        f_t = torch.from_numpy(features).cuda()
        c_t = torch.from_numpy(centers).cuda()
        dist_t = torch.norm(f_t - c_t, dim=1)
        dist = dist_t.cpu().numpy()
    else:
        dist = np.linalg.norm(features - centers, axis=1)

    out = pd.DataFrame(
        {
            "record_id": record_ids,
            "cluster_id": labels.astype(int),
            "cluster_distance": dist.astype(float),
        }
    )
    return out, model
