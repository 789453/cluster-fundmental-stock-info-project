from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse

logger = logging.getLogger(__name__)


def build_industry_graph(
    snapshot_df: pd.DataFrame,
    l1_weight: float = 0.4,
    l2_weight: float = 0.7,
    l3_weight: float = 1.0,
) -> sparse.csr_matrix:
    """
    Build a graph based on SW industry hierarchy.
    
    Weights:
      same l3: l3_weight (default 1.0)
      same l2: l2_weight (default 0.7)
      same l1: l1_weight (default 0.4)
    """
    N = len(snapshot_df)
    adj = sparse.lil_matrix((N, N), dtype=np.float32)
    
    # We use grouping to speed up the process
    for level, weight in [("l1_name", l1_weight), ("l2_name", l2_weight), ("l3_name", l3_weight)]:
        if level not in snapshot_df.columns:
            continue
            
        groups = snapshot_df.groupby(level).indices
        for members in groups.values():
            if len(members) < 2:
                continue
            # For each group, create a clique (or update existing weights)
            # Since we want the maximum weight for the most specific level:
            for i in members:
                for j in members:
                    if i != j:
                        adj[i, j] = max(adj[i, j], weight)
                        
    return adj.tocsr()
