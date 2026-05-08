from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse

logger = logging.getLogger(__name__)


def build_return_corr_graph(
    returns_panel: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    window: int = 60,
    k: int = 20,
) -> sparse.csr_matrix:
    """
    Build a graph based on rolling return correlation.
    
    Args:
        returns_panel: DataFrame with ts_code, trade_date, ret_1d.
        snapshot_df: The target cross-section for node ordering.
        window: Correlation window.
        k: Neighbors per node.
    """
    # 1. Pivot returns to (Time, Stocks)
    pivot_ret = returns_panel.pivot(index="trade_date", columns="ts_code", values="ret_1d")
    
    # 2. Compute correlation matrix
    # Note: For 5000 stocks, this is 5000x5000, which is manageable (~100MB)
    corr_mat = pivot_ret.tail(window).corr()
    
    # 3. Align with snapshot
    ts_codes = snapshot_df["ts_code"].tolist()
    corr_mat = corr_mat.reindex(index=ts_codes, columns=ts_codes).fillna(0)
    
    # 4. Sparsify: keep top-k correlations per row
    N = len(ts_codes)
    adj = sparse.lil_matrix((N, N), dtype=np.float32)
    
    mat_values = corr_mat.values
    for i in range(N):
        row = mat_values[i]
        # Get indices of top k+1 (including self)
        top_indices = np.argsort(-row)[:k+1]
        for idx in top_indices:
            if idx != i:
                adj[i, idx] = row[idx]
                
    return adj.tocsr()
