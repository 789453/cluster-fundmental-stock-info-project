from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from .faiss_knn import build_faiss_knn_graph

logger = logging.getLogger(__name__)


def build_style_graph(
    snapshot_df: pd.DataFrame,
    style_cols: list[str] | None = None,
    k: int = 20,
    use_gpu: bool = True,
    tau: float = 1.0,
) -> sparse.csr_matrix:
    """
    Build a graph based on Barra style distance.
    Weight = exp(- ||style_i - style_j||_2 / tau)
    """
    if style_cols is None:
        style_cols = [
            "Size", "Value", "Momentum", "Volatility", "Liquidity", 
            "Profitability", "Growth", "Leverage"
        ]
        
    df = snapshot_df.copy()
    avail = [c for c in style_cols if c in df.columns]
    
    if not avail:
        logger.warning("No style features found for graph building.")
        return sparse.csr_matrix((len(df), len(df)))

    X = df[avail].fillna(0).values
    
    # We use L2 distance for style similarity
    return build_faiss_knn_graph(X, k=k, use_gpu=use_gpu, metric="l2")
