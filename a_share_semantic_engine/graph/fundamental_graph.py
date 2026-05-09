from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from .faiss_knn import build_faiss_knn_graph
from ..features.cross_section import zscore_series

logger = logging.getLogger(__name__)


def build_fundamental_graph(
    snapshot_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    k: int = 20,
    use_gpu: bool = True,
) -> sparse.csr_matrix:
    """
    Build a graph based on fundamental feature similarity.
    """
    if feature_cols is None:
        feature_cols = [
            "roe", "roa", "roic", "grossprofit_margin", "netprofit_margin",
            "debt_to_assets", "assets_to_eqt", "current_ratio", "quick_ratio",
            "tr_yoy", "or_yoy", "netprofit_yoy",
        ]
        
    df = snapshot_df.copy()
    avail = [c for c in feature_cols if c in df.columns]
    
    if not avail:
        logger.warning("No fundamental features found for graph building.")
        return sparse.csr_matrix((len(df), len(df)))

    # Standardize features
    X = df[avail].copy()
    for col in avail:
        X[col] = zscore_series(X[col].fillna(X[col].median()))
        
    return build_faiss_knn_graph(X.values, k=k, use_gpu=use_gpu)
