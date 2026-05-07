from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ..data.validation import validate_matrix
from .common import l2_normalize, pad_or_trim


def reduce_npy_view(matrix: np.ndarray, compact_dim: int, random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    validate_matrix("npy_view", matrix)
    X = np.asarray(matrix, dtype=np.float32)
    if X.shape[1] <= compact_dim:
        Z = pad_or_trim(X, compact_dim)
        Z = l2_normalize(Z)
        return Z, {"reducer": None, "explained_variance": None}
    reducer = PCA(n_components=compact_dim, random_state=random_state, svd_solver="randomized")
    Z = reducer.fit_transform(X).astype(np.float32)
    Z = pad_or_trim(Z, compact_dim)
    Z = l2_normalize(Z)
    explained = float(reducer.explained_variance_ratio_.sum())
    return Z, {"reducer": reducer, "explained_variance": explained}
