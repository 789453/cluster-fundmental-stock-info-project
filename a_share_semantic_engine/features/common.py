from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return (x / norm).astype(np.float32)


def safe_text(series) -> list[str]:
    return series.fillna("").astype(str).tolist()


def standardize_columns(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def pad_or_trim(x: np.ndarray, target_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    if d == target_dim:
        return x
    if d > target_dim:
        return x[:, :target_dim].astype(np.float32)
    out = np.zeros((n, target_dim), dtype=np.float32)
    out[:, :d] = x
    return out
