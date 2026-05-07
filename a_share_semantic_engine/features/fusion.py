from __future__ import annotations

import numpy as np

from .common import l2_normalize


def mean_fuse(mats: list[np.ndarray]) -> np.ndarray:
    if not mats:
        raise ValueError("empty mats")
    stacked = np.stack(mats, axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def weighted_concat(text_mat: np.ndarray, json_mat: np.ndarray, anchor_mat: np.ndarray, quality_mat: np.ndarray, weights: dict) -> np.ndarray:
    parts = [
        text_mat * float(weights["text_weight"]),
        json_mat * float(weights["json_weight"]),
        anchor_mat * float(weights["anchor_weight"]),
        quality_mat * float(weights["quality_weight"]),
    ]
    x = np.concatenate(parts, axis=1).astype(np.float32)
    return l2_normalize(x)
