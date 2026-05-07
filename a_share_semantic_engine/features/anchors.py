from __future__ import annotations

import pandas as pd
import numpy as np

from .common import standardize_columns


def encode_anchors(df: pd.DataFrame, anchor_fields: list[str]) -> tuple[np.ndarray, list[str]]:
    frame = pd.get_dummies(df[anchor_fields].fillna("unknown").astype(str), prefix=anchor_fields, dtype=float)
    x = frame.to_numpy(dtype=np.float32)
    x = standardize_columns(x)
    return x, frame.columns.tolist()
