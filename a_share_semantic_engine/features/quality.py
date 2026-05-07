from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .common import standardize_columns

RISK_PATTERNS = [
    "退市",
    "审计",
    "重大经营不确定",
    "持续亏损",
    "风险",
    "现金流",
    "无法表示意见",
    "经营能力",
]


def _count_uncertainties(x: str) -> int:
    if pd.isna(x):
        return 0
    try:
        obj = json.loads(x)
        if isinstance(obj, list):
            return len(obj)
        return 0
    except Exception:
        return 0


def _confidence_to_score(x: str) -> float:
    x = str(x).lower()
    return {"high": 1.0, "medium": 0.5, "low": 0.2}.get(x, 0.0)


def _risk_keyword_score(text: str) -> int:
    text = "" if pd.isna(text) else str(text)
    return sum(1 for p in RISK_PATTERNS if p in text)


def build_quality_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    out = pd.DataFrame(index=df.index)
    out["confidence_score"] = df["confidence"].map(_confidence_to_score).astype(float)
    out["uncertainty_count"] = df.get("uncertainties_json", "").map(_count_uncertainties)
    out["source_note_len"] = df["source_quality_note"].fillna("").astype(str).str.len().astype(float)
    out["risk_keyword_score"] = df["source_quality_note"].map(_risk_keyword_score).astype(float)
    out["stability_penalty"] = (out["uncertainty_count"] + out["risk_keyword_score"]) / 10.0
    matrix = standardize_columns(out.to_numpy(dtype=np.float32))
    return out, matrix
