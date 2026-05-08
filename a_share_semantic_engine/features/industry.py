from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_industry_features(
    snapshot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build industry-related features including one-hot encoding and industry ranks.
    """
    df = snapshot_df.copy()
    
    # Ensure SW industry IDs are present
    sw_cols = ["l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name"]
    for col in sw_cols:
        if col not in df.columns:
            logger.warning(f"Industry column {col} missing from snapshot.")

    # One-hot encoding for L1 industry (optional, can be memory intensive for high L)
    # Usually better to keep as category/ID for analysis
    
    # Industry size rank
    if "total_mv" in df.columns and "l1_name" in df.columns:
        df["industry_size_rank"] = df.groupby("l1_name")["total_mv"].transform(lambda x: x.rank(pct=True))
    
    # Industry value rank
    if "pe_ttm" in df.columns and "l1_name" in df.columns:
        df["industry_value_rank"] = df.groupby("l1_name")["pe_ttm"].transform(lambda x: x.rank(pct=True))

    logger.info("Industry features built.")
    return df
