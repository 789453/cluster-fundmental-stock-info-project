from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_daily_basic_features(
    snapshot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build liquidity, valuation, and size features from daily_basic data.
    """
    df = snapshot_df.copy()
    
    basic_cols = [
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share",
        "total_mv", "circ_mv",
    ]
    
    avail = [c for c in basic_cols if c in df.columns]
    for c in avail:
        df[c] = df[c].astype(np.float32)

    if "total_mv" in df.columns:
        df["log_total_mv"] = np.log(df["total_mv"].replace(0, np.nan)).astype(np.float32)
    
    if "circ_mv" in df.columns:
        df["log_circ_mv"] = np.log(df["circ_mv"].replace(0, np.nan)).astype(np.float32)

    logger.info("Daily basic features: %d columns", len(avail))
    return df
