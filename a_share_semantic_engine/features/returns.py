from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_returns_features(
    price_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build return features from price panel.

    Args:
        price_df: DataFrame with ts_code, trade_date, close, high, low columns.
        windows: list of return windows. Default [1, 5, 20, 60, 120, 252].

    Returns:
        DataFrame with return features added.
    """
    if windows is None:
        windows = [1, 5, 20, 60, 120, 252]

    df = price_df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = df.groupby("ts_code")

    for w in windows:
        col = f"ret_{w}d"
        # Use close to close return
        df[col] = grouped["close"].pct_change(periods=w).astype(np.float32)

        rev_col = f"reversal_{w}d"
        df[rev_col] = (-df[col]).astype(np.float32)

    df["log_close"] = np.log(df["close"]).astype(np.float32)

    # Volatility and Drawdown
    vol_windows = [20, 60]
    for w in vol_windows:
        # Realized volatility
        df[f"realized_vol_{w}d"] = (
            grouped["ret_1d"]
            .transform(lambda x: x.rolling(w, min_periods=max(w // 2, 5)).std() * np.sqrt(252))
        ).astype(np.float32)
        
        # Downside volatility
        df[f"downside_vol_{w}d"] = (
            grouped["ret_1d"]
            .transform(lambda x: x[x < 0].rolling(w, min_periods=max(w // 2, 5)).std() * np.sqrt(252))
        ).astype(np.float32)

        # Max drawdown
        df[f"max_drawdown_{w}d"] = (
            grouped["close"]
            .transform(lambda x: (x / x.rolling(w, min_periods=1).max() - 1).rolling(w, min_periods=1).min())
        ).astype(np.float32)

    # High-Low Range
    if "high" in df.columns and "low" in df.columns:
        df["high_low_range_20d"] = (
            (grouped["high"].transform(lambda x: x.rolling(20).max()) - 
             grouped["low"].transform(lambda x: x.rolling(20).min())) / 
            df["close"]
        ).astype(np.float32)

    logger.info("Built return features: %s", df.columns.tolist())
    return df
