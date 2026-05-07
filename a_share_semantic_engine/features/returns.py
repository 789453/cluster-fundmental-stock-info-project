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
        price_df: DataFrame with ts_code, trade_date, close columns.
        windows: list of return windows. Default [1, 5, 20, 60, 120, 252].

    Returns:
        DataFrame with return features added.
    """
    if windows is None:
        windows = [1, 5, 20, 60, 120, 252]

    df = price_df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = df.groupby("ts_code")["close"]

    for w in windows:
        col = f"ret_{w}d"
        prev = grouped.shift(w)
        df[col] = (df["close"] - prev) / prev
        df[col] = df[col].astype(np.float32)

        rev_col = f"reversal_{w}d"
        df[rev_col] = (-df[col]).astype(np.float32)

    df["log_close"] = np.log(df["close"]).astype(np.float32)

    vol_windows = [20, 60]
    for w in vol_windows:
        ret_col = f"ret_{w}d"
        if ret_col in df.columns:
            df[f"realized_vol_{w}d"] = (
                df.groupby("ts_code")[ret_col]
                .transform(lambda x: x.rolling(w, min_periods=max(w // 2, 5)).std())
            ).astype(np.float32)

    logger.info("Built return features: %s", df.columns.tolist())
    return df


def build_liquidity_features(
    daily_df: pd.DataFrame,
    basic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build liquidity and valuation features.

    Args:
        daily_df: DataFrame with ts_code, trade_date, vol, amount columns.
        basic_df: DataFrame with turnover_rate, pe, pb, ps, dv_ratio, total_mv, circ_mv columns.

    Returns:
        DataFrame with liquidity/valuation features merged.
    """
    merge_cols = ["ts_code", "trade_date"]
    avail_basic = [c for c in basic_df.columns if c in [
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share",
        "total_mv", "circ_mv",
    ]]
    basic_sub = basic_df[merge_cols + avail_basic].copy()

    df = daily_df.merge(basic_sub, on=merge_cols, how="left")

    for c in avail_basic:
        df[c] = df[c].astype(np.float32)

    if "total_mv" in df.columns and "circ_mv" in df.columns:
        df["log_total_mv"] = np.log(df["total_mv"].replace(0, np.nan)).astype(np.float32)
        df["log_circ_mv"] = np.log(df["circ_mv"].replace(0, np.nan)).astype(np.float32)

    return df


def build_financial_features(
    snapshot_df: pd.DataFrame,
    fina_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build financial quality/growth/leverage features from snapshot.

    Args:
        snapshot_df: DataFrame with financial indicator columns.
        fina_cols: list of financial columns to keep.

    Returns:
        DataFrame with financial features.
    """
    if fina_cols is None:
        fina_cols = [
            "roe", "roe_waa", "roa", "roic",
            "grossprofit_margin", "netprofit_margin", "profit_to_gr",
            "debt_to_assets", "assets_to_eqt",
            "current_ratio", "quick_ratio",
            "ocfps", "cfps", "fcff", "fcfe",
            "tr_yoy", "or_yoy", "netprofit_yoy",
            "basic_eps_yoy", "q_sales_yoy", "q_roe",
        ]

    df = snapshot_df.copy()
    available = [c for c in fina_cols if c in df.columns]

    for c in available:
        df[c] = df[c].astype(np.float32)

    logger.info("Financial features: %d columns", len(available))
    return df
