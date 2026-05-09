from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_price_feature_panel(
    warehouse_db: str | Path,
    trade_date: str,
    ts_codes: list[str],
    windows: Sequence[int] = (5, 20, 60, 120, 252),
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build price feature panel from continuous market data.

    Args:
        warehouse_db: path to warehouse.duckdb
        trade_date: target trade date (YYYYMMDD)
        ts_codes: list of stock codes to include
        windows: return windows to compute

    Returns:
        price_features: DataFrame at ts_code level with ret_*, realized_vol_*,
                       momentum_*, turnover_*, beta_* columns
        returns_matrix: float32 matrix (N, T), aligned with ts_codes
    """
    from ..data.lakehouse import LakeHouse

    with LakeHouse(warehouse_db) as lh:
        start_date = str(int(trade_date) - windows[-1] * 4 if len(trade_date) == 8 else 0)
        daily = lh.fetch_stock_daily(
            trade_date=None,
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=trade_date,
            columns=["ts_code", "trade_date", "close", "pct_chg", "vol", "amount"],
        )

    if daily.empty:
        logger.warning("No daily data for panel build")
        return pd.DataFrame(), np.array([])

    daily = daily.sort_values(["ts_code", "trade_date"])

    price_features = []
    returns_list = []
    all_dates = sorted(daily["trade_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    T = len(all_dates)

    grouped = daily.groupby("ts_code")

    for ts_code, grp in grouped:
        grp = grp.sort_values("trade_date")
        closes = grp["close"].values.astype(np.float32)
        rets = np.diff(closes) / closes[:-1].clip(min=1e-8)
        rets = np.insert(rets, 0, 0.0)

        features = {"ts_code": ts_code}

        for w in windows:
            if len(grp) >= w:
                cum_ret = closes[-1] / closes[-w - 1] - 1 if len(closes) > w else 0.0
            else:
                cum_ret = closes[-1] / closes[0] - 1 if len(closes) > 1 else 0.0
            features[f"ret_{w}d"] = cum_ret

        if len(grp) >= 20:
            realized_vol = rets[-20:].std() * np.sqrt(252) if len(rets) >= 20 else 0.0
        else:
            realized_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
        features["realized_vol_20d"] = realized_vol

        if len(grp) >= 60:
            realized_vol_60 = rets[-60:].std() * np.sqrt(252) if len(rets) >= 60 else 0.0
        else:
            realized_vol_60 = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
        features["realized_vol_60d"] = realized_vol_60

        momentum_raw = features.get("ret_252d", 0.0) - features.get("ret_20d", 0.0)
        features["momentum_252_20"] = momentum_raw

        reversal_20d = -features.get("ret_20d", 0.0)
        features["short_reversal_20d"] = reversal_20d

        vol_arr = grp["vol"].values.astype(np.float32)
        if len(vol_arr) >= 20:
            turnover_mean = np.nanmean(vol_arr[-20:]) if not np.isnan(vol_arr[-20:]).all() else 0.0
        else:
            turnover_mean = np.nanmean(vol_arr) if not np.isnan(vol_arr).all() else 0.0
        features["turnover_mean_20d"] = turnover_mean

        amount_arr = grp["amount"].values.astype(np.float32)
        if len(amount_arr) >= 20:
            amount_mean = np.nanmean(amount_arr[-20:]) if not np.isnan(amount_arr[-20:]).all() else 0.0
        else:
            amount_mean = np.nanmean(amount_arr) if not np.isnan(amount_arr).all() else 0.0
        features["amount_mean_20d"] = amount_mean

        if len(rets) >= 252:
            market_ret = np.diff(closes[:252]) / closes[:252][:-1].clip(min=1e-8)
            stock_ret = rets[-252:] if len(rets) >= 252 else rets
            if len(stock_ret) > 1 and len(market_ret) > 1:
                cov = np.cov(stock_ret[-252:], market_ret[-252:])[0, 1]
                var = np.var(market_ret)
                beta = cov / var if var > 0 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0
        features["beta_252d"] = beta

        price_features.append(features)

        row_ret = np.zeros(T, dtype=np.float32)
        for d in grp["trade_date"].values:
            if d in date_to_idx:
                idx = date_to_idx[d]
                ts_idx = grp["trade_date"].values.tolist().index(d)
                if ts_idx < len(rets):
                    row_ret[idx] = rets[ts_idx]
        returns_list.append(row_ret)

    price_df = pd.DataFrame(price_features)
    returns_matrix = np.stack(returns_list, axis=0).astype(np.float32) if returns_list else np.array([])

    logger.info("Built price panel: %d stocks, %d time steps, %d features",
                len(price_df), T, len(price_df.columns) - 1)

    return price_df, returns_matrix


def build_returns_matrix(
    warehouse_db: str | Path,
    ts_codes: list[str],
    end_date: str,
    window: int = 120,
    price_col: str = "close",
) -> np.ndarray:
    """
    Build returns matrix from continuous price data.

    Args:
        warehouse_db: path to warehouse.duckdb
        ts_codes: list of stock codes
        end_date: end date YYYYMMDD
        window: number of periods to include
        price_col: price column to use

    Returns:
        returns_matrix: float32 array shape (N, T)
    """
    from ..data.lakehouse import LakeHouse

    start_date = str(int(end_date) - window * 4)

    with LakeHouse(warehouse_db) as lh:
        daily = lh.fetch_stock_daily(
            trade_date=None,
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            columns=["ts_code", "trade_date", price_col],
        )

    if daily.empty:
        return np.array([])

    daily = daily.sort_values(["ts_code", "trade_date"])

    all_dates = sorted(daily["trade_date"].unique())
    T = min(window, len(all_dates))
    date_to_idx = {d: i for i, d in enumerate(all_dates[-T:])}

    returns_list = []
    grouped = daily.groupby("ts_code")

    for ts_code in ts_codes:
        grp = grouped.get_group(ts_code).sort_values("trade_date") if ts_code in grouped.groups else pd.DataFrame()
        prices = grp[price_col].values.astype(np.float32)
        row = np.zeros(T, dtype=np.float32)

        if len(prices) > 1:
            rets = np.diff(prices) / prices[:-1].clip(min=1e-8)
            dates = grp["trade_date"].values
            for i, d in enumerate(dates[1:]):
                if d in date_to_idx:
                    row[date_to_idx[d]] = rets[i]

        returns_list.append(row)

    return np.stack(returns_list, axis=0).astype(np.float32) if returns_list else np.array([])