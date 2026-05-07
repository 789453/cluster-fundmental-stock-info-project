from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from ..data.asof_join import cross_section_winsorize, industry_neutralize

logger = logging.getLogger(__name__)


def build_barra_style_exposures(
    df: pd.DataFrame,
    price_panel_df: pd.DataFrame | None = None,
    industry_col: str = "l1_name",
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
) -> pd.DataFrame:
    """
    Build Barra-style cross-sectional style exposures.

    Per the spec:
      Size:          zscore(log(total_mv))
      NonlinearSize: residual after regressing log(total_mv) on Size^2
      Value:         zscore(1/pb, 1/pe_ttm, 1/ps_ttm, dv_ratio)
      Momentum:       zscore(ret_252d - ret_20d)  [exclude recent 20d]
      ShortReversal: zscore(-ret_20d)
      Volatility:    zscore(realized_vol_60d)
      Liquidity:     zscore(turnover_rate)
      Profitability: zscore(roe, roa, roic, grossprofit_margin)
      Growth:        zscore(netprofit_yoy, or_yoy, basic_eps_yoy)
      Leverage:      zscore(debt_to_assets, assets_to_eqt)
      EarningsQuality: zscore(ocf_to_profit, salescash_to_or)

    Process per date:
      1. winsorize at 1%/99%
      2. fill missing with industry median
      3. zscore across full universe
      4. zscore within SW-L1 industry

    Args:
        df: snapshot DataFrame with trade_date, valuation, financial, return columns.
        price_panel_df: optional DataFrame with longer return history (for momentum calc).
        industry_col: column name for industry grouping.
        winsorize_lower: lower quantile.
        winsorize_upper: upper quantile.

    Returns:
        DataFrame with style exposure columns added.
    """
    result = df.copy()

    for col in result.columns:
        if result[col].dtype in (np.float32, np.float64, np.float16):
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    price_cols = ["ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
                  "realized_vol_20d", "realized_vol_60d", "momentum_raw"]
    val_cols = ["total_mv", "circ_mv", "pe_ttm", "pb", "ps_ttm", "dv_ratio"]
    fin_cols = ["roe", "roa", "roic", "grossprofit_margin", "netprofit_margin"]
    growth_cols = ["netprofit_yoy", "or_yoy", "tr_yoy", "basic_eps_yoy", "q_sales_yoy"]
    lev_cols = ["debt_to_assets", "assets_to_eqt"]
    liq_cols = ["turnover_rate", "turnover_rate_f", "volume_ratio"]
    qual_cols = ["ocf_to_profit", "salescash_to_or"]

    if price_panel_df is not None and len(price_panel_df) > 0:
        pm = price_panel_df.copy()
        if "ret_252d" in pm.columns and "ret_20d" in pm.columns:
            pm["momentum_raw"] = pm["ret_252d"] - pm["ret_20d"]
        elif "ret_252d" in pm.columns:
            pm["momentum_raw"] = pm["ret_252d"]
        if "momentum_raw" in pm.columns:
            result = result.merge(
                pm[["ts_code", "trade_date", "momentum_raw"]],
                on=["ts_code", "trade_date"],
                how="left",
                suffixes=("", "_pm"),
            )
            if "momentum_raw_pm" in result.columns:
                result["momentum_raw"] = result.pop("momentum_raw_pm")

    all_style_src = (
        val_cols + price_cols + fin_cols + growth_cols +
        lev_cols + liq_cols + qual_cols + ["momentum_raw"]
    )
    avail = [c for c in all_style_src if c in result.columns]
    result = cross_section_winsorize(result, avail, winsorize_lower, winsorize_upper, group_col=None)

    result = _fill_missing_with_industry_median(result, avail, industry_col)

    style_defs = {
        "Size":          ["total_mv"],
        "LogSize":       ["circ_mv"],
        "NonlinearSize": ["total_mv"],
        "Value":         ["pb", "pe_ttm", "ps_ttm", "dv_ratio"],
        "Momentum":      ["momentum_raw"],
        "ShortReversal": ["ret_20d"],
        "Volatility":    ["realized_vol_60d", "realized_vol_20d"],
        "Liquidity":     ["turnover_rate", "turnover_rate_f"],
        "Profitability": ["roe", "roa", "roic", "grossprofit_margin"],
        "Growth":        ["netprofit_yoy", "or_yoy", "basic_eps_yoy"],
        "Leverage":       ["debt_to_assets", "assets_to_eqt"],
        "EarningsQuality":["ocf_to_profit", "salescash_to_or"],
    }

    for style_name, src_cols in style_defs.items():
        actual_cols = [c for c in src_cols if c in result.columns]
        if not actual_cols:
            result[style_name] = np.nan
            result[f"{style_name}_sw_neutral"] = np.nan
            continue

        combined = result[actual_cols].mean(axis=1, skipna=True)
        mu = combined.mean()
        sigma = combined.std()
        sigma = sigma if sigma != 0 else 1.0
        z_full = ((combined - mu) / sigma).astype(np.float32)

        result[style_name] = z_full
        neutral = industry_neutralize(result, style_name, industry_col=industry_col, method="zscore")
        result[f"{style_name}_sw_neutral"] = neutral.astype(np.float32)

    neutral_cols = [c for c in result.columns if c.endswith("_sw_neutral")]

    for c in result.columns:
        if c in style_defs or c in neutral_cols:
            result[c] = result[c].astype(np.float32)

    logger.info(
        "Barra style exposures built: %d total, %d neutral",
        len([s for s in style_defs if s in result.columns]),
        len(neutral_cols),
    )
    return result


def _fill_missing_with_industry_median(
    df: pd.DataFrame,
    cols: list[str],
    industry_col: str,
) -> pd.DataFrame:
    result = df.copy()
    if industry_col not in result.columns:
        return result

    for col in cols:
        if col not in result.columns:
            continue
        missing = result[col].isna()
        if missing.sum() == 0:
            continue
        medians = result.groupby(industry_col)[col].transform("median")
        result.loc[missing, col] = medians[missing]
        still_missing = result[col].isna()
        if still_missing.sum() > 0:
            global_median = result[col].median()
            result.loc[still_missing, col] = global_median

    return result


def compute_beta(
    returns: np.ndarray,
    market_returns: np.ndarray,
    window: int = 252,
) -> np.ndarray:
    """
    Compute rolling beta for each stock.

    Args:
        returns: (N, T) return series.
        market_returns: (T,) market return series.
        window: rolling window.

    Returns:
        beta: (N,) array of betas.
    """
    N, T = returns.shape
    betas = np.zeros(N, dtype=np.float32)
    market_var = np.nanvar(market_returns)

    if market_var == 0:
        return betas

    for i in range(N):
        stock_ret = returns[i]
        roll_stock = stock_ret[-window:] if T >= window else stock_ret
        roll_market = market_returns[-window:] if len(market_returns) >= window else market_returns
        min_len = min(len(roll_stock), len(roll_market))
        cov = np.nanmean((roll_stock[:min_len] - np.nanmean(roll_stock[:min_len])) *
                         (roll_market[:min_len] - np.nanmean(roll_market[:min_len])))
        betas[i] = cov / market_var

    return betas
