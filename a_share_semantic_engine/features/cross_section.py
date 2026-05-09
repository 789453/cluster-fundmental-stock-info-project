from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a single series."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
    """Compute z-score for a single series."""
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or pd.isna(sigma):
        return s - mu
    return (s - mu) / sigma


def industry_neutralize(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "industry",
    method: str = "zscore",
) -> pd.Series:
    """
    Neutralize a factor by industry.
    
    Args:
        df: DataFrame containing the factor and industry columns.
        factor_col: The factor to neutralize.
        industry_col: The industry grouping column.
        method: 'zscore' or 'rank'.
    """
    if industry_col not in df.columns:
        logger.warning(f"Industry column {industry_col} not found for neutralization. Skipping.")
        return df[factor_col]

    grp = df.groupby(industry_col)[factor_col]
    if method == "zscore":
        grp_mean = grp.transform("mean")
        grp_std = grp.transform("std")
        neutral = df[factor_col] - grp_mean
        neutral = neutral / grp_std.replace(0, 1)
    elif method == "rank":
        neutral = grp.transform(lambda x: x.rank(pct=True))
    else:
        raise ValueError(f"Unknown neutralization method: {method}")
    return neutral


def cross_section_winsorize(
    df: pd.DataFrame,
    factor_cols: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Winsorize multiple columns in a cross-section.
    
    Args:
        df: The cross-section DataFrame.
        factor_cols: Columns to winsorize.
        lower: Lower quantile.
        upper: Upper quantile.
        group_col: Optional column to group by before winsorizing (e.g. industry).
    """
    result = df.copy()
    for col in factor_cols:
        if col not in result.columns:
            continue
        if group_col and group_col in result.columns:
            result[col] = result.groupby(group_col)[col].transform(
                lambda x: x.clip(x.quantile(lower), x.quantile(upper))
            )
        else:
            lo = result[col].quantile(lower)
            hi = result[col].quantile(upper)
            result[col] = result[col].clip(lower=lo, upper=hi)
    return result


def fill_missing_with_industry_median(
    df: pd.DataFrame,
    cols: list[str],
    industry_col: str,
) -> pd.DataFrame:
    """Fill missing values with industry median, then global median."""
    result = df.copy()
    if industry_col not in result.columns:
        logger.warning(f"Industry column {industry_col} not found for missing fill.")
        for col in cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].median())
        return result

    for col in cols:
        if col not in result.columns:
            continue
        missing = result[col].isna()
        if missing.sum() == 0:
            continue
        
        # Industry median
        medians = result.groupby(industry_col)[col].transform("median")
        result.loc[missing, col] = medians[missing]
        
        # Still missing (if an entire industry is NaN)
        still_missing = result[col].isna()
        if still_missing.sum() > 0:
            global_median = result[col].median()
            result.loc[still_missing, col] = global_median

    return result
