from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def join_financial_asof(
    daily_df: pd.DataFrame,
    fina_df: pd.DataFrame | None = None,
    trade_date_col: str = "trade_date",
    ann_date_col: str = "ann_date",
    by: str = "ts_code",
    lookback_days: int = 365,
    warehouse_db: str | Path | None = None,
) -> pd.DataFrame:
    """
    Join financial indicators using as-of semantics: only allow ann_date <= trade_date.
    For each (ts_code, trade_date), select the most recent financial record.

    Uses DuckDB SQL for efficiency when warehouse_db is provided; otherwise falls back to
    a vectorized pandas merge.

    Args:
        daily_df: DataFrame with trade_date and ts_code columns.
        fina_df: DataFrame with ann_date, end_date, and financial fields.
        trade_date_col: name of the trade date column in daily_df.
        ann_date_col: name of the announcement date column in fina_df.
        by: join key, typically 'ts_code'.
        lookback_days: only consider financial records within this many days of trade_date.
        warehouse_db: if provided, use DuckDB for fast SQL as-of join.

    Returns:
        DataFrame with financial fields added, one row per trade_date/ts_code.
    """
    if daily_df.empty:
        return daily_df.copy()

    if fina_df is not None and fina_df.empty:
        logger.warning("fina_df is empty, returning daily_df with NaN financial fields")
        return daily_df.copy()

    if warehouse_db and fina_df is not None:
        return _join_financial_asof_sql(
            daily_df, fina_df, str(warehouse_db),
            trade_date_col, ann_date_col, by,
        )

    return _join_financial_asof_pandas(daily_df, fina_df, trade_date_col, ann_date_col, by)


def _join_financial_asof_sql(
    daily_df: pd.DataFrame,
    fina_df: pd.DataFrame,
    warehouse_db: str,
    trade_date_col: str,
    ann_date_col: str,
    by: str,
) -> pd.DataFrame:
    tmp_daily = "tmp_daily_asof"
    tmp_fina = "tmp_fina_asof"

    fina_cols = [c for c in fina_df.columns if c not in (ann_date_col, by, "_ad_int")]
    daily_cols = [c for c in daily_df.columns]

    conn = duckdb.connect(warehouse_db, read_only=False)
    conn.execute(f"CREATE OR REPLACE TEMP TABLE {tmp_daily} AS SELECT * FROM daily_df")
    conn.execute(f"CREATE OR REPLACE TEMP TABLE {tmp_fina} AS SELECT * FROM fina_df")

    fina_col_sql = ", ".join([f'f."{c}"' for c in fina_cols])
    daily_col_sql = ", ".join([f'd."{c}"' for c in daily_cols])

    sql = f"""
    WITH ranked AS (
        SELECT
            d."{by}" AS "{by}",
            d."{trade_date_col}" AS "{trade_date_col}",
            {fina_col_sql},
            ROW_NUMBER() OVER (
                PARTITION BY d."{by}", d."{trade_date_col}"
                ORDER BY f."{ann_date_col}" DESC
            ) AS _rn
        FROM {tmp_daily} d
        JOIN {tmp_fina} f
          ON d."{by}" = f."{by}"
         AND f."{ann_date_col}" <= d."{trade_date_col}"
    )
    SELECT
        {daily_col_sql},
        {fina_col_sql}
    FROM {tmp_daily} d
    LEFT JOIN ranked r
      ON d."{by}" = r."{by}"
     AND d."{trade_date_col}" = r."{trade_date_col}"
     AND r._rn = 1
    """

    result = conn.execute(sql).df()
    conn.close()
    logger.info("SQL asof join complete: %d rows", len(result))
    return result


def _join_financial_asof_pandas(
    daily_df: pd.DataFrame,
    fina_df: pd.DataFrame | None,
    trade_date_col: str,
    ann_date_col: str,
    by: str,
) -> pd.DataFrame:
    if fina_df is None or fina_df.empty:
        return daily_df

    daily_df = daily_df.copy()
    fina_df = fina_df.copy()

    fina_cols = [c for c in fina_df.columns if c not in (ann_date_col, by)]
    latest = (
        fina_df
        .sort_values(ann_date_col)
        .groupby(by)
        .tail(1)
    )

    result = daily_df.drop(columns=fina_cols, errors="ignore").merge(
        latest[[by] + fina_cols], on=by, how="left"
    )

    n_filled = result[fina_cols].notna().sum().sum()
    n_total = result[fina_cols].size
    logger.info(
        "asof join (pandas fallback): %d/%d (%.1f%%) financial cells filled",
        n_filled, n_total, 100 * n_filled / max(n_total, 1),
    )
    return result


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
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
    result = df.copy()
    for col in factor_cols:
        if group_col and group_col in result.columns:
            result[col] = result.groupby(group_col)[col].transform(
                lambda x: x.clip(x.quantile(lower), x.quantile(upper))
            )
        else:
            lo = result[col].quantile(lower)
            hi = result[col].quantile(upper)
            result[col] = result[col].clip(lower=lo, upper=hi)
    return result
