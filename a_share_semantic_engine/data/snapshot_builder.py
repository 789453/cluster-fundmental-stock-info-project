from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .asof_join import cross_section_winsorize, industry_neutralize, join_financial_asof
from .lakehouse import LakeHouse

logger = logging.getLogger(__name__)


def build_stock_snapshot(
    trade_date: str,
    warehouse_db: str | Path,
    include_st: bool = False,
    include_suspended: bool = False,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build a single-day stock cross-section snapshot.

    Must include: ts_code, trade_date, close, return, daily_basic fields,
    申万行业 (SW), financial-as-of fields.

    Args:
        trade_date: YYYYMMDD string.
        warehouse_db: path to warehouse.duckdb.
        include_st: whether to include *ST stocks.
        include_suspended: whether to include suspended stocks.
        output_path: if provided, save snapshot to this path.

    Returns:
        DataFrame with one row per stock.
    """
    logger.info("Building stock snapshot for trade_date=%s", trade_date)

    with LakeHouse(warehouse_db) as lh:
        daily = lh.fetch_stock_daily(trade_date=trade_date)
        basic = lh.fetch_stock_daily_basic(trade_date=trade_date)
        stock_info = lh.fetch_stock_basic_snapshot()
        sw_member = lh.fetch_sw_member()

        fina_cols = [
            "ts_code", "ann_date", "end_date",
            "roe", "roe_waa", "roa", "roic",
            "grossprofit_margin", "netprofit_margin", "profit_to_gr",
            "debt_to_assets", "assets_to_eqt",
            "current_ratio", "quick_ratio",
            "ocfps", "cfps", "fcff", "fcfe",
            "tr_yoy", "or_yoy", "netprofit_yoy", "dt_netprofit_yoy",
            "basic_eps_yoy", "q_sales_yoy", "q_roe", "q_dt_roe", "q_ocf_to_sales",
        ]
        fina = lh.fetch_fina_indicator(columns=fina_cols)

    if daily.empty:
        logger.warning("No daily data for trade_date=%s", trade_date)
        return pd.DataFrame()

    n_daily = len(daily)
    logger.info("  daily records: %d", n_daily)

    df = daily[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]].copy()

    basic_cols = [
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share",
        "total_mv", "circ_mv",
    ]
    df = df.merge(basic[["ts_code", "trade_date"] + basic_cols], on=["ts_code", "trade_date"], how="left")

    stock_info_map = stock_info[["ts_code", "name", "area", "industry", "market", "list_date", "act_name"]].copy()
    df = df.merge(stock_info_map, on="ts_code", how="left")

    if not include_st:
        df = df[~df["name"].str.contains("ST", na=False)]

    df = df[df["vol"] > 0]

    td_str = str(trade_date)
    df["trade_date"] = df["trade_date"].astype(str)

    # Add record_id from semantic dataset if possible for alignment
    semantic_path = Path("artifacts/a_share_semantic_dataset/parquet/records-all.parquet")
    if semantic_path.exists():
        # Use pandas for an efficient as-of join with semantic records
        try:
            semantic_records = pd.read_parquet(semantic_path, columns=["record_id", "stock_code", "asof_date"])
            semantic_records["asof_date_fmt"] = semantic_records["asof_date"].str.replace("-", "")
            
            # We want the latest record_id for each stock where asof_date <= trade_date
            s = semantic_records[semantic_records["asof_date_fmt"] <= td_str].copy()
            if not s.empty:
                s = s.sort_values("asof_date_fmt", ascending=False).drop_duplicates("stock_code")
                df = df.merge(s[["stock_code", "record_id"]], left_on="ts_code", right_on="stock_code", how="left")
                df.drop(columns=["stock_code"], inplace=True)
                logger.info("  joined record_id from semantic dataset (as-of join via pandas)")
            else:
                df["record_id"] = df["ts_code"] + "_" + df["trade_date"]
                logger.warning("  no semantic records found for asof_date <= %s", td_str)
        except Exception as e:
            logger.warning("  failed to join record_id via pandas: %s. Using fallback.", e)
            df["record_id"] = df["ts_code"] + "_" + df["trade_date"]
    else:
        df["record_id"] = df["ts_code"] + "_" + df["trade_date"]

    sw_active = sw_member[
        (sw_member["in_date"].astype(str) <= td_str) &
        ((sw_member["out_date"].isna()) | (sw_member["out_date"].astype(str) > td_str) | (sw_member["out_date"] == 0))
    ].copy()
    sw_active["in_date"] = sw_active["in_date"].astype(str)

    sw_cols = ["ts_code", "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name"]
    sw_latest = sw_active.sort_values("in_date").groupby("ts_code").last().reset_index()[sw_cols]
    df = df.merge(sw_latest, on="ts_code", how="left")

    df = join_financial_asof(
        df, fina,
        trade_date_col="trade_date",
        ann_date_col="ann_date",
        by="ts_code",
    )

    float_cols = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, np.float16)]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)

    df["record_key"] = df["ts_code"] + "_" + df["trade_date"].astype(str)

    missing_report = _build_missing_report(df, trade_date)
    logger.info("  snapshot shape: %s, missing report:\n%s", df.shape, missing_report)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, compression="zstd")
        logger.info("  saved snapshot to %s", output_path)

    return df


def _build_missing_report(df: pd.DataFrame, trade_date: str) -> str:
    lines = []
    total = len(df)
    lines.append(f"trade_date: {trade_date}, total stocks: {total}")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = 100 * n_missing / max(total, 1)
        if n_missing > 0:
            lines.append(f"  {col}: {n_missing}/{total} ({pct:.1f}%)")
    return "\n".join(lines) if lines else "no missing values"


def build_snapshot_panel(
    start_date: str,
    end_date: str,
    warehouse_db: str | Path,
    include_st: bool = False,
    include_suspended: bool = False,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build snapshots for a date range.

    Args:
        start_date: YYYYMMDD start.
        end_date: YYYYMMDD end.
        warehouse_db: path to warehouse.duckdb.
        include_st: include *ST stocks.
        include_suspended: include suspended stocks.
        output_dir: if provided, save each snapshot parquet under this directory.

    Returns:
        DataFrame with all snapshots stacked.
    """
    with LakeHouse(warehouse_db) as lh:
        trade_dates = lh.get_trade_dates(start_date, end_date)

    logger.info("Building snapshot panel from %s to %s (%d trading days)", start_date, end_date, len(trade_dates))

    snapshots = []
    for td in tqdm(trade_dates, desc="building snapshots"):
        path = None
        if output_dir:
            path = Path(output_dir) / f"trade_date={td}" / "stock_snapshot.parquet"
        snap = build_stock_snapshot(
            trade_date=td,
            warehouse_db=warehouse_db,
            include_st=include_st,
            include_suspended=include_suspended,
            output_path=path,
        )
        if not snap.empty:
            snapshots.append(snap)

    if not snapshots:
        logger.warning("No snapshots built")
        return pd.DataFrame()

    panel = pd.concat(snapshots, ignore_index=True)
    logger.info("Snapshot panel shape: %s", panel.shape)
    return panel
