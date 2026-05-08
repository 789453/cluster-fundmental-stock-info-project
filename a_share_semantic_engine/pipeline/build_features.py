from __future__ import annotations
import argparse
import logging
from pathlib import Path

import pandas as pd

from ..features.returns import build_returns_features
from ..features.financial import build_financial_features
from ..features.industry import build_industry_features
from ..features.daily_basic import build_daily_basic_features
from ..features.barra_style import build_barra_style_exposures
from ..core.config import load_config
from ..core.logging_utils import setup_logging
from ..data.lakehouse import LakeHouse

def main():
    parser = argparse.ArgumentParser(description="Build features for a date range")
    parser.add_argument("--start-date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--config", type=str, default="configs/research.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = Path(cfg["paths"]["output_root"]) / "logs" / f"build_features_{args.start_date}_{args.end_date}.log"
    logger = setup_logging("build_features", log_path)

    warehouse_db = cfg["paths"]["warehouse_db"]
    gold_root = Path(cfg["paths"]["gold_root"])
    marts_root = Path(cfg["paths"]["marts_root"])

    with LakeHouse(warehouse_db) as lh:
        trade_dates = lh.get_trade_dates(args.start_date, args.end_date)
        
    logger.info(f"Building features for {len(trade_dates)} trading days")

    for td in trade_dates:
        snapshot_path = gold_root / f"trade_date={td}" / "stock_snapshot.parquet"
        if not snapshot_path.exists():
            logger.warning(f"Snapshot not found for {td}, skipping features.")
            continue
            
        df = pd.read_parquet(snapshot_path)
        
        # 1. Basic & Financial Features
        df = build_daily_basic_features(df)
        df = build_financial_features(df)
        df = build_industry_features(df)
        
        # 2. Returns Features (requires panel for volatility, but we can do it row-wise if pre-joined)
        # Note: build_returns_features expects a panel, so we might need to load price history
        # For simplicity in this script, we assume snapshot already has basic return fields
        # and we focus on cross-sectional style exposures here.
        
        # 3. Barra Style Exposures
        style_df = build_barra_style_exposures(df, industry_col="l1_name")
        
        # Save enriched snapshot
        feature_path = gold_root / f"trade_date={td}" / "factor_snapshot.parquet"
        style_df.to_parquet(feature_path, index=False, compression="zstd")
        
        # Save to marts for research panel
        marts_dir = marts_root / "research_panel"
        marts_dir.mkdir(parents=True, exist_ok=True)
        # We append or save per-day files that can be combined later
        style_df.to_parquet(marts_dir / f"stock_feature_{td}.parquet", index=False)

    logger.info("Feature building complete.")

if __name__ == "__main__":
    main()
