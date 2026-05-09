from __future__ import annotations
import argparse
import logging
from pathlib import Path

from ..data.snapshot_builder import build_stock_snapshot
from ..core.config import load_config
from ..core.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Build single-day stock snapshot")
    parser.add_argument("--trade-date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--config", type=str, default="configs/research.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["paths"]["gold_root"]) / f"trade_date={args.trade_date}"
    output_path = output_dir / "stock_snapshot.parquet"
    
    log_path = Path(cfg["paths"]["output_root"]) / "logs" / f"build_snapshot_{args.trade_date}.log"
    logger = setup_logging("build_snapshot", log_path)

    warehouse_db = cfg["paths"]["warehouse_db"]
    
    build_stock_snapshot(
        trade_date=args.trade_date,
        warehouse_db=warehouse_db,
        output_path=output_path,
        include_st=cfg["project"].get("include_st", False),
        include_suspended=cfg["project"].get("include_suspended", False)
    )

if __name__ == "__main__":
    main()
