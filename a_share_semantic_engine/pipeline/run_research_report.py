from __future__ import annotations
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse

from ..research.report_builder import build_research_report
from ..viz.plots import save_all_figures
from ..graph.graph_metrics import compute_graph_metrics
from ..core.config import load_config
from ..core.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Generate research report")
    parser.add_argument("--trade-date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--config", type=str, default="configs/research.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = Path(cfg["paths"]["output_root"]) / "logs" / f"run_report_{args.trade_date}.log"
    logger = setup_logging("run_report", log_path)

    gold_root = Path(cfg["paths"]["gold_root"])
    export_dir = Path(cfg["paths"]["output_root"]) / "exports"
    report_dir = Path(cfg["paths"]["output_root"]) / "reports" / f"report_{args.trade_date}"
    figure_dir = report_dir / "figures"
    
    # 1. Load Data
    snapshot_path = gold_root / f"trade_date={args.trade_date}" / "factor_snapshot.parquet"
    cluster_path = export_dir / f"cluster_result_{args.trade_date}.parquet"
    graph_path = gold_root / "graphs" / f"trade_date={args.trade_date}" / "graph_fused.csr.npz"
    
    if not all(p.exists() for p in [snapshot_path, cluster_path, graph_path]):
        logger.error("Missing required files for report generation.")
        return

    df = pd.read_parquet(snapshot_path)
    cluster_df = pd.read_parquet(cluster_path)
    adj = sparse.load_npz(graph_path)
    
    # 2. Compute Graph Metrics
    logger.info("Computing graph metrics...")
    metrics_df = compute_graph_metrics(adj, df["ts_code"].tolist(), labels=cluster_df["cluster_id"].values, snap_df=df)
    
    # 3. Build Report
    logger.info("Building report...")
    report = build_research_report(
        trade_date=args.trade_date,
        snapshot_df=df,
        cluster_df=cluster_df,
        graph_stats_df=metrics_df,
        output_dir=report_dir
    )
    
    # 4. Generate Figures
    logger.info("Generating figures...")
    save_all_figures(
        cluster_labels=cluster_df["cluster_id"].values,
        degrees=metrics_df["degree"].values,
        style_df=df,
        industry_arr=df["l1_name"].values,
        output_dir=figure_dir,
        cluster_profile_df=pd.DataFrame(report["cluster_stats"])
    )
    
    logger.info(f"Research report and figures generated in {report_dir}")

if __name__ == "__main__":
    main()
