from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_research_report(
    trade_date: str,
    snapshot_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    style_df: pd.DataFrame | None = None,
    graph_stats_df: pd.DataFrame | None = None,
    community_labels: np.ndarray | None = None,
    modularity: float = 0.0,
    nmi: float = 0.0,
    purity: float = 0.0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build a structured research report for a single trade_date.

    Args:
        trade_date: YYYYMMDD string.
        snapshot_df: snapshot with stock info, industry, features.
        cluster_df: DataFrame with ts_code, cluster_id, cluster_distance.
        style_df: optional DataFrame with Barra style exposures.
        graph_stats_df: optional DataFrame with per-stock graph metrics.
        community_labels: optional array of community labels aligned with snapshot rows.
        modularity: graph modularity score.
        nmi: NMI between semantic clusters and SW industry.
        purity: cluster purity vs SW industry.
        output_dir: if provided, save report JSON and figures here.

    Returns:
        Dictionary report.
    """
    report: dict[str, Any] = {
        "trade_date": trade_date,
        "generated_by": "a_share_semantic_engine",
        "universe_size": len(snapshot_df),
        "modularity": float(modularity),
        "nmi_sw_l1": float(nmi),
        "purity_sw_l1": float(purity),
    }

    snapshot_df = snapshot_df.copy()

    if "cluster_id" in cluster_df.columns:
        # Ensure ts_code is string and remove potential duplicates
        cluster_sub = cluster_df[["ts_code", "cluster_id", "cluster_distance"]].copy()
        cluster_sub["ts_code"] = cluster_sub["ts_code"].astype(str)
        
        snap_copy = snapshot_df.copy()
        snap_copy["ts_code"] = snap_copy["ts_code"].astype(str)
        
        # Merge using ts_code
        merged = snap_copy.merge(cluster_sub, on="ts_code", how="left")
        
        # Fill missing clusters with -1
        if "cluster_id" in merged.columns:
            merged["cluster_id"] = merged["cluster_id"].fillna(-1).astype(int)
        else:
            merged["cluster_id"] = -1
    else:
        merged = snapshot_df.copy()
        merged["cluster_id"] = -1

    # Count actual clusters (excluding -1)
    unique_cids = sorted([int(c) for c in merged["cluster_id"].unique() if c >= 0])
    n_clusters = len(unique_cids)
    report["n_clusters"] = n_clusters
    report["n_unclustered"] = int((merged["cluster_id"] < 0).sum())

    cluster_stats = []
    if "cluster_id" in merged.columns:
        for cid in range(n_clusters):
            mask = merged["cluster_id"] == cid
            cnt = int(mask.sum())
            sub = merged[mask]
            ind_counts = sub["l1_name"].value_counts() if "l1_name" in sub.columns else pd.Series()
            sw_counts = sub["industry"].value_counts() if "industry" in sub.columns else pd.Series()
            avg_ret = float(sub["pct_chg"].mean()) if "pct_chg" in sub.columns else 0.0
            avg_pe = float(sub["pe_ttm"].mean()) if "pe_ttm" in sub.columns else 0.0
            avg_roe = float(sub["roe"].mean()) if "roe" in sub.columns else 0.0
            cluster_stats.append({
                "cluster_id": cid,
                "count": cnt,
                "avg_return_1d": avg_ret,
                "avg_pe_ttm": avg_pe,
                "avg_roe": avg_roe,
                "top_sw_l1": ind_counts.index[0] if len(ind_counts) > 0 else "",
                "top_sw_l1_pct": float(ind_counts.iloc[0] / cnt) if len(ind_counts) > 0 and cnt > 0 else 0.0,
                "top_industry": sw_counts.index[0] if len(sw_counts) > 0 else "",
                "top_industry_pct": float(sw_counts.iloc[0] / cnt) if len(sw_counts) > 0 and cnt > 0 else 0.0,
            })

    report["cluster_stats"] = cluster_stats

    if "l1_name" in merged.columns:
        industry_stats = (
            merged.groupby("l1_name")
            .agg(
                count=("ts_code", "count"),
                avg_return=("pct_chg", "mean"),
                avg_pe=("pe_ttm", "mean"),
                avg_roe=("roe", "mean"),
            )
            .reset_index()
            .rename(columns={"l1_name": "sw_l1_name"})
            .to_dict("records")
        )
        report["industry_stats"] = industry_stats

    if style_df is not None and not style_df.empty:
        style_cols = [c for c in style_df.columns if c not in ["ts_code", "trade_date", "cluster_id"]]
        cluster_style = (
            style_df.groupby("cluster_id")[style_cols]
            .mean()
            .reset_index()
            .to_dict("records")
        )
        report["cluster_style_exposures"] = cluster_style

    if graph_stats_df is not None and not graph_stats_df.empty:
        graph_summary = {
            "avg_degree": float(graph_stats_df["degree"].mean()) if "degree" in graph_stats_df.columns else 0.0,
            "median_degree": float(graph_stats_df["degree"].median()) if "degree" in graph_stats_df.columns else 0.0,
            "avg_pagerank": float(graph_stats_df["pagerank"].mean()) if "pagerank" in graph_stats_df.columns else 0.0,
            "isolated_count": int((graph_stats_df["degree"] == 0).sum()) if "degree" in graph_stats_df.columns else 0,
        }
        report["graph_summary"] = graph_summary

    missing_summary = {}
    for col in ["close", "pct_chg", "pe_ttm", "roe", "l1_name"]:
        if col in merged.columns:
            miss = int(merged[col].isna().sum())
            missing_summary[col] = {"missing": miss, "total": len(merged), "pct": f"{100*miss/max(len(merged),1):.1f}%"}
    report["data_quality"] = missing_summary

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "research_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Research report saved to %s", report_path)

    return report


def print_report_summary(report: dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  Research Report — {report.get('trade_date', 'N/A')}")
    print(f"{'='*60}")
    print(f"  Universe size:     {report.get('universe_size', 'N/A')}")
    print(f"  Clusters:           {report.get('n_clusters', 'N/A')}")
    print(f"  Modularity:         {report.get('modularity', 0):.4f}")
    print(f"  NMI (vs SW-L1):     {report.get('nmi_sw_l1', 0):.4f}")
    print(f"  Purity (vs SW-L1):  {report.get('purity_sw_l1', 0):.4f}")
    if "graph_summary" in report:
        gs = report["graph_summary"]
        print(f"  Avg Degree:         {gs.get('avg_degree', 0):.2f}")
        print(f"  Isolated Nodes:     {gs.get('isolated_count', 0)}")
    print(f"\n  Top Clusters by Size:")
    cs = report.get("cluster_stats", [])
    top = sorted(cs, key=lambda x: x["count"], reverse=True)[:5]
    for c in top:
        print(f"    Cluster {c['cluster_id']:>3}: {c['count']:>4} stocks | "
              f"Industry: {c['top_sw_l1']:<10} ({c['top_sw_l1_pct']:>4.0%}) | "
              f"ROE: {c['avg_roe']:>7.2f}%")
    print(f"{'='*60}\n")
