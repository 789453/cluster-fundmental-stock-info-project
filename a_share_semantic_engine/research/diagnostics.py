from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_data_quality_report(
    snap_df: pd.DataFrame,
    style_df: pd.DataFrame | None = None,
    vector_coverage: float = 1.0,
    output_path: Path | None = None,
) -> dict:
    """
    Build data quality report per the spec:
    - missing_rate per field
    - inf count
    - dtype check
    - financial announcement lag distribution
    - vector coverage score
    """
    report: dict = {
        "total_stocks": len(snap_df),
        "total_fields": len(snap_df.columns),
        "vector_coverage": float(vector_coverage),
    }

    missing_section = {}
    inf_section = {}
    dtype_section = {}

    for col in snap_df.columns:
        n_total = len(snap_df)
        n_missing = int(snap_df[col].isna().sum())
        n_inf = 0
        if snap_df[col].dtype in (np.float32, np.float64, np.float16):
            n_inf = int(np.isinf(snap_df[col]).sum())

        if n_missing > 0 or n_inf > 0:
            missing_section[col] = {
                "missing": n_missing,
                "pct": f"{100 * n_missing / max(n_total, 1):.2f}%",
            }
        if n_inf > 0:
            inf_section[col] = {"inf_count": n_inf}
        dtype_section[col] = str(snap_df[col].dtype)

    report["missing_by_field"] = missing_section
    report["inf_by_field"] = inf_section
    report["dtypes"] = dtype_section

    quality_flags = []
    for col, info in missing_section.items():
        pct = float(info["pct"].rstrip("%"))
        if pct > 50:
            quality_flags.append(f"HIGH: {col} missing {info['pct']}")
        elif pct > 20:
            quality_flags.append(f"MEDIUM: {col} missing {info['pct']}")

    report["quality_flags"] = quality_flags

    if "ann_date" in snap_df.columns and "trade_date" in snap_df.columns:
        ann_lag_days = []
        for _, row in snap_df.iterrows():
            try:
                ann = str(row.get("ann_date", ""))
                td = str(row.get("trade_date", ""))
                if len(ann) == 8 and len(td) == 8:
                    lag = int(td) - int(ann)
                    ann_lag_days.append(lag)
            except Exception:
                pass
        if ann_lag_days:
            report["financial_announcement_lag"] = {
                "mean_days": float(np.mean(ann_lag_days)),
                "median_days": float(np.median(ann_lag_days)),
                "p90_days": float(np.percentile(ann_lag_days, 90)),
                "max_days": int(max(ann_lag_days)),
            }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("data quality report saved to %s", output_path)

    logger.info("data quality: %d fields with missing, %d quality flags",
                len(missing_section), len(quality_flags))
    return report


def build_graph_quality_report(
    graphs: dict[str, tuple],
    ts_codes: list[str],
    output_path: Path | None = None,
) -> dict:
    """
    Build graph quality report:
    - per graph: edge_count, avg_degree, isolated_ratio, density
    """
    report: dict = {"graphs": {}}
    N = len(ts_codes)

    for name, (adj, *_) in graphs.items():
        from scipy import sparse as sp
        if not isinstance(adj, sp.csr_matrix):
            adj = sp.csr_matrix(adj)

        n_edges = int(adj.nnz) // 2
        degrees = np.asarray(adj.sum(axis=1)).flatten()
        avg_deg = float(degrees.mean())
        isolated = int(np.sum(degrees == 0))
        density = float(2 * n_edges / max(N * (N - 1), 1))

        report["graphs"][name] = {
            "edge_count": n_edges,
            "avg_degree": round(avg_deg, 3),
            "isolated_count": isolated,
            "isolated_ratio": f"{100 * isolated / max(N, 1):.2f}%",
            "density": round(density, 6),
        }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("graph quality report saved to %s", output_path)

    return report
