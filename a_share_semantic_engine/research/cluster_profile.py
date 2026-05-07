from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_cluster_profile(
    labels: np.ndarray,
    ts_codes: list[str],
    snap_df: pd.DataFrame,
    style_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build per-cluster profile: size, industry composition, avg financial features, top stocks.

    Args:
        labels: cluster labels aligned with ts_codes.
        ts_codes: list of stock codes.
        snap_df: snapshot with stock features.
        style_df: optional DataFrame with style exposures.

    Returns:
        DataFrame with cluster profiles.
    """
    n_clusters = int(labels.max()) + 1
    records = []

    for cid in range(n_clusters):
        mask = labels == cid
        cnt = int(mask.sum())
        if cnt == 0:
            continue

        cluster_stocks = snap_df[mask].copy()
        rec: dict = {
            "cluster_id": cid,
            "count": cnt,
        }

        if "l1_name" in cluster_stocks.columns:
            ind_counts = cluster_stocks["l1_name"].value_counts()
            rec["top_sw_l1"] = ind_counts.index[0] if len(ind_counts) > 0 else ""
            rec["top_sw_l1_pct"] = float(ind_counts.iloc[0] / cnt) if len(ind_counts) > 0 else 0.0
            rec["sw_l1_entropy"] = float(-(ind_counts / cnt * np.log(ind_counts / cnt + 1e-10)).sum())

        if "industry" in cluster_stocks.columns:
            ind2 = cluster_stocks["industry"].value_counts()
            rec["top_industry"] = ind2.index[0] if len(ind2) > 0 else ""
            rec["top_industry_pct"] = float(ind2.iloc[0] / cnt) if len(ind2) > 0 else 0.0

        for col in ["pct_chg", "roe", "pe_ttm", "pb", "total_mv", "turnover_rate"]:
            if col in cluster_stocks.columns:
                vals = cluster_stocks[col].dropna()
                rec[f"avg_{col}"] = float(vals.mean()) if len(vals) > 0 else float("nan")
                rec[f"std_{col}"] = float(vals.std()) if len(vals) > 0 else float("nan")

        if style_df is not None and len(style_df) == len(labels):
            cluster_styles = style_df[mask]
            style_cols = [c for c in cluster_styles.columns if c not in ["ts_code", "trade_date", "cluster_id"]]
            for sc in style_cols:
                vals = cluster_styles[sc].dropna()
                rec[f"avg_{sc}"] = float(vals.mean()) if len(vals) > 0 else float("nan")

        top_stock_mask = cluster_stocks.nlargest(3, "close")["ts_code"].tolist() if "close" in cluster_stocks.columns else []
        rec["prototype_stocks"] = ",".join(top_stock_mask)

        records.append(rec)

    df = pd.DataFrame(records)
    logger.info("cluster_profile: %d clusters", len(df))
    return df


def build_cluster_diagnostics(
    labels: np.ndarray,
    features: np.ndarray,
    adj: pd.DataFrame | None = None,
) -> dict:
    """
    Compute cluster-level structural diagnostics.

    Args:
        labels: cluster labels.
        features: feature matrix aligned with labels.
        adj: optional adjacency info (DataFrame with src_ts_code, dst_ts_code, weight).

    Returns:
        dict with silhouette, davies_bouldin, calinski_harabasz, modularity estimates.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    diag: dict = {}

    try:
        sil = silhouette_score(features, labels, sample_size=min(5000, len(features)))
        diag["silhouette_score"] = float(sil)
    except Exception as e:
        logger.warning("silhouette_score failed: %s", e)
        diag["silhouette_score"] = float("nan")

    try:
        db = davies_bouldin_score(features, labels)
        diag["davies_bouldin_score"] = float(db)
    except Exception as e:
        logger.warning("davies_bouldin_score failed: %s", e)
        diag["davies_bouldin_score"] = float("nan")

    try:
        ch = calinski_harabasz_score(features, labels)
        diag["calinski_harabasz_score"] = float(ch)
    except Exception as e:
        logger.warning("calinski_harabasz_score failed: %s", e)
        diag["calinski_harabasz_score"] = float("nan")

    diag["n_clusters"] = int(labels.max()) + 1
    diag["cluster_size_std"] = float(pd.Series(labels).value_counts().std())
    diag["cluster_size_min"] = int(pd.Series(labels).value_counts().min())
    diag["cluster_size_max"] = int(pd.Series(labels).value_counts().max())

    return diag
