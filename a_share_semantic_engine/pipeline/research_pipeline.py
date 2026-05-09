from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from ..data.snapshot_builder import build_stock_snapshot
from ..features.barra_style import build_barra_style_exposures
from ..features.clustering import (
    cluster_summary, compute_modularity, compute_nmi, compute_purity,
    run_hdbscan, run_kmeans, run_leiden, run_louvain,
)
from ..graph.edge_table import build_edge_table, save_all_graphs
from ..graph.graph_builder import (
    build_csr_from_knn, build_fundamental_graph, build_industry_graph,
    build_return_corr_graph, build_semantic_knn, build_style_graph,
    fuse_multiplex_graphs,
)
from ..graph.graph_metrics import compute_graph_metrics, compute_conductance
from ..graph.smoothing import graph_smooth
from ..research.cluster_profile import build_cluster_diagnostics, build_cluster_profile
from ..research.diagnostics import build_data_quality_report, build_graph_quality_report
from ..research.report_builder import build_research_report, print_report_summary
from ..viz.plots import save_all_figures

logger = logging.getLogger(__name__)


def run_research_pipeline(
    trade_date: str,
    warehouse_db: str | Path,
    npy_root: str | Path,
    output_dir: str | Path,
    k_semantic: int = 30,
    k_industry: int = 20,
    k_fundamental: int = 20,
    k_return: int = 20,
    k_style: int = 20,
    n_clusters: int | None = None,
    cluster_method: str = "kmeans",
    fusion_weights: dict[str, float] | None = None,
    use_cuda: bool = True,
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> dict:
    t0 = time.time()
    output_dir = Path(output_dir)
    
    if run_dir:
        run_dir = Path(run_dir)
    elif resume:
        # Find latest run for this trade_date
        existing_runs = sorted(output_dir.glob(f"run_{trade_date}_*"))
        if existing_runs:
            run_dir = existing_runs[-1]
            logger.info("Resuming from latest run: %s", run_dir)
        else:
            run_id = f"run_{trade_date}_{int(time.time())}"
            run_dir = output_dir / run_id
    else:
        run_id = f"run_{trade_date}_{int(time.time())}"
        run_dir = output_dir / run_id

    # Ensure all subdirectories exist
    for sub in ["reports", "exports", "figures", "graphs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Research pipeline | trade_date=%s | cluster_method=%s", trade_date, cluster_method)
    logger.info("=" * 60)

    # 1. Snapshot
    snap_path = run_dir / "snapshot.parquet"
    if snap_path.exists():
        logger.info("[1/9] Loading existing snapshot from %s", snap_path)
        snap = pd.read_parquet(snap_path)
    else:
        logger.info("[1/9] Building stock snapshot...")
        from ..data.snapshot_builder import build_stock_snapshot
        snap = build_stock_snapshot(trade_date, warehouse_db)
        snap.to_parquet(snap_path)
    
    ts_codes = snap["ts_code"].tolist()
    N = len(snap)
    logger.info("  snapshot: %d stocks", N)
    
    if n_clusters is None:
        n_clusters = max(10, int(N / 50))

    # 2. Barra-style exposures
    style_path = run_dir / "snapshot_with_style.parquet"
    if style_path.exists():
        logger.info("[2/9] Loading existing style exposures from %s", style_path)
        snap = pd.read_parquet(style_path)
    else:
        logger.info("[2/9] Calculating Barra-style exposures...")
        snap = build_barra_style_exposures(snap, industry_col="l1_name")
        snap.to_parquet(style_path)

    style_cols = [c for c in snap.columns if c.endswith("_sw_neutral") or c in [
        "Size", "NonlinearSize", "Value", "Momentum", "ShortReversal",
        "Volatility", "Liquidity", "Profitability", "Growth", "Leverage", "EarningsQuality",
    ]]

    # 3. Semantic Vectors
    feat_mat_path = run_dir / "feature_matrix.npy"
    if feat_mat_path.exists():
        logger.info("[3/9] Loading existing feature matrix from %s", feat_mat_path)
        feat_matrix = np.load(feat_mat_path)
    else:
        logger.info("[3/9] Loading semantic vectors...")
        from ..graph.graph_builder import VectorStore
        vs = VectorStore(npy_root)
        available_views = ["full_text", "profile_text", "product_text", "raw_json"]
        view_to_use = None
        for v in available_views:
            try:
                vs.load_view(v)
                view_to_use = v
                logger.info("  using view '%s'", v)
                break
            except FileNotFoundError:
                continue

        if view_to_use:
            vectors, row_ids, stock_codes_vec = vs.load_view(view_to_use)
            key_to_vec_idx = {}
            for i, rid in enumerate(row_ids):
                if rid: key_to_vec_idx[rid] = i
            for i, code in enumerate(stock_codes_vec):
                if code: key_to_vec_idx[code] = i
                
            feat_matrix = np.zeros((N, vectors.shape[1]), dtype=np.float32)
            matched_count = 0
            for i, row in snap.iterrows():
                idx = None
                if "record_id" in row and row["record_id"] in key_to_vec_idx:
                    idx = key_to_vec_idx[row["record_id"]]
                elif row["ts_code"] in key_to_vec_idx:
                    idx = key_to_vec_idx[row["ts_code"]]
                if idx is not None:
                    feat_matrix[i] = vectors[idx]
                    matched_count += 1
            if matched_count > 0:
                logger.info("  aligned %d/%d stocks with semantic vectors", matched_count, N)
            else:
                feat_matrix = vectors[:N]
        else:
            fcols = ["close", "pct_chg", "turnover_rate", "pe_ttm", "roe", "total_mv"]
            feat_matrix = snap[[c for c in fcols if c in snap.columns]].values.astype(np.float32)
            feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

        feat_matrix = vs.l2_normalize(feat_matrix)
        np.save(feat_mat_path, feat_matrix)

    # 4. Building all graphs
    graph_dir = run_dir / "graphs"
    if graph_dir.exists() and any(graph_dir.glob("*.npz")):
        logger.info("[4/9] Loading existing graphs from %s", graph_dir)
        from ..graph.edge_table import load_all_graphs
        loaded_csrs = load_all_graphs(graph_dir)
        # Note: We reconstruct the graph tuples for the pipeline
        graphs = {k: (v, None, None, None) for k, v in loaded_csrs.items()}
    else:
        logger.info("[4/9] Building all graphs...")
        graphs: dict[str, tuple] = {}

        neigh_sem, dist_sem, w_sem = build_semantic_knn(
            feat_matrix, k=k_semantic, min_sim=0.0, mutual=True,
            backend="faiss_gpu" if (use_cuda) else "exact",
        )
        sem_csr, _, _ = build_csr_from_knn(neigh_sem, dist_sem, N)
        graphs["semantic"] = (sem_csr, neigh_sem, dist_sem, w_sem)

        sw_l1 = snap["l1_name"].fillna("").tolist()
        sw_l2 = snap["l2_name"].fillna("").tolist()
        sw_l3 = snap["l3_name"].fillna("").tolist()
        neigh_ind, dist_ind, w_ind = build_industry_graph(ts_codes, sw_l1, sw_l2, sw_l3)
        ind_csr, _, _ = build_csr_from_knn(neigh_ind, dist_ind, N)
        graphs["industry"] = (ind_csr, neigh_ind, dist_ind, w_ind)

        fin_cols = [c for c in snap.columns if c in [
            "roe", "roa", "roic", "grossprofit_margin", "debt_to_assets",
            "turnover_rate", "pe_ttm", "pb", "netprofit_margin",
        ]]
        if len(fin_cols) >= 3:
            fin_mat = snap[fin_cols].values.astype(np.float32)
            fin_mat = np.nan_to_num(fin_mat, nan=0.0)
            neigh_fin, dist_fin, w_fin = build_fundamental_graph(fin_mat, k=k_fundamental)
            fin_csr, _, _ = build_csr_from_knn(neigh_fin, dist_fin, N)
            graphs["fundamental"] = (fin_csr, neigh_fin, dist_fin, w_fin)
        else:
            graphs["fundamental"] = graphs["semantic"]

        ret_cols = [f"ret_{d}d" for d in [1, 5, 20, 60]]
        avail_ret = [c for c in ret_cols if c in snap.columns]
        if len(avail_ret) >= 2:
            ret_mat = snap[avail_ret].values.astype(np.float32)
            ret_mat = np.nan_to_num(ret_mat, nan=0.0)
            neigh_ret, dist_ret, w_ret = build_return_corr_graph(ret_mat.T, k=k_return)
            ret_csr, _, _ = build_csr_from_knn(neigh_ret, dist_ret, N)
            graphs["return_corr"] = (ret_csr, neigh_ret, dist_ret, w_ret)
        else:
            graphs["return_corr"] = graphs["semantic"]

        if len(style_cols) > 0:
            style_mat = snap[style_cols].values.astype(np.float32)
            style_mat = np.nan_to_num(style_mat, nan=0.0)
            neigh_sty, dist_sty, w_sty = build_style_graph(style_mat, k=k_style)
            sty_csr, _, _ = build_csr_from_knn(neigh_sty, dist_sty, N)
            graphs["style"] = (sty_csr, neigh_sty, dist_sty, w_sty)
        else:
            graphs["style"] = graphs["semantic"]

        # 5. Fusing graphs
        logger.info("[5/9] Fusing graphs...")
        if fusion_weights is None:
            fusion_weights = {"semantic": 0.35, "industry": 0.20, "fundamental": 0.20, "return_corr": 0.15, "style": 0.10}
        graph_dict_fuse = {k: (v[1], v[2]) for k, v in graphs.items() if v[1] is not None}
        if not graph_dict_fuse:
             # If loading from cache, we only have CSR. Use CSR for fusion if possible, or skip
             fused_csr = graphs["semantic"][0] # fallback
        else:
             fused_neigh, fused_dists, fused_sim = fuse_multiplex_graphs(graph_dict_fuse, fusion_weights)
             fused_csr, _, _ = build_csr_from_knn(fused_neigh, fused_dists, N)
        
        graphs["fused"] = (fused_csr, None, None, None)
        save_all_graphs({k: v[0] for k, v in graphs.items()}, run_dir / "graphs")

    logger.info("[6/9] Clustering (%s)...", cluster_method)
    if cluster_method == "kmeans":
        labels, dists = run_kmeans(feat_matrix, n_clusters=n_clusters)
        community_labels = None
    elif cluster_method == "hdbscan":
        labels, probs = run_hdbscan(feat_matrix)
        community_labels = None
        dists = probs
    elif cluster_method in ("leiden", "louvain"):
        community_labels = run_leiden(fused_csr) if cluster_method == "leiden" else run_louvain(fused_csr)
        labels = community_labels
        dists = np.zeros(N, dtype=np.float32)
    else:
        labels, dists = run_kmeans(feat_matrix, n_clusters=n_clusters)
        community_labels = None

    n_actual_clusters = int(labels.max()) + 1
    logger.info("  clusters: %d", n_actual_clusters)

    # Ensure we have the fused CSR for metrics
    fused_csr = graphs["fused"][0]

    logger.info("[7/9] Computing graph + cluster metrics...")
    snap["cluster_id"] = labels

    style_df = snap[["ts_code", "trade_date", "cluster_id"] + style_cols].copy() if style_cols else None

    graph_metrics_df = compute_graph_metrics(
        fused_csr, ts_codes,
        labels=labels,
        style_df=style_df,
        snap_df=snap,
    )

    diag = build_cluster_diagnostics(labels, feat_matrix)
    logger.info("  silhouette=%.3f davies_bouldin=%.3f calinski=%.0f",
                diag.get("silhouette_score", 0), diag.get("davies_bouldin_score", 0), diag.get("calinski_harabasz_score", 0))

    modularity = compute_modularity(fused_csr, labels)
    conductance = compute_conductance(fused_csr, labels)

    sw_labels = np.zeros(N, dtype=np.int32)
    if "l1_name" in snap.columns:
        unique_ind = {n: i for i, n in enumerate(snap["l1_name"].fillna("UNKNOWN").unique())}
        sw_labels = np.array([unique_ind.get(n, 0) for n in snap["l1_name"].fillna("UNKNOWN")])

    nmi_l1 = compute_nmi(labels, sw_labels)
    purity_l1 = compute_purity(labels, sw_labels)

    sw_l2_labels = np.zeros(N, dtype=np.int32)
    if "l2_name" in snap.columns:
        unique_l2 = {n: i for i, n in enumerate(snap["l2_name"].fillna("UNKNOWN").unique())}
        sw_l2_labels = np.array([unique_l2.get(n, 0) for n in snap["l2_name"].fillna("UNKNOWN")])
    nmi_l2 = compute_nmi(labels, sw_l2_labels)

    cluster_df = pd.DataFrame({
        "ts_code": ts_codes,
        "cluster_id": labels,
        "cluster_distance": dists,
    })
    if community_labels is not None:
        cluster_df["community_id"] = community_labels

    logger.info("[8/9] Building reports...")
    cluster_panel = cluster_summary(labels, ts_codes, snap, industry_col="l1_name")

    data_q_report = build_data_quality_report(
        snap, style_df,
        vector_coverage=1.0 if view_to_use else 0.0,
        output_path=run_dir / "reports" / "data_quality_report.json",
    )

    graph_q_report = build_graph_quality_report(
        {k: (v[0],) for k, v in graphs.items()},
        ts_codes,
        output_path=run_dir / "reports" / "graph_quality_report.json",
    )

    cluster_profile_df = build_cluster_profile(labels, ts_codes, snap, style_df)

    report = build_research_report(
        trade_date=trade_date,
        snapshot_df=snap,
        cluster_df=cluster_df,
        style_df=style_df,
        graph_stats_df=graph_metrics_df,
        community_labels=community_labels,
        modularity=modularity,
        nmi=nmi_l1,
        purity=purity_l1,
        output_dir=run_dir / "reports",
    )

    report["cluster_diagnostics"] = diag
    report["conductance"] = float(conductance)
    report["nmi_sw_l2"] = float(nmi_l2)
    with open(run_dir / "reports" / "research_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("[9/9] Saving outputs + figures...")
    cluster_df.to_csv(run_dir / "exports" / "cluster_result.csv", index=False)
    cluster_df.to_parquet(run_dir / "exports" / "cluster_result.parquet", index=False, compression="zstd")
    graph_metrics_df.to_csv(run_dir / "exports" / "graph_stats.csv", index=False)
    graph_metrics_df.to_parquet(run_dir / "exports" / "graph_stats.parquet", index=False, compression="zstd")
    cluster_panel.to_csv(run_dir / "exports" / "cluster_panel.csv", index=False)
    cluster_profile_df.to_csv(run_dir / "exports" / "cluster_profile.csv", index=False)
    if style_df is not None:
        style_df.to_csv(run_dir / "exports" / "style_exposure_panel.csv", index=False)
        style_df.to_parquet(run_dir / "exports" / "style_exposure_panel.parquet", index=False, compression="zstd")
    snap[["ts_code", "name", "industry", "l1_name", "close", "pct_chg", "roe", "pe_ttm", "cluster_id"]].to_csv(
        run_dir / "exports" / "stock_snapshot_preview.csv", index=False
    )

    edge_tables = {}
    for name, adj in {k: v[0] for k, v in graphs.items()}.items():
        et = build_edge_table(adj, ts_codes, edge_type=name)
        p = run_dir / "exports" / f"edge_table_{name}.parquet"
        et.to_parquet(p, index=False, compression="zstd")
        edge_tables[name] = p

    save_all_figures(labels, graph_metrics_df["degree"].values, style_df, np.array(sw_l1), run_dir / "figures")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline done in %.1fs | %d stocks | %d clusters | modularity=%.4f | NMI=%.4f",
                elapsed, N, n_actual_clusters, modularity, nmi_l1)
    logger.info("Outputs: %s", run_dir)
    logger.info("=" * 60)

    print_report_summary(report)

    return report
