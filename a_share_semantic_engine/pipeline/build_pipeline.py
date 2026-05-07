from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.config import save_config
from ..core.logging_utils import setup_logging
from ..data.loaders import load_dataset
from ..data.validation import validate_required_columns, validate_row_alignment
from ..features.anchors import encode_anchors
from ..features.clustering import run_clustering
from ..features.common import l2_normalize
from ..features.fusion import mean_fuse, weighted_concat
from ..features.npy_features import reduce_npy_view
from ..features.prototypes import build_prototypes, compute_prototype_exposure
from ..features.quality import build_quality_features
from ..features.text_embedding import build_view_embeddings_from_csv
from ..graph.fusion import fuse_graphs
from ..graph.knn_graph import build_knn_graph
from ..graph.smoothing import graph_smooth
from ..graph.stats import compute_graph_stats
from ..pipeline.export import save_joblib, save_sparse_graph, save_table, plot_cluster_sizes, plot_degree_hist
from ..utils.io import ensure_dir, write_json


def _build_view_features(dataset, cfg, logger):
    view_cfg = cfg["views"]
    selected_views = list(dict.fromkeys(view_cfg["text_views"] + view_cfg["json_views"]))

    csv_views = [v for v in selected_views if not dataset.has_view(v)]
    npy_views = [v for v in selected_views if dataset.has_view(v)]

    matrices = {}
    reducers = {}

    if csv_views:
        logger.info("building csv text embeddings for %s views", len(csv_views))
        csv_mats, csv_models = build_view_embeddings_from_csv(dataset.records, csv_views, cfg, logger)
        matrices.update(csv_mats)
        reducers.update(csv_models)

    if npy_views:
        compact_dim = int(cfg["embedding"]["compact_dim"])
        random_state = int(cfg["project"]["random_state"])
        for view in tqdm(npy_views, desc="reduce_npy_views"):
            mat = dataset.get_view_matrix(view)
            Z, reducer = reduce_npy_view(mat, compact_dim=compact_dim, random_state=random_state)
            matrices[view] = Z
            reducers[view] = reducer
            logger.info("npy compact view=%s shape=%s explained=%s", view, tuple(Z.shape), reducer["explained_variance"])

    return matrices, reducers


def _group_features(mats: dict[str, np.ndarray], cfg: dict):
    text_views = [v for v in cfg["views"]["text_views"] if v in mats]
    json_views = [v for v in cfg["views"]["json_views"] if v in mats]

    text_mat = mean_fuse([mats[v] for v in text_views]) if text_views else None
    json_mat = mean_fuse([mats[v] for v in json_views]) if json_views else None

    if text_mat is None and json_mat is None:
        raise ValueError("no feature matrices available")
    if text_mat is None:
        text_mat = json_mat.copy()
    if json_mat is None:
        json_mat = text_mat.copy()

    product_views = [v for v in ("product_text", "core_products_services_json", "application_scenarios_json") if v in mats]
    model_chain_views = [v for v in ("model_text", "chain_text", "business_model_json", "industry_chain_position_json", "business_logic_json") if v in mats]
    theme_views = [v for v in ("theme_text", "structural_themes_json") if v in mats]

    z_product = mean_fuse([mats[v] for v in product_views]) if product_views else text_mat.copy()
    z_model_chain = mean_fuse([mats[v] for v in model_chain_views]) if model_chain_views else text_mat.copy()
    z_theme = mean_fuse([mats[v] for v in theme_views]) if theme_views else text_mat.copy()

    return {
        "z_text": l2_normalize(text_mat),
        "z_json": l2_normalize(json_mat),
        "z_product": l2_normalize(z_product),
        "z_model_chain": l2_normalize(z_model_chain),
        "z_theme": l2_normalize(z_theme),
    }


def run_pipeline(cfg: dict) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["paths"]["output_root"]) / f"run_{ts}"
    ensure_dir(run_dir)

    logger = setup_logging("pipeline", run_dir / "logs" / "run.log")
    save_config(cfg, run_dir / "config_snapshot.yaml")

    logger.info("load dataset")
    dataset = load_dataset(cfg, logger)
    validate_required_columns(dataset.records)
    if dataset.row_ids_by_view:
        validate_row_alignment(dataset.records, dataset.row_ids_by_view)

    record_ids = dataset.records["record_id"].astype(str).tolist()
    base_df = dataset.records[["record_id", "stock_code", "stock_name", "asof_date"]].copy()

    logger.info("build view features")
    mats, reducers = _build_view_features(dataset, cfg, logger)

    logger.info("build anchor and quality features")
    anchor_mat, anchor_names = encode_anchors(dataset.records, cfg["views"]["anchor_fields"])
    quality_df, quality_mat = build_quality_features(dataset.records)

    grouped = _group_features(mats, cfg)
    z_semantic_core = weighted_concat(
        text_mat=grouped["z_text"],
        json_mat=grouped["z_json"],
        anchor_mat=anchor_mat,
        quality_mat=quality_mat,
        weights=cfg["fusion"],
    )

    logger.info("run clustering")
    cluster_df, cluster_model = run_clustering(z_semantic_core, cfg, record_ids)

    logger.info("build prototypes")
    prototypes = build_prototypes(z_semantic_core, cluster_df["cluster_id"].to_numpy())
    proto_df = compute_prototype_exposure(z_semantic_core, prototypes, record_ids)

    logger.info("build graphs")
    graph_cfg = cfg["graph"]
    semantic_graph, _ = build_knn_graph(grouped["z_text"], k=int(graph_cfg["k"]), min_sim=float(graph_cfg["min_sim"]), mutual=bool(graph_cfg["mutual"]))
    product_graph, _ = build_knn_graph(grouped["z_product"], k=int(graph_cfg["k"]), min_sim=float(graph_cfg["min_sim"]), mutual=bool(graph_cfg["mutual"]))
    model_chain_graph, _ = build_knn_graph(grouped["z_model_chain"], k=int(graph_cfg["k"]), min_sim=float(graph_cfg["min_sim"]), mutual=bool(graph_cfg["mutual"]))
    theme_graph, _ = build_knn_graph(grouped["z_theme"], k=int(graph_cfg["k"]), min_sim=float(graph_cfg["min_sim"]), mutual=bool(graph_cfg["mutual"]))

    fused_graph = fuse_graphs(
        {
            "semantic_graph": semantic_graph,
            "product_graph": product_graph,
            "model_chain_graph": model_chain_graph,
            "theme_graph": theme_graph,
        },
        cfg["graph"]["weights"],
    )

    graph_stats = compute_graph_stats(fused_graph, record_ids)
    z_smoothed = graph_smooth(
        fused_graph,
        z_semantic_core,
        alpha=float(graph_cfg["smooth_alpha"]),
        steps=int(graph_cfg["smooth_steps"]),
    )

    logger.info("assemble export tables")
    feature_cols = [f"sem_{i:03d}" for i in range(z_semantic_core.shape[1])]
    smooth_cols = [f"sem_smooth_{i:03d}" for i in range(z_smoothed.shape[1])]
    feature_df = pd.concat(
        [
            base_df,
            pd.DataFrame(z_semantic_core, columns=feature_cols),
            pd.DataFrame(z_smoothed, columns=smooth_cols),
            quality_df.reset_index(drop=True),
        ],
        axis=1,
    )

    cluster_export = base_df.merge(cluster_df, on="record_id", how="left")
    proto_export = base_df.merge(proto_df, on="record_id", how="left")
    graph_export = base_df.merge(graph_stats, on="record_id", how="left")

    exp_cfg = cfg["export"]
    save_table(feature_df, run_dir / "exports" / "semantic_feature_table", exp_cfg["write_parquet"], exp_cfg["write_csv"])
    save_table(cluster_export, run_dir / "exports" / "cluster_result", exp_cfg["write_parquet"], exp_cfg["write_csv"])
    save_table(proto_export, run_dir / "exports" / "prototype_exposure", exp_cfg["write_parquet"], exp_cfg["write_csv"])
    save_table(graph_export, run_dir / "exports" / "graph_stats", exp_cfg["write_parquet"], exp_cfg["write_csv"])

    save_sparse_graph(fused_graph, run_dir / "graphs" / "fused_graph.npz")
    save_joblib(cluster_model, run_dir / "models" / "cluster_model.joblib")
    save_joblib(reducers, run_dir / "models" / "reducers.joblib")
    save_joblib({"prototypes": prototypes, "anchor_names": anchor_names}, run_dir / "models" / "semantic_state.joblib")

    if exp_cfg["write_figures"]:
        plot_cluster_sizes(cluster_export, run_dir / "figures" / "cluster_sizes.png")
        plot_degree_hist(graph_export, run_dir / "figures" / "degree_hist.png")

    report = {
        "run_dir": str(run_dir.resolve()),
        "rows": int(len(base_df)),
        "num_feature_dims": int(z_semantic_core.shape[1]),
        "num_clusters": int(cluster_export["cluster_id"].nunique()),
        "graph_edges": int(fused_graph.nnz),
        "avg_degree": float(graph_export["degree"].mean()),
        "views_used": sorted(mats.keys()),
    }
    write_json(report, run_dir / "reports" / "run_report.json")
    logger.info("pipeline done | %s", report)
    return report
