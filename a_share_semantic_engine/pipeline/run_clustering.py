from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..features.clustering import run_kmeans, run_leiden
from ..semantic.vector_store import load_semantic_vectors, align_vectors_to_snapshot
from ..core.config import load_config
from ..core.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run clustering and community detection")
    parser.add_argument("--trade-date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--method", type=str, choices=["kmeans", "leiden", "louvain"], default="kmeans")
    parser.add_argument("--config", type=str, default="configs/research.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = Path(cfg["paths"]["output_root"]) / "logs" / f"run_clustering_{args.trade_date}.log"
    logger = setup_logging("run_clustering", log_path)

    gold_root = Path(cfg["paths"]["gold_root"])
    npy_root = Path(cfg["paths"]["npy_root"])
    
    snapshot_path = gold_root / f"trade_date={args.trade_date}" / "factor_snapshot.parquet"
    if not snapshot_path.exists():
        logger.error(f"Factor snapshot not found for {args.trade_date}")
        return

    df = pd.read_parquet(snapshot_path)
    record_ids = df["record_id"].tolist()
    
    if args.method == "kmeans":
        logger.info("Running FAISS KMeans...")
        view_name = cfg["views"]["text_views"][0]
        vector_block = load_semantic_vectors(npy_root, view_name)
        aligned_mat = align_vectors_to_snapshot(vector_block, df)
        labels, dists = run_kmeans(aligned_mat, n_clusters=int(cfg["cluster"]["n_clusters"]), use_gpu=cfg["project"].get("use_gpu", True))
        cluster_df = pd.DataFrame({"record_id": record_ids, "cluster_id": labels, "cluster_distance": dists})
    else:
        logger.info(f"Running community detection ({args.method})...")
        graph_path = gold_root / "graphs" / f"trade_date={args.trade_date}" / "graph_fused.csr.npz"
        if not graph_path.exists():
            logger.error(f"Fused graph not found at {graph_path}")
            return
        adj = sparse.load_npz(graph_path)
        labels = run_leiden(adj) if args.method == "leiden" else run_leiden(adj) # Leiden/Louvain handled by run_leiden fallback
        cluster_df = pd.DataFrame({"record_id": record_ids, "cluster_id": labels})

    # Save clustering results
    export_dir = Path(cfg["paths"]["output_root"]) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    cluster_df.to_parquet(export_dir / f"cluster_result_{args.trade_date}.parquet", index=False)
    
    logger.info(f"Clustering complete. Results saved to {export_dir}")

if __name__ == "__main__":
    main()
