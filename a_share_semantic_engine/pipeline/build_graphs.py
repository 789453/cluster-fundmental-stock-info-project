from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..graph.faiss_knn import build_faiss_knn_graph
from ..graph.industry_graph import build_industry_graph
from ..graph.fundamental_graph import build_fundamental_graph
from ..graph.style_graph import build_style_graph
from ..graph.multiplex import build_multiplex_graph
from ..semantic.vector_store import load_semantic_vectors, align_vectors_to_snapshot
from ..core.config import load_config
from ..core.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Build graphs for a specific date")
    parser.add_argument("--trade-date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--config", type=str, default="configs/research.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = Path(cfg["paths"]["output_root"]) / "logs" / f"build_graphs_{args.trade_date}.log"
    logger = setup_logging("build_graphs", log_path)

    gold_root = Path(cfg["paths"]["gold_root"])
    npy_root = Path(cfg["paths"]["npy_root"])
    
    snapshot_path = gold_root / f"trade_date={args.trade_date}" / "factor_snapshot.parquet"
    if not snapshot_path.exists():
        logger.error(f"Factor snapshot not found for {args.trade_date}")
        return

    df = pd.read_parquet(snapshot_path)
    ts_codes = df["ts_code"].tolist()
    
    graphs = {}
    
    # 1. Semantic Graph
    logger.info("Building semantic graph...")
    view_name = cfg["views"]["text_views"][0] # Use primary text view
    vector_block = load_semantic_vectors(npy_root, view_name)
    aligned_mat = align_vectors_to_snapshot(vector_block, df)
    graphs["semantic"] = build_faiss_knn_graph(aligned_mat, k=30, use_gpu=cfg["project"].get("use_gpu", True))
    
    # 2. Industry Graph
    logger.info("Building industry graph...")
    graphs["industry"] = build_industry_graph(df)
    
    # 3. Fundamental Graph
    logger.info("Building fundamental graph...")
    graphs["fundamental"] = build_fundamental_graph(df, k=20, use_gpu=cfg["project"].get("use_gpu", True))
    
    # 4. Style Graph
    logger.info("Building style graph...")
    graphs["style"] = build_style_graph(df, k=20, use_gpu=cfg["project"].get("use_gpu", True))
    
    # 5. Multiplex Graph
    logger.info("Fusing graphs...")
    fused_graph = build_multiplex_graph(graphs, weights=cfg["graph"]["weights"])
    
    # Save graphs
    graph_dir = gold_root / "graphs" / f"trade_date={args.trade_date}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    for name, adj in graphs.items():
        sparse.save_npz(graph_dir / f"graph_{name}.csr.npz", adj)
        
    sparse.save_npz(graph_dir / "graph_fused.csr.npz", fused_graph)
    
    logger.info(f"Graphs saved to {graph_dir}")

if __name__ == "__main__":
    main()
