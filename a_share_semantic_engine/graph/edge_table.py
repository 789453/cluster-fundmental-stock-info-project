from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def build_edge_table(
    adj: sparse.csr_matrix,
    ts_codes: list[str],
    edge_type: str = "generic",
) -> pd.DataFrame:
    """
    Build edge table from CSR adjacency matrix (upper triangular, no self-loops).

    Args:
        adj: CSR matrix (N, N), assumed symmetric for undirected graphs.
        ts_codes: list of stock codes aligned with rows/cols.
        edge_type: one of 'semantic', 'industry', 'fundamental', 'return_corr', 'style', 'fused'.

    Returns:
        DataFrame with src_ts_code, dst_ts_code, weight, rank, edge_type columns.
    """
    N = adj.shape[0]
    rows, cols = adj.nonzero()
    all_weights = np.asarray(adj[rows, cols]).flatten()

    mask_upper = rows < cols
    rows_u = rows[mask_upper]
    cols_u = cols[mask_upper]
    weights_u = all_weights[mask_upper]

    records = []
    for i, (r, c, w) in enumerate(zip(rows_u, cols_u, weights_u)):
        records.append({
            "src_ts_code": ts_codes[r],
            "dst_ts_code": ts_codes[c],
            "weight": float(w),
            "rank": i + 1,
            "edge_type": edge_type,
        })

    df = pd.DataFrame(records)
    logger.info("edge_table (%s): %d edges", edge_type, len(df))
    return df


def save_edge_tables(
    graphs: dict[str, sparse.csr_matrix],
    ts_codes: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Save multiple edge tables to parquet.

    Args:
        graphs: {name: adj_csr} for each graph type.
        ts_codes: list of stock codes.
        output_dir: directory to save parquet files.

    Returns:
        dict of {edge_type: path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, adj in graphs.items():
        et = build_edge_table(adj, ts_codes, edge_type=name)
        p = output_dir / f"edge_table_{name}.parquet"
        et.to_parquet(p, index=False, compression="zstd")
        paths[name] = p
        logger.info("saved edge_table_%s: %d rows", name, len(et))

    return paths


def save_all_graphs(
    graphs: dict[str, sparse.csr_matrix],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Save multiple CSR graphs as .npz files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, adj in graphs.items():
        p = output_dir / f"graph_{name}.csr.npz"
        sparse.save_npz(str(p), adj)
        paths[name] = p

    return paths
