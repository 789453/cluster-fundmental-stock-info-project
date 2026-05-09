from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import read_json
from ..features.common import l2_normalize

logger = logging.getLogger(__name__)


@dataclass
class VectorBlock:
    view_name: str
    asof_date: str
    ts_codes: list[str]
    matrix: np.ndarray  # float32, shape = [N, 1024]
    meta: dict[str, Any]


def load_semantic_vectors(
    npy_root: str | Path,
    view_name: str,
    asof_date: str | None = None,
    dtype: str = "float32",
    normalize: bool = True,
) -> VectorBlock:
    """
    Load 1024-dimensional semantic vectors for a given view.
    
    Args:
        npy_root: Root directory of npy views.
        view_name: Name of the view (e.g., 'product_text').
        asof_date: Optional date filter (currently assuming 'all' files are used).
        dtype: Data type to load.
        normalize: Whether to apply L2 normalization.
    """
    npy_root = Path(npy_root)
    view_dir = npy_root / view_name
    
    # We follow the '*-all.npy' convention from the data spec
    npy_path = view_dir / f"{view_name}-all.npy"
    meta_path = view_dir / f"{view_name}-all.meta.json"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    logger.info(f"Loading vectors for view={view_name} from {npy_path}")
    
    # Use mmap_mode='r' for memory efficiency
    matrix = np.load(npy_path, mmap_mode="r").astype(dtype)
    meta = read_json(meta_path)
    
    row_ids = meta["row_ids"]
    
    if normalize:
        matrix = l2_normalize(matrix)
    
    return VectorBlock(
        view_name=view_name,
        asof_date=asof_date or "all",
        ts_codes=row_ids,  # In this dataset, record_id is often the identifier
        matrix=matrix,
        meta=meta
    )


def align_vectors_to_snapshot(
    vector_block: VectorBlock,
    snapshot_df: pd.DataFrame,
    id_col: str = "record_id",
) -> np.ndarray:
    """
    Align vector matrix to the order of stocks in a snapshot.
    
    Args:
        vector_block: The loaded vector block.
        snapshot_df: The target snapshot DataFrame.
        id_col: The ID column to use for alignment (e.g., 'record_id').
    """
    snapshot_ids = snapshot_df[id_col].astype(str).tolist()
    vector_id_to_idx = {rid: i for i, rid in enumerate(vector_block.ts_codes)}
    
    indices = []
    missing_count = 0
    for rid in snapshot_ids:
        if rid in vector_id_to_idx:
            indices.append(vector_id_to_idx[rid])
        else:
            indices.append(-1)
            missing_count += 1
            
    if missing_count > 0:
        logger.warning(f"Missing vectors for {missing_count}/{len(snapshot_ids)} records in view {vector_block.view_name}")

    # Create aligned matrix
    dim = vector_block.matrix.shape[1]
    aligned_matrix = np.zeros((len(snapshot_ids), dim), dtype=np.float32)
    
    for i, idx in enumerate(indices):
        if idx != -1:
            aligned_matrix[i] = vector_block.matrix[idx]
            
    return aligned_matrix
