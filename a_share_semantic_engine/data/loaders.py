from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import duckdb

from ..core.exceptions import DataArtifactMissingError, InvalidValueError
from ..utils.io import read_json
from .dataset import SemanticDataset


def load_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataArtifactMissingError(f"Records file not found: {path}")
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise InvalidValueError(f"Unsupported record format: {path.suffix}")
    return df


def load_duckdb_metadata(db_path: str | Path, stock_codes: list[str]) -> pd.DataFrame:
    """Load stock metadata (industry, area, etc.) from DuckDB."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    
    codes_str = "', '".join(stock_codes)
    query = f"""
        SELECT ts_code, name, area, industry, market, list_date
        FROM silver.fact_stock_basic_snapshot
        WHERE ts_code IN ('{codes_str}')
    """
    
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        df = conn.execute(query).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _load_single_npy_view(npy_root: Path, view: str) -> tuple[np.ndarray, list[str]]:
    npy_path = npy_root / view / f"{view}-all.npy"
    meta_path = npy_root / view / f"{view}-all.meta.json"
    if not npy_path.exists() or not meta_path.exists():
        raise DataArtifactMissingError(f"Missing NPY artifacts for view={view}")
    mat = np.load(npy_path, mmap_mode="r")
    meta = read_json(meta_path)
    row_ids = meta["row_ids"]
    return np.asarray(mat), row_ids


def maybe_load_npy_views(records: pd.DataFrame, npy_root: str | Path, selected_views: list[str], logger) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    npy_root = Path(npy_root) if npy_root else None
    if not npy_root or not npy_root.exists():
        logger.info("npy root missing or empty; fallback to csv text embedding")
        return {}, {}

    loaded = {}
    row_ids_by_view = {}
    for view in selected_views:
        try:
            mat, row_ids = _load_single_npy_view(npy_root, view)
            loaded[view] = mat
            row_ids_by_view[view] = row_ids
            logger.info("loaded npy view=%s shape=%s", view, tuple(mat.shape))
        except Exception as exc:
            logger.warning("skip npy view=%s reason=%s", view, exc)
    return loaded, row_ids_by_view


def load_dataset(cfg: dict[str, Any], logger) -> SemanticDataset:
    paths = cfg["paths"]
    view_cfg = cfg["views"]
    selected_views = list(dict.fromkeys(view_cfg["text_views"] + view_cfg["json_views"]))
    
    # Try to load from records_path if it exists, otherwise fallback to records_csv
    records_path = paths.get("records_path") or paths.get("records_csv")
    records = load_records(records_path)
    
    # Optionally enrich with DuckDB metadata
    db_path = paths.get("warehouse_db")
    if db_path:
        logger.info("enriching records with DuckDB metadata from %s", db_path)
        meta_df = load_duckdb_metadata(db_path, records["stock_code"].unique().tolist())
        if not meta_df.empty:
            # Merge metadata back to records if needed, or just keep it for anchor features
            # Here we just log it; the pipeline can use it later if it joins records with more info
            logger.info("loaded metadata for %d stocks from DuckDB", len(meta_df))
            # Merge logic could go here if we want to overwrite stock_name or add industry
            records = pd.merge(records, meta_df, left_on="stock_code", right_on="ts_code", how="left", suffixes=("", "_db"))
    
    npy_views, row_ids_by_view = maybe_load_npy_views(records, paths.get("npy_root", ""), selected_views, logger)
    return SemanticDataset(records=records, npy_views=npy_views, row_ids_by_view=row_ids_by_view)
