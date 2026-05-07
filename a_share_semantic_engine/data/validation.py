from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.exceptions import InvalidValueError, RowAlignmentError
from .schema import REQUIRED_COLUMNS


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InvalidValueError(f"missing required columns: {missing}")


def validate_row_alignment(records: pd.DataFrame, row_ids_by_view: dict[str, list[str]]) -> None:
    record_ids = records["record_id"].astype(str).tolist()
    for view, row_ids in row_ids_by_view.items():
        if list(map(str, row_ids)) != record_ids:
            raise RowAlignmentError(f"row_id mismatch for view={view}")


def validate_matrix(name: str, matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise InvalidValueError(f"{name} must be 2D")
    if not np.isfinite(matrix).all():
        raise InvalidValueError(f"{name} contains non-finite values")
