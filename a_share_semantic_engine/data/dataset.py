from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np


@dataclass
class SemanticDataset:
    records: pd.DataFrame
    npy_views: dict[str, np.ndarray] = field(default_factory=dict)
    row_ids_by_view: dict[str, list[str]] = field(default_factory=dict)

    def list_views(self) -> list[str]:
        return sorted(self.npy_views.keys())

    def has_view(self, view: str) -> bool:
        return view in self.npy_views

    def get_record(self, record_id: str) -> dict[str, Any]:
        row = self.records.loc[self.records["record_id"] == record_id]
        if row.empty:
            raise KeyError(record_id)
        return row.iloc[0].to_dict()

    def get_view_matrix(self, view: str) -> np.ndarray:
        return self.npy_views[view]

    @property
    def num_rows(self) -> int:
        return len(self.records)
