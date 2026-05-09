from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_financial_features(
    snapshot_df: pd.DataFrame,
    fina_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build financial quality, growth, leverage, and profitability features from snapshot.
    
    The snapshot_df should already contain financial indicators joined with as-of logic.
    """
    if fina_cols is None:
        fina_cols = [
            # Profitability
            "roe", "roe_waa", "roa", "roic",
            "grossprofit_margin", "netprofit_margin", "profit_to_gr",
            # Growth
            "tr_yoy", "or_yoy", "netprofit_yoy", "dt_netprofit_yoy",
            "basic_eps_yoy", "q_sales_yoy",
            # Quality
            "ocf_to_or", "ocf_to_profit", "salescash_to_or",
            # Leverage
            "debt_to_assets", "assets_to_eqt",
            "current_ratio", "quick_ratio",
            # Cash Flow
            "ocfps", "cfps", "fcff", "fcfe",
        ]

    df = snapshot_df.copy()
    available = [c for c in fina_cols if c in df.columns]

    for c in available:
        df[c] = df[c].astype(np.float32)

    # Add derived stability features if possible (requires panel data)
    # For now, we just pass through the indicators from the snapshot
    
    logger.info("Financial features: %d columns", len(available))
    return df
