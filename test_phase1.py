import logging
from pathlib import Path
from a_share_semantic_engine.data.snapshot_builder import build_stock_snapshot

logging.basicConfig(level=logging.INFO)

trade_date = "20260423"
warehouse_db = "D:/Trading/data_ever_26_3_14/data/meta/warehouse.duckdb"
output_path = "outputs/test_snapshot_20260423.parquet"

df = build_stock_snapshot(
    trade_date=trade_date,
    warehouse_db=warehouse_db,
    output_path=output_path
)

print(f"Snapshot built. Shape: {df.shape}")
print(df.head())
