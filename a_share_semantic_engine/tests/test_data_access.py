import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

WAREHOUSE_DB = Path(r"D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb")


class TestTushareCatalog:
    def test_list_tables(self):
        from a_share_semantic_engine.data.tushare_catalog import list_tables, get_table_spec
        tables = list_tables()
        assert "fact_stock_daily" in tables
        assert "fact_stock_daily_basic" in tables
        assert "fact_stock_sw_member" in tables

    def test_get_table_spec(self):
        from a_share_semantic_engine.data.tushare_catalog import get_table_spec
        spec = get_table_spec("fact_stock_daily")
        assert spec is not None
        assert spec.pk_cols == ("ts_code", "trade_date")


class TestLakeHouse:
    def test_connect_and_query(self):
        from a_share_semantic_engine.data.lakehouse import LakeHouse
        with LakeHouse(WAREHOUSE_DB) as lh:
            latest = lh.get_latest_trade_date()
            assert latest == "20260423", f"Expected 20260423, got {latest}"

    def test_fetch_stock_daily(self):
        from a_share_semantic_engine.data.lakehouse import LakeHouse
        with LakeHouse(WAREHOUSE_DB) as lh:
            df = lh.fetch_stock_daily(trade_date="20260423")
            assert len(df) > 0
            assert "ts_code" in df.columns
            assert "close" in df.columns

    def test_fetch_stock_basic_snapshot(self):
        from a_share_semantic_engine.data.lakehouse import LakeHouse
        with LakeHouse(WAREHOUSE_DB) as lh:
            df = lh.fetch_stock_basic_snapshot()
            assert len(df) > 0
            assert "industry" in df.columns
            assert "area" in df.columns
            assert "market" in df.columns


class TestSnapshotBuilder:
    def test_build_stock_snapshot_shape(self):
        from a_share_semantic_engine.data.snapshot_builder import build_stock_snapshot
        snap = build_stock_snapshot("20260423", WAREHOUSE_DB)
        assert len(snap) > 4000, f"Expected >4000 stocks, got {len(snap)}"
        assert "ts_code" in snap.columns
        assert "trade_date" in snap.columns
        assert "close" in snap.columns
        assert "l1_name" in snap.columns

    def test_snapshot_no_st_stocks(self):
        from a_share_semantic_engine.data.snapshot_builder import build_stock_snapshot
        snap = build_stock_snapshot("20260423", WAREHOUSE_DB, include_st=False)
        names = snap["name"].fillna("")
        assert not names.str.contains("ST", na=False).any(), "Should not contain ST stocks"

    def test_snapshot_float_dtypes(self):
        from a_share_semantic_engine.data.snapshot_builder import build_stock_snapshot
        snap = build_stock_snapshot("20260423", WAREHOUSE_DB)
        float_cols = [c for c in snap.columns if snap[c].dtype in (np.float64, np.float32, np.float16)]
        for c in float_cols:
            assert snap[c].dtype == np.float32, f"Column {c} should be float32"


class TestAsOfJoin:
    def test_asof_join_pandas_fallback(self):
        from a_share_semantic_engine.data.asof_join import _join_financial_asof_pandas
        daily = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20260423", "20260423"],
            "close": [10.0, 20.0],
        })
        fina = pd.DataFrame({
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "ann_date": ["20260401", "20260420", "20260415"],
            "roe": [0.1, 0.15, 0.2],
        })
        result = _join_financial_asof_pandas(daily, fina, "trade_date", "ann_date", "ts_code")
        assert len(result) == 2
        assert "roe" in result.columns
        assert result.loc[result["ts_code"] == "000001.SZ", "roe"].iloc[0] == 0.15
        assert result.loc[result["ts_code"] == "000002.SZ", "roe"].iloc[0] == 0.2

    def test_asof_join_empty_fina(self):
        from a_share_semantic_engine.data.asof_join import _join_financial_asof_pandas
        daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260423"], "close": [10.0]})
        result = _join_financial_asof_pandas(daily, pd.DataFrame(), "trade_date", "ann_date", "ts_code")
        assert len(result) == 1
        assert "roe" not in result.columns or result["close"].iloc[0] == 10.0
