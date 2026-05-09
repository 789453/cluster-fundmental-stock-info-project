from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator
import logging

import duckdb
import pandas as pd

from .tushare_catalog import CATALOG, get_table_spec

logger = logging.getLogger(__name__)


class LakeHouse:
    def __init__(self, db_path: str | Path, read_only: bool = True):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB not found: {self.db_path}")
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._read_only = read_only

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path), read_only=self._read_only)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "LakeHouse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def fetch_table(
        self,
        table_name: str,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        order_by: list[str] | None = None,
    ) -> pd.DataFrame:
        spec = get_table_spec(table_name)
        if spec is None:
            raise ValueError(f"Unknown table: {table_name}")

        col_list = ", ".join(columns) if columns else "*"

        where_clauses: list[str] = []
        params: list[Any] = []
        if filters:
            for k, v in filters.items():
                if isinstance(v, (list, tuple)):
                    placeholders = ", ".join(["?" for _ in v])
                    where_clauses.append(f"{k} IN ({placeholders})")
                    params.extend(v)
                else:
                    where_clauses.append(f"{k} = ?")
                    params.append(v)

        where_str = ""
        if where_clauses:
            where_str = "WHERE " + " AND ".join(where_clauses)

        order_str = ""
        if order_by:
            order_str = "ORDER BY " + ", ".join(order_by)

        limit_str = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT {col_list}
            FROM {spec.full_name}
            {where_str}
            {order_str}
            {limit_str}
        """.strip()

        logger.debug("LakeHouse query: %s with params %s", query, params)
        conn = self.connect()
        result = conn.execute(query, params).df()
        logger.info("fetched %d rows from %s", len(result), table_name)
        return result

    def fetch_stock_daily(
        self,
        trade_date: str | None = None,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        filters: dict[str, Any] = {}
        if trade_date:
            filters["trade_date"] = trade_date
        if ts_codes:
            filters["ts_code"] = ts_codes

        if start_date:
            date_filter = f"trade_date >= '{start_date}'"
            if "trade_date" in filters:
                filters["trade_date"] = f"= '{trade_date}'"
            else:
                filters["trade_date"] = trade_date
        if end_date:
            if "trade_date" in filters and "=" in str(filters["trade_date"]):
                pass
            else:
                filters["trade_date"] = f"= '{end_date}'"

        if columns is None:
            columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]

        conn = self.connect()
        parts: list[str] = []
        params: list[Any] = []

        if trade_date:
            parts.append("trade_date = ?")
            params.append(trade_date)
        if start_date:
            parts.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            parts.append("trade_date <= ?")
            params.append(end_date)
        if ts_codes:
            codes_str = ", ".join([f"'{c}'" for c in ts_codes])
            parts.append(f"ts_code IN ({codes_str})")

        where_str = "WHERE " + " AND ".join(parts) if parts else ""
        col_str = ", ".join(columns)

        query = f"""
            SELECT {col_str}
            FROM silver.fact_stock_daily
            {where_str}
            ORDER BY ts_code, trade_date
        """
        return conn.execute(query, params).df()

    def fetch_stock_daily_basic(
        self,
        trade_date: str | None = None,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if columns is None:
            columns = [
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
                "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
                "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv",
            ]

        conn = self.connect()
        parts: list[str] = []
        params: list[Any] = []

        if trade_date:
            parts.append("trade_date = ?")
            params.append(trade_date)
        if start_date:
            parts.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            parts.append("trade_date <= ?")
            params.append(end_date)
        if ts_codes:
            codes_str = ", ".join([f"'{c}'" for c in ts_codes])
            parts.append(f"ts_code IN ({codes_str})")

        where_str = "WHERE " + " AND ".join(parts) if parts else ""
        col_str = ", ".join(columns)

        query = f"""
            SELECT {col_str}
            FROM silver.fact_stock_daily_basic
            {where_str}
            ORDER BY ts_code, trade_date
        """
        return conn.execute(query, params).df()

    def fetch_fina_indicator(
        self,
        ts_codes: list[str] | None = None,
        ann_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if columns is None:
            columns = [
                "ts_code", "ann_date", "end_date",
                "roe", "roe_waa", "roa", "roic",
                "grossprofit_margin", "netprofit_margin", "profit_to_gr",
                "debt_to_assets", "assets_to_eqt",
                "current_ratio", "quick_ratio",
                "ocfps", "cfps", "fcff", "fcfe",
                "tr_yoy", "or_yoy", "netprofit_yoy", "dt_netprofit_yoy",
                "basic_eps_yoy", "q_sales_yoy", "q_roe", "q_dt_roe", "q_ocf_to_sales",
            ]

        conn = self.connect()
        parts: list[str] = []
        params: list[Any] = []

        if ts_codes:
            codes_str = ", ".join([f"'{c}'" for c in ts_codes])
            parts.append(f"ts_code IN ({codes_str})")
        if ann_date:
            parts.append("ann_date = ?")
            params.append(ann_date)
        if end_date:
            parts.append("end_date = ?")
            params.append(end_date)

        where_str = "WHERE " + " AND ".join(parts) if parts else ""
        col_str = ", ".join(columns)

        query = f"""
            SELECT {col_str}
            FROM silver.fact_stock_fina_indicator
            {where_str}
            ORDER BY ts_code, ann_date
        """
        return conn.execute(query, params).df()

    def fetch_sw_member(
        self,
        ts_codes: list[str] | None = None,
        l1_code: str | None = None,
        l2_code: str | None = None,
        l3_code: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if columns is None:
            columns = [
                "ts_code", "name",
                "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
                "in_date", "out_date", "is_new",
            ]

        conn = self.connect()
        parts: list[str] = []
        params: list[Any] = []

        if ts_codes:
            codes_str = ", ".join([f"'{c}'" for c in ts_codes])
            parts.append(f"ts_code IN ({codes_str})")
        if l1_code:
            parts.append("l1_code = ?")
            params.append(l1_code)
        if l2_code:
            parts.append("l2_code = ?")
            params.append(l2_code)
        if l3_code:
            parts.append("l3_code = ?")
            params.append(l3_code)

        where_str = "WHERE " + " AND ".join(parts) if parts else ""
        col_str = ", ".join(columns)

        query = f"""
            SELECT {col_str}
            FROM silver.fact_stock_sw_member
            {where_str}
            ORDER BY ts_code, in_date
        """
        return conn.execute(query, params).df()

    def fetch_stock_basic_snapshot(
        self,
        ts_codes: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if columns is None:
            columns = [
                "ts_code", "symbol", "name", "area", "industry",
                "market", "list_date", "act_name", "act_ent_type",
            ]

        conn = self.connect()
        parts: list[str] = []
        params: list[Any] = []

        if ts_codes:
            codes_str = ", ".join([f"'{c}'" for c in ts_codes])
            parts.append(f"ts_code IN ({codes_str})")

        where_str = "WHERE " + " AND ".join(parts) if parts else ""
        col_str = ", ".join(columns)

        query = f"""
            SELECT {col_str}
            FROM silver.fact_stock_basic_snapshot
            {where_str}
            ORDER BY ts_code
        """
        return conn.execute(query, params).df()

    def get_latest_trade_date(self) -> str:
        conn = self.connect()
        result = conn.execute(
            "SELECT MAX(trade_date) FROM silver.fact_stock_daily"
        ).fetchone()
        if result is None or result[0] is None:
            return "20260423"
        return str(result[0])

    def get_trade_dates(self, start_date: str, end_date: str) -> list[str]:
        conn = self.connect()
        result = conn.execute(
            "SELECT DISTINCT trade_date FROM silver.fact_stock_daily "
            "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
            [start_date, end_date],
        ).fetchall()
        return [str(r[0]) for r in result]
