from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TableSpec:
    schema: str
    name: str
    pk_cols: tuple[str, ...]
    date_cols: tuple[str, ...]
    fields: dict[str, str]
    partition_cols: tuple[str, ...] = ()
    description: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"


CATALOG: dict[str, TableSpec] = {}


def _register(spec: TableSpec) -> None:
    CATALOG[spec.name] = spec


_register(TableSpec(
    schema="silver",
    name="fact_stock_daily",
    pk_cols=("ts_code", "trade_date"),
    date_cols=("trade_date",),
    partition_cols=("trade_date",),
    fields={
        "ts_code": "VARCHAR",
        "trade_date": "VARCHAR",
        "open": "FLOAT",
        "high": "FLOAT",
        "low": "FLOAT",
        "close": "FLOAT",
        "pre_close": "FLOAT",
        "change": "FLOAT",
        "pct_chg": "FLOAT",
        "vol": "FLOAT",
        "amount": "FLOAT",
    },
    description="股票日线行情（stock_daily），2005-01-04 至 2026-04-23，15,500,774条",
))

_register(TableSpec(
    schema="silver",
    name="fact_stock_daily_basic",
    pk_cols=("ts_code", "trade_date"),
    date_cols=("trade_date",),
    partition_cols=("trade_date",),
    fields={
        "ts_code": "VARCHAR",
        "trade_date": "VARCHAR",
        "close": "FLOAT",
        "turnover_rate": "FLOAT",
        "turnover_rate_f": "FLOAT",
        "volume_ratio": "FLOAT",
        "pe": "FLOAT",
        "pe_ttm": "FLOAT",
        "pb": "FLOAT",
        "ps": "FLOAT",
        "ps_ttm": "FLOAT",
        "dv_ratio": "FLOAT",
        "dv_ttm": "FLOAT",
        "total_share": "FLOAT",
        "float_share": "FLOAT",
        "free_share": "FLOAT",
        "total_mv": "FLOAT",
        "circ_mv": "FLOAT",
    },
    description="股票每日指标（stock_daily_basic），含换手率、PE、PB、PS、股息率、总股本、流通股本、总市值、流通市值",
))

_register(TableSpec(
    schema="silver",
    name="fact_stock_fina_indicator",
    pk_cols=("ts_code", "ann_date", "end_date"),
    date_cols=("ann_date", "end_date"),
    partition_cols=("ann_date",),
    fields={
        "ts_code": "VARCHAR",
        "ann_date": "VARCHAR",
        "end_date": "VARCHAR",
        "eps": "DOUBLE",
        "dt_eps": "DOUBLE",
        "profit_dedt": "DOUBLE",
        "gross_margin": "DOUBLE",
        "current_ratio": "DOUBLE",
        "quick_ratio": "DOUBLE",
        "roe": "DOUBLE",
        "roe_waa": "DOUBLE",
        "roa": "DOUBLE",
        "roic": "DOUBLE",
        "debt_to_assets": "DOUBLE",
        "grossprofit_margin": "DOUBLE",
        "netprofit_margin": "DOUBLE",
        "profit_to_gr": "DOUBLE",
        "tr_yoy": "DOUBLE",
        "or_yoy": "DOUBLE",
        "netprofit_yoy": "DOUBLE",
        "dt_netprofit_yoy": "DOUBLE",
        "basic_eps_yoy": "DOUBLE",
        "q_sales_yoy": "DOUBLE",
        "q_profit_yoy": "DOUBLE",
        "ocf_to_or": "DOUBLE",
        "ocf_to_profit": "DOUBLE",
        "salescash_to_or": "DOUBLE",
        "assets_to_eqt": "DOUBLE",
        "ocfps": "DOUBLE",
        "cfps": "DOUBLE",
        "fcff": "DOUBLE",
        "fcfe": "DOUBLE",
        "update_flag": "VARCHAR",
    },
    description="财务指标数据（stock_fina_indicator），1988-12-31 至 2026-03-31，季度数据。重要：必须按 ann_date <= trade_date 做 as-of join",
))

_register(TableSpec(
    schema="silver",
    name="fact_stock_sw_member",
    pk_cols=("ts_code", "in_date"),
    date_cols=("in_date",),
    partition_cols=(),
    fields={
        "l1_code": "VARCHAR",
        "l1_name": "VARCHAR",
        "l2_code": "VARCHAR",
        "l2_name": "VARCHAR",
        "l3_code": "VARCHAR",
        "l3_name": "VARCHAR",
        "ts_code": "VARCHAR",
        "name": "VARCHAR",
        "in_date": "VARCHAR",
        "out_date": "INTEGER",
        "is_new": "VARCHAR",
        "snapshot_date": "BIGINT",
    },
    description="申万行业成分构成（stock_sw_member），含一、二、三级行业代码和名称，以及股票纳入/剔除日期",
))

_register(TableSpec(
    schema="silver",
    name="fact_stock_basic_snapshot",
    pk_cols=("ts_code",),
    date_cols=(),
    partition_cols=(),
    fields={
        "ts_code": "VARCHAR",
        "symbol": "VARCHAR",
        "name": "VARCHAR",
        "area": "VARCHAR",
        "industry": "VARCHAR",
        "cnspell": "VARCHAR",
        "market": "VARCHAR",
        "list_date": "VARCHAR",
        "act_name": "VARCHAR",
        "act_ent_type": "VARCHAR",
    },
    description="股票基础信息快照（stock_basic_snapshot），5,502条，含地域、行业、市场类型",
))


def get_table_spec(name: str) -> TableSpec | None:
    return CATALOG.get(name)


def list_tables() -> list[str]:
    return list(CATALOG.keys())
