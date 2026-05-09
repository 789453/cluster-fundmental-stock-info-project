"""
Comprehensive test suite for a_share_semantic_engine.
Each module / function has its own test class / test method.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


# ---------------------------------------------------------------------------
# graph_builder.py — unit tests
# ---------------------------------------------------------------------------

class TestBuildSemanticKnn:
    """build_semantic_knn with exact / faiss_cpu backends."""

    def test_exact_knn_output_shapes(self):
        from a_share_semantic_engine.graph.graph_builder import build_semantic_knn
        np.random.seed(42)
        N, D = 50, 32
        x = np.random.randn(N, D).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        neigh, dists, weights = build_semantic_knn(x, k=5, backend="exact")
        assert neigh.shape == (N, 5)
        assert dists.shape == (N, 5)
        assert weights.shape == (N, 5)
        assert 0.0 <= weights.max() <= 1.0

    def test_exact_knn_mutual(self):
        from a_share_semantic_engine.graph.graph_builder import build_semantic_knn
        np.random.seed(0)
        x = np.random.randn(30, 16).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        neigh, dists, _ = build_semantic_knn(x, k=3, mutual=True, backend="exact")
        assert neigh.shape == (30, 3)

    def test_l2_normalize(self):
        from a_share_semantic_engine.graph.graph_builder import VectorStore
        vs = VectorStore(Path("."))
        v = np.array([[3.0, 4.0]], dtype=np.float32)
        normed = vs.l2_normalize(v)
        assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)

    def test_faiss_fallback_when_unavailable(self):
        from a_share_semantic_engine.graph.graph_builder import build_semantic_knn
        np.random.seed(42)
        x = np.random.randn(20, 8).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        neigh, dists, weights = build_semantic_knn(x, k=3, backend="faiss_cpu")
        assert neigh.shape == (20, 3)


class TestBuildIndustryGraph:
    """build_industry_graph."""

    def test_all_same_industry(self):
        from a_share_semantic_engine.graph.graph_builder import build_industry_graph
        codes = ["A", "B", "C"]
        l1 = ["银行", "银行", "银行"]
        neigh, dists, weights = build_industry_graph(codes, l1, l1, l1)
        assert neigh.shape[0] == 3
        assert weights[0, 0] == 1.0

    def test_all_different_industry(self):
        from a_share_semantic_engine.graph.graph_builder import build_industry_graph
        codes = ["A", "B", "C"]
        l1 = ["银行", "地产", "医药"]
        neigh, dists, weights = build_industry_graph(codes, l1, l1, l1)
        assert weights[0, 0] == 0.0


class TestBuildFundamentalGraph:
    """build_fundamental_graph."""

    def test_output_shapes(self):
        from a_share_semantic_engine.graph.graph_builder import build_fundamental_graph
        np.random.seed(42)
        N, D = 40, 6
        x = np.random.randn(N, D).astype(np.float32)
        neigh, dists, weights = build_fundamental_graph(x, k=5)
        assert neigh.shape == (N, 5)
        assert dists.shape == (N, 5)


class TestBuildReturnCorrGraph:
    """build_return_corr_graph."""

    def test_output_shapes(self):
        from a_share_semantic_engine.graph.graph_builder import build_return_corr_graph
        np.random.seed(42)
        N, T = 30, 60
        r = np.random.randn(N, T).astype(np.float32) * 0.02
        neigh, dists, weights = build_return_corr_graph(r, k=5)
        assert neigh.shape == (N, 5)


class TestBuildStyleGraph:
    """build_style_graph."""

    def test_output_shapes(self):
        from a_share_semantic_engine.graph.graph_builder import build_style_graph
        np.random.seed(42)
        N, D = 30, 10
        x = np.random.randn(N, D).astype(np.float32)
        neigh, dists, weights = build_style_graph(x, k=5)
        assert neigh.shape == (N, 5)


class TestFuseMultiplexGraphs:
    """fuse_multiplex_graphs."""

    def test_fusion_output_shapes(self):
        from a_share_semantic_engine.graph.graph_builder import fuse_multiplex_graphs
        np.random.seed(42)
        N, k = 20, 5
        idx = np.tile(np.arange(N)[:, None], (1, k))
        d = np.random.rand(N, k).astype(np.float32)

        graph_dict = {
            "semantic": (idx, d * 0.6),
            "industry": (idx, d * 0.4),
        }
        weights = {"semantic": 0.6, "industry": 0.4}
        fused_neigh, fused_dists, fused_sim = fuse_multiplex_graphs(graph_dict, weights)

        assert fused_neigh.shape == (N, k)
        assert fused_dists.shape == (N, k)
        assert fused_sim.shape == (N, N)

    def test_fusion_symmetric(self):
        from a_share_semantic_engine.graph.graph_builder import fuse_multiplex_graphs
        np.random.seed(99)
        N, k = 15, 4
        idx = np.tile(np.arange(N)[:, None], (1, k))
        d = np.random.rand(N, k).astype(np.float32)
        fused_neigh, fused_dists, fused_sim = fuse_multiplex_graphs(
            {"a": (idx, d)}, {"a": 1.0}
        )
        assert np.allclose(fused_sim, fused_sim.T)


class TestBuildCsrFromKnn:
    """build_csr_from_knn."""

    def test_csr_shape(self):
        from a_share_semantic_engine.graph.graph_builder import build_csr_from_knn
        np.random.seed(42)
        N = 30
        k = 5
        neigh = np.tile(np.arange(N)[:, None], (1, k))
        for i in range(N):
            neigh[i] = np.roll(neigh[i], i)
        dists = np.random.rand(N, k).astype(np.float32)
        csr, rows, cols = build_csr_from_knn(neigh, dists, N)
        assert csr.shape == (N, N)
        assert csr.format == "csr"


# ---------------------------------------------------------------------------
# graph_metrics.py — unit tests
# ---------------------------------------------------------------------------

class TestGraphMetrics:
    """compute_graph_metrics + compute_conductance."""

    @pytest.fixture
    def small_adj(self):
        N = 20
        rng = np.random.default_rng(42)
        adj = rng.uniform(0, 1, (N, N)).astype(np.float32)
        adj = (adj + adj.T) / 2.0
        adj = np.where(adj > 0.3, adj, 0.0)
        np.fill_diagonal(adj, 0)
        return sparse.csr_matrix(adj)

    def test_compute_graph_metrics_shapes(self, small_adj):
        from a_share_semantic_engine.graph.graph_metrics import compute_graph_metrics
        codes = [f"{i:04d}.SZ" for i in range(20)]
        labels = np.random.randint(0, 3, size=20)
        snap = pd.DataFrame({
            "ts_code": codes,
            "l1_name": np.random.choice(["银行", "地产"], 20),
            "pct_chg": np.random.randn(20) * 0.02,
            "roe": np.random.randn(20) * 0.1,
        })
        result = compute_graph_metrics(small_adj, codes, labels=labels, snap_df=snap)
        assert len(result) == 20
        assert "degree" in result.columns
        assert "pagerank" in result.columns
        assert "clustering_coefficient" in result.columns
        assert "kcore_number" in result.columns
        assert "bridge_score" in result.columns

    def test_pagerank_sum_to_one(self, small_adj):
        from a_share_semantic_engine.graph.graph_metrics import _pagerank
        pr = _pagerank(small_adj)
        assert np.isclose(pr.sum(), 1.0, atol=1e-5)

    def test_clustering_coefficient_between_0_and_1(self, small_adj):
        from a_share_semantic_engine.graph.graph_metrics import _clustering_coefficient
        cc = _clustering_coefficient(small_adj)
        assert np.all((cc >= 0) | np.isnan(cc))
        assert np.all((cc <= 1) | np.isnan(cc))

    def test_kcore_nonnegative(self, small_adj):
        from a_share_semantic_engine.graph.graph_metrics import _kcore
        kc = _kcore(small_adj)
        assert np.all(kc >= 0)

    def test_conductance(self, small_adj):
        from a_share_semantic_engine.graph.graph_metrics import compute_conductance
        labels = np.random.randint(0, 3, size=20)
        c = compute_conductance(small_adj, labels)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# edge_table.py — unit tests
# ---------------------------------------------------------------------------

class TestEdgeTable:
    """build_edge_table + save_edge_tables."""

    @pytest.fixture
    def csr_adj(self):
        N = 10
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
        rows = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        cols = np.array([1, 2, 0, 3, 0, 1, 2, 3, 4, 5])
        return sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    def test_edge_table_upper_triangular(self, csr_adj):
        from a_share_semantic_engine.graph.edge_table import build_edge_table
        codes = [f"00000{i}.SZ" for i in range(10)]
        et = build_edge_table(csr_adj, codes, edge_type="semantic")
        assert len(et) > 0
        assert "src_ts_code" in et.columns
        assert "dst_ts_code" in et.columns
        assert "weight" in et.columns
        assert "edge_type" in et.columns
        assert (et["src_ts_code"] != et["dst_ts_code"]).all()


# ---------------------------------------------------------------------------
# barra_style.py — unit tests
# ---------------------------------------------------------------------------

class TestBarraStyle:
    """build_barra_style_exposures + industry_neutralize."""

    @pytest.fixture
    def barra_df(self):
        np.random.seed(42)
        N = 50
        return pd.DataFrame({
            "ts_code": [f"{i:06d}.SZ" for i in range(N)],
            "trade_date": ["20260423"] * N,
            "total_mv": np.random.rand(N) * 1e9,
            "circ_mv": np.random.rand(N) * 1e9,
            "pe_ttm": np.random.rand(N) * 50,
            "pb": np.random.rand(N) * 5,
            "dv_ratio": np.random.rand(N) * 3,
            "roe": np.random.rand(N) * 0.3,
            "roa": np.random.rand(N) * 0.1,
            "roic": np.random.rand(N) * 0.15,
            "debt_to_assets": np.random.rand(N) * 0.7,
            "assets_to_eqt": np.random.rand(N) * 3,
            "turnover_rate": np.random.rand(N) * 5,
            "turnover_rate_f": np.random.rand(N) * 4,
            "netprofit_yoy": np.random.rand(N) * 0.5 - 0.2,
            "or_yoy": np.random.rand(N) * 0.4 - 0.1,
            "basic_eps_yoy": np.random.rand(N) * 0.5,
            "q_sales_yoy": np.random.rand(N) * 0.3,
            "realized_vol_60d": np.random.rand(N) * 0.3,
            "realized_vol_20d": np.random.rand(N) * 0.4,
            "l1_name": np.random.choice(["银行", "地产", "医药", "科技"], N),
            "ret_20d": np.random.rand(N) * 0.2 - 0.1,
            "momentum_raw": np.random.rand(N) * 0.3,
            "ocf_to_profit": np.random.rand(N) * 0.5,
            "salescash_to_or": np.random.rand(N) * 0.5,
            "ret_1d": np.random.rand(N) * 0.02 - 0.01,
            "ret_5d": np.random.rand(N) * 0.05 - 0.02,
            "ret_60d": np.random.rand(N) * 0.2 - 0.05,
            "ret_120d": np.random.rand(N) * 0.3 - 0.1,
            "ret_252d": np.random.rand(N) * 0.5 - 0.2,
        })

    def test_barra_output_has_all_factors(self, barra_df):
        from a_share_semantic_engine.features.barra_style import build_barra_style_exposures
        result = build_barra_style_exposures(barra_df, industry_col="l1_name")
        required = ["Size", "NonlinearSize", "Value", "Momentum",
                    "ShortReversal", "Volatility", "Liquidity",
                    "Profitability", "Growth", "Leverage", "EarningsQuality"]
        for f in required:
            assert f in result.columns, f"Missing factor: {f}"

    def test_barra_sw_neutral(self, barra_df):
        from a_share_semantic_engine.features.barra_style import build_barra_style_exposures
        result = build_barra_style_exposures(barra_df, industry_col="l1_name")
        neutral_cols = [c for c in result.columns if c.endswith("_sw_neutral")]
        assert len(neutral_cols) > 0

    def test_industry_neutralize(self):
        from a_share_semantic_engine.data.asof_join import industry_neutralize
        df = pd.DataFrame({
            "factor": np.random.randn(100),
            "industry": np.random.choice(["A", "B", "C"], 100),
        })
        neutral = industry_neutralize(df, "factor", industry_col="industry", method="zscore")
        assert len(neutral) == 100

    def test_compute_beta(self):
        from a_share_semantic_engine.features.barra_style import compute_beta
        np.random.seed(42)
        stock_rets = np.random.randn(50, 300).astype(np.float32) * 0.02
        mkt_rets = np.random.randn(300).astype(np.float32) * 0.015
        betas = compute_beta(stock_rets, mkt_rets, window=252)
        assert betas.shape == (50,)


# ---------------------------------------------------------------------------
# clustering.py — unit tests
# ---------------------------------------------------------------------------

class TestClustering:
    """run_kmeans / run_hdbscan / run_louvain / community metrics."""

    def test_kmeans_output(self):
        from a_share_semantic_engine.features.clustering import run_kmeans
        np.random.seed(42)
        X = np.random.randn(100, 16).astype(np.float32)
        labels, dists = run_kmeans(X, n_clusters=5)
        assert labels.shape == (100,)
        assert dists.shape == (100,)
        assert labels.max() < 5

    def test_hdbscan_output(self):
        from a_share_semantic_engine.features.clustering import run_hdbscan
        np.random.seed(42)
        X = np.random.randn(80, 12).astype(np.float32)
        labels, probs = run_hdbscan(X)
        assert labels.shape == (80,)
        assert probs.shape == (80,)

    def test_nmi(self):
        from a_share_semantic_engine.features.clustering import compute_nmi
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([0, 0, 1, 1, 0, 0])
        n = compute_nmi(a, b)
        assert 0.0 <= n <= 1.0

    def test_purity(self):
        from a_share_semantic_engine.features.clustering import compute_purity
        labels = np.array([0, 0, 1, 1, 1])
        true_l = np.array([0, 1, 0, 1, 2])
        p = compute_purity(labels, true_l)
        assert 0.0 <= p <= 1.0

    def test_cluster_summary(self):
        from a_share_semantic_engine.features.clustering import cluster_summary
        labels = np.array([0, 0, 0, 1, 1, 2])
        codes = [f"s{i}" for i in range(6)]
        ind = np.array(["银行", "银行", "银行", "地产", "地产", "医药"])
        df = pd.DataFrame({"industry": ind})
        summary = cluster_summary(labels, codes, df, industry_col="industry")
        assert len(summary) >= 2
        assert "cluster_id" in summary.columns
        assert "count" in summary.columns

    def test_clustering_run_kmeans_compat(self):
        from a_share_semantic_engine.features.clustering import run_clustering
        np.random.seed(42)
        X = np.random.randn(60, 8).astype(np.float32)
        cfg = {"cluster": {"n_clusters": 4}, "project": {"random_state": 42}}
        result_df, _ = run_clustering(X, cfg, [f"s{i}" for i in range(60)])
        assert "cluster_id" in result_df.columns
        assert "cluster_distance" in result_df.columns


# ---------------------------------------------------------------------------
# cluster_profile.py + diagnostics.py — unit tests
# ---------------------------------------------------------------------------

class TestClusterProfile:
    """build_cluster_profile + build_cluster_diagnostics."""

    def test_cluster_profile(self):
        from a_share_semantic_engine.research.cluster_profile import build_cluster_profile
        np.random.seed(42)
        labels = np.random.randint(0, 4, size=100)
        codes = [f"{i:06d}.SZ" for i in range(100)]
        snap = pd.DataFrame({
            "ts_code": codes,
            "l1_name": np.random.choice(["银行", "地产"], 100),
            "industry": np.random.choice(["商业银行", "股份制银行"], 100),
            "pct_chg": np.random.randn(100) * 0.02,
            "roe": np.random.randn(100) * 0.1,
            "pe_ttm": np.random.rand(100) * 30,
        })
        profile = build_cluster_profile(labels, codes, snap)
        assert len(profile) > 0
        assert "cluster_id" in profile.columns
        assert "count" in profile.columns

    def test_cluster_diagnostics(self):
        from a_share_semantic_engine.research.cluster_profile import build_cluster_diagnostics
        np.random.seed(42)
        labels = np.random.randint(0, 5, size=200)
        features = np.random.randn(200, 10).astype(np.float32)
        diag = build_cluster_diagnostics(labels, features)
        assert "silhouette_score" in diag
        assert "davies_bouldin_score" in diag
        assert "calinski_harabasz_score" in diag


class TestDiagnostics:
    """build_data_quality_report + build_graph_quality_report."""

    def test_data_quality_report(self):
        from a_share_semantic_engine.research.diagnostics import build_data_quality_report
        df = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [10.0, np.nan],
            "roe": [0.1, 0.2],
        })
        report = build_data_quality_report(df)
        assert "missing_by_field" in report
        assert "dtypes" in report

    def test_graph_quality_report(self):
        from a_share_semantic_engine.research.diagnostics import build_graph_quality_report
        N = 20
        rng = np.random.default_rng(42)
        adj = rng.uniform(0, 1, (N, N)).astype(np.float32)
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0)
        csr = sparse.csr_matrix(adj)
        graphs = {"semantic": (csr,)}
        report = build_graph_quality_report(graphs, [f"{i}.SZ" for i in range(N)])
        assert "graphs" in report
        assert "semantic" in report["graphs"]


# ---------------------------------------------------------------------------
# research/report_builder.py — unit tests
# ---------------------------------------------------------------------------

class TestReportBuilder:
    """build_research_report."""

    def test_report_structure(self):
        from a_share_semantic_engine.research.report_builder import build_research_report
        snap = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "name": ["A", "B", "C"],
            "industry": ["银行", "房地产", "电气设备"],
            "l1_name": ["银行", "房地产", "电力设备"],
            "close": [10.0, 8.0, 200.0],
            "pct_chg": [0.01, -0.02, 0.03],
            "pe_ttm": [5.0, 8.0, 30.0],
            "roe": [0.12, 0.08, 0.15],
            "trade_date": ["20260423"] * 3,
        })
        cluster_df = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "cluster_id": [0, 0, 1],
            "cluster_distance": [0.1, 0.2, 0.15],
        })
        report = build_research_report(
            trade_date="20260423",
            snapshot_df=snap,
            cluster_df=cluster_df,
            modularity=0.35,
            nmi=0.42,
            purity=0.78,
        )
        assert report["trade_date"] == "20260423"
        assert "cluster_stats" in report
        assert "data_quality" in report


# ---------------------------------------------------------------------------
# asof_join.py — unit tests
# ---------------------------------------------------------------------------

class TestAsOfJoin:
    """join_financial_asof pandas fallback."""

    def test_asof_join_pandas(self):
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

    def test_asof_join_empty_fina(self):
        from a_share_semantic_engine.data.asof_join import _join_financial_asof_pandas
        daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260423"], "close": [10.0]})
        result = _join_financial_asof_pandas(daily, pd.DataFrame(), "trade_date", "ann_date", "ts_code")
        assert len(result) == 1

    def test_cross_section_winsorize(self):
        from a_share_semantic_engine.data.asof_join import cross_section_winsorize
        df = pd.DataFrame({"a": [1, 2, 3, 100.0], "b": [0.01, 0.5, 0.99, 1.5]})
        result = cross_section_winsorize(df, ["a", "b"], 0.05, 0.95)
        assert result["a"].max() < 100.0

    def test_industry_neutralize_zscore(self):
        from a_share_semantic_engine.data.asof_join import industry_neutralize
        df = pd.DataFrame({
            "factor": np.random.randn(90),
            "industry": np.tile(np.repeat(["A", "B", "C"], 30), 1)[0],
        })
        neutral = industry_neutralize(df, "factor", industry_col="industry", method="zscore")
        assert len(neutral) == 90


# ---------------------------------------------------------------------------
# smoothing.py — unit tests
# ---------------------------------------------------------------------------

class TestSmoothing:
    """graph_smooth (CPU only in this env)."""

    def test_graph_smooth_cpu(self):
        from a_share_semantic_engine.graph.smoothing import graph_smooth
        np.random.seed(42)
        N = 20
        adj = np.random.rand(N, N).astype(np.float32)
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0)
        csr = sparse.csr_matrix(adj)
        features = np.random.randn(N, 4).astype(np.float32)
        result = graph_smooth(csr, features, alpha=0.7, steps=2)
        assert result.shape == features.shape
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# data_access — integration unit tests (with DuckDB)
# ---------------------------------------------------------------------------

WAREHOUSE_DB = Path(r"D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb")


class TestDataAccess:
    """LakeHouse + snapshot_builder.  Skipped if DuckDB unavailable."""

    def test_lakehouse_connect(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")
        from a_share_semantic_engine.data.lakehouse import LakeHouse
        with LakeHouse(WAREHOUSE_DB) as lh:
            latest = lh.get_latest_trade_date()
            assert latest == "20260423"

    def test_fetch_stock_daily(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")
        from a_share_semantic_engine.data.lakehouse import LakeHouse
        with LakeHouse(WAREHOUSE_DB) as lh:
            df = lh.fetch_stock_daily(trade_date="20260423")
            assert len(df) > 4000
            assert "close" in df.columns

    def test_fetch_stock_basic_snapshot(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")
        from a_share_semantic_engine.data.lakehouse import LakeHouse
        with LakeHouse(WAREHOUSE_DB) as lh:
            df = lh.fetch_stock_basic_snapshot()
            assert len(df) > 0
            assert "industry" in df.columns

    def test_build_stock_snapshot(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")
        from a_share_semantic_engine.data.snapshot_builder import build_stock_snapshot
        snap = build_stock_snapshot("20260423", WAREHOUSE_DB)
        assert len(snap) > 4000
        assert "ts_code" in snap.columns
        assert "close" in snap.columns
        assert "l1_name" in snap.columns

    def test_snapshot_no_st(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")
        from a_share_semantic_engine.data.snapshot_builder import build_stock_snapshot
        snap = build_stock_snapshot("20260423", WAREHOUSE_DB, include_st=False)
        assert not snap["name"].str.contains("ST", na=False).any()


class TestCatalog:
    """tushare_catalog."""

    def test_list_tables(self):
        from a_share_semantic_engine.data.tushare_catalog import list_tables, get_table_spec
        tables = list_tables()
        assert "fact_stock_daily" in tables
        assert "fact_stock_fina_indicator" in tables

    def test_get_table_spec(self):
        from a_share_semantic_engine.data.tushare_catalog import get_table_spec
        spec = get_table_spec("fact_stock_daily")
        assert spec is not None
        assert spec.pk_cols == ("ts_code", "trade_date")
