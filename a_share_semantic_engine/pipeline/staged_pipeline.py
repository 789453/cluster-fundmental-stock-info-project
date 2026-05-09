"""
Staged research pipeline — each stage is independently runnable and cacheable.

Design principles:
  - Each Stage class has run() / validate() / load_cached()
  - Outputs cached under run_dir/stage{N}_{name}/
  - A failed validate() halts the pipeline (no silent corruption)
  - Can run any stage individually via: python -m staged_pipeline --stage N
"""
from __future__ import annotations
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base Stage
# ---------------------------------------------------------------------------

class Stage:
    name: str = "base"
    cache_subdir: str = ""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.stage_dir = self.output_dir / self.cache_subdir
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_data: dict, context: dict) -> dict:
        raise NotImplementedError

    def validate(self, output: dict) -> bool:
        return True

    def load_cached(self) -> dict | None:
        manifest = self.stage_dir / "manifest.json"
        if not manifest.exists():
            return None
        data = {"manifest": json.loads(manifest.read_text(encoding="utf-8"))}
        for f in self.stage_dir.glob("*.npy"):
            data[f.stem] = np.load(f)
        for f in self.stage_dir.glob("*.parquet"):
            data[f.stem] = pd.read_parquet(f)
        for f in self.stage_dir.glob("*.csv"):
            data[f.stem] = pd.read_csv(f)
        for f in self.stage_dir.glob("*.npz"):
            data[f.stem] = sparse.load_npz(str(f)).tocsr()
        for f in self.stage_dir.glob("*.json"):
            if f.name == "manifest.json":
                continue
            data[f.stem] = json.loads(f.read_text(encoding="utf-8"))

        if self.name == "graph_builder":
            graphs = {}
            for key, val in data.items():
                if key.startswith("graph_") and sparse.issparse(val):
                    graphs[key.replace("graph_", "", 1)] = val.tocsr()
            if graphs:
                data["graphs"] = graphs

        if self.name == "fusion" and "fused_csr" in data:
            data["fused_csr"] = data["fused_csr"].tocsr()

        logger.info("[%s] loaded from cache: %s", self.name, self.stage_dir)
        return data

    def _save(self, output: dict) -> None:
        manifest: dict[str, Any] = {"stage": self.name, "saved_at": time.time()}

        for key, val in output.items():
            if isinstance(val, np.ndarray):
                p = self.stage_dir / f"{key}.npy"
                np.save(p, val)
                manifest[f"{key}.npy"] = str(p)
            elif isinstance(val, pd.DataFrame):
                p = self.stage_dir / f"{key}.parquet"
                val.to_parquet(p, index=False, compression="zstd")
                manifest[f"{key}.parquet"] = str(p)
            elif isinstance(val, sparse.csr_matrix):
                p = self.stage_dir / f"{key}.npz"
                sparse.save_npz(str(p), val)
                manifest[f"{key}.npz"] = str(p)
            elif isinstance(val, (np.int32, np.int64)):
                manifest[key] = int(val)
            elif isinstance(val, (np.float32, np.float64)):
                manifest[key] = float(val)
            elif isinstance(val, (dict, list, tuple)):
                p = self.stage_dir / f"{key}.json"
                p.write_text(json.dumps(val, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                manifest[f"{key}.json"] = str(p)
            elif val is None or isinstance(val, (str, int, float, bool)):
                manifest[key] = val

        (self.stage_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Stage 1: Snapshot
# ---------------------------------------------------------------------------

class SnapshotStage(Stage):
    name = "snapshot"
    cache_subdir = "stage1_snapshot"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..data.snapshot_builder import build_stock_snapshot
        trade_date = context["trade_date"]
        warehouse_db = context["warehouse_db"]

        logger.info("[Stage1] building snapshot for %s", trade_date)
        df = build_stock_snapshot(trade_date, warehouse_db)

        manifest = {
            "trade_date": trade_date,
            "n_stocks": len(df),
            "columns": df.columns.tolist(),
        }
        self._save({"df": df, "manifest": manifest})

        return {"df": df, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        df = output["df"]
        assert len(df) > 4000, f"Expected >4000 stocks, got {len(df)}"
        assert "ts_code" in df.columns
        assert "close" in df.columns
        assert "l1_name" in df.columns
        assert df["l1_name"].isna().sum() < len(df) * 0.5, "Too many missing l1_name"
        logger.info("[Stage1] validate OK: %d stocks, %d cols", len(df), len(df.columns))
        return True


# ---------------------------------------------------------------------------
# Stage 2: Barra Style Exposures
# ---------------------------------------------------------------------------

class BarraStage(Stage):
    name = "barra"
    cache_subdir = "stage2_barra"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..features.barra_style import build_barra_style_exposures

        df: pd.DataFrame = input_data["snapshot"]["df"]
        logger.info("[Stage2] building Barra exposures for %d stocks", len(df))
        result = build_barra_style_exposures(df, industry_col="l1_name")

        neutral_cols = [c for c in result.columns if c.endswith("_sw_neutral")]
        raw_cols = [c for c in result.columns if c not in ["ts_code", "trade_date", "cluster_id"] and c not in neutral_cols]

        manifest = {
            "n_stocks": len(result),
            "n_style_factors": len(neutral_cols),
            "style_factors": neutral_cols,
        }
        self._save({"df": result, "manifest": manifest})

        return {"df": result, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        df = output["df"]
        critical = ["Size", "Value", "Volatility", "Leverage"]
        all_factors = ["Size", "NonlinearSize", "Value", "Momentum", "ShortReversal",
                       "Volatility", "Liquidity", "Profitability", "Growth",
                       "Leverage", "EarningsQuality"]

        missing = [f for f in critical if f not in df.columns]
        assert not missing, f"Missing critical Barra factors: {missing}"

        all_missing = [f for f in all_factors if f not in df.columns]
        logger.info("[Stage2] validate OK: %d/%d factors present",
                    len(all_factors) - len(all_missing), len(all_factors))
        return True


# ---------------------------------------------------------------------------
# Stage 2b: Price Features (continuous market data)
# ---------------------------------------------------------------------------

class PriceFeatureStage(Stage):
    name = "price_features"
    cache_subdir = "stage2b_price_features"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..features.price_panel import build_price_feature_panel, build_returns_matrix

        snap: pd.DataFrame = input_data["snapshot"]["df"]
        ts_codes = snap["ts_code"].tolist()
        trade_date = context["trade_date"]
        warehouse_db = context["warehouse_db"]

        logger.info("[Stage2b] building price features for %d stocks", len(ts_codes))

        price_features, returns_matrix = build_price_feature_panel(
            warehouse_db=warehouse_db,
            trade_date=trade_date,
            ts_codes=ts_codes,
            windows=(5, 20, 60, 120, 252),
        )

        returns_mask = (returns_matrix != 0).any(axis=1) if returns_matrix.size > 0 else np.array([])

        manifest = {
            "n_stocks": len(ts_codes),
            "n_timesteps": returns_matrix.shape[1] if returns_matrix.size > 0 else 0,
            "features": list(price_features.columns),
        }

        if price_features is not None and not price_features.empty:
            self._save({
                "price_features": price_features,
                "returns_matrix": returns_matrix,
                "returns_mask": returns_mask,
                "manifest": manifest,
            })

        return {
            "price_features": price_features,
            "returns_matrix": returns_matrix,
            "manifest": manifest,
        }

    def validate(self, output: dict) -> bool:
        pf = output.get("price_features")
        if pf is None or pf.empty:
            logger.warning("[Stage2b] No price features generated")
            return True
        assert len(pf) > 4000, f"Expected >4000 stocks, got {len(pf)}"
        logger.info("[Stage2b] validate OK: %d stocks, %d features", len(pf), len(pf.columns) - 1)
        return True


# ---------------------------------------------------------------------------
# Stage 3: Semantic Vectors
# ---------------------------------------------------------------------------

class SemanticStage(Stage):
    name = "semantic"
    cache_subdir = "stage3_semantic"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..graph.graph_builder import VectorStore

        npy_root = context["npy_root"]
        df: pd.DataFrame = input_data["snapshot"]["df"]
        ts_codes = df["ts_code"].tolist()

        logger.info("[Stage3] loading semantic vectors from %s", npy_root)
        vs = VectorStore(npy_root)

        available_views = ["full_text", "profile_text", "product_text", "raw_json"]
        view_used = None
        feat_matrix = None

        for v in available_views:
            try:
                vectors, row_ids, stock_codes = vs.load_view(v)
                idx_map = vs.get_index(v, ts_codes)
                if idx_map is not None and len(idx_map) == len(ts_codes) and np.all(idx_map >= 0):
                    feat_matrix = vectors[idx_map]
                    view_used = v
                    logger.info("  view '%s' aligned: shape=%s", v, feat_matrix.shape)
                    break
                else:
                    feat_matrix = vectors[:len(ts_codes)]
                    view_used = v + "_partial"
                    logger.info("  view '%s' partial: using first %d rows", v, len(ts_codes))
                    break
            except FileNotFoundError:
                continue

        if feat_matrix is None:
            logger.warning("[Stage3] no NPY views found, using snapshot features as fallback")
            fcols = ["close", "pct_chg", "turnover_rate", "pe_ttm", "roe", "total_mv"]
            raw = df[[c for c in fcols if c in df.columns]].values.astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0)
            mu = raw.mean(axis=0, keepdims=True)
            sigma = raw.std(axis=0, keepdims=True)
            sigma = np.where(sigma == 0, 1.0, sigma)
            feat_matrix = ((raw - mu) / sigma).astype(np.float32)
            feat_matrix = feat_matrix / (np.linalg.norm(feat_matrix, axis=1, keepdims=True) + 1e-8)
            view_used = None
        logger.info("[Stage3] feat_matrix: shape=%s dtype=%s", feat_matrix.shape, feat_matrix.dtype)

        manifest = {
            "view_used": view_used,
            "shape": list(feat_matrix.shape),
            "dtype": str(feat_matrix.dtype),
            "l2_normalized": True,
        }
        self._save({"feat_matrix": feat_matrix, "manifest": manifest})

        return {"feat_matrix": feat_matrix, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        fm = output["feat_matrix"]
        assert isinstance(fm, np.ndarray), f"Expected ndarray, got {type(fm)}"
        assert fm.dtype == np.float32, f"Expected float32, got {fm.dtype}"
        assert fm.ndim == 2, f"Expected 2D, got {fm.ndim}D"
        assert fm.shape[0] > 4000, f"Expected >4000 rows, got {fm.shape[0]}"
        norms = np.linalg.norm(fm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "Vectors not L2-normalized"
        logger.info("[Stage3] validate OK: shape=%s, all L2-normalized", fm.shape)
        return True


# ---------------------------------------------------------------------------
# Stage 4: Graph Construction (all 5 graphs)
# ---------------------------------------------------------------------------

class GraphBuilderStage(Stage):
    name = "graph_builder"
    cache_subdir = "stage4_graphs"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..graph.graph_builder import (
            build_semantic_knn, build_industry_graph,
            build_fundamental_graph, build_return_corr_graph,
            build_style_graph, build_csr_from_knn,
        )

        snap: pd.DataFrame = input_data["snapshot"]["df"]
        style: pd.DataFrame = input_data["barra"]["df"]
        feat_matrix: np.ndarray = input_data["semantic"]["feat_matrix"]

        ts_codes = snap["ts_code"].tolist()
        N = len(ts_codes)

        k_sem = context.get("k_semantic", 30)

        logger.info("[Stage4] building graphs for N=%d", N)

        neigh_sem, dist_sem, w_sem = build_semantic_knn(
            feat_matrix, k=k_sem, min_sim=0.0, mutual=True,
            backend=context.get("knn_backend", "exact"),
        )
        sem_csr, _, _ = build_csr_from_knn(neigh_sem, dist_sem, N)
        graphs = {"semantic": sem_csr}
        logger.info("  semantic: edges=%d avg_deg=%.3f", sem_csr.nnz // 2, sem_csr.sum() / N)

        sw_l1 = snap["l1_name"].fillna("").tolist()
        sw_l2 = snap["l2_name"].fillna("").tolist()
        sw_l3 = snap["l3_name"].fillna("").tolist()
        neigh_ind, dist_ind, w_ind = build_industry_graph(ts_codes, sw_l1, sw_l2, sw_l3)
        ind_csr, _, _ = build_csr_from_knn(neigh_ind, dist_ind, N)
        graphs["industry"] = ind_csr
        logger.info("  industry: edges=%d avg_deg=%.3f", ind_csr.nnz // 2, ind_csr.sum() / N)

        fin_cols = [c for c in snap.columns if c in [
            "roe", "roa", "roic", "grossprofit_margin", "debt_to_assets",
            "turnover_rate", "pe_ttm", "pb", "netprofit_margin",
        ]]
        if len(fin_cols) >= 3:
            fin_mat = np.nan_to_num(snap[fin_cols].values.astype(np.float32), nan=0.0)
            neigh_fin, dist_fin, w_fin = build_fundamental_graph(fin_mat, k=20)
            fin_csr, _, _ = build_csr_from_knn(neigh_fin, dist_fin, N)
            graphs["fundamental"] = fin_csr
            logger.info("  fundamental: edges=%d", fin_csr.nnz // 2)

        ret_cols = [f"ret_{d}d" for d in [1, 5, 20, 60]]
        avail_ret = [c for c in ret_cols if c in snap.columns]
        if len(avail_ret) >= 2:
            logger.warning("[Stage4] return_corr graph using single-day snapshot returns, not continuous panel. Consider building PriceFeatureStage first for proper return correlation.")
            ret_mat = np.nan_to_num(snap[avail_ret].values.astype(np.float32), nan=0.0)
            if ret_mat.shape[0] == len(ts_codes):
                neigh_ret, dist_ret, w_ret = build_return_corr_graph(ret_mat, k=20)
                ret_csr, _, _ = build_csr_from_knn(neigh_ret, dist_ret, N)
                graphs["return_corr"] = ret_csr

        style_cols = [c for c in style.columns if c.endswith("_sw_neutral")]
        if len(style_cols) >= 3:
            sty_mat = np.nan_to_num(style[style_cols].values.astype(np.float32), nan=0.0)
            neigh_sty, dist_sty, w_sty = build_style_graph(sty_mat, k=20)
            sty_csr, _, _ = build_csr_from_knn(neigh_sty, dist_sty, N)
            graphs["style"] = sty_csr

        manifest = {
            "graphs": list(graphs.keys()),
            "n_nodes": N,
        }

        from ..graph.edge_table import save_all_graphs, save_edge_tables

        save_all_graphs(graphs, self.stage_dir / "csr")
        save_edge_tables(graphs, ts_codes, self.stage_dir / "edges")

        graph_quality = []
        for name, adj in graphs.items():
            rows, cols = adj.nonzero()
            w = np.asarray(adj[rows, cols]).flatten()
            graph_quality.append({
                "graph_name": name,
                "n_nodes": N,
                "nnz": adj.nnz,
                "edge_count": len(rows) // 2,
                "avg_degree": float(adj.sum() / N),
                "isolated_count": int((np.diff(adj.indptr) == 0).sum()),
                "weight_mean": float(w.mean()) if len(w) > 0 else 0.0,
            })
        graph_quality_df = pd.DataFrame(graph_quality)
        graph_quality_df.to_parquet(self.stage_dir / "graph_quality.parquet", index=False)

        self._save({f"graph_{k}": v for k, v in graphs.items()} | {"manifest": manifest})

        return {"graphs": graphs, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        graphs = output["graphs"]
        assert "semantic" in graphs, "Missing semantic graph"
        assert "industry" in graphs, "Missing industry graph"

        for name, adj in graphs.items():
            assert isinstance(adj, sparse.csr_matrix), f"{name}: expected CSR, got {type(adj)}"
            N = adj.shape[0]
            n_edges = adj.nnz // 2
            avg_deg = adj.sum() / N
            logger.info("[Stage4] %s: N=%d edges=%d avg_deg=%.3f", name, N, n_edges, avg_deg)
            assert avg_deg > 1.0, f"{name}: avg_deg={avg_deg:.3f} too low (likely bad weights)"
            assert n_edges > 0, f"{name}: zero edges"

        logger.info("[Stage4] validate OK")
        return True


# ---------------------------------------------------------------------------
# Stage 5: Graph Fusion
# ---------------------------------------------------------------------------

class GraphFusionStage(Stage):
    name = "fusion"
    cache_subdir = "stage5_fusion"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..graph.fusion import fuse_sparse_graphs

        graphs_raw = input_data.get("graph_builder")
        if graphs_raw is None:
            graphs_raw = input_data.get("graphs")
        if isinstance(graphs_raw, dict) and "graphs" in graphs_raw:
            graphs: dict[str, sparse.csr_matrix] = graphs_raw["graphs"]
        elif isinstance(graphs_raw, dict):
            graphs = {k: v for k, v in graphs_raw.items() if isinstance(v, sparse.csr_matrix)}
        else:
            graphs = graphs_raw
        fusion_weights = context.get("fusion_weights", {
            "semantic": 0.35, "industry": 0.20, "fundamental": 0.20,
            "return_corr": 0.15, "style": 0.10,
        })

        logger.info("[Stage5] fusing %d graphs with weights %s", len(graphs), fusion_weights)

        N = next(iter(graphs.values())).shape[0]
        topk = context.get("fusion_topk", 30)

        fused_csr = fuse_sparse_graphs(
            graphs,
            fusion_weights,
            normalize="max",
            topk=topk,
            min_weight=1e-6,
        )

        avg_deg = fused_csr.sum() / N
        logger.info("[Stage5] fused graph: edges=%d avg_deg=%.3f", fused_csr.nnz // 2, avg_deg)

        from ..graph.edge_table import save_all_graphs, save_edge_tables

        save_all_graphs({"fused": fused_csr}, self.stage_dir / "csr")
        snap_for_fusion: pd.DataFrame = input_data.get("snapshot", {}).get("df", pd.DataFrame())
        if not snap_for_fusion.empty:
            save_edge_tables({"fused": fused_csr}, snap_for_fusion["ts_code"].tolist(), self.stage_dir / "edges")

        manifest = {
            "avg_degree": float(avg_deg),
            "n_edges": fused_csr.nnz // 2,
            "weights_used": fusion_weights,
            "topk": topk,
        }
        self._save({"fused_csr": fused_csr, "manifest": manifest})

        return {"fused_csr": fused_csr, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        fused = output["fused_csr"]
        avg_deg = fused.sum() / fused.shape[0]
        assert avg_deg > 1.0, f"[Stage5] FAILED: fused avg_deg={avg_deg:.3f} <= 1.0 — bad fusion weights!"
        logger.info("[Stage5] validate OK: avg_deg=%.3f", avg_deg)
        return True


# ---------------------------------------------------------------------------
# Stage 6: Clustering
# ---------------------------------------------------------------------------

class ClusteringStage(Stage):
    name = "clustering"
    cache_subdir = "stage6_clustering"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..features.clustering import (
            run_kmeans, run_hdbscan, run_leiden, run_louvain,
            run_clustering,
        )

        feat_matrix: np.ndarray = input_data["semantic"]["feat_matrix"]
        fused_csr: sparse.csr_matrix = input_data["fusion"]["fused_csr"]
        snap: pd.DataFrame = input_data["snapshot"]["df"]

        N = len(feat_matrix)
        method = context.get("cluster_method", "kmeans")
        n_clusters = context.get("n_clusters") or max(10, N // 50)

        logger.info("[Stage6] clustering: method=%s n_clusters=%s", method, n_clusters)

        if method == "kmeans":
            labels, dists = run_kmeans(feat_matrix, n_clusters=n_clusters)
            community_labels = None
        elif method == "hdbscan":
            labels, probs = run_hdbscan(feat_matrix)
            community_labels = None
            dists = probs
        elif method == "leiden":
            community_labels = run_leiden(fused_csr)
            labels = community_labels
            dists = np.zeros(N, dtype=np.float32)
        elif method == "louvain":
            community_labels = run_louvain(fused_csr)
            labels = community_labels
            dists = np.zeros(N, dtype=np.float32)
        else:
            labels, dists = run_kmeans(feat_matrix, n_clusters=n_clusters)
            community_labels = None

        n_actual = int(labels.max()) + 1
        logger.info("[Stage6] clusters found: %d", n_actual)

        ts_codes = snap["ts_code"].tolist()
        cluster_df = pd.DataFrame({
            "ts_code": ts_codes,
            "cluster_id": labels.astype(np.int32),
            "cluster_distance": dists.astype(np.float32),
        })
        if community_labels is not None:
            cluster_df["community_id"] = np.array(community_labels).astype(np.int32)

        manifest = {
            "method": method,
            "n_clusters": n_actual,
            "n_stocks": N,
        }
        self._save({"labels": labels, "cluster_df": cluster_df, "manifest": manifest})

        return {"labels": labels, "cluster_df": cluster_df, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        labels = output["labels"]
        cluster_df = output["cluster_df"]

        assert isinstance(labels, np.ndarray)
        assert len(labels) > 4000
        n_unique = len(np.unique(labels[labels >= 0]))
        assert n_unique >= 2, f"Expected >=2 clusters, got {n_unique}"

        assert "cluster_id" in cluster_df.columns
        assert cluster_df["cluster_id"].notna().all()
        assert (cluster_df["cluster_id"] >= 0).all()

        logger.info("[Stage6] validate OK: %d clusters for %d stocks",
                    n_unique, len(labels))
        return True


# ---------------------------------------------------------------------------
# Stage 7: Graph Metrics
# ---------------------------------------------------------------------------

class GraphMetricsStage(Stage):
    name = "graph_metrics"
    cache_subdir = "stage7_graph_metrics"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..graph.graph_metrics import compute_graph_metrics

        fused_csr: sparse.csr_matrix = input_data["fusion"]["fused_csr"]
        snap: pd.DataFrame = input_data["snapshot"]["df"]
        barra: pd.DataFrame = input_data["barra"]["df"]
        labels: np.ndarray = input_data["clustering"]["labels"]

        ts_codes = snap["ts_code"].tolist()

        logger.info("[Stage7] computing graph metrics...")
        style_cols = [c for c in barra.columns if c.endswith("_sw_neutral")]
        barra_sel_cols = ["ts_code", "trade_date"]
        if "cluster_id" in barra.columns:
            barra_sel_cols.append("cluster_id")
        style_df = barra[barra_sel_cols + style_cols].copy() if style_cols else None

        metrics_df = compute_graph_metrics(
            fused_csr, ts_codes,
            labels=labels,
            style_df=style_df,
            snap_df=snap,
        )

        manifest = {
            "n_stocks": len(metrics_df),
            "columns": metrics_df.columns.tolist(),
        }
        self._save({"metrics_df": metrics_df, "manifest": manifest})

        return {"metrics_df": metrics_df, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        df = output["metrics_df"]
        required = ["degree", "pagerank", "clustering_coefficient", "kcore_number"]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing graph metric columns: {missing}"
        assert df["degree"].notna().all()
        avg_deg = df["degree"].mean()
        assert avg_deg > 1.0, f"avg_deg={avg_deg:.3f} too low"
        logger.info("[Stage7] validate OK: avg_deg=%.3f", avg_deg)
        return True


# ---------------------------------------------------------------------------
# Stage 8: Report
# ---------------------------------------------------------------------------

class ReportStage(Stage):
    name = "report"
    cache_subdir = "stage8_report"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..features.clustering import compute_modularity, compute_nmi, compute_purity
        from ..graph.graph_metrics import compute_conductance
        from ..research.cluster_profile import build_cluster_diagnostics, build_cluster_profile
        from ..research.diagnostics import build_data_quality_report, build_graph_quality_report
        from ..research.report_builder import build_research_report
        from ..graph.edge_table import build_edge_table

        snap: pd.DataFrame = input_data["snapshot"]["df"]
        barra: pd.DataFrame = input_data["barra"]["df"]
        cluster_df: pd.DataFrame = input_data["clustering"]["cluster_df"]
        metrics_df: pd.DataFrame = input_data["graph_metrics"]["metrics_df"]
        fused_csr: sparse.csr_matrix = input_data["fusion"]["fused_csr"]
        graphs_raw = input_data.get("graph_builder")
        if graphs_raw is None:
            graphs_raw = input_data.get("graphs")
        if isinstance(graphs_raw, dict) and "graphs" in graphs_raw:
            graphs: dict = graphs_raw["graphs"]
        elif isinstance(graphs_raw, dict):
            graphs = {k: v for k, v in graphs_raw.items() if isinstance(v, sparse.csr_matrix)}
        else:
            graphs = graphs_raw
        trade_date: str = context["trade_date"]

        if cluster_df is None or cluster_df.empty or "cluster_id" not in cluster_df.columns:
            raise ValueError(f"cluster_df is invalid: {cluster_df}")

        if "cluster_distance" not in cluster_df.columns:
            logger.warning("[Stage8] cluster_distance missing from cluster_df, adding default")
            cluster_df = cluster_df.copy()
            cluster_df["cluster_distance"] = 0.0

        labels: np.ndarray = cluster_df["cluster_id"].values
        snap = snap.copy()
        snap["cluster_id"] = labels

        logger.info("[Stage8] computing metrics...")
        logger.info("[Stage8] labels: len=%d unique=%s", len(labels), len(np.unique(labels)) if len(labels) > 0 else "EMPTY")
        logger.info("[Stage8] DEBUG: cluster_df columns=%s", list(cluster_df.columns))
        logger.info("[Stage8] DEBUG: cluster_df head=\n%s", cluster_df.head())

        modularity = compute_modularity(fused_csr, labels)
        conductance = compute_conductance(fused_csr, labels)

        sw_labels = np.zeros(len(snap), dtype=np.int32)
        if "l1_name" in snap.columns:
            uniq = {n: i for i, n in enumerate(snap["l1_name"].fillna("UNKNOWN").unique())}
            sw_labels = np.array([uniq.get(n, 0) for n in snap["l1_name"].fillna("UNKNOWN")])

        nmi = compute_nmi(labels, sw_labels)
        purity = compute_purity(labels, sw_labels)
        diag = build_cluster_diagnostics(labels, input_data["semantic"]["feat_matrix"])

        sw_l2 = np.zeros(len(snap), dtype=np.int32)
        if "l2_name" in snap.columns:
            uniq2 = {n: i for i, n in enumerate(snap["l2_name"].fillna("UNKNOWN").unique())}
            sw_l2 = np.array([uniq2.get(n, 0) for n in snap["l2_name"].fillna("UNKNOWN")])
        nmi_l2 = compute_nmi(labels, sw_l2)

        style_cols = [c for c in barra.columns if c.endswith("_sw_neutral")]
        barra_sel_cols = ["ts_code", "trade_date"]
        if "cluster_id" in barra.columns:
            barra_sel_cols.append("cluster_id")
        style_df = barra[barra_sel_cols + style_cols].copy() if style_cols else None

        if style_df is not None and "cluster_id" not in style_df.columns:
            style_df = style_df.copy()
            style_df["cluster_id"] = labels

        cluster_profile = build_cluster_profile(labels, snap["ts_code"].tolist(), snap, style_df)

        snap["trade_date"] = trade_date

        report = build_research_report(
            trade_date=trade_date,
            snapshot_df=snap,
            cluster_df=cluster_df,
            style_df=style_df,
            graph_stats_df=metrics_df,
            modularity=modularity,
            nmi=nmi,
            purity=purity,
            output_dir=self.stage_dir,
        )

        report["cluster_diagnostics"] = diag
        report["conductance"] = float(conductance)
        report["nmi_sw_l2"] = float(nmi_l2)
        report["cluster_profile"] = cluster_profile.to_dict("records")

        report_path = self.stage_dir / "research_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[Stage8] report saved to %s", report_path)

        data_q = build_data_quality_report(
            snap, style_df,
            vector_coverage=1.0 if input_data["semantic"]["manifest"].get("view_used") else 0.0,
            output_path=self.stage_dir / "data_quality_report.json",
        )

        graph_q = build_graph_quality_report(
            {k: (v,) for k, v in graphs.items()},
            snap["ts_code"].tolist(),
            output_path=self.stage_dir / "graph_quality_report.json",
        )

        edge_df = build_edge_table(fused_csr, snap["ts_code"].tolist(), edge_type="fused")
        edge_df.to_parquet(self.stage_dir / "edge_table_fused.parquet", index=False, compression="zstd")

        manifest = {
            "trade_date": trade_date,
            "modularity": float(modularity),
            "nmi": float(nmi),
            "n_clusters": len(cluster_profile),
        }
        self._save({"manifest": manifest})

        return {"report": report, "manifest": manifest}

    def validate(self, output: dict) -> bool:
        report = output["report"]
        assert report["n_clusters"] >= 2, f"n_clusters={report['n_clusters']} too low"
        assert report["modularity"] >= 0, f"modularity={report['modularity']} negative"
        logger.info("[Stage8] validate OK: n_clusters=%d modularity=%.4f nmi=%.4f",
                    report["n_clusters"], report["modularity"], report["nmi"])
        return True


# ---------------------------------------------------------------------------
# Stage 9: Visualization (read-only from saved artifacts)
# ---------------------------------------------------------------------------

class VisualizationStage(Stage):
    name = "visualization"
    cache_subdir = "stage9_visualization"

    def run(self, input_data: dict, context: dict) -> dict:
        from ..graph.edge_table import load_all_graphs
        from ..viz.plots import plot_cluster_sizes, plot_degree_histogram

        snap_dir = self.output_dir / "stage1_snapshot"
        graphs_dir = self.output_dir / "stage4_graphs"
        fusion_dir = self.output_dir / "stage5_fusion"
        cluster_dir = self.output_dir / "stage6_clustering"
        metrics_dir = self.output_dir / "stage7_graph_metrics"

        snap_path = snap_dir / "df.parquet"
        if snap_path.exists():
            snap = pd.read_parquet(snap_path)
        else:
            snap = pd.DataFrame()

        cluster_path = cluster_dir / "cluster_df.parquet"
        if cluster_path.exists():
            cluster_df = pd.read_parquet(cluster_path)
        else:
            cluster_df = pd.DataFrame()

        metrics_path = metrics_dir / "metrics_df.parquet"
        if metrics_path.exists():
            metrics_df = pd.read_parquet(metrics_path)
        else:
            metrics_df = pd.DataFrame()

        fused_csr = None
        fused_path = fusion_dir / "csr" / "graph_fused.csr.npz"
        if fused_path.exists():
            fused_csr = sparse.load_npz(str(fused_path)).tocsr()

        fig_dir = self.stage_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if not cluster_df.empty:
            sizes = cluster_df["cluster_id"].value_counts().sort_index().values
            plot_cluster_sizes(sizes.tolist(), fig_dir / "01_cluster_size_bar.png")

        if not metrics_df.empty and "degree" in metrics_df.columns:
            plot_degree_histogram(metrics_df["degree"].values, fig_dir / "02_graph_degree_distribution.png")

        manifest = {
            "figures_generated": len(list(fig_dir.glob("*.png"))),
            "snapshot_rows": len(snap),
            "cluster_rows": len(cluster_df),
            "metrics_rows": len(metrics_df),
        }
        self._save({"manifest": manifest})

        return {"manifest": manifest}

    def validate(self, output: dict) -> bool:
        logger.info("[Stage9] visualization OK")
        return True


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

STAGE_CLASSES = [
    SnapshotStage,
    BarraStage,
    PriceFeatureStage,
    SemanticStage,
    GraphBuilderStage,
    GraphFusionStage,
    ClusteringStage,
    GraphMetricsStage,
    ReportStage,
    VisualizationStage,
]


def run_staged_pipeline(
    trade_date: str,
    warehouse_db: str | Path,
    npy_root: str | Path,
    output_dir: str | Path,
    cluster_method: str = "kmeans",
    n_clusters: int | None = None,
    fusion_weights: dict[str, float] | None = None,
    use_cuda: bool = False,
    start_from_stage: int = 1,
) -> dict:
    """
    Run the full research pipeline in 8 stages.

    Each stage is independently validated; failure halts the pipeline.
    Each stage caches its output under output_dir/stage{N}_{name}/
    """
    output_dir = Path(output_dir)
    context = {
        "trade_date": trade_date,
        "warehouse_db": str(warehouse_db),
        "npy_root": str(npy_root),
        "cluster_method": cluster_method,
        "n_clusters": n_clusters,
        "fusion_weights": fusion_weights or {
            "semantic": 0.35, "industry": 0.20, "fundamental": 0.20,
            "return_corr": 0.15, "style": 0.10,
        },
        "knn_backend": "faiss_gpu" if use_cuda else "exact",
        "use_cuda": use_cuda,
    }

    stage_data: dict[str, Any] = {}

    if start_from_stage > 1:
        for stage_cls in STAGE_CLASSES[:start_from_stage - 1]:
            stage = stage_cls(output_dir)
            cached = stage.load_cached()
            if cached is None:
                raise FileNotFoundError(f"Required cache missing for previous stage: {stage.name}")
            stage_data[stage.name] = cached

    for stage_cls in STAGE_CLASSES[max(0, start_from_stage - 1):]:
        stage = stage_cls(output_dir)
        stage_num = STAGE_CLASSES.index(stage_cls) + 1

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"[Pipeline] Stage {stage_num}/{len(STAGE_CLASSES)}: {stage.name}")
        logger.info("=" * 60)

        cached = stage.load_cached()
        if cached is not None:
            logger.info("[Stage %d] <<< SKIPPED (cache found) >>>", stage_num)
            stage_data[stage.name] = cached
            continue

        t0 = time.time()

        try:
            output = stage.run(stage_data, context)
        except Exception as e:
            logger.error("[Stage %d] FAILED with: %s", stage_num, e)
            raise

        try:
            ok = stage.validate(output)
        except AssertionError as e:
            logger.error("[Stage %d] VALIDATION FAILED: %s", stage_num, e)
            raise RuntimeError(f"Stage {stage_num} ({stage.name}) validation failed: {e}") from e

        stage_data[stage.name] = output

        elapsed = time.time() - t0
        logger.info("[Stage %d] COMPLETE in %.1fs", stage_num, elapsed)

    return stage_data
