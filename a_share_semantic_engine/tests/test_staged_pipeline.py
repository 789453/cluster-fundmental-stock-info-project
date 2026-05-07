"""
Tests for the staged pipeline.
Each Stage class is tested INDEPENDENTLY — no cross-stage dependencies.
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


# ---------------------------------------------------------------------------
# Stage 1: SnapshotStage
# ---------------------------------------------------------------------------

WAREHOUSE_DB = Path(r"D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb")


class TestSnapshotStage:
    def test_snapshot_run_and_validate(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        from a_share_semantic_engine.pipeline.staged_pipeline import SnapshotStage

        with tempfile.TemporaryDirectory() as tmpdir:
            stage = SnapshotStage(output_dir=tmpdir)
            ctx = {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)}
            output = stage.run({}, ctx)

            assert stage.validate(output) is True
            df = output["df"]
            assert len(df) > 4000
            assert "ts_code" in df.columns
            assert "l1_name" in df.columns


# ---------------------------------------------------------------------------
# Stage 2: BarraStage
# ---------------------------------------------------------------------------

class TestBarraStage:
    def test_barra_run_and_validate(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        from a_share_semantic_engine.pipeline.staged_pipeline import BarraStage, SnapshotStage

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_stage = SnapshotStage(output_dir=tmpdir)
            snap_out = snap_stage.run({}, {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)})

            barra_stage = BarraStage(output_dir=tmpdir)
            barra_out = barra_stage.run({"snapshot": snap_out}, {})

            assert barra_stage.validate(barra_out) is True
            df = barra_out["df"]
            for f in ["Size", "Value", "Momentum", "Volatility", "Leverage"]:
                assert f in df.columns


# ---------------------------------------------------------------------------
# Stage 3: SemanticStage
# ---------------------------------------------------------------------------

class TestSemanticStage:
    def test_semantic_fallback_run_and_validate(self):
        from a_share_semantic_engine.pipeline.staged_pipeline import SemanticStage, SnapshotStage

        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_stage = SnapshotStage(output_dir=tmpdir)
            snap_out = snap_stage.run({}, {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)})

            sem_stage = SemanticStage(output_dir=tmpdir)
            sem_out = sem_stage.run({"snapshot": snap_out}, {"npy_root": "nonexistent"})

            assert sem_stage.validate(sem_out) is True
            fm = sem_out["feat_matrix"]
            assert fm.dtype == np.float32
            assert fm.shape[0] == len(snap_out["df"])


# ---------------------------------------------------------------------------
# Stage 4: GraphBuilderStage
# ---------------------------------------------------------------------------

class TestGraphBuilderStage:
    @pytest.fixture
    def mini_data(self):
        from a_share_semantic_engine.pipeline.staged_pipeline import (
            SnapshotStage, BarraStage, SemanticStage,
        )
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_stage = SnapshotStage(output_dir=tmpdir)
            snap_out = snap_stage.run({}, {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)})

            barra_stage = BarraStage(output_dir=tmpdir)
            barra_out = barra_stage.run({"snapshot": snap_out}, {})

            sem_stage = SemanticStage(output_dir=tmpdir)
            sem_out = sem_stage.run({"snapshot": snap_out}, {"npy_root": "nonexistent"})

            return {
                "snapshot": snap_out,
                "barra": barra_out,
                "semantic": sem_out,
                "tmpdir": tmpdir,
            }

    def test_graph_builder_run_and_validate(self, mini_data):
        from a_share_semantic_engine.pipeline.staged_pipeline import GraphBuilderStage

        with tempfile.TemporaryDirectory() as tmpdir:
            stage = GraphBuilderStage(output_dir=tmpdir)
            output = stage.run(
                {"snapshot": mini_data["snapshot"], "barra": mini_data["barra"], "semantic": mini_data["semantic"]},
                {"k_semantic": 10, "knn_backend": "exact"},
            )

            assert stage.validate(output) is True
            graphs = output["graphs"]
            assert "semantic" in graphs
            assert "industry" in graphs


# ---------------------------------------------------------------------------
# Stage 5: FusionStage — CRITICAL test for avg_deg > 1
# ---------------------------------------------------------------------------

class TestFusionStage:
    def test_fused_graph_avg_degree_must_be_greater_than_one(self):
        from a_share_semantic_engine.pipeline.staged_pipeline import GraphBuilderStage

        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            from a_share_semantic_engine.pipeline.staged_pipeline import (
                SnapshotStage, BarraStage, SemanticStage, GraphFusionStage,
            )

            snap_out = SnapshotStage(output_dir=tmpdir).run(
                {}, {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)}
            )
            barra_out = BarraStage(output_dir=tmpdir).run({"snapshot": snap_out}, {})
            sem_out = SemanticStage(output_dir=tmpdir).run(
                {"snapshot": snap_out}, {"npy_root": "nonexistent"}
            )
            gb_out = GraphBuilderStage(output_dir=tmpdir).run(
                {"snapshot": snap_out, "barra": barra_out, "semantic": sem_out},
                {"k_semantic": 20, "knn_backend": "exact"},
            )

            fusion = GraphFusionStage(output_dir=tmpdir)
            fused = fusion.run({"graphs": gb_out}, {})

            avg_deg = fused["fused_csr"].sum() / fused["fused_csr"].shape[0]
            print(f"  fused avg_deg = {avg_deg:.4f}")
            assert avg_deg > 1.0, f"FAILED: fused avg_deg={avg_deg:.4f} <= 1.0 — bad fusion!"
            assert fusion.validate(fused) is True


# ---------------------------------------------------------------------------
# Stage 6: ClusteringStage
# ---------------------------------------------------------------------------

class TestClusteringStage:
    def test_clustering_run_and_validate(self):
        from a_share_semantic_engine.pipeline.staged_pipeline import (
            SnapshotStage, BarraStage, SemanticStage,
            GraphBuilderStage, GraphFusionStage, ClusteringStage,
        )
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_out = SnapshotStage(output_dir=tmpdir).run(
                {}, {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)}
            )
            barra_out = BarraStage(output_dir=tmpdir).run({"snapshot": snap_out}, {})
            sem_out = SemanticStage(output_dir=tmpdir).run(
                {"snapshot": snap_out}, {"npy_root": "nonexistent"}
            )
            gb_out = GraphBuilderStage(output_dir=tmpdir).run(
                {"snapshot": snap_out, "barra": barra_out, "semantic": sem_out},
                {"k_semantic": 10, "knn_backend": "exact"},
            )
            fused_out = GraphFusionStage(output_dir=tmpdir).run(
                {"graphs": gb_out}, {}
            )

            stage = ClusteringStage(output_dir=tmpdir)
            out = stage.run(
                {"semantic": sem_out, "fusion": fused_out, "snapshot": snap_out},
                {"cluster_method": "kmeans", "n_clusters": 20},
            )

            assert stage.validate(out) is True
            labels = out["labels"]
            assert len(np.unique(labels[labels >= 0])) >= 2


# ---------------------------------------------------------------------------
# Stage 7: GraphMetricsStage
# ---------------------------------------------------------------------------

class TestGraphMetricsStage:
    def test_graph_metrics_validate_checks_avg_deg(self):
        from a_share_semantic_engine.pipeline.staged_pipeline import (
            SnapshotStage, BarraStage, SemanticStage,
            GraphBuilderStage, GraphFusionStage,
            ClusteringStage, GraphMetricsStage,
        )
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_out = SnapshotStage(output_dir=tmpdir).run(
                {}, {"trade_date": "20260423", "warehouse_db": str(WAREHOUSE_DB)}
            )
            barra_out = BarraStage(output_dir=tmpdir).run({"snapshot": snap_out}, {})
            sem_out = SemanticStage(output_dir=tmpdir).run(
                {"snapshot": snap_out}, {"npy_root": "nonexistent"}
            )
            gb_out = GraphBuilderStage(output_dir=tmpdir).run(
                {"snapshot": snap_out, "barra": barra_out, "semantic": sem_out},
                {"k_semantic": 10, "knn_backend": "exact"},
            )
            fused_out = GraphFusionStage(output_dir=tmpdir).run(
                {"graphs": gb_out}, {}
            )
            cl_out = ClusteringStage(output_dir=tmpdir).run(
                {"semantic": sem_out, "fusion": fused_out, "snapshot": snap_out},
                {"cluster_method": "kmeans", "n_clusters": 20},
            )

            stage = GraphMetricsStage(output_dir=tmpdir)
            out = stage.run(
                {"fusion": fused_out, "snapshot": snap_out, "barra": barra_out, "clustering": cl_out},
                {},
            )

            assert stage.validate(out) is True
            df = out["metrics_df"]
            assert "degree" in df.columns
            assert df["degree"].mean() > 1.0


# ---------------------------------------------------------------------------
# Full staged pipeline run
# ---------------------------------------------------------------------------

class TestFullStagedPipeline:
    def test_staged_pipeline_end_to_end(self):
        if not WAREHOUSE_DB.exists():
            pytest.skip("DuckDB not available")

        from a_share_semantic_engine.pipeline.staged_pipeline import run_staged_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_staged_pipeline(
                trade_date="20260423",
                warehouse_db=str(WAREHOUSE_DB),
                npy_root="nonexistent",
                output_dir=tmpdir,
                cluster_method="kmeans",
                start_from_stage=1,
            )

            assert "report" in result
            report = result["report"]
            assert report["n_clusters"] >= 2
            assert report["modularity"] >= 0
