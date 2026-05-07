from pathlib import Path

from a_share_semantic_engine.pipeline.build_pipeline import run_pipeline


def test_pipeline_smoke(base_config, tmp_path):
    cfg = dict(base_config)
    cfg["paths"] = dict(base_config["paths"])
    cfg["paths"]["output_root"] = str(tmp_path / "outputs")
    report = run_pipeline(cfg)
    run_dir = Path(report["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "exports" / "semantic_feature_table.csv").exists()
    assert (run_dir / "graphs" / "fused_graph.npz").exists()
