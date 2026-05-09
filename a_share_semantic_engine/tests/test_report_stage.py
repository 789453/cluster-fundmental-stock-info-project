import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from scipy import sparse
from a_share_semantic_engine.pipeline.staged_pipeline import ReportStage


def test_report_stage():
    stage_data = {
        "snapshot": {
            "df": pd.DataFrame({
                "ts_code": [f"{i:06d}.SZ" for i in range(100)],
                "close": np.random.randn(100) * 10 + 20,
                "l1_name": np.random.choice(["银行", "地产", "医药"], 100),
            })
        },
        "barra": {
            "df": pd.DataFrame({
                "ts_code": [f"{i:06d}.SZ" for i in range(100)],
                "trade_date": "20240101",
                "size_sw_neutral": np.random.randn(100),
                "value_sw_neutral": np.random.randn(100),
                "mom1m_sw_neutral": np.random.randn(100),
            })
        },
        "semantic": {
            "feat_matrix": np.random.randn(100, 50).astype(np.float32),
            "manifest": {"view_used": "full_text"}
        },
        "graph_builder": {
            "graphs": {
                "semantic": sparse.random(100, 100, density=0.1, format='csr'),
                "industry": sparse.random(100, 100, density=0.1, format='csr'),
            }
        },
        "fusion": {
            "fused_csr": sparse.random(100, 100, density=0.15, format='csr'),
            "manifest": {}
        },
        "clustering": {
            "labels": np.random.randint(0, 5, 100),
            "cluster_df": pd.DataFrame({
                "ts_code": [f"{i:06d}.SZ" for i in range(100)],
                "cluster_id": np.random.randint(0, 5, 100),
            }),
            "manifest": {}
        },
        "graph_metrics": {
            "metrics_df": pd.DataFrame({
                "ts_code": [f"{i:06d}.SZ" for i in range(100)],
                "degree": np.random.randint(1, 20, 100),
                "pagerank": np.random.randn(100),
            })
        }
    }

    context = {"trade_date": "20240101"}

    output_dir = "d:/Trading/a_share_semantic_engine_project/test_report_output"
    os.makedirs(output_dir, exist_ok=True)

    stage = ReportStage(output_dir)
    result = stage.run(stage_data, context)
    print(f"ReportStage.run() succeeded!")
    print(f"  n_clusters: {result.get('report', {}).get('n_clusters', 'N/A')}")
    assert result.get("report", {}).get("n_clusters", 0) >= 2, f"Expected >=2 clusters, got {result.get('report', {}).get('n_clusters')}"


if __name__ == "__main__":
    test_report_stage()
    print("[PASS]")
