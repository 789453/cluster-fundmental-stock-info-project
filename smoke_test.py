from a_share_semantic_engine.core.config import load_config
from a_share_semantic_engine.pipeline.build_pipeline import run_pipeline

if __name__ == "__main__":
    cfg = load_config("configs/base.yaml")
    report = run_pipeline(cfg)
    print(report)
