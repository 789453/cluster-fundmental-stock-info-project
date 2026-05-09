from __future__ import annotations

import argparse
import sys

from .core.config import load_config
from .core.logging_utils import setup_logging
from .pipeline.build_pipeline import run_pipeline
from .pipeline.research_pipeline import run_research_pipeline
from .pipeline.staged_pipeline import run_staged_pipeline, STAGE_CLASSES
from .data.loaders import load_records
from .data.validation import validate_required_columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share semantic engine")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("check", "run", "smoke"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", type=str, default="configs/base.yaml")

    p_staged = sub.add_parser("staged")
    p_staged.add_argument("--trade-date", type=str, required=True)
    p_staged.add_argument("--config", type=str, default="configs/base.yaml")
    p_staged.add_argument("--cluster-method", type=str, default="kmeans")
    p_staged.add_argument("--start-from", type=int, default=1)
    p_staged.add_argument("--use-cuda", action="store_true", default=False)

    p_research = sub.add_parser("research")
    p_research.add_argument("--trade-date", type=str, required=True)
    p_research.add_argument("--config", type=str, default="configs/base.yaml")
    p_research.add_argument("--k-semantic", type=int, default=30)
    p_research.add_argument("--cluster-method", type=str, default="kmeans")
    p_research.add_argument("--use-cuda", action="store_true", default=False)
    p_research.add_argument("--run-dir", type=str, default=None)
    p_research.add_argument("--resume", action="store_true", default=False)

    return parser


def cmd_check(config_path: str) -> int:
    cfg = load_config(config_path)
    logger = setup_logging("check", None)
    records_path = cfg["paths"].get("records_path") or cfg["paths"].get("records_csv")
    df = load_records(records_path)
    validate_required_columns(df)
    logger.info("dataset rows=%s cols=%s", len(df), len(df.columns))
    print(f"check ok | rows={len(df)} cols={len(df.columns)}")
    return 0


def cmd_run(config_path: str) -> int:
    cfg = load_config(config_path)
    report = run_pipeline(cfg)
    print(f"run complete | output_dir={report['run_dir']}")
    return 0


def cmd_smoke(config_path: str) -> int:
    cfg = load_config(config_path)
    report = run_pipeline(cfg)
    print(f"smoke complete | output_dir={report['run_dir']}")
    return 0


def cmd_staged(trade_date: str, config_path: str, start_from: int, cluster_method: str, use_cuda: bool) -> int:
    cfg = load_config(config_path)
    logger = setup_logging("staged", None)
    paths = cfg["paths"]
    warehouse_db = paths.get("warehouse_db")
    npy_root = paths.get("npy_root")
    output_root = paths.get("output_root", "outputs")

    if not warehouse_db:
        print("ERROR: warehouse_db not configured", file=sys.stderr)
        return 1
    if not npy_root:
        print("ERROR: npy_root not configured", file=sys.stderr)
        return 1

    logger.info("Starting staged pipeline | trade_date=%s | from_stage=%d | method=%s",
                 trade_date, start_from, cluster_method)

    result = run_staged_pipeline(
        trade_date=trade_date,
        warehouse_db=warehouse_db,
        npy_root=npy_root,
        output_dir=output_root,
        cluster_method=cluster_method,
        start_from_stage=start_from,
        use_cuda=use_cuda,
    )

    report = result.get("report", {})
    report_payload = report.get("report", report)
    n_clusters = report_payload.get("n_clusters", 0)
    modularity = report_payload.get("modularity", 0.0)
    nmi = report_payload.get("nmi", 0.0)
    print(f"staged complete | n_clusters={n_clusters} | modularity={modularity:.4f} | nmi={nmi:.4f}")
    return 0


def cmd_research(trade_date: str, config_path: str, k_semantic: int, cluster_method: str, use_cuda: bool, run_dir: str | None = None, resume: bool = False) -> int:
    cfg = load_config(config_path)
    logger = setup_logging("research", None)
    paths = cfg["paths"]
    warehouse_db = paths.get("warehouse_db")
    npy_root = paths.get("npy_root")
    output_root = paths.get("output_root", "outputs")

    if not warehouse_db:
        print("ERROR: warehouse_db not configured in paths", file=sys.stderr)
        return 1
    if not npy_root:
        print("ERROR: npy_root not configured in paths", file=sys.stderr)
        return 1

    report = run_research_pipeline(
        trade_date=trade_date,
        warehouse_db=warehouse_db,
        npy_root=npy_root,
        output_dir=output_root,
        k_semantic=k_semantic,
        cluster_method=cluster_method,
        use_cuda=use_cuda,
        run_dir=run_dir,
        resume=resume,
    )
    print(f"research complete | trade_date={trade_date}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        return cmd_check(args.config)
    if args.command == "run":
        return cmd_run(args.config)
    if args.command == "smoke":
        return cmd_smoke(args.config)
    if args.command == "staged":
        return cmd_staged(args.trade_date, args.config, args.start_from, args.cluster_method, args.use_cuda)
    if args.command == "research":
        return cmd_research(args.trade_date, args.config, args.k_semantic, args.cluster_method, args.use_cuda, args.run_dir, args.resume)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
