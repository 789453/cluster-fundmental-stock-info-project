from __future__ import annotations

import argparse

from .core.config import load_config
from .core.logging_utils import setup_logging
from .pipeline.build_pipeline import run_pipeline
from .data.loaders import load_records
from .data.validation import validate_required_columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share semantic engine")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("check", "run", "smoke"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", type=str, default="configs/base.yaml")

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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        return cmd_check(args.config)
    if args.command == "run":
        return cmd_run(args.config)
    if args.command == "smoke":
        return cmd_smoke(args.config)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
