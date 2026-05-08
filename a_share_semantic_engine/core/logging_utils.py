from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(name: str, log_file: str | Path | None):
    # Setup the requested logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Also setup the base package logger
    base_logger = logging.getLogger("a_share_semantic_engine")
    base_logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    handlers.append(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)

    for h in handlers:
        if not logger.handlers:
            logger.addHandler(h)
        if not base_logger.handlers:
            base_logger.addHandler(h)

    logger.propagate = False
    base_logger.propagate = False
    return logger
