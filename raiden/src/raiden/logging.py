"""Logging helpers: files on the persistent volume + optional W&B."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from raiden.paths import logs_dir


def setup_logging(run_name: str = "raiden") -> logging.Logger:
    log_root = logs_dir()
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    logfile = log_root / f"{run_name}-{stamp}.log"
    logger = logging.getLogger("raiden")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("log file: %s", logfile)
    return logger


def choose_report_to(requested: str = "auto") -> list[str]:
    if requested == "none":
        return []
    if requested == "wandb":
        return ["wandb"] if os.environ.get("WANDB_API_KEY") else []
    if requested == "tensorboard":
        return ["tensorboard"]
    # auto
    sinks = ["tensorboard"]
    if os.environ.get("WANDB_API_KEY"):
        sinks.append("wandb")
    return sinks


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
