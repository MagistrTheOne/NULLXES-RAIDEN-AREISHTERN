"""Checkpoint resume helpers for RunPod restarts."""

from __future__ import annotations

import logging
from pathlib import Path

from raiden.paths import latest_checkpoint

logger = logging.getLogger("raiden")


def resolve_resume(spec: str | None, output_dir: str) -> str | None:
    """
    spec:
      - None / "false" / "none" → start fresh
      - "auto" → newest checkpoint-* under output_dir
      - path → that path if it exists
    """
    if spec in (None, "", "false", "none", "None"):
        return None
    if spec == "auto":
        found = latest_checkpoint(Path(output_dir))
        if found is None:
            logger.info("resume=auto: no checkpoint in %s", output_dir)
            return None
        logger.info("resume=auto: %s", found)
        return str(found)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"resume path does not exist: {path}")
    return str(path)
