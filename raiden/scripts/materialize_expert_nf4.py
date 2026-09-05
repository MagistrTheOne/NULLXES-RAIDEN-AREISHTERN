#!/usr/bin/env python3
"""Pod entry: one-time packed-expert NF4 cache. See docs/RUNBOOK.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.materialize_experts import main

if __name__ == "__main__":
    raise SystemExit(main())
