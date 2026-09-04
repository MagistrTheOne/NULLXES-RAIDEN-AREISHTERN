"""Compatibility shim.

`score_style` is a negative residue detector, not a RAIDEN personality score.
Use `assistant_residue` / `behavior` for the real suite.
"""

from __future__ import annotations

from raiden.eval.assistant_residue import (  # noqa: F401
    CORPORATE_PATTERNS,
    RESIDUE_PATTERNS,
    aggregate_residue as aggregate_style,
    score_residue as score_style,
)
