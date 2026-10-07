"""Deterministic random-state control.

Author: 晨星
"""

from __future__ import annotations

import os
import random

import numpy as np

_CURRENT_SEED = 0


def set_all(seed: int) -> np.random.Generator:
    """Seed Python, NumPy legacy APIs, and return the canonical generator."""
    global _CURRENT_SEED
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _CURRENT_SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def current_seed() -> int:
    """Return the most recently configured global seed."""
    return _CURRENT_SEED


__all__ = ["current_seed", "set_all"]
