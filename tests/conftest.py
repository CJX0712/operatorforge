"""Shared deterministic fixtures for the OperatorForge test suite.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from data.pde import generate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng() -> np.random.Generator:
    """Return an isolated deterministic random generator."""
    return np.random.default_rng(20261007)


@pytest.fixture
def advdiff_dataset():
    """Create a compact independent train/test operator-learning dataset."""
    return generate_dataset("advdiff1d", n_train=8, n_test=3, grid_size=16, seed=71)


@pytest.fixture(autouse=True)
def isolate_opf_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Prevent developer-machine ENV_OPF values from leaking into tests."""
    for name in tuple(__import__("os").environ):
        if name.startswith("ENV_OPF_"):
            monkeypatch.delenv(name, raising=False)
    yield
