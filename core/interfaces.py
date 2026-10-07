"""Structural interfaces shared across operator implementations.

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class OperatorModel(Protocol):
    """Minimal contract implemented by every learned or numerical operator."""

    name: str

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit the operator to paired discretized functions."""

    def predict(self, x: FloatArray) -> FloatArray:
        """Map discretized input functions to output functions."""


@runtime_checkable
class ParameterizedModel(Protocol):
    """Optional parameter-vector contract used by generic optimizers."""

    def parameters(self) -> dict[str, FloatArray]:
        """Return mutable named model parameters."""


__all__ = ["FloatArray", "OperatorModel", "ParameterizedModel"]
