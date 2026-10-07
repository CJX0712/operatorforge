"""Pointwise MLP baseline with a shared nonlinear random-feature readout.

The model intentionally sees only ``(u(x), x)`` at each point and therefore
cannot represent general non-local operators.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from core.errors import ModelError
from core.types import FloatArray
from operators._base import (
    RidgeMap,
    pointwise_fit,
    pointwise_predict,
    validate_predict_input,
    validate_training_pair,
)


@dataclass(slots=True)
class PointwiseMLPOperator:
    """Two-layer pointwise random-feature MLP baseline."""

    width: int = 24
    ridge: float = 1.0e-4
    seed: int = 0
    name: str = field(default="mlp", init=False)
    _weights: FloatArray | None = field(default=None, init=False, repr=False)
    _bias: FloatArray | None = field(default=None, init=False, repr=False)
    _readout: RidgeMap | None = field(default=None, init=False, repr=False)
    _spatial_rank: int | None = field(default=None, init=False)

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit a pointwise map with no access to neighbouring values."""
        inputs, targets = validate_training_pair(x, y)
        self._spatial_rank = inputs.ndim - 1
        raw = _pointwise_inputs(inputs)
        rng = np.random.default_rng(self.seed)
        self._weights = rng.normal(
            scale=1.0 / np.sqrt(raw.shape[-1]), size=(raw.shape[-1], self.width)
        )
        self._bias = rng.uniform(-0.5, 0.5, size=self.width)
        hidden = np.tanh(raw @ self._weights + self._bias)
        features = np.concatenate((raw, hidden), axis=-1)
        self._readout = pointwise_fit(features, targets, self.ridge)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Apply the local operator independently at every point."""
        if (
            self._spatial_rank is None
            or self._weights is None
            or self._bias is None
            or self._readout is None
        ):
            raise ModelError("Pointwise MLP must be fitted before prediction.")
        inputs = validate_predict_input(x, self._spatial_rank)
        raw = _pointwise_inputs(inputs)
        hidden = np.tanh(raw @ self._weights + self._bias)
        return pointwise_predict(self._readout, np.concatenate((raw, hidden), axis=-1))


def _pointwise_inputs(inputs: FloatArray) -> FloatArray:
    if inputs.ndim == 2:
        coordinate = np.linspace(0.0, 1.0, inputs.shape[1], endpoint=False)
        coordinate = np.broadcast_to(coordinate, inputs.shape)
        return np.stack((inputs, inputs**2, coordinate), axis=-1)
    axis = np.linspace(0.0, 1.0, inputs.shape[1])
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    return np.stack(
        (
            inputs,
            inputs**2,
            np.broadcast_to(x_grid, inputs.shape),
            np.broadcast_to(y_grid, inputs.shape),
        ),
        axis=-1,
    )


def available_mlp() -> bool:
    """Return availability of the NumPy pointwise baseline."""
    return True


MLP = PointwiseMLPOperator

__all__ = ["MLP", "PointwiseMLPOperator", "available_mlp"]
