"""Local convolutional encoder-decoder baseline.

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
class ConvolutionalOperator:
    """Small translation-equivariant local convolution feature model."""

    ridge: float = 5.0e-5
    name: str = field(default="cnn", init=False)
    _readout: RidgeMap | None = field(default=None, init=False, repr=False)
    _spatial_rank: int | None = field(default=None, init=False)

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit the decoder over a fixed local multiscale convolution bank."""
        inputs, targets = validate_training_pair(x, y)
        self._spatial_rank = inputs.ndim - 1
        self._readout = pointwise_fit(_convolution_features(inputs), targets, self.ridge)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict using resolution-independent local kernels."""
        if self._spatial_rank is None or self._readout is None:
            raise ModelError("CNN must be fitted before prediction.")
        inputs = validate_predict_input(x, self._spatial_rank)
        return pointwise_predict(self._readout, _convolution_features(inputs))


def _convolution_features(inputs: FloatArray) -> FloatArray:
    if inputs.ndim == 2:
        shifts = [np.roll(inputs, offset, axis=1) for offset in (-3, -2, -1, 0, 1, 2, 3)]
        smooth = sum(shifts[2:5]) / 3.0
        laplacian = shifts[2] - 2.0 * shifts[3] + shifts[4]
        return np.stack((*shifts, smooth, laplacian, inputs**2), axis=-1)
    padded = np.pad(inputs, ((0, 0), (2, 2), (2, 2)), mode="edge")
    center = padded[:, 2:-2, 2:-2]
    north = padded[:, 1:-3, 2:-2]
    south = padded[:, 3:-1, 2:-2]
    west = padded[:, 2:-2, 1:-3]
    east = padded[:, 2:-2, 3:-1]
    north2 = padded[:, :-4, 2:-2]
    south2 = padded[:, 4:, 2:-2]
    west2 = padded[:, 2:-2, :-4]
    east2 = padded[:, 2:-2, 4:]
    laplacian = north + south + west + east - 4.0 * center
    return np.stack(
        (center, north, south, west, east, north2, south2, west2, east2, laplacian, center**2),
        axis=-1,
    )


def available_cnn() -> bool:
    """Return availability of the local convolution baseline."""
    return True


CNN = ConvolutionalOperator

__all__ = ["CNN", "ConvolutionalOperator", "available_cnn"]
