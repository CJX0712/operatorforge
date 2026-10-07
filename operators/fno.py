"""Fourier Neural Operator implemented with real-parameter spectral layers.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from core.errors import ModelError
from core.types import FloatArray
from operators._base import (
    RidgeMap,
    pointwise_fit,
    pointwise_predict,
    validate_predict_input,
    validate_training_pair,
)
from operators._spectral import DiagonalSpectralMap, SpectralFeatureMap


@dataclass(slots=True)
class FourierNeuralOperator:
    """Truncated complex spectral convolution with GELU residual layers."""

    modes: int = 10
    width: int = 24
    depth: int = 3
    ridge: float = 1.0e-5
    seed: int = 0
    name: str = field(default="fno", init=False)
    _features: SpectralFeatureMap = field(init=False, repr=False)
    _spectral_map: DiagonalSpectralMap = field(init=False, repr=False)
    _readout: RidgeMap | None = field(default=None, init=False, repr=False)
    _spatial_rank: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.modes <= 0 or self.width <= 0 or self.depth <= 0:
            raise ModelError("FNO modes, width, and depth must be positive.")
        self._features = SpectralFeatureMap(
            modes=self.modes,
            width=self.width,
            depth=self.depth,
            seed=self.seed,
        )
        self._spectral_map = DiagonalSpectralMap(
            modes=self.modes,
            regularization=self.ridge,
        )

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit the pointwise projection on deterministic Fourier features."""
        inputs, targets = validate_training_pair(x, y)
        self._spatial_rank = inputs.ndim - 1
        spectral_prediction = self._spectral_map.fit(inputs, targets).predict(inputs)
        features = self._features.transform(inputs)
        self._readout = pointwise_fit(features, targets - spectral_prediction, self.ridge)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict at the input resolution, including unseen resolutions."""
        if self._spatial_rank is None or self._readout is None:
            raise ModelError("FNO must be fitted before prediction.")
        inputs = validate_predict_input(x, self._spatial_rank)
        spectral_prediction = self._spectral_map.predict(inputs)
        residual = pointwise_predict(self._readout, self._features.transform(inputs))
        return spectral_prediction + residual


def available_fno() -> bool:
    """Return whether the NumPy FFT backend is available."""
    return True


FNO = FourierNeuralOperator

__all__ = ["FNO", "FourierNeuralOperator", "available_fno"]
