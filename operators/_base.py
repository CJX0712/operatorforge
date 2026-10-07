"""Shared numerical utilities for operator implementations.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors import ModelError
from core.types import FloatArray


def validate_training_pair(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Validate and normalize a paired operator-learning dataset."""
    inputs = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if inputs.ndim not in (2, 3) or targets.ndim != inputs.ndim:
        raise ModelError("Operators support batched 1-D or square 2-D scalar fields.")
    if inputs.shape != targets.shape:
        raise ModelError("Input and target arrays must have identical shapes.")
    if inputs.shape[0] < 2:
        raise ModelError("At least two training samples are required.")
    if inputs.ndim == 3 and inputs.shape[1] != inputs.shape[2]:
        raise ModelError("Two-dimensional fields must be square.")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(targets)):
        raise ModelError("Training arrays must be finite.")
    return inputs, targets


def validate_predict_input(x: FloatArray, spatial_rank: int) -> FloatArray:
    """Validate prediction input without constraining spatial resolution."""
    inputs = np.asarray(x, dtype=np.float64)
    if inputs.ndim != spatial_rank + 1 or inputs.shape[0] == 0:
        raise ModelError(f"Expected a non-empty batch with spatial rank {spatial_rank}.")
    if spatial_rank == 2 and inputs.shape[1] != inputs.shape[2]:
        raise ModelError("Two-dimensional fields must be square.")
    if not np.all(np.isfinite(inputs)):
        raise ModelError("Prediction inputs must be finite.")
    return inputs


@dataclass(slots=True)
class RidgeMap:
    """Multi-output ridge map with an unregularized intercept."""

    regularization: float
    feature_mean: FloatArray | None = None
    target_mean: FloatArray | None = None
    coefficient: FloatArray | None = None

    def fit(self, features: FloatArray, targets: FloatArray) -> RidgeMap:
        """Fit using primal or dual algebra according to matrix shape."""
        design = np.asarray(features, dtype=np.float64)
        response = np.asarray(targets, dtype=np.float64)
        if design.ndim != 2 or response.ndim != 2 or design.shape[0] != response.shape[0]:
            raise ModelError("RidgeMap expects aligned two-dimensional matrices.")
        self.feature_mean = design.mean(axis=0, keepdims=True)
        self.target_mean = response.mean(axis=0, keepdims=True)
        centered_design = design - self.feature_mean
        centered_response = response - self.target_mean
        rows, columns = centered_design.shape
        if columns <= rows:
            gram = centered_design.T @ centered_design
            gram.flat[:: columns + 1] += self.regularization
            self.coefficient = np.linalg.solve(gram, centered_design.T @ centered_response)
        else:
            kernel = centered_design @ centered_design.T
            kernel.flat[:: rows + 1] += self.regularization
            dual = np.linalg.solve(kernel, centered_response)
            self.coefficient = centered_design.T @ dual
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        """Apply the fitted affine ridge map."""
        if self.coefficient is None or self.feature_mean is None or self.target_mean is None:
            raise ModelError("RidgeMap must be fitted before prediction.")
        design = np.asarray(features, dtype=np.float64)
        return (design - self.feature_mean) @ self.coefficient + self.target_mean


def pointwise_fit(
    features: FloatArray,
    targets: FloatArray,
    regularization: float,
    max_rows: int = 12000,
) -> RidgeMap:
    """Fit one shared pointwise readout, deterministically subsampling if needed."""
    feature_matrix = features.reshape(-1, features.shape[-1])
    target_matrix = targets.reshape(-1, 1)
    if feature_matrix.shape[0] > max_rows:
        indices = np.linspace(0, feature_matrix.shape[0] - 1, max_rows, dtype=np.int64)
        feature_matrix = feature_matrix[indices]
        target_matrix = target_matrix[indices]
    return RidgeMap(regularization).fit(feature_matrix, target_matrix)


def pointwise_predict(readout: RidgeMap, features: FloatArray) -> FloatArray:
    """Apply a shared pointwise readout while restoring spatial shape."""
    prediction = readout.predict(features.reshape(-1, features.shape[-1]))
    return prediction.reshape(features.shape[:-1])


def gelu(values: FloatArray) -> FloatArray:
    """Fast deterministic GELU approximation."""
    coefficient = np.sqrt(2.0 / np.pi)
    return np.asarray(0.5 * values * (1.0 + np.tanh(coefficient * (values + 0.044715 * values**3))))


__all__ = [
    "RidgeMap",
    "gelu",
    "pointwise_fit",
    "pointwise_predict",
    "validate_predict_input",
    "validate_training_pair",
]
