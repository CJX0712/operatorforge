"""Classical RBF operator using train-only PCA and kernel ridge.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge

from core.errors import ModelError
from core.types import FloatArray
from operators._base import validate_predict_input, validate_training_pair


@dataclass(slots=True)
class RBFOperator:
    """PCA-reduced Gaussian RBF interpolation operator."""

    latent_dim: int = 20
    alpha: float = 2.0e-3
    gamma: float | None = None
    seed: int = 0
    name: str = field(default="rbf", init=False)
    _input_pca: PCA | None = field(default=None, init=False, repr=False)
    _target_pca: PCA | None = field(default=None, init=False, repr=False)
    _kernel: KernelRidge | None = field(default=None, init=False, repr=False)
    _input_shape: tuple[int, ...] | None = field(default=None, init=False)
    _output_shape: tuple[int, ...] | None = field(default=None, init=False)

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit PCA statistics and RBF weights exclusively on training samples."""
        inputs, targets = validate_training_pair(x, y)
        flat_inputs = inputs.reshape(inputs.shape[0], -1)
        flat_targets = targets.reshape(targets.shape[0], -1)
        component_count = min(self.latent_dim, inputs.shape[0] - 1, flat_inputs.shape[1])
        target_components = min(self.latent_dim, targets.shape[0] - 1, flat_targets.shape[1])
        self._input_pca = PCA(
            n_components=component_count, svd_solver="full", random_state=self.seed
        )
        self._target_pca = PCA(
            n_components=target_components, svd_solver="full", random_state=self.seed
        )
        input_codes = self._input_pca.fit_transform(flat_inputs)
        target_codes = self._target_pca.fit_transform(flat_targets)
        scale = float(np.mean(np.var(input_codes, axis=0)))
        gamma = self.gamma if self.gamma is not None else 1.0 / max(component_count * scale, 1.0e-8)
        self._kernel = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma)
        self._kernel.fit(input_codes, target_codes)
        self._input_shape = inputs.shape[1:]
        self._output_shape = targets.shape[1:]
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Interpolate target PCA coefficients with the fitted RBF kernel."""
        if (
            self._input_pca is None
            or self._target_pca is None
            or self._kernel is None
            or self._input_shape is None
            or self._output_shape is None
        ):
            raise ModelError("RBF operator must be fitted before prediction.")
        inputs = validate_predict_input(x, len(self._input_shape))
        if inputs.shape[1:] != self._input_shape:
            raise ModelError("RBF baseline requires its training resolution.")
        input_codes = self._input_pca.transform(inputs.reshape(inputs.shape[0], -1))
        target_codes = self._kernel.predict(input_codes)
        prediction = self._target_pca.inverse_transform(target_codes)
        return prediction.reshape((inputs.shape[0], *self._output_shape))


def available_rbf() -> bool:
    """Return whether the scikit-learn RBF backend is available."""
    return True


RBF = RBFOperator

__all__ = ["RBF", "RBFOperator", "available_rbf"]
