"""DeepONet branch/trunk operator implemented with deterministic ridge training.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.decomposition import PCA

from core.errors import ModelError
from core.types import FloatArray
from operators._base import RidgeMap, validate_predict_input, validate_training_pair


@dataclass(slots=True)
class DeepONetOperator:
    """Branch encoder and learned trunk basis joined by an inner product."""

    latent_dim: int = 16
    width: int = 32
    ridge: float = 1.0e-4
    seed: int = 0
    name: str = field(default="deeponet", init=False)
    _input_pca: PCA | None = field(default=None, init=False, repr=False)
    _target_pca: PCA | None = field(default=None, init=False, repr=False)
    _branch_weight: FloatArray | None = field(default=None, init=False, repr=False)
    _branch_bias: FloatArray | None = field(default=None, init=False, repr=False)
    _coefficient_map: RidgeMap | None = field(default=None, init=False, repr=False)
    _input_shape: tuple[int, ...] | None = field(default=None, init=False)
    _output_shape: tuple[int, ...] | None = field(default=None, init=False)

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit branch embeddings and the trunk coefficient projection."""
        inputs, targets = validate_training_pair(x, y)
        flat_inputs = inputs.reshape(inputs.shape[0], -1)
        flat_targets = targets.reshape(targets.shape[0], -1)
        component_count = min(
            self.latent_dim, inputs.shape[0] - 1, flat_inputs.shape[1], flat_targets.shape[1]
        )
        self._input_pca = PCA(
            n_components=component_count, svd_solver="full", random_state=self.seed
        )
        self._target_pca = PCA(
            n_components=component_count, svd_solver="full", random_state=self.seed
        )
        input_codes = self._input_pca.fit_transform(flat_inputs)
        target_codes = self._target_pca.fit_transform(flat_targets)
        rng = np.random.default_rng(self.seed)
        self._branch_weight = rng.normal(
            scale=1.0 / np.sqrt(component_count), size=(component_count, self.width)
        )
        self._branch_bias = rng.uniform(-0.5, 0.5, size=self.width)
        branch = np.tanh(input_codes @ self._branch_weight + self._branch_bias)
        branch_features = np.concatenate((input_codes, input_codes**2, branch), axis=1)
        self._coefficient_map = RidgeMap(self.ridge).fit(branch_features, target_codes)
        self._input_shape = inputs.shape[1:]
        self._output_shape = targets.shape[1:]
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Evaluate branch coefficients against the fitted trunk basis."""
        if (
            self._input_pca is None
            or self._target_pca is None
            or self._branch_weight is None
            or self._branch_bias is None
            or self._coefficient_map is None
            or self._input_shape is None
            or self._output_shape is None
        ):
            raise ModelError("DeepONet must be fitted before prediction.")
        inputs = validate_predict_input(x, len(self._input_shape))
        if inputs.shape[1:] != self._input_shape:
            raise ModelError("DeepONet baseline requires its training resolution.")
        input_codes = self._input_pca.transform(inputs.reshape(inputs.shape[0], -1))
        branch = np.tanh(input_codes @ self._branch_weight + self._branch_bias)
        features = np.concatenate((input_codes, input_codes**2, branch), axis=1)
        target_codes = self._coefficient_map.predict(features)
        prediction = self._target_pca.inverse_transform(target_codes)
        return prediction.reshape((inputs.shape[0], *self._output_shape))


def available_deeponet() -> bool:
    """Return whether scikit-learn PCA is importable."""
    return True


DeepONet = DeepONetOperator

__all__ = ["DeepONet", "DeepONetOperator", "available_deeponet"]
