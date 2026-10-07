"""Behavioral, determinism, and discretization tests for all operators.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest

from core.errors import ModelError
from data.pde import periodic_grf
from operators._base import RidgeMap, pointwise_fit, pointwise_predict
from operators.cnn import ConvolutionalOperator
from operators.coarse import CoarseSolverOperator
from operators.deeponet import DeepONetOperator
from operators.fno import FourierNeuralOperator
from operators.mlp import PointwiseMLPOperator
from operators.opfuse import OpFuseOperator
from operators.rbf import RBFOperator


def _model_factories() -> dict[str, Any]:
    return {
        "cnn": lambda: ConvolutionalOperator(ridge=1.0e-4),
        "coarse": lambda: CoarseSolverOperator(task_name="advdiff1d", coarse_size_1d=8),
        "deeponet": lambda: DeepONetOperator(latent_dim=4, width=5, ridge=1.0e-4, seed=9),
        "fno": lambda: FourierNeuralOperator(modes=3, width=4, depth=1, ridge=1.0e-4, seed=9),
        "mlp": lambda: PointwiseMLPOperator(width=5, ridge=1.0e-4, seed=9),
        "opfuse": lambda: OpFuseOperator(modes=3, width=4, depth=1, ridge=1.0e-4, seed=9),
        "rbf": lambda: RBFOperator(latent_dim=4, alpha=1.0e-3, seed=9),
    }


def _array_state(value: object, prefix: str = "root") -> dict[str, np.ndarray]:
    """Collect deterministic ndarray state from nested dataclass model objects."""
    collected: dict[str, np.ndarray] = {}
    if isinstance(value, np.ndarray):
        collected[prefix] = value.copy()
    elif is_dataclass(value) and not isinstance(value, type):
        for definition in fields(value):
            collected.update(
                _array_state(getattr(value, definition.name), f"{prefix}.{definition.name}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            collected.update(_array_state(item, f"{prefix}[{index}]"))
    return collected


class TestOperatorBehavior:
    """Run each public operator through realistic fit/predict paths."""

    @pytest.mark.parametrize(("name", "factory"), _model_factories().items())
    def test_fit_predict_returns_finite_values_with_expected_shape(
        self, name: str, factory, advdiff_dataset
    ) -> None:
        model = factory()
        returned = model.fit(advdiff_dataset.x_train, advdiff_dataset.y_train)
        prediction = model.predict(advdiff_dataset.x_test)

        assert returned is model
        assert model.name == name
        assert prediction.shape == advdiff_dataset.y_test.shape
        assert np.all(np.isfinite(prediction))

    @pytest.mark.parametrize(("_name", "factory"), _model_factories().items())
    def test_predict_before_fit_raises_e200(self, _name: str, factory) -> None:
        with pytest.raises(ModelError) as error:
            factory().predict(np.zeros((1, 16)))
        assert str(error.value).startswith("E200:")

    @pytest.mark.parametrize(
        ("inputs", "targets", "message"),
        [
            (np.empty((0, 16)), np.empty((0, 16)), "At least two"),
            (np.ones((1, 16)), np.ones((1, 16)), "At least two"),
            (np.ones((2, 16)), np.ones((2, 15)), "identical shapes"),
            (np.ones((2, 4, 5)), np.ones((2, 4, 5)), "square"),
            (np.full((2, 16), np.nan), np.ones((2, 16)), "finite"),
        ],
    )
    def test_training_boundaries_fail_explicitly(
        self, inputs: np.ndarray, targets: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ModelError, match=message):
            ConvolutionalOperator().fit(inputs, targets)

    def test_constant_input_field_does_not_produce_nan(self) -> None:
        inputs = np.ones((6, 16))
        targets = np.full((6, 16), 0.25)
        for factory in _model_factories().values():
            model = factory().fit(inputs, targets)
            assert np.all(np.isfinite(model.predict(np.ones((2, 16)))))

    def test_fixed_resolution_baselines_reject_changed_grid(self, advdiff_dataset) -> None:
        for model in (
            DeepONetOperator(latent_dim=3, width=4),
            RBFOperator(latent_dim=3),
        ):
            model.fit(advdiff_dataset.x_train, advdiff_dataset.y_train)
            with pytest.raises(ModelError, match="training resolution"):
                model.predict(np.ones((2, 32)))


class TestDeterminismAndDiscretization:
    """Prove reproducible parameters and mesh-independent inference."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: FourierNeuralOperator(modes=3, width=4, depth=2, ridge=1.0e-4, seed=77),
            lambda: OpFuseOperator(modes=3, width=4, depth=2, ridge=1.0e-4, seed=77),
            lambda: DeepONetOperator(latent_dim=4, width=5, ridge=1.0e-4, seed=77),
        ],
    )
    def test_same_seed_fit_produces_bitwise_equal_state_and_predictions(
        self, factory, advdiff_dataset
    ) -> None:
        first = factory().fit(advdiff_dataset.x_train, advdiff_dataset.y_train)
        second = factory().fit(advdiff_dataset.x_train, advdiff_dataset.y_train)
        first_state = _array_state(first)
        second_state = _array_state(second)

        assert first_state.keys() == second_state.keys()
        assert first_state
        for name in first_state:
            assert np.array_equal(first_state[name], second_state[name]), name
        assert np.array_equal(
            first.predict(advdiff_dataset.x_test), second.predict(advdiff_dataset.x_test)
        )

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: FourierNeuralOperator(modes=6, width=6, depth=1, ridge=1.0e-4, seed=31),
            lambda: OpFuseOperator(modes=6, width=6, depth=1, ridge=1.0e-4, seed=31),
        ],
    )
    def test_operator_trained_at_64_predicts_at_256(self, factory) -> None:
        x_train = periodic_grf(7, 64, seed=100)
        y_train = 0.5 * x_train + 0.2 * np.roll(x_train, 1, axis=1)
        x_super = periodic_grf(2, 256, seed=101)
        model = factory().fit(x_train, y_train)

        prediction = model.predict(x_super)

        assert prediction.shape == (2, 256)
        assert np.all(np.isfinite(prediction))

    def test_fully_disabled_opfuse_is_numerically_equivalent_to_fno(self) -> None:
        x_train = periodic_grf(8, 32, seed=51)
        y_train = 0.6 * x_train + 0.25 * np.roll(x_train, 1, axis=1)
        x_test = periodic_grf(3, 32, seed=52)
        parameters = {"modes": 5, "width": 6, "depth": 2, "ridge": 2.0e-4, "seed": 12}
        fno = FourierNeuralOperator(**parameters).fit(x_train, y_train)
        opfuse = OpFuseOperator(
            **parameters,
            no_gate=True,
            no_markov=True,
            no_highfreq=True,
        ).fit(x_train, y_train)

        expected = fno.predict(x_test)
        actual = opfuse.predict(x_test)
        relative_difference = np.linalg.norm(actual - expected) / max(
            np.linalg.norm(expected), 1.0e-15
        )
        assert relative_difference <= 1.0e-10


class TestTrainOnlyFitting:
    """Ensure PCA and ridge statistics never inspect the test fold."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DeepONetOperator(latent_dim=3, width=4, seed=5),
            lambda: RBFOperator(latent_dim=3, seed=5),
        ],
    )
    def test_pca_means_equal_training_fold_means(self, factory, advdiff_dataset) -> None:
        model = factory().fit(advdiff_dataset.x_train, advdiff_dataset.y_train)
        assert np.allclose(
            model._input_pca.mean_, advdiff_dataset.x_train.reshape(8, -1).mean(axis=0)
        )
        assert np.allclose(
            model._target_pca.mean_, advdiff_dataset.y_train.reshape(8, -1).mean(axis=0)
        )

    def test_ridge_map_statistics_are_train_only(self) -> None:
        features = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        targets = np.array([[1.0], [3.0], [5.0]])
        fitted = RidgeMap(1.0e-6).fit(features, targets)

        assert np.array_equal(fitted.feature_mean, features.mean(axis=0, keepdims=True))
        assert np.array_equal(fitted.target_mean, targets.mean(axis=0, keepdims=True))

    def test_pointwise_fit_subsamples_deterministically_and_restores_shape(self) -> None:
        features = np.arange(60.0).reshape(3, 10, 2)
        targets = (features[..., 0] * 0.5)[..., None]
        first = pointwise_fit(features, targets[..., 0], 1.0e-4, max_rows=7)
        second = pointwise_fit(features, targets[..., 0], 1.0e-4, max_rows=7)

        assert np.array_equal(first.coefficient, second.coefficient)
        assert pointwise_predict(first, features).shape == (3, 10)
