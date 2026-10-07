"""Tests for evaluation metrics, optimization, and training orchestration.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import EvaluationError, TrainingError
from eval.metrics import (
    evaluate_prediction,
    parseval_error,
    relative_h1,
    relative_l2,
    spectral_tail_error,
)
from training.loops import fit_operator, relative_l2_loss
from training.optim import Adam


class TestEvaluationMetrics:
    """Verify metric definitions on analytic and invalid inputs."""

    def test_identity_predictions_have_zero_error(self, rng: np.random.Generator) -> None:
        target = rng.normal(size=(3, 32))
        assert relative_l2(target, target) == 0.0
        assert relative_h1(target, target) == 0.0
        assert spectral_tail_error(target, target) == 0.0

    def test_relative_l2_matches_known_scale_error(self) -> None:
        target = np.array([[3.0, 4.0], [6.0, 8.0]])
        prediction = 1.5 * target
        assert relative_l2(prediction, target) == pytest.approx(0.5)
        assert relative_l2_loss(prediction, target) == pytest.approx(0.5)

    @pytest.mark.parametrize("length", [15, 16, 63, 64])
    def test_parseval_holds_for_odd_and_even_real_signals(
        self, rng: np.random.Generator, length: int
    ) -> None:
        values = rng.normal(size=(4, length))
        spectrum = np.fft.rfft(values, axis=1)
        weights = np.full(spectrum.shape[1], 2.0)
        weights[0] = 1.0
        if length % 2 == 0:
            weights[-1] = 1.0
        spatial_energy = np.sum(np.abs(values) ** 2, axis=1)
        frequency_energy = np.sum(weights[None, :] * np.abs(spectrum) ** 2, axis=1) / length
        relative_residual = np.max(
            np.abs(spatial_energy - frequency_energy) / np.maximum(spatial_energy, 1.0e-15)
        )

        assert relative_residual <= 1.0e-12
        assert parseval_error(values) <= 1.0e-12

    def test_two_dimensional_h1_and_spectral_tail_are_finite(
        self, rng: np.random.Generator
    ) -> None:
        target = rng.normal(size=(2, 8, 8))
        prediction = target + 0.01 * rng.normal(size=target.shape)
        assert 0.0 < relative_h1(prediction, target) < 1.0
        assert spectral_tail_error(prediction, target, tail_modes=3) > 0.0

    @pytest.mark.parametrize(
        "call",
        [
            lambda: relative_l2(np.empty((0, 4)), np.empty((0, 4))),
            lambda: relative_l2(np.zeros((1, 4)), np.zeros((1, 5))),
            lambda: relative_h1(np.zeros(4), np.zeros(4)),
            lambda: spectral_tail_error(np.zeros((1, 4)), np.zeros((1, 4)), 0),
            lambda: parseval_error(np.zeros(4)),
            lambda: relative_l2(np.full((1, 4), np.nan), np.zeros((1, 4))),
        ],
    )
    def test_metric_error_paths_raise_e400(self, call) -> None:
        with pytest.raises(EvaluationError) as error:
            call()
        assert str(error.value).startswith("E400:")

    def test_evaluate_prediction_populates_all_fields(self) -> None:
        target = np.ones((2, 8))
        result = evaluate_prediction(
            "model", "task", 7, target.copy(), target, superresolution_l2=0.25, elapsed_sec=0.1
        )
        assert result.model_name == "model"
        assert result.task_name == "task"
        assert result.seed == 7
        assert result.relative_l2 == 0.0
        assert result.superresolution_l2 == 0.25
        assert result.elapsed_sec == 0.1


class TestAdam:
    """Check optimizer mathematics and defensive validation."""

    def test_first_adam_step_matches_bias_corrected_formula(self) -> None:
        parameter = np.array([1.0, -2.0])
        optimizer = Adam(learning_rate=0.1)
        optimizer.step({"weight": parameter}, {"weight": np.array([2.0, -4.0])})

        assert np.allclose(parameter, np.array([0.9, -1.9]), atol=1.0e-9)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: Adam(learning_rate=0.0),
            lambda: Adam(beta1=-0.1),
            lambda: Adam(beta2=1.0),
        ],
    )
    def test_invalid_hyperparameters_raise_e300(self, factory) -> None:
        with pytest.raises(TrainingError) as error:
            factory()
        assert str(error.value).startswith("E300:")

    @pytest.mark.parametrize(
        ("parameters", "gradients", "message"),
        [
            ({"a": np.ones(2)}, {"b": np.ones(2)}, "names differ"),
            ({"a": np.ones(2)}, {"a": np.ones(3)}, "shape mismatch"),
            ({"a": np.ones(2)}, {"a": np.array([np.nan, 1.0])}, "Non-finite"),
        ],
    )
    def test_invalid_gradients_raise_e300(
        self,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        message: str,
    ) -> None:
        with pytest.raises(TrainingError, match=message):
            Adam().step(parameters, gradients)


class TestTrainingLoop:
    """Exercise the generic OperatorModel orchestration path."""

    def test_fit_operator_records_closed_form_training_result(self) -> None:
        class IdentityModel:
            name = "identity"

            def fit(self, x: np.ndarray, y: np.ndarray):
                self.fitted = True
                return self

            def predict(self, x: np.ndarray) -> np.ndarray:
                return x.copy()

        inputs = np.arange(12.0).reshape(3, 4)
        model = IdentityModel()
        result = fit_operator(model, inputs, inputs.copy(), "identity-task", 12)

        assert model.fitted
        assert result.loss_history == (0.0,)
        assert result.epochs == 1
        assert result.metadata == {"optimizer": "closed-form-ridge"}
        assert result.elapsed_sec >= 0.0

    def test_relative_l2_loss_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shapes must match"):
            relative_l2_loss(np.zeros((2, 3)), np.zeros((2, 4)))
