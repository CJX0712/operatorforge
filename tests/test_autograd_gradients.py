"""Layer-wise finite-difference tests for the real reverse-mode AD tape.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from training.autograd import (
    Tensor,
    complex_spectral_linear,
    gated_softshrink,
    gradient_check,
    irfft,
    markov_correction,
    rfft,
    sigmoid,
    softshrink,
)

GRADIENT_TOLERANCE = 1.0e-6


def _assert_gradient(function: Callable[[Tensor], Tensor], value: np.ndarray) -> None:
    error = gradient_check(function, value)
    assert error <= GRADIENT_TOLERANCE, f"maximum symmetric relative error was {error:.3e}"


class TestPrimitiveTape:
    """Verify primitive operators, broadcasting, and error paths."""

    def test_composed_arithmetic_and_gelu_gradient(self, rng: np.random.Generator) -> None:
        value = rng.normal(size=(2, 3)) + 0.4
        fixed = Tensor(rng.normal(size=(2, 3)) + 1.3)

        _assert_gradient(lambda variable: ((variable * fixed + 0.2) / fixed).gelu().mean(), value)

    def test_broadcast_bias_and_reduction_gradient(self, rng: np.random.Generator) -> None:
        value = rng.normal(size=(1, 4)) + 0.3
        fixed = Tensor(rng.normal(size=(3, 4)))

        _assert_gradient(lambda bias: (fixed + bias).sum(axis=0).mean(), value)

    def test_rfft_irfft_roundtrip_and_gradient(self, rng: np.random.Generator) -> None:
        value = rng.normal(size=(2, 8))
        transformed = rfft(Tensor(value))
        reconstructed = irfft(transformed, 8)

        assert np.allclose(reconstructed.data, value, rtol=0.0, atol=1.0e-12)
        _assert_gradient(lambda variable: (irfft(rfft(variable), 8) * variable).mean(), value)

    def test_non_scalar_backward_requires_seed_gradient(self) -> None:
        with pytest.raises(ValueError, match="gradient is required"):
            Tensor(np.ones(2), requires_grad=True).backward()

    def test_division_by_zero_and_invalid_irfft_raise(self) -> None:
        with pytest.raises(ZeroDivisionError):
            _ = Tensor(1.0) / Tensor(0.0)
        with pytest.raises(ValueError, match="real/imaginary"):
            irfft(Tensor(np.ones((2, 3))), 4)

    def test_sigmoid_is_finite_for_extreme_inputs(self) -> None:
        values = sigmoid(Tensor(np.array([-1000.0, 0.0, 1000.0]))).data
        assert np.array_equal(values[[0, 2]], np.array([0.0, 1.0]))
        assert values[1] == 0.5

    def test_softshrink_zeroes_inactive_values(self) -> None:
        output = softshrink(Tensor(np.array([-2.0, -0.2, 0.2, 2.0])), Tensor(0.5))
        assert np.array_equal(output.data, np.array([-1.5, 0.0, 0.0, 1.5]))


class TestLayerGradients:
    """Check every required scientific layer independently."""

    def test_complex_spectral_linear_input_gradient(self, rng: np.random.Generator) -> None:
        value = rng.normal(size=(2, 3, 2)) + 0.2
        weight_real = Tensor(rng.normal(size=(3, 2)))
        weight_imag = Tensor(rng.normal(size=(3, 2)))
        _assert_gradient(
            lambda spectrum: complex_spectral_linear(spectrum, weight_real, weight_imag).mean(),
            value,
        )

    def test_complex_spectral_linear_real_weight_gradient(self, rng: np.random.Generator) -> None:
        spectrum = Tensor(rng.normal(size=(2, 3, 2)) + 0.1)
        weight_imag = Tensor(rng.normal(size=(3, 2)))
        _assert_gradient(
            lambda weight_real: complex_spectral_linear(spectrum, weight_real, weight_imag).mean(),
            rng.normal(size=(3, 2)),
        )

    def test_complex_spectral_linear_imaginary_weight_gradient(
        self, rng: np.random.Generator
    ) -> None:
        spectrum = Tensor(rng.normal(size=(2, 3, 2)) + 0.1)
        weight_real = Tensor(rng.normal(size=(3, 2)))
        _assert_gradient(
            lambda weight_imag: complex_spectral_linear(spectrum, weight_real, weight_imag).mean(),
            rng.normal(size=(3, 2)),
        )

    def test_adaptive_soft_gate_spectrum_gradient(self, rng: np.random.Generator) -> None:
        theta = Tensor(np.full((1, 3, 1), np.log(19.0)))
        threshold = Tensor(np.asarray(0.03))
        value = rng.normal(size=(2, 3, 2)) + np.sign(rng.normal(size=(2, 3, 2))) * 0.4
        _assert_gradient(
            lambda spectrum: gated_softshrink(spectrum, theta, threshold).mean(), value
        )

    def test_adaptive_soft_gate_theta_gradient(self, rng: np.random.Generator) -> None:
        spectrum = Tensor(rng.normal(size=(2, 3, 2)) + 1.0)
        threshold = Tensor(np.asarray(0.03))
        _assert_gradient(
            lambda theta: gated_softshrink(spectrum, theta, threshold).mean(),
            np.full((1, 3, 1), 1.7),
        )

    def test_adaptive_soft_gate_threshold_gradient(self, rng: np.random.Generator) -> None:
        spectrum = Tensor(rng.normal(size=(2, 3, 2)) + 1.5)
        theta = Tensor(np.full((1, 3, 1), 1.7))
        _assert_gradient(
            lambda threshold: gated_softshrink(spectrum, theta, threshold).mean(),
            np.asarray(0.03),
        )

    def test_markov_state_gradient(self, rng: np.random.Generator) -> None:
        gain = Tensor(np.array([0.13, -0.07, 0.2]))
        _assert_gradient(
            lambda state: markov_correction(state, gain).mean(),
            rng.normal(size=(4, 3)),
        )

    def test_markov_gain_gradient(self, rng: np.random.Generator) -> None:
        state = Tensor(rng.normal(size=(4, 3)))
        _assert_gradient(
            lambda gain: markov_correction(state, gain).mean(),
            np.array([0.13, -0.07, 0.2]),
        )

    def test_deeponet_branch_projection_gradient(self, rng: np.random.Generator) -> None:
        branch_input = Tensor(rng.normal(size=(3, 4)))
        trunk_code = Tensor(rng.normal(size=(3, 5)))
        _assert_gradient(
            lambda branch_weight: ((branch_input @ branch_weight) * trunk_code).mean(),
            rng.normal(size=(4, 5)),
        )

    def test_deeponet_trunk_projection_gradient(self, rng: np.random.Generator) -> None:
        branch_code = Tensor(rng.normal(size=(3, 5)))
        coordinates = Tensor(rng.normal(size=(3, 2)))
        _assert_gradient(
            lambda trunk_weight: (branch_code * (coordinates @ trunk_weight)).mean(),
            rng.normal(size=(2, 5)),
        )

    def test_cnn_kernel_gradient_via_im2col(self, rng: np.random.Generator) -> None:
        neighborhoods = Tensor(rng.normal(size=(7, 3)))
        _assert_gradient(
            lambda kernel: ((neighborhoods @ kernel) * (neighborhoods @ kernel)).mean(),
            rng.normal(size=(3, 2)) + 0.3,
        )

    def test_pointwise_linear_weight_and_bias_gradients(self, rng: np.random.Generator) -> None:
        points = Tensor(rng.normal(size=(6, 3)))
        bias = Tensor(rng.normal(size=(1, 2)))
        _assert_gradient(
            lambda weight: (points @ weight + bias).gelu().mean(),
            rng.normal(size=(3, 2)) + 0.2,
        )
        weight = Tensor(rng.normal(size=(3, 2)))
        _assert_gradient(
            lambda variable_bias: (points @ weight + variable_bias).gelu().mean(),
            rng.normal(size=(1, 2)) + 0.2,
        )
