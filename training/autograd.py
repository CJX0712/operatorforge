"""A compact real-valued reverse-mode automatic differentiation tape.

Complex Fourier coefficients are represented by a final dimension ``(real,
imag)``.  Consequently all parameters and gradients remain real.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from core.types import FloatArray

Backward = Callable[[FloatArray], None]


def _unbroadcast(gradient: FloatArray, target_shape: tuple[int, ...]) -> FloatArray:
    result = gradient
    while result.ndim > len(target_shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return np.asarray(result)


@dataclass(eq=False)
class Tensor:
    """Real ndarray plus a dynamic reverse-mode graph."""

    data: FloatArray
    requires_grad: bool = False
    grad: FloatArray | None = None
    _parents: tuple[Tensor, ...] = field(default_factory=tuple, repr=False)
    _backward: Backward = field(default=lambda _gradient: None, repr=False)

    def __init__(self, data: FloatArray | float, requires_grad: bool = False) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward = lambda _gradient: None

    @classmethod
    def _result(
        cls,
        data: FloatArray,
        parents: tuple[Tensor, ...],
        backward: Backward,
    ) -> Tensor:
        result = cls(data, any(parent.requires_grad for parent in parents))
        result._parents = parents
        result._backward = backward
        return result

    def _accumulate(self, gradient: FloatArray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.asarray(gradient, dtype=np.float64)
        else:
            self.grad += gradient

    def backward(self, gradient: FloatArray | None = None) -> None:
        """Backpropagate from this tensor through the recorded graph."""
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("A gradient is required for non-scalar tensors.")
            gradient = np.ones_like(self.data)
        ordered: list[Tensor] = []
        visited: set[int] = set()

        def visit(tensor: Tensor) -> None:
            identity = id(tensor)
            if identity in visited:
                return
            visited.add(identity)
            for parent in tensor._parents:
                visit(parent)
            ordered.append(tensor)

        visit(self)
        self.grad = np.asarray(gradient, dtype=np.float64)
        for tensor in reversed(ordered):
            if tensor.grad is not None:
                tensor._backward(tensor.grad)

    def __add__(self, other: Tensor | FloatArray | float) -> Tensor:
        right = ensure_tensor(other)
        data = self.data + right.data

        def backward(gradient: FloatArray) -> None:
            self._accumulate(_unbroadcast(gradient, self.data.shape))
            right._accumulate(_unbroadcast(gradient, right.data.shape))

        return Tensor._result(data, (self, right), backward)

    def __radd__(self, other: Tensor | FloatArray | float) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Tensor | FloatArray | float) -> Tensor:
        return self + (-ensure_tensor(other))

    def __rsub__(self, other: Tensor | FloatArray | float) -> Tensor:
        return ensure_tensor(other) - self

    def __mul__(self, other: Tensor | FloatArray | float) -> Tensor:
        right = ensure_tensor(other)
        data = self.data * right.data

        def backward(gradient: FloatArray) -> None:
            self._accumulate(_unbroadcast(gradient * right.data, self.data.shape))
            right._accumulate(_unbroadcast(gradient * self.data, right.data.shape))

        return Tensor._result(data, (self, right), backward)

    def __rmul__(self, other: Tensor | FloatArray | float) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | FloatArray | float) -> Tensor:
        right = ensure_tensor(other)
        if np.any(right.data == 0.0):
            raise ZeroDivisionError("Tensor division by zero.")
        data = self.data / right.data

        def backward(gradient: FloatArray) -> None:
            self._accumulate(_unbroadcast(gradient / right.data, self.data.shape))
            right._accumulate(
                _unbroadcast(-gradient * self.data / (right.data**2), right.data.shape)
            )

        return Tensor._result(data, (self, right), backward)

    def __matmul__(self, other: Tensor | FloatArray) -> Tensor:
        right = ensure_tensor(other)
        data = self.data @ right.data

        def backward(gradient: FloatArray) -> None:
            self._accumulate(gradient @ np.swapaxes(right.data, -1, -2))
            right._accumulate(np.swapaxes(self.data, -1, -2) @ gradient)

        return Tensor._result(data, (self, right), backward)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        data = np.asarray(self.data.sum(axis=axis, keepdims=keepdims))

        def backward(gradient: FloatArray) -> None:
            expanded = gradient
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for current_axis in sorted(item % self.data.ndim for item in axes):
                    expanded = np.expand_dims(expanded, current_axis)
            self._accumulate(np.broadcast_to(expanded, self.data.shape))

        return Tensor._result(data, (self,), backward)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.data.shape[item] for item in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def reshape(self, *shape: int) -> Tensor:
        data = self.data.reshape(*shape)

        def backward(gradient: FloatArray) -> None:
            self._accumulate(gradient.reshape(self.data.shape))

        return Tensor._result(data, (self,), backward)

    def relu(self) -> Tensor:
        data = np.maximum(self.data, 0.0)

        def backward(gradient: FloatArray) -> None:
            self._accumulate(gradient * (self.data > 0.0))

        return Tensor._result(data, (self,), backward)

    def gelu(self) -> Tensor:
        coefficient = np.sqrt(2.0 / np.pi)
        inner = coefficient * (self.data + 0.044715 * self.data**3)
        tangent = np.tanh(inner)
        data = 0.5 * self.data * (1.0 + tangent)

        def backward(gradient: FloatArray) -> None:
            inner_derivative = coefficient * (1.0 + 3.0 * 0.044715 * self.data**2)
            derivative = (
                0.5 * (1.0 + tangent) + 0.5 * self.data * (1.0 - tangent**2) * inner_derivative
            )
            self._accumulate(gradient * derivative)

        return Tensor._result(data, (self,), backward)


def ensure_tensor(value: Tensor | FloatArray | float) -> Tensor:
    """Convert arrays and scalars to constant tensors."""
    return value if isinstance(value, Tensor) else Tensor(value)


@lru_cache(maxsize=16)
def _rfft_basis(length: int) -> NDArray[np.complex128]:
    return np.fft.rfft(np.eye(length, dtype=np.float64), axis=1)


@lru_cache(maxsize=16)
def _irfft_bases(length: int) -> tuple[FloatArray, FloatArray]:
    frequency_count = length // 2 + 1
    real_basis = np.empty((frequency_count, length), dtype=np.float64)
    imag_basis = np.empty((frequency_count, length), dtype=np.float64)
    for index in range(frequency_count):
        real_impulse = np.zeros(frequency_count, dtype=np.complex128)
        imag_impulse = np.zeros(frequency_count, dtype=np.complex128)
        real_impulse[index] = 1.0
        imag_impulse[index] = 1.0j
        real_basis[index] = np.fft.irfft(real_impulse, n=length)
        imag_basis[index] = np.fft.irfft(imag_impulse, n=length)
    return real_basis, imag_basis


def rfft(tensor: Tensor) -> Tensor:
    """Real FFT encoded as ``[..., frequency, real_or_imag]``."""
    length = tensor.data.shape[-1]
    basis = _rfft_basis(length)
    transformed = np.fft.rfft(tensor.data, axis=-1)
    data = np.stack((transformed.real, transformed.imag), axis=-1)

    def backward(gradient: FloatArray) -> None:
        input_gradient = gradient[..., 0] @ basis.real.T + gradient[..., 1] @ basis.imag.T
        tensor._accumulate(input_gradient)

    return Tensor._result(data, (tensor,), backward)


def irfft(tensor: Tensor, length: int) -> Tensor:
    """Inverse real FFT from a real/imaginary stacked representation."""
    if tensor.data.shape[-1] != 2:
        raise ValueError("irfft expects a final real/imaginary dimension of size two.")
    real_basis, imag_basis = _irfft_bases(length)
    data = tensor.data[..., 0] @ real_basis + tensor.data[..., 1] @ imag_basis

    def backward(gradient: FloatArray) -> None:
        real_gradient = gradient @ real_basis.T
        imag_gradient = gradient @ imag_basis.T
        tensor._accumulate(np.stack((real_gradient, imag_gradient), axis=-1))

    return Tensor._result(data, (tensor,), backward)


def complex_spectral_linear(spectrum: Tensor, weight_real: Tensor, weight_imag: Tensor) -> Tensor:
    """Apply a complex channel matrix using only real tensor operations."""
    if spectrum.data.shape[-1] != 2:
        raise ValueError("Spectrum must use a final real/imaginary dimension.")
    input_real = spectrum.data[..., 0]
    input_imag = spectrum.data[..., 1]
    output_real = input_real @ weight_real.data - input_imag @ weight_imag.data
    output_imag = input_real @ weight_imag.data + input_imag @ weight_real.data
    data = np.stack((output_real, output_imag), axis=-1)

    def backward(gradient: FloatArray) -> None:
        gradient_real = gradient[..., 0]
        gradient_imag = gradient[..., 1]
        input_gradient_real = gradient_real @ np.swapaxes(
            weight_real.data, -1, -2
        ) + gradient_imag @ np.swapaxes(weight_imag.data, -1, -2)
        input_gradient_imag = -gradient_real @ np.swapaxes(
            weight_imag.data, -1, -2
        ) + gradient_imag @ np.swapaxes(weight_real.data, -1, -2)
        spectrum._accumulate(np.stack((input_gradient_real, input_gradient_imag), axis=-1))

        input_real_flat = input_real.reshape(-1, input_real.shape[-1])
        input_imag_flat = input_imag.reshape(-1, input_imag.shape[-1])
        gradient_real_flat = gradient_real.reshape(-1, gradient_real.shape[-1])
        gradient_imag_flat = gradient_imag.reshape(-1, gradient_imag.shape[-1])
        real_gradient = (
            input_real_flat.T @ gradient_real_flat + input_imag_flat.T @ gradient_imag_flat
        )
        imag_gradient = (
            -input_imag_flat.T @ gradient_real_flat + input_real_flat.T @ gradient_imag_flat
        )
        weight_real._accumulate(real_gradient)
        weight_imag._accumulate(imag_gradient)

    return Tensor._result(data, (spectrum, weight_real, weight_imag), backward)


def sigmoid(tensor: Tensor) -> Tensor:
    """Numerically stable sigmoid."""
    data = np.empty_like(tensor.data)
    positive = tensor.data >= 0.0
    data[positive] = 1.0 / (1.0 + np.exp(-tensor.data[positive]))
    negative_exp = np.exp(tensor.data[~positive])
    data[~positive] = negative_exp / (1.0 + negative_exp)

    def backward(gradient: FloatArray) -> None:
        tensor._accumulate(gradient * data * (1.0 - data))

    return Tensor._result(data, (tensor,), backward)


def softshrink(tensor: Tensor, threshold: Tensor) -> Tensor:
    """Differentiable almost-everywhere soft shrinkage with positive threshold."""
    absolute = np.abs(tensor.data)
    data = np.sign(tensor.data) * np.maximum(absolute - threshold.data, 0.0)
    active = absolute > threshold.data

    def backward(gradient: FloatArray) -> None:
        tensor._accumulate(gradient * active)
        threshold_gradient = -gradient * np.sign(tensor.data) * active
        threshold._accumulate(_unbroadcast(threshold_gradient, threshold.data.shape))

    return Tensor._result(data, (tensor, threshold), backward)


def gated_softshrink(spectrum: Tensor, theta: Tensor, threshold: Tensor) -> Tensor:
    """Adaptive per-mode gain followed by soft thresholding."""
    return softshrink(spectrum * sigmoid(theta), threshold)


def markov_correction(discarded_state: Tensor, gain: Tensor) -> Tensor:
    """First-order Markov closure term for truncated spectral state."""
    return discarded_state * gain


def gradient_check(
    function: Callable[[Tensor], Tensor],
    value: FloatArray,
    epsilon: float = 1.0e-6,
) -> float:
    """Compare tape gradients with centered finite differences.

    The returned maximum symmetric relative error is suitable for the G3
    ``<= 1e-6`` invariant.
    """
    initial = np.asarray(value, dtype=np.float64)
    variable = Tensor(initial.copy(), requires_grad=True)
    output = function(variable)
    if output.data.size != 1:
        raise ValueError("gradient_check requires a scalar-valued function.")
    output.backward()
    if variable.grad is None:
        raise RuntimeError("The checked function did not propagate a gradient.")
    numerical = np.zeros_like(initial)
    for index in np.ndindex(initial.shape):
        positive = initial.copy()
        negative = initial.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        positive_value = float(function(Tensor(positive)).data)
        negative_value = float(function(Tensor(negative)).data)
        numerical[index] = (positive_value - negative_value) / (2.0 * epsilon)
    denominator = np.maximum(np.abs(variable.grad) + np.abs(numerical), 1.0e-12)
    return float(np.max(np.abs(variable.grad - numerical) / denominator))


__all__ = [
    "Tensor",
    "complex_spectral_linear",
    "ensure_tensor",
    "gated_softshrink",
    "gradient_check",
    "irfft",
    "markov_correction",
    "rfft",
    "sigmoid",
    "softshrink",
]
