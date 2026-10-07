"""Deterministic Fourier feature blocks shared by spectral architectures.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from core.types import FloatArray
from operators._base import gelu


@dataclass(slots=True)
class SpectralFeatureMap:
    """Fixed Fourier layers whose pointwise readout is learned by ridge."""

    modes: int = 10
    width: int = 24
    depth: int = 3
    seed: int = 0
    gate: bool = False
    markov: bool = False
    high_frequency: bool = False
    _spatial_rank: int | None = field(default=None, init=False)
    _lift: FloatArray | None = field(default=None, init=False)
    _pointwise: list[FloatArray] = field(default_factory=list, init=False)
    _spectral_real: list[FloatArray] = field(default_factory=list, init=False)
    _spectral_imag: list[FloatArray] = field(default_factory=list, init=False)
    _theta: FloatArray | None = field(default=None, init=False)
    _threshold: float = field(default=1.0e-3, init=False)
    _markov_gain: FloatArray | None = field(default=None, init=False)

    def initialize(self, spatial_rank: int) -> None:
        """Initialize all real-valued spectral parameters deterministically."""
        if self._spatial_rank is not None:
            if self._spatial_rank != spatial_rank:
                raise ValueError("A spectral feature map cannot change spatial rank.")
            return
        self._spatial_rank = spatial_rank
        rng = np.random.default_rng(self.seed)
        input_channels = 5 if spatial_rank == 1 else 7
        self._lift = rng.normal(
            scale=0.55 / np.sqrt(input_channels), size=(input_channels, self.width)
        )
        mode_shape = (self.modes,) if spatial_rank == 1 else (self.modes, self.modes)
        identity = np.eye(self.width, dtype=np.float64)
        for _ in range(self.depth):
            pointwise = 0.22 * identity + rng.normal(
                scale=0.045 / np.sqrt(self.width), size=(self.width, self.width)
            )
            spectral_real = rng.normal(
                scale=0.10 / np.sqrt(self.width), size=(*mode_shape, self.width, self.width)
            )
            spectral_real += 0.55 * identity
            spectral_imag = rng.normal(
                scale=0.035 / np.sqrt(self.width), size=(*mode_shape, self.width, self.width)
            )
            self._pointwise.append(pointwise)
            self._spectral_real.append(spectral_real)
            self._spectral_imag.append(spectral_imag)
        self._theta = np.full(mode_shape, np.log(0.95 / 0.05), dtype=np.float64)
        self._markov_gain = rng.normal(scale=0.08, size=self.width)

    def transform(self, inputs: FloatArray) -> FloatArray:
        """Return multiscale spectral features at the input resolution."""
        spatial_rank = inputs.ndim - 1
        self.initialize(spatial_rank)
        base_features = _input_features(inputs)
        if self._lift is None:
            raise RuntimeError("Spectral feature map initialization failed.")
        state = base_features @ self._lift
        collected = [base_features, state]
        for layer_index in range(self.depth):
            spectral, discarded = self._spectral_convolution(state, layer_index)
            update = spectral + state @ self._pointwise[layer_index]
            if self.markov and self._markov_gain is not None:
                update += discarded * self._markov_gain
            if self.high_frequency:
                update += 0.35 * _depthwise_high_frequency(state)
            state = state + 0.45 * gelu(update)
            collected.append(state)
        if self.high_frequency:
            collected.append(_depthwise_high_frequency(state))
        return np.concatenate(collected, axis=-1)

    def _spectral_convolution(
        self, state: FloatArray, layer_index: int
    ) -> tuple[FloatArray, FloatArray]:
        if self._spatial_rank == 1:
            return self._spectral_convolution_1d(state, layer_index)
        return self._spectral_convolution_2d(state, layer_index)

    def _spectral_convolution_1d(
        self, state: FloatArray, layer_index: int
    ) -> tuple[FloatArray, FloatArray]:
        length = state.shape[1]
        spectrum = np.fft.rfft(state, axis=1)
        count = min(self.modes, spectrum.shape[1])
        selected = spectrum[:, :count, :]
        if self.gate and self._theta is not None:
            gains = 1.0 / (1.0 + np.exp(-self._theta[:count]))
            selected = np.sign(selected.real) * np.maximum(
                np.abs(selected.real) * gains[None, :, None] - self._threshold, 0.0
            ) + 1j * np.sign(selected.imag) * np.maximum(
                np.abs(selected.imag) * gains[None, :, None] - self._threshold, 0.0
            )
        weight_real = self._spectral_real[layer_index][:count]
        weight_imag = self._spectral_imag[layer_index][:count]
        output_real = np.einsum("nmc,mco->nmo", selected.real, weight_real)
        output_real -= np.einsum("nmc,mco->nmo", selected.imag, weight_imag)
        output_imag = np.einsum("nmc,mco->nmo", selected.real, weight_imag)
        output_imag += np.einsum("nmc,mco->nmo", selected.imag, weight_real)
        output_low = output_real + 1j * output_imag
        output_spectrum = np.zeros_like(spectrum)
        output_spectrum[:, :count, :] = output_low
        output = np.fft.irfft(output_spectrum, n=length, axis=1).real
        discarded_spectrum = spectrum.copy()
        discarded_spectrum[:, :count, :] = 0.0
        discarded = np.fft.irfft(discarded_spectrum, n=length, axis=1).real
        return np.asarray(output), np.asarray(discarded)

    def _spectral_convolution_2d(
        self, state: FloatArray, layer_index: int
    ) -> tuple[FloatArray, FloatArray]:
        height, width = state.shape[1:3]
        spectrum = np.fft.rfft2(state, axes=(1, 2))
        row_count = min(self.modes, height)
        column_count = min(self.modes, spectrum.shape[2])
        selected = spectrum[:, :row_count, :column_count, :]
        if self.gate and self._theta is not None:
            gains = 1.0 / (1.0 + np.exp(-self._theta[:row_count, :column_count]))
            scaled = selected * gains[None, :, :, None]
            selected = np.sign(scaled.real) * np.maximum(
                np.abs(scaled.real) - self._threshold, 0.0
            ) + 1j * np.sign(scaled.imag) * np.maximum(np.abs(scaled.imag) - self._threshold, 0.0)
        weight_real = self._spectral_real[layer_index][:row_count, :column_count]
        weight_imag = self._spectral_imag[layer_index][:row_count, :column_count]
        output_real = np.einsum("nxyc,xyco->nxyo", selected.real, weight_real)
        output_real -= np.einsum("nxyc,xyco->nxyo", selected.imag, weight_imag)
        output_imag = np.einsum("nxyc,xyco->nxyo", selected.real, weight_imag)
        output_imag += np.einsum("nxyc,xyco->nxyo", selected.imag, weight_real)
        output_low = output_real + 1j * output_imag
        output_spectrum = np.zeros_like(spectrum)
        output_spectrum[:, :row_count, :column_count, :] = output_low
        output = np.fft.irfft2(output_spectrum, s=(height, width), axes=(1, 2)).real
        discarded_spectrum = spectrum.copy()
        discarded_spectrum[:, :row_count, :column_count, :] = 0.0
        discarded = np.fft.irfft2(discarded_spectrum, s=(height, width), axes=(1, 2)).real
        return np.asarray(output), np.asarray(discarded)


@dataclass(slots=True)
class DiagonalSpectralMap:
    """Learn a translation-equivariant complex multiplier using real parameters."""

    modes: int
    regularization: float
    include_high_frequency: bool = False
    gate: bool = False
    weight_real: FloatArray | None = field(default=None, init=False)
    weight_imag: FloatArray | None = field(default=None, init=False)
    intercept_real: FloatArray | None = field(default=None, init=False)
    intercept_imag: FloatArray | None = field(default=None, init=False)
    mask: NDArray[np.bool_] | None = field(default=None, init=False)
    training_shape: tuple[int, ...] | None = field(default=None, init=False)

    def fit(self, inputs: FloatArray, targets: FloatArray) -> DiagonalSpectralMap:
        """Fit independent complex spectral multipliers without complex gradients."""
        spatial_axes = tuple(range(1, inputs.ndim))
        input_spectrum = np.fft.rfftn(inputs, axes=spatial_axes)
        target_spectrum = np.fft.rfftn(targets, axes=spatial_axes)
        self.training_shape = tuple(inputs.shape[1:])
        self.mask = self._mode_mask(input_spectrum.shape[1:])
        gated_input = self._apply_gate(input_spectrum)
        input_mean = gated_input.mean(axis=0)
        target_mean = target_spectrum.mean(axis=0)
        centered_input = gated_input - input_mean
        centered_target = target_spectrum - target_mean
        denominator = np.sum(np.abs(centered_input) ** 2, axis=0) + self.regularization
        numerator = np.sum(np.conj(centered_input) * centered_target, axis=0)
        weight = numerator / denominator
        weight = np.where(self.mask, weight, 0.0)
        intercept = np.where(self.mask, target_mean - weight * input_mean, 0.0)
        self.weight_real = np.asarray(weight.real)
        self.weight_imag = np.asarray(weight.imag)
        self.intercept_real = np.asarray(intercept.real)
        self.intercept_imag = np.asarray(intercept.imag)
        return self

    def predict(self, inputs: FloatArray) -> FloatArray:
        """Apply the learned real-parameterized multiplier at any 1-D resolution."""
        if (
            self.weight_real is None
            or self.weight_imag is None
            or self.intercept_real is None
            or self.intercept_imag is None
            or self.training_shape is None
        ):
            raise RuntimeError("DiagonalSpectralMap must be fitted before prediction.")
        spatial_axes = tuple(range(1, inputs.ndim))
        input_spectrum = self._apply_gate(np.fft.rfftn(inputs, axes=spatial_axes))
        output_spectrum = np.zeros_like(input_spectrum)
        if inputs.ndim == 2:
            count = min(input_spectrum.shape[1], self.weight_real.shape[0])
            input_real = input_spectrum[:, :count].real
            input_imag = input_spectrum[:, :count].imag
            weight_real = self.weight_real[:count]
            weight_imag = self.weight_imag[:count]
            resolution_scale = inputs.shape[1] / self.training_shape[0]
            output_spectrum[:, :count] = (
                input_real * weight_real
                - input_imag * weight_imag
                + resolution_scale * self.intercept_real[:count]
            ) + 1j * (
                input_real * weight_imag
                + input_imag * weight_real
                + resolution_scale * self.intercept_imag[:count]
            )
        else:
            rows = min(input_spectrum.shape[1], self.weight_real.shape[0])
            columns = min(input_spectrum.shape[2], self.weight_real.shape[1])
            input_real = input_spectrum[:, :rows, :columns].real
            input_imag = input_spectrum[:, :rows, :columns].imag
            weight_real = self.weight_real[:rows, :columns]
            weight_imag = self.weight_imag[:rows, :columns]
            output_spectrum[:, :rows, :columns] = (
                input_real * weight_real
                - input_imag * weight_imag
                + self.intercept_real[:rows, :columns]
            ) + 1j * (
                input_real * weight_imag
                + input_imag * weight_real
                + self.intercept_imag[:rows, :columns]
            )
        return np.asarray(
            np.fft.irfftn(output_spectrum, s=inputs.shape[1:], axes=spatial_axes).real
        )

    def _mode_mask(self, spectral_shape: tuple[int, ...]) -> NDArray[np.bool_]:
        if self.include_high_frequency:
            return np.ones(spectral_shape, dtype=bool)
        if len(spectral_shape) == 1:
            mask = np.zeros(spectral_shape, dtype=bool)
            mask[: min(self.modes, spectral_shape[0])] = True
            return mask
        row_frequency = np.abs(np.fft.fftfreq(spectral_shape[0]) * spectral_shape[0])
        column_frequency = np.arange(spectral_shape[1])
        return (row_frequency[:, None] < self.modes) & (column_frequency[None, :] < self.modes)

    def _apply_gate(self, spectrum: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if not self.gate:
            return spectrum
        gains = np.full(spectrum.shape[1:], 0.95, dtype=np.float64)
        scaled = spectrum * gains[None, ...]
        threshold = 1.0e-3
        return np.sign(scaled.real) * np.maximum(
            np.abs(scaled.real) - threshold, 0.0
        ) + 1j * np.sign(scaled.imag) * np.maximum(np.abs(scaled.imag) - threshold, 0.0)


@dataclass(slots=True)
class LiftedSpectralMap:
    """Nonlinear channel lift followed by real-parameter spectral contraction."""

    modes: int
    regularization: float
    gate: bool = True
    markov: bool = True
    high_frequency: bool = True
    vanilla: bool = False
    weight_real: FloatArray | None = field(default=None, init=False)
    weight_imag: FloatArray | None = field(default=None, init=False)
    feature_mean_real: FloatArray | None = field(default=None, init=False)
    feature_mean_imag: FloatArray | None = field(default=None, init=False)
    target_mean_real: FloatArray | None = field(default=None, init=False)
    target_mean_imag: FloatArray | None = field(default=None, init=False)
    training_shape: tuple[int, ...] | None = field(default=None, init=False)

    def fit(self, inputs: FloatArray, targets: FloatArray) -> LiftedSpectralMap:
        """Fit cross-channel complex weights independently for every mode."""
        spatial_axes = tuple(range(1, inputs.ndim))
        lifted = self._lift(inputs)
        feature_spectrum = np.fft.rfftn(lifted, axes=spatial_axes)
        target_spectrum = np.fft.rfftn(targets, axes=spatial_axes)
        feature_spectrum = self._apply_gate(feature_spectrum)
        spectral_shape = target_spectrum.shape[1:]
        channel_count = lifted.shape[-1]
        weights = np.zeros((*spectral_shape, channel_count), dtype=np.complex128)
        feature_mean = feature_spectrum.mean(axis=0)
        target_mean = target_spectrum.mean(axis=0)
        mask = self._mode_mask(spectral_shape)
        for index in np.ndindex(spectral_shape):
            if not mask[index]:
                continue
            design = feature_spectrum[(slice(None), *index, slice(None))]
            response = target_spectrum[(slice(None), *index)]
            centered_design = design - feature_mean[index]
            centered_response = response - target_mean[index]
            gram = centered_design.conj().T @ centered_design
            gram.flat[:: channel_count + 1] += self.regularization
            right_hand_side = centered_design.conj().T @ centered_response
            weights[index] = np.linalg.solve(gram, right_hand_side)
        self.weight_real = np.asarray(weights.real)
        self.weight_imag = np.asarray(weights.imag)
        self.feature_mean_real = np.asarray(feature_mean.real)
        self.feature_mean_imag = np.asarray(feature_mean.imag)
        self.target_mean_real = np.asarray(target_mean.real)
        self.target_mean_imag = np.asarray(target_mean.imag)
        self.training_shape = tuple(inputs.shape[1:])
        return self

    def predict(self, inputs: FloatArray) -> FloatArray:
        """Apply the learned nonlinear spectral operator."""
        if (
            self.weight_real is None
            or self.weight_imag is None
            or self.feature_mean_real is None
            or self.feature_mean_imag is None
            or self.target_mean_real is None
            or self.target_mean_imag is None
            or self.training_shape is None
        ):
            raise RuntimeError("LiftedSpectralMap must be fitted before prediction.")
        spatial_axes = tuple(range(1, inputs.ndim))
        lifted = self._lift(inputs)
        feature_spectrum = self._apply_gate(np.fft.rfftn(lifted, axes=spatial_axes))
        output_spectrum = np.zeros(feature_spectrum.shape[:-1], dtype=np.complex128)
        current_shape = output_spectrum.shape[1:]
        stored_shape = self.weight_real.shape[:-1]
        common_shape = tuple(
            min(current, stored)
            for current, stored in zip(current_shape, stored_shape, strict=True)
        )
        resolution_scale = float(np.prod(inputs.shape[1:]) / np.prod(self.training_shape))
        for index in np.ndindex(common_shape):
            feature = feature_spectrum[(slice(None), *index, slice(None))]
            centered_real = feature.real - self.feature_mean_real[index] * resolution_scale
            centered_imag = feature.imag - self.feature_mean_imag[index] * resolution_scale
            weight_real = self.weight_real[index]
            weight_imag = self.weight_imag[index]
            output_real = centered_real @ weight_real - centered_imag @ weight_imag
            output_imag = centered_real @ weight_imag + centered_imag @ weight_real
            output_spectrum[(slice(None), *index)] = (
                output_real + self.target_mean_real[index] * resolution_scale
            ) + 1j * (output_imag + self.target_mean_imag[index] * resolution_scale)
        return np.asarray(
            np.fft.irfftn(output_spectrum, s=inputs.shape[1:], axes=spatial_axes).real
        )

    def _lift(self, inputs: FloatArray) -> FloatArray:
        if self.vanilla:
            return inputs[..., None]
        channels: list[FloatArray] = [inputs, inputs**2, inputs**3]
        if inputs.ndim == 2:
            spacing = 1.0 / inputs.shape[1]
            derivative = (np.roll(inputs, -1, axis=1) - np.roll(inputs, 1, axis=1)) / (
                2.0 * spacing
            )
            if self.markov:
                channels.extend((derivative, inputs * derivative))
            if self.high_frequency:
                smooth = (
                    np.roll(inputs, 1, axis=1) + 2.0 * inputs + np.roll(inputs, -1, axis=1)
                ) / 4.0
                channels.extend((inputs - smooth, derivative**2))
        else:
            inverse = 1.0 / np.maximum(inputs, 1.0e-8)
            neighborhood = (
                np.roll(inputs, 1, axis=1)
                + np.roll(inputs, -1, axis=1)
                + np.roll(inputs, 1, axis=2)
                + np.roll(inputs, -1, axis=2)
            ) / 4.0
            channels.append(inverse)
            if self.markov:
                channels.extend((neighborhood, 1.0 / np.maximum(neighborhood, 1.0e-8)))
            if self.high_frequency:
                channels.append(inputs - neighborhood)
        return np.stack(channels, axis=-1)

    def _mode_mask(self, spectral_shape: tuple[int, ...]) -> NDArray[np.bool_]:
        if self.vanilla:
            mask = np.zeros(spectral_shape, dtype=bool)
            mask[: min(self.modes, spectral_shape[0])] = True
            return mask
        if self.high_frequency:
            return np.ones(spectral_shape, dtype=bool)
        if len(spectral_shape) == 1:
            mask = np.zeros(spectral_shape, dtype=bool)
            mask[: min(self.modes, spectral_shape[0])] = True
            return mask
        row_frequency = np.abs(np.fft.fftfreq(spectral_shape[0]) * spectral_shape[0])
        column_frequency = np.arange(spectral_shape[1])
        return (row_frequency[:, None] < self.modes) & (column_frequency[None, :] < self.modes)

    def _apply_gate(self, spectrum: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if not self.gate or self.vanilla:
            return spectrum
        scaled = 0.95 * spectrum
        threshold = 1.0e-3
        return np.sign(scaled.real) * np.maximum(
            np.abs(scaled.real) - threshold, 0.0
        ) + 1j * np.sign(scaled.imag) * np.maximum(np.abs(scaled.imag) - threshold, 0.0)


def _input_features(inputs: FloatArray) -> FloatArray:
    if inputs.ndim == 2:
        length = inputs.shape[1]
        coordinate = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float64)
        coordinate = np.broadcast_to(coordinate, inputs.shape)
        return np.stack(
            (
                inputs,
                inputs**2,
                coordinate,
                np.sin(2.0 * np.pi * coordinate),
                np.cos(2.0 * np.pi * coordinate),
            ),
            axis=-1,
        )
    height, width = inputs.shape[1:3]
    x_axis = np.linspace(0.0, 1.0, height, dtype=np.float64)
    y_axis = np.linspace(0.0, 1.0, width, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    x_grid = np.broadcast_to(x_grid, inputs.shape)
    y_grid = np.broadcast_to(y_grid, inputs.shape)
    return np.stack(
        (
            inputs,
            inputs**2,
            x_grid,
            y_grid,
            np.sin(np.pi * x_grid),
            np.sin(np.pi * y_grid),
            np.sin(np.pi * x_grid) * np.sin(np.pi * y_grid),
        ),
        axis=-1,
    )


def _depthwise_high_frequency(state: FloatArray) -> FloatArray:
    if state.ndim == 3:
        smooth = (np.roll(state, 1, axis=1) + 2.0 * state + np.roll(state, -1, axis=1)) / 4.0
        return state - smooth
    padded = np.pad(state, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    smooth = (
        4.0 * padded[:, 1:-1, 1:-1]
        + padded[:, :-2, 1:-1]
        + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
    ) / 8.0
    return state - smooth


__all__ = ["DiagonalSpectralMap", "LiftedSpectralMap", "SpectralFeatureMap"]
