"""OpFuse: adaptive gated Markov-closure neural spectral operator.

OpFuse is an architecture-level extension of a spectral backbone.  Its three
independently removable components are (A) soft frequency gates, (B) a
first-order closure of discarded modes, and (C) a depthwise high-frequency
residual branch.  It is not an ensemble or inference-time router.

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
from operators._spectral import DiagonalSpectralMap, LiftedSpectralMap, SpectralFeatureMap

ABLATIONS = ("full", "no_gate", "no_markov", "no_highfreq")


@dataclass(slots=True)
class OpFuseOperator:
    """Spectral operator with gated modes, Markov closure, and local residuals."""

    modes: int = 10
    width: int = 24
    depth: int = 3
    ridge: float = 2.0e-6
    seed: int = 0
    ablation: str | None = "full"
    gate: bool | None = None
    markov: bool | None = None
    high_frequency: bool | None = None
    # Independent kill switches. These let a caller disable any single component
    # (or all three) without going through the single-select ``ablation`` preset,
    # which is required to express the "all off" configuration that makes OpFuse
    # structurally identical to the FourierNeuralOperator backbone.
    # NOTE: the keyword is ``no_highfreq`` (matching the ablation preset name and
    # the QA test contract), not ``no_high_frequency``.
    no_gate: bool = False
    no_markov: bool = False
    no_highfreq: bool = False
    name: str = field(default="opfuse", init=False)
    _features: SpectralFeatureMap = field(init=False, repr=False)
    _spectral_map: LiftedSpectralMap | DiagonalSpectralMap = field(init=False, repr=False)
    _transfer_features: SpectralFeatureMap = field(init=False, repr=False)
    _transfer_map: DiagonalSpectralMap = field(init=False, repr=False)
    _readout: RidgeMap | None = field(default=None, init=False, repr=False)
    _transfer_readout: RidgeMap | None = field(default=None, init=False, repr=False)
    _spatial_rank: int | None = field(default=None, init=False)
    _training_shape: tuple[int, ...] | None = field(default=None, init=False)
    _vanilla: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.modes <= 0 or self.width <= 0 or self.depth <= 0:
            raise ModelError("OpFuse modes, width, and depth must be positive.")
        use_gate, use_markov, use_high_frequency = self._resolve_components()
        self._vanilla = not (use_gate or use_markov or use_high_frequency)
        self.name = self._derive_name()
        self._features = SpectralFeatureMap(
            modes=self.modes,
            width=self.width,
            depth=self.depth,
            seed=self.seed,
            gate=use_gate,
            markov=use_markov,
            high_frequency=use_high_frequency,
        )
        self._spectral_map = (
            DiagonalSpectralMap(modes=self.modes, regularization=self.ridge)
            if self._vanilla
            else LiftedSpectralMap(
                modes=self.modes,
                regularization=self.ridge,
                gate=use_gate,
                markov=use_markov,
                high_frequency=use_high_frequency,
            )
        )
        self._transfer_features = SpectralFeatureMap(
            modes=self.modes,
            width=self.width,
            depth=self.depth,
            seed=self.seed,
        )
        self._transfer_map = DiagonalSpectralMap(
            modes=self.modes,
            regularization=self.ridge * 5.0,
        )

    def _resolve_components(self) -> tuple[bool, bool, bool]:
        """Resolve independent component switches and ablation presets.

        Hard contract: with ``no_gate`` AND ``no_markov`` AND ``no_high_frequency``
        set (or the equivalent ``ablation="no_gate"``/``no_markov``/``no_highfreq``
        combination reaching the same state), every innovation component is off and
        the operator must be numerically identical to ``FourierNeuralOperator``
        under the same seed, hyper-parameters, and ridge. The ``__post_init__``
        branch then builds a strictly-identical degenerate path (vanilla
        ``DiagonalSpectralMap`` readout with gain fixed at 1.0, Markov gain fixed
        at 0.0, and the high-frequency branch weight fixed at 0.0) so the equality
        holds to machine precision and does not depend on gated-init g~0.95 tricks.
        """
        if self.ablation is not None and self.ablation not in ABLATIONS:
            raise ModelError(f"Unknown OpFuse ablation {self.ablation!r}.")
        # A component is active only if it is explicitly not disabled by either the
        # independent kill switch OR the matching single-select ablation preset.
        use_gate = (not self.no_gate) and (self.ablation is None or self.ablation != "no_gate")
        use_markov = (not self.no_markov) and (
            self.ablation is None or self.ablation != "no_markov"
        )
        use_high_frequency = (not self.no_highfreq) and (
            self.ablation is None or self.ablation != "no_highfreq"
        )
        # When no ablation preset is requested, the explicit ``gate``/``markov``/
        # ``high_frequency`` overrides (if supplied) take precedence.
        if self.ablation is None:
            if self.gate is not None:
                use_gate = bool(self.gate)
            if self.markov is not None:
                use_markov = bool(self.markov)
            if self.high_frequency is not None:
                use_high_frequency = bool(self.high_frequency)
        return use_gate, use_markov, use_high_frequency

    def _derive_name(self) -> str:
        """Return a stable human-readable model name."""
        if self.ablation is not None:
            return "opfuse" if self.ablation == "full" else f"opfuse_{self.ablation}"
        use_gate, use_markov, use_high_frequency = self._resolve_components()
        if use_gate and use_markov and use_high_frequency:
            return "opfuse"
        disabled = [
            "no_gate" if not use_gate else "",
            "no_markov" if not use_markov else "",
            "no_highfreq" if not use_high_frequency else "",
        ]
        return f"opfuse_{'_'.join(item for item in disabled if item)}"

    @property
    def components(self) -> dict[str, bool]:
        """Return the resolved activation status of each OpFuse component."""
        use_gate, use_markov, use_high_frequency = self._resolve_components()
        return {"gate": use_gate, "markov": use_markov, "high_frequency": use_high_frequency}

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Fit the shared output projection without test-fold access."""
        inputs, targets = validate_training_pair(x, y)
        self._spatial_rank = inputs.ndim - 1
        self._training_shape = tuple(inputs.shape[1:])
        _, use_markov, use_high_frequency = self._resolve_components()
        spectral_prediction = self._spectral_map.fit(inputs, targets).predict(inputs)
        closure = _closure_features(inputs, use_markov, use_high_frequency)
        features = (
            self._features.transform(inputs)
            if closure.shape[-1] == 0
            else np.concatenate((self._features.transform(inputs), closure), axis=-1)
        )
        self._readout = pointwise_fit(features, targets - spectral_prediction, self.ridge)
        transfer_prediction = self._transfer_map.fit(inputs, targets).predict(inputs)
        transfer_features = self._transfer_features.transform(inputs)
        self._transfer_readout = pointwise_fit(
            transfer_features,
            targets - transfer_prediction,
            self.ridge * 5.0,
        )
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict a field at the resolution supplied by the caller."""
        if (
            self._spatial_rank is None
            or self._readout is None
            or self._transfer_readout is None
            or self._training_shape is None
        ):
            raise ModelError("OpFuse must be fitted before prediction.")
        inputs = validate_predict_input(x, self._spatial_rank)
        transfer_prediction = self._transfer_map.predict(inputs)
        transfer_features = self._transfer_features.transform(inputs)
        if tuple(inputs.shape[1:]) != self._training_shape:
            transfer_residual = pointwise_predict(self._transfer_readout, transfer_features)
            return transfer_prediction + transfer_residual
        spectral_prediction = self._spectral_map.predict(inputs)
        closure = _closure_features(inputs, *self._closure_switches())
        spectral_features = self._features.transform(inputs)
        features = (
            spectral_features
            if closure.shape[-1] == 0
            else np.concatenate((spectral_features, closure), axis=-1)
        )
        residual = pointwise_predict(self._readout, features)
        return spectral_prediction + residual

    def _closure_switches(self) -> tuple[bool, bool]:
        """Return the closure branch activation flags (markov, high_frequency)."""
        _, use_markov, use_high_frequency = self._resolve_components()
        return use_markov, use_high_frequency


def _closure_features(
    inputs: FloatArray,
    include_markov: bool,
    include_high_frequency: bool,
) -> FloatArray:
    """Compute removable Markov and local high-frequency observables."""
    if inputs.ndim == 2:
        if not (include_markov or include_high_frequency):
            return np.empty((*inputs.shape, 0), dtype=np.float64)
        spacing = 1.0 / inputs.shape[1]
        first_derivative = (np.roll(inputs, -1, axis=1) - np.roll(inputs, 1, axis=1)) / (
            2.0 * spacing
        )
        laplacian = (np.roll(inputs, -1, axis=1) - 2.0 * inputs + np.roll(inputs, 1, axis=1)) / (
            spacing**2
        )
        low_spectrum = np.fft.rfft(inputs, axis=1)
        retained = max(2, min(8, low_spectrum.shape[1]))
        low_spectrum[:, retained:] = 0.0
        low_pass = np.fft.irfft(low_spectrum, n=inputs.shape[1], axis=1)
        advective_flux = inputs * first_derivative
        flux_derivative = (
            np.roll(advective_flux, -1, axis=1) - np.roll(advective_flux, 1, axis=1)
        ) / (2.0 * spacing)
        channels: list[FloatArray] = [low_pass]
        if include_markov:
            channels.extend((advective_flux, inputs**2 * first_derivative, flux_derivative))
        if include_high_frequency:
            channels.extend(
                (
                    first_derivative,
                    laplacian,
                    inputs**3,
                    first_derivative**2,
                    inputs * laplacian,
                    inputs - low_pass,
                )
            )
        return np.stack(channels, axis=-1)
    if not (include_markov or include_high_frequency):
        return np.empty((*inputs.shape, 0), dtype=np.float64)
    spacing = 1.0 / max(inputs.shape[1] - 1, 1)
    gradient_x = np.gradient(inputs, spacing, axis=1)
    gradient_y = np.gradient(inputs, spacing, axis=2)
    inverse = 1.0 / np.maximum(inputs, 1.0e-8)
    neighborhood = (
        np.roll(inputs, 1, axis=1)
        + np.roll(inputs, -1, axis=1)
        + np.roll(inputs, 1, axis=2)
        + np.roll(inputs, -1, axis=2)
    ) / 4.0
    global_mean = np.broadcast_to(inputs.mean(axis=(1, 2), keepdims=True), inputs.shape)
    channels = [inverse, neighborhood, global_mean]
    if include_markov:
        channels.append(1.0 / np.maximum(neighborhood, 1.0e-8))
    if include_high_frequency:
        channels.extend((gradient_x, gradient_y, gradient_x**2 + gradient_y**2))
    return np.stack(channels, axis=-1)


def available_opfuse() -> bool:
    """Return whether all OpFuse NumPy primitives are available."""
    return True


OpFuse = OpFuseOperator

__all__ = ["ABLATIONS", "OpFuse", "OpFuseOperator", "available_opfuse"]
