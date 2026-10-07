"""Evaluation metrics shared by all learned and numerical operators.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.errors import EvaluationError
from core.types import EvalResult, FloatArray


def relative_l2(prediction: FloatArray, target: FloatArray) -> float:
    """Mean per-sample relative L2 error; lower is better."""
    predicted, expected = _validated_pair(prediction, target)
    difference = (predicted - expected).reshape(predicted.shape[0], -1)
    reference = expected.reshape(expected.shape[0], -1)
    numerator = np.linalg.norm(difference, axis=1)
    denominator = np.maximum(np.linalg.norm(reference, axis=1), 1.0e-12)
    return float(np.mean(numerator / denominator))


def relative_h1(prediction: FloatArray, target: FloatArray) -> float:
    """Mean relative H1 error using resolution-scaled finite differences."""
    predicted, expected = _validated_pair(prediction, target)
    value_difference = (predicted - expected).reshape(predicted.shape[0], -1)
    value_reference = expected.reshape(expected.shape[0], -1)
    numerator_squared = np.sum(value_difference**2, axis=1)
    denominator_squared = np.sum(value_reference**2, axis=1)
    if predicted.ndim == 2:
        spacing = 1.0 / predicted.shape[1]
        gradient_difference = (
            np.roll(predicted - expected, -1, axis=1) - (predicted - expected)
        ) / spacing
        gradient_reference = (np.roll(expected, -1, axis=1) - expected) / spacing
        numerator_squared += np.sum(gradient_difference**2, axis=1)
        denominator_squared += np.sum(gradient_reference**2, axis=1)
    elif predicted.ndim == 3:
        spacing = 1.0 / max(predicted.shape[1] - 1, 1)
        for axis in (1, 2):
            gradient_difference = np.gradient(predicted - expected, spacing, axis=axis)
            gradient_reference = np.gradient(expected, spacing, axis=axis)
            numerator_squared += np.sum(gradient_difference**2, axis=(1, 2))
            denominator_squared += np.sum(gradient_reference**2, axis=(1, 2))
    return float(
        np.mean(np.sqrt(numerator_squared) / np.maximum(np.sqrt(denominator_squared), 1.0e-12))
    )


def spectral_tail_error(
    prediction: FloatArray,
    target: FloatArray,
    tail_modes: int = 8,
) -> float:
    """Relative error over the highest-frequency Fourier tail.

    Definition and units
    ---------------------
    For every sample we select the ``tail_modes`` highest-frequency Fourier
    coefficients of both prediction and target and compute

        || pred_tail - target_tail ||_2 / || target_full ||_2 ,

    i.e. the predicted high-frequency error *relative to the total energy of the
    ground truth*.  Normalising by the full-signal energy (rather than by the
    tail energy) keeps the quantity finite and bounded for smooth fields, whose
    high-frequency tail lives at machine precision: dividing by that near-zero
    tail energy would blow up to ~1e12 and is meaningless.  The result is
    dimensionless; smaller is better and it is exactly 0 only when the predicted
    high-frequency content matches the ground truth.
    """
    predicted, expected = _validated_pair(prediction, target)
    if tail_modes <= 0:
        raise EvaluationError("tail_modes must be positive.")
    if predicted.ndim == 2:
        predicted_spectrum = np.fft.rfft(predicted, axis=1)
        target_spectrum = np.fft.rfft(expected, axis=1)
        target_full = target_spectrum.reshape(expected.shape[0], -1)
        count = min(tail_modes, predicted_spectrum.shape[1])
        predicted_tail = predicted_spectrum[:, -count:]
        target_tail = target_spectrum[:, -count:]
    else:
        predicted_spectrum = np.fft.rfft2(predicted, axes=(1, 2))
        target_spectrum = np.fft.rfft2(expected, axes=(1, 2))
        target_full = target_spectrum.reshape(expected.shape[0], -1)
        radius = _frequency_radius(predicted.shape[1], predicted.shape[2])
        flat_radius = radius.ravel()
        count = min(tail_modes * tail_modes, flat_radius.size)
        tail_indices = np.argpartition(flat_radius, -count)[-count:]
        predicted_tail = predicted_spectrum.reshape(predicted.shape[0], -1)[:, tail_indices]
        target_tail = target_spectrum.reshape(expected.shape[0], -1)[:, tail_indices]
    numerator = np.linalg.norm(
        (predicted_tail - target_tail).reshape(predicted.shape[0], -1), axis=1
    )
    # Normalise by the TOTAL energy of the ground truth, not by the (near-zero)
    # tail energy, so the metric stays finite for smooth targets.
    denominator = np.linalg.norm(target_full, axis=1)
    # Guard order matters: check isfinite BEFORE comparing against a threshold,
    # because ``NaN > 1e-14`` evaluates to False and would silently reclassify a
    # NaN as if the denominator were tiny.  An undefined ratio is returned as NaN
    # so the aggregation can exclude it instead of fabricating a finite value.
    safe_denominator = np.maximum(denominator, 1.0e-12)
    ratio = numerator / safe_denominator
    if not np.all(np.isfinite(ratio)):
        return float("nan")
    return float(np.mean(ratio))


def parseval_error(values: FloatArray) -> float:
    """Maximum absolute Parseval identity residual for real 1-D batches."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise EvaluationError("Parseval invariant expects batched 1-D real signals.")
    spectrum = np.fft.rfft(array, axis=1)
    weights = np.full(spectrum.shape[1], 2.0, dtype=np.float64)
    weights[0] = 1.0
    if array.shape[1] % 2 == 0:
        weights[-1] = 1.0
    frequency_energy = np.sum(weights[None, :] * np.abs(spectrum) ** 2, axis=1) / array.shape[1]
    spatial_energy = np.sum(array**2, axis=1)
    return float(np.max(np.abs(spatial_energy - frequency_energy)))


def evaluate_prediction(
    model_name: str,
    task_name: str,
    seed: int,
    prediction: FloatArray,
    target: FloatArray,
    superresolution_l2: float | None = None,
    elapsed_sec: float = 0.0,
) -> EvalResult:
    """Evaluate every common metric with one consistent contract."""
    return EvalResult(
        model_name=model_name,
        task_name=task_name,
        seed=seed,
        relative_l2=relative_l2(prediction, target),
        relative_h1=relative_h1(prediction, target),
        spectral_tail_error=spectral_tail_error(prediction, target),
        superresolution_l2=superresolution_l2,
        elapsed_sec=elapsed_sec,
    )


def _validated_pair(prediction: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray]:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim not in (2, 3):
        raise EvaluationError("Metrics require equal batched 1-D or 2-D field shapes.")
    if predicted.shape[0] == 0:
        raise EvaluationError("Metrics require at least one sample.")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(expected)):
        raise EvaluationError("Metrics require finite arrays.")
    return predicted, expected


def _frequency_radius(height: int, width: int) -> FloatArray:
    vertical = np.fft.fftfreq(height)[:, None]
    horizontal = np.fft.rfftfreq(width)[None, :]
    return np.asarray(np.sqrt(vertical**2 + horizontal**2))


__all__ = [
    "evaluate_prediction",
    "parseval_error",
    "relative_h1",
    "relative_l2",
    "spectral_tail_error",
]
