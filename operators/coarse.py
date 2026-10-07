"""Fair fixed-budget coarse-grid numerical solver baseline.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from scipy.ndimage import zoom

from core.errors import ModelError
from core.types import FloatArray
from data.pde import advdiff_exact, burgers_spectral_rk4, darcy_finite_difference
from operators._base import validate_predict_input, validate_training_pair


@dataclass(slots=True)
class CoarseSolverOperator:
    """Solve on a deliberately coarse grid and interpolate to the requested grid."""

    task_name: str
    coarse_size_1d: int = 8
    coarse_size_2d: int = 6
    name: str = field(default="coarse", init=False)
    _spatial_rank: int | None = field(default=None, init=False)

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        """Validate the task; the numerical solver has no learned parameters."""
        inputs, _ = validate_training_pair(x, y)
        self._spatial_rank = inputs.ndim - 1
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Downsample, solve at fixed budget, and interpolate back."""
        if self._spatial_rank is None:
            raise ModelError("Coarse solver must be fitted before prediction.")
        inputs = validate_predict_input(x, self._spatial_rank)
        if self.task_name in {"burgers1d", "advdiff1d"}:
            coarse = _periodic_resample_1d(inputs, self.coarse_size_1d)
            if self.task_name == "burgers1d":
                solved = burgers_spectral_rk4(coarse, time_step=2.0e-3)
            else:
                solved = advdiff_exact(coarse)
            return _periodic_resample_1d(solved, inputs.shape[1])
        if self.task_name == "darcy2d":
            ratio = self.coarse_size_2d / inputs.shape[1]
            coarse = zoom(inputs, (1.0, ratio, ratio), order=1, mode="nearest", prefilter=False)
            coarse = np.where(coarse >= 1.75, 3.0, 0.5)
            solved = darcy_finite_difference(coarse)
            output_ratio = inputs.shape[1] / solved.shape[1]
            return np.asarray(
                zoom(
                    solved,
                    (1.0, output_ratio, output_ratio),
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
            )
        raise ModelError(f"Unsupported coarse solver task {self.task_name!r}.")


def _periodic_resample_1d(values: FloatArray, output_size: int) -> FloatArray:
    input_size = values.shape[1]
    spectrum = np.fft.rfft(values, axis=1)
    output_spectrum = np.zeros((values.shape[0], output_size // 2 + 1), dtype=np.complex128)
    copied = min(spectrum.shape[1], output_spectrum.shape[1])
    output_spectrum[:, :copied] = spectrum[:, :copied]
    return np.asarray(
        np.fft.irfft(output_spectrum, n=output_size, axis=1) * (output_size / input_size)
    )


def available_coarse() -> bool:
    """Return whether SciPy numerical solvers are available."""
    return True


CoarseSolver = CoarseSolverOperator

__all__ = ["CoarseSolver", "CoarseSolverOperator", "available_coarse"]
