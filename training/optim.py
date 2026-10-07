"""Deterministic optimizers.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.errors import TrainingError
from core.types import FloatArray


@dataclass
class Adam:
    """Adam optimizer with bias correction and in-place updates."""

    learning_rate: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    _step: int = 0
    _first: dict[str, FloatArray] = field(default_factory=dict)
    _second: dict[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise TrainingError("Adam learning_rate must be positive.")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise TrainingError("Adam beta values must lie in [0, 1).")

    def step(
        self,
        parameters: dict[str, FloatArray],
        gradients: dict[str, FloatArray],
    ) -> None:
        """Apply one bias-corrected Adam update."""
        if set(parameters) != set(gradients):
            raise TrainingError("Adam parameter and gradient names differ.")
        self._step += 1
        for name in sorted(parameters):
            parameter = parameters[name]
            gradient = np.asarray(gradients[name], dtype=np.float64)
            if parameter.shape != gradient.shape:
                raise TrainingError(f"Gradient shape mismatch for {name}.")
            if not np.all(np.isfinite(gradient)):
                raise TrainingError(f"Non-finite gradient for {name}.")
            first = self._first.setdefault(name, np.zeros_like(parameter))
            second = self._second.setdefault(name, np.zeros_like(parameter))
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient**2
            corrected_first = first / (1.0 - self.beta1**self._step)
            corrected_second = second / (1.0 - self.beta2**self._step)
            parameter -= (
                self.learning_rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)
            )


__all__ = ["Adam"]
