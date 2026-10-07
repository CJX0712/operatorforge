"""Shared dataclasses for datasets, training, evaluation, and reports.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OperatorTask:
    """Definition of a function-to-function learning task."""

    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    coordinates: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PDEDataset:
    """Independent train/test samples and train-only normalization state."""

    task: OperatorTask
    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    x_mean: FloatArray
    x_scale: FloatArray
    y_mean: FloatArray
    y_scale: FloatArray
    train_seed: int
    test_seed: int

    def __post_init__(self) -> None:
        if self.train_seed == self.test_seed:
            raise ValueError("Training and test seeds must be different.")
        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("Training input and target sample counts differ.")
        if self.x_test.shape[0] != self.y_test.shape[0]:
            raise ValueError("Test input and target sample counts differ.")

    def normalized_train(self) -> tuple[FloatArray, FloatArray]:
        """Return data normalized with statistics fitted on training data only."""
        return (
            (self.x_train - self.x_mean) / self.x_scale,
            (self.y_train - self.y_mean) / self.y_scale,
        )

    def normalized_test(self) -> tuple[FloatArray, FloatArray]:
        """Return test data transformed by train-only statistics."""
        return (
            (self.x_test - self.x_mean) / self.x_scale,
            (self.y_test - self.y_mean) / self.y_scale,
        )


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Training outcome for one operator and one seed."""

    model_name: str
    task_name: str
    seed: int
    loss_history: tuple[float, ...]
    epochs: int
    elapsed_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Metric values for one fitted operator."""

    model_name: str
    task_name: str
    seed: int
    relative_l2: float
    relative_h1: float
    spectral_tail_error: float
    superresolution_l2: float | None = None
    elapsed_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkReport:
    """Serializable benchmark results and gate decisions."""

    project: str
    author: str
    configuration: dict[str, Any]
    results: list[dict[str, Any]]
    aggregates: list[dict[str, Any]]
    gates: dict[str, dict[str, Any]]
    ablations: list[dict[str, Any]]
    failure_cases: list[dict[str, Any]]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return asdict(self)
