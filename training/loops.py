"""Training orchestration shared by closed-form and iterative operators.

Author: 晨星
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from core.interfaces import OperatorModel
from core.types import FloatArray, TrainResult


def relative_l2_loss(prediction: FloatArray, target: FloatArray) -> float:
    """Return mean per-sample relative L2 loss."""
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.shape != expected.shape:
        raise ValueError("Prediction and target shapes must match.")
    flat_difference = (predicted - expected).reshape(predicted.shape[0], -1)
    flat_target = expected.reshape(expected.shape[0], -1)
    numerator = np.linalg.norm(flat_difference, axis=1)
    denominator = np.maximum(np.linalg.norm(flat_target, axis=1), 1.0e-12)
    return float(np.mean(numerator / denominator))


def fit_operator(
    model: OperatorModel,
    x_train: FloatArray,
    y_train: FloatArray,
    task_name: str,
    seed: int,
) -> TrainResult:
    """Fit an operator and record deterministic quality metadata."""
    start = perf_counter()
    model.fit(x_train, y_train)
    elapsed = perf_counter() - start
    training_loss = relative_l2_loss(model.predict(x_train), y_train)
    return TrainResult(
        model_name=model.name,
        task_name=task_name,
        seed=seed,
        loss_history=(training_loss,),
        epochs=1,
        elapsed_sec=elapsed,
        metadata={"optimizer": "closed-form-ridge"},
    )


__all__ = ["fit_operator", "relative_l2_loss"]
