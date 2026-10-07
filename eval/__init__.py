"""Evaluation API.

Author: 晨星
"""

from eval.metrics import (
    evaluate_prediction,
    parseval_error,
    relative_h1,
    relative_l2,
    spectral_tail_error,
)

__all__ = [
    "evaluate_prediction",
    "parseval_error",
    "relative_h1",
    "relative_l2",
    "spectral_tail_error",
]
