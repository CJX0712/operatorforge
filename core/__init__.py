"""Core contracts for OperatorForge.

Author: 晨星
"""

from core.config import GATE_THRESHOLDS, BenchmarkConfig
from core.errors import OperatorForgeError
from core.types import BenchmarkReport, EvalResult, OperatorTask, PDEDataset, TrainResult

__all__ = [
    "GATE_THRESHOLDS",
    "BenchmarkConfig",
    "BenchmarkReport",
    "EvalResult",
    "OperatorForgeError",
    "OperatorTask",
    "PDEDataset",
    "TrainResult",
]
