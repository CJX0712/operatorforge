"""OperatorForge public package facade.

Author: 晨星
"""

from core.config import GATE_THRESHOLDS, BenchmarkConfig
from core.types import BenchmarkReport, EvalResult, OperatorTask, PDEDataset, TrainResult
from operators.registry import available_operators, create_operator
from pipeline.benchmark import BenchmarkPipeline

__all__ = [
    "GATE_THRESHOLDS",
    "BenchmarkConfig",
    "BenchmarkPipeline",
    "BenchmarkReport",
    "EvalResult",
    "OperatorTask",
    "PDEDataset",
    "TrainResult",
    "available_operators",
    "create_operator",
]

__author__ = "晨星"
__version__ = "0.1.0"
