"""Operator registry with deterministic construction.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.errors import ModelError
from core.interfaces import OperatorModel
from operators.cnn import ConvolutionalOperator, available_cnn
from operators.coarse import CoarseSolverOperator, available_coarse
from operators.deeponet import DeepONetOperator, available_deeponet
from operators.fno import FourierNeuralOperator, available_fno
from operators.mlp import PointwiseMLPOperator, available_mlp
from operators.opfuse import OpFuseOperator, available_opfuse
from operators.rbf import RBFOperator, available_rbf

Factory = Callable[..., OperatorModel]
Availability = Callable[[], bool]

_REGISTRY: dict[str, tuple[Factory, Availability]] = {
    "cnn": (ConvolutionalOperator, available_cnn),
    "coarse": (CoarseSolverOperator, available_coarse),
    "deeponet": (DeepONetOperator, available_deeponet),
    "fno": (FourierNeuralOperator, available_fno),
    "mlp": (PointwiseMLPOperator, available_mlp),
    "opfuse": (OpFuseOperator, available_opfuse),
    "rbf": (RBFOperator, available_rbf),
}


def available_operators() -> tuple[str, ...]:
    """Return available registered names in stable lexical order."""
    return tuple(name for name in sorted(_REGISTRY) if _REGISTRY[name][1]())


def create_operator(name: str, **parameters: Any) -> OperatorModel:
    """Construct a registered operator after availability validation."""
    normalized = name.strip().lower()
    if normalized not in _REGISTRY:
        raise ModelError(f"Unknown operator {name!r}; available: {available_operators()}.")
    factory, availability = _REGISTRY[normalized]
    if not availability():
        raise ModelError(f"Operator {normalized!r} is unavailable in this environment.")
    try:
        return factory(**parameters)
    except TypeError as exc:
        raise ModelError(f"Invalid parameters for {normalized}: {exc}") from exc


def register_operator(name: str, factory: Factory, availability: Availability) -> None:
    """Register a uniquely named operator factory."""
    normalized = name.strip().lower()
    if not normalized:
        raise ModelError("Operator names cannot be empty.")
    if normalized in _REGISTRY:
        raise ModelError(f"Operator {normalized!r} is already registered.")
    _REGISTRY[normalized] = (factory, availability)


__all__ = ["available_operators", "create_operator", "register_operator"]
