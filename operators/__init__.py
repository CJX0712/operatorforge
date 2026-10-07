"""Operator implementations and registry.

Author: 晨星
"""

from operators.cnn import CNN, ConvolutionalOperator
from operators.coarse import CoarseSolver, CoarseSolverOperator
from operators.deeponet import DeepONet, DeepONetOperator
from operators.fno import FNO, FourierNeuralOperator
from operators.mlp import MLP, PointwiseMLPOperator
from operators.opfuse import ABLATIONS, OpFuse, OpFuseOperator
from operators.rbf import RBF, RBFOperator
from operators.registry import available_operators, create_operator

__all__ = [
    "ABLATIONS",
    "CNN",
    "FNO",
    "MLP",
    "RBF",
    "CoarseSolver",
    "CoarseSolverOperator",
    "ConvolutionalOperator",
    "DeepONet",
    "DeepONetOperator",
    "FourierNeuralOperator",
    "OpFuse",
    "OpFuseOperator",
    "PointwiseMLPOperator",
    "RBFOperator",
    "available_operators",
    "create_operator",
]
