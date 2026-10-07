"""Deterministic PDE data generators.

Author: 晨星
"""

from data.pde import (
    TASKS,
    advdiff_exact,
    advdiff_spectral,
    burgers_spectral_rk4,
    darcy_finite_difference,
    generate_dataset,
    periodic_grf,
)

__all__ = [
    "TASKS",
    "advdiff_exact",
    "advdiff_spectral",
    "burgers_spectral_rk4",
    "darcy_finite_difference",
    "generate_dataset",
    "periodic_grf",
]
