"""Scientific gold-standard tests for PDE generators and solvers.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import DataError
from data.pde import (
    advdiff_spectral,
    burgers_spectral_rk4,
    darcy_coefficients,
    darcy_finite_difference,
    generate_dataset,
    periodic_grf,
)


def _independent_advdiff_solution(
    initial: np.ndarray,
    final_time: float,
    velocity: float,
    diffusivity: float,
) -> np.ndarray:
    grid_size = initial.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=1.0 / grid_size)
    multiplier = np.exp(
        -(1j * velocity * wave_numbers + diffusivity * wave_numbers**2) * final_time
    )
    return np.fft.irfft(np.fft.rfft(initial, axis=-1) * multiplier, n=grid_size, axis=-1)


def _discrete_darcy_forcing(permeability: np.ndarray, solution: np.ndarray) -> np.ndarray:
    """Apply the same conservative five-point operator to a manufactured field."""
    grid_size = permeability.shape[0]
    h_squared = (1.0 / (grid_size - 1)) ** 2
    center = permeability[1:-1, 1:-1]

    def harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return 2.0 * left * right / np.maximum(left + right, 1.0e-15)

    east = harmonic(center, permeability[1:-1, 2:])
    west = harmonic(center, permeability[1:-1, :-2])
    north = harmonic(center, permeability[2:, 1:-1])
    south = harmonic(center, permeability[:-2, 1:-1])
    forcing = np.zeros_like(solution)
    forcing[1:-1, 1:-1] = (
        (east + west + north + south) * solution[1:-1, 1:-1]
        - east * solution[1:-1, 2:]
        - west * solution[1:-1, :-2]
        - north * solution[2:, 1:-1]
        - south * solution[:-2, 1:-1]
    ) / h_squared
    return forcing


class TestAdvectionDiffusion:
    """Compare the numerical path against an independently written exact formula."""

    @pytest.mark.parametrize(
        ("final_time", "velocity", "diffusivity"),
        [(0.0, 0.7, 0.006), (0.31, -1.2, 0.0), (0.17, 0.4, 2.5)],
    )
    def test_spectral_solver_matches_analytic_fourier_gold_standard(
        self,
        rng: np.random.Generator,
        final_time: float,
        velocity: float,
        diffusivity: float,
    ) -> None:
        initial = rng.normal(size=(3, 64))
        expected = _independent_advdiff_solution(initial, final_time, velocity, diffusivity)

        actual = advdiff_spectral(initial, final_time, velocity, diffusivity)

        relative_error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-15)
        assert relative_error <= 1.0e-8

    def test_extreme_diffusion_is_finite_and_preserves_mean(self, rng: np.random.Generator) -> None:
        initial = rng.normal(size=(2, 64))
        solution = advdiff_spectral(initial, final_time=2.0, diffusivity=1.0e6)

        assert np.all(np.isfinite(solution))
        assert np.allclose(solution.mean(axis=1), initial.mean(axis=1), atol=1.0e-12)


class TestBurgers:
    """Verify fourth-order time convergence and invariant constant states."""

    def test_rk4_error_contracts_when_time_step_is_halved(self) -> None:
        coordinate = np.linspace(0.0, 1.0, 64, endpoint=False)
        initial = (0.35 * np.sin(2.0 * np.pi * coordinate))[None, :]
        reference = burgers_spectral_rk4(
            initial, final_time=0.04, viscosity=0.001, time_step=1.25e-4
        )
        coarse = burgers_spectral_rk4(initial, final_time=0.04, viscosity=0.001, time_step=2.0e-3)
        fine = burgers_spectral_rk4(initial, final_time=0.04, viscosity=0.001, time_step=1.0e-3)
        coarse_error = np.linalg.norm(coarse - reference)
        fine_error = np.linalg.norm(fine - reference)

        assert fine_error < coarse_error
        assert coarse_error / max(fine_error, 1.0e-30) >= 8.0

    def test_constant_field_is_an_exact_stationary_solution(self) -> None:
        initial = np.full((2, 32), 0.25)
        solution = burgers_spectral_rk4(initial, final_time=0.1, viscosity=1.0e-6, time_step=0.005)
        assert np.array_equal(solution, initial)


class TestDarcy:
    """Use method of manufactured solutions and explicit residual checks."""

    def test_manufactured_solution_is_recovered_with_small_residual(self) -> None:
        grid_size = 18
        axis = np.linspace(0.0, 1.0, grid_size)
        x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
        permeability = 1.4 + 0.2 * np.sin(np.pi * x_grid) * np.sin(np.pi * y_grid)
        expected = np.sin(np.pi * x_grid) * np.sin(2.0 * np.pi * y_grid)
        forcing = _discrete_darcy_forcing(permeability, expected)

        actual = darcy_finite_difference(permeability[None, ...], forcing=forcing[None, ...])[0]

        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        reconstructed_forcing = _discrete_darcy_forcing(permeability, actual)
        interior_residual = reconstructed_forcing[1:-1, 1:-1] - forcing[1:-1, 1:-1]
        relative_residual = np.linalg.norm(interior_residual) / np.linalg.norm(forcing[1:-1, 1:-1])
        assert relative_error <= 1.0e-6
        assert relative_residual <= 1.0e-10

    def test_generated_coefficients_are_positive_binary_and_deterministic(self) -> None:
        first = darcy_coefficients(3, 8, 19)
        second = darcy_coefficients(3, 8, 19)
        assert np.array_equal(first, second)
        assert set(np.unique(first)) == {0.5, 3.0}


class TestDatasetIntegrityAndBoundaries:
    """Protect train/test isolation and explicit invalid-input behavior."""

    @pytest.mark.parametrize("task_name", ["burgers1d", "advdiff1d", "darcy2d"])
    def test_generated_train_and_test_samples_are_disjoint_and_train_normalized(
        self, task_name: str
    ) -> None:
        grid_size = 8 if task_name == "darcy2d" else 16
        dataset = generate_dataset(task_name, n_train=5, n_test=3, grid_size=grid_size, seed=13)
        train_rows = {row.tobytes() for row in dataset.x_train}
        test_rows = {row.tobytes() for row in dataset.x_test}

        assert train_rows.isdisjoint(test_rows)
        assert dataset.train_seed != dataset.test_seed
        assert np.array_equal(
            dataset.x_mean,
            dataset.x_train.mean(axis=tuple(range(dataset.x_train.ndim)), keepdims=True),
        )
        assert np.array_equal(
            dataset.y_mean,
            dataset.y_train.mean(axis=tuple(range(dataset.y_train.ndim)), keepdims=True),
        )
        assert np.all(dataset.x_scale > 0.0)
        assert np.all(dataset.y_scale > 0.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: periodic_grf(0, 16, 1),
            lambda: periodic_grf(1, 1, 1),
            lambda: advdiff_spectral(np.empty((0, 0))),
            lambda: advdiff_spectral(np.ones((1, 8)), final_time=-1.0),
            lambda: burgers_spectral_rk4(np.ones((1, 8)), viscosity=0.0),
            lambda: darcy_coefficients(1, 5, 1),
            lambda: darcy_finite_difference(np.ones((1, 4, 5))),
            lambda: darcy_finite_difference(np.zeros((1, 6, 6))),
            lambda: generate_dataset("unknown"),
            lambda: generate_dataset("advdiff1d", n_train=0),
        ],
    )
    def test_invalid_pde_inputs_raise_e100(self, call) -> None:
        with pytest.raises(DataError) as error:
            call()
        assert str(error.value).startswith("E100:")
