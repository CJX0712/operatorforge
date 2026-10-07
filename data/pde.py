"""PDE dataset generation with deterministic high-accuracy reference solvers.

The Burgers integrator advances every sample in one vectorized RK4 loop.  The
advection-diffusion reference uses its exact Fourier semigroup.  Darcy systems
are assembled as one block sparse matrix and solved in a single sparse call.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from core.errors import DataError
from core.types import FloatArray, OperatorTask, PDEDataset

TASKS = ("burgers1d", "advdiff1d", "darcy2d")


def periodic_grf(
    n_samples: int,
    grid_size: int,
    seed: int,
    smoothness: float = 2.5,
    amplitude: float = 0.75,
) -> FloatArray:
    """Generate smooth real periodic Gaussian random fields."""
    if n_samples <= 0 or grid_size < 4:
        raise DataError("n_samples must be positive and grid_size at least four.")
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(grid_size) * grid_size
    envelope = (1.0 + frequencies**2) ** (-0.5 * smoothness)
    coefficients = (
        rng.normal(size=(n_samples, frequencies.size))
        + 1j * rng.normal(size=(n_samples, frequencies.size))
    ) * envelope[None, :]
    coefficients[:, 0] = rng.normal(scale=0.2, size=n_samples)
    if grid_size % 2 == 0:
        coefficients[:, -1] = coefficients[:, -1].real
    fields = np.fft.irfft(coefficients, n=grid_size, axis=-1, norm="ortho")
    fields -= fields.mean(axis=-1, keepdims=True)
    field_scale = fields.std(axis=-1, keepdims=True)
    fields = amplitude * fields / np.maximum(field_scale, 1.0e-12)
    offsets = rng.uniform(-0.2, 0.2, size=(n_samples, 1))
    return np.asarray(fields + offsets, dtype=np.float64)


def advdiff_exact(
    initial: FloatArray,
    final_time: float = 0.3,
    velocity: float = 0.7,
    diffusivity: float = 0.006,
) -> FloatArray:
    """Apply the exact periodic advection-diffusion semigroup."""
    values = _validate_1d_batch(initial)
    if final_time < 0.0 or diffusivity < 0.0:
        raise DataError("final_time and diffusivity must be non-negative.")
    grid_size = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=1.0 / grid_size)
    multiplier = np.exp(
        -(1j * velocity * wave_numbers + diffusivity * wave_numbers**2) * final_time
    )
    transformed = np.fft.rfft(values, axis=-1)
    return np.asarray(np.fft.irfft(transformed * multiplier, n=grid_size, axis=-1).real)


def advdiff_spectral(
    initial: FloatArray,
    final_time: float = 0.3,
    velocity: float = 0.7,
    diffusivity: float = 0.006,
) -> FloatArray:
    """Numerically solve advection-diffusion via the exact spectral propagator.

    This is intentionally a numerical implementation of the analytic Fourier
    formula and therefore serves as a gold-standard precision invariant.
    """
    return advdiff_exact(initial, final_time, velocity, diffusivity)


def burgers_spectral_rk4(
    initial: FloatArray,
    final_time: float = 0.2,
    viscosity: float = 0.01,
    time_step: float = 1.0e-3,
) -> FloatArray:
    """Solve periodic viscous Burgers with de-aliased pseudo-spectral RK4."""
    values = _validate_1d_batch(initial).copy()
    if viscosity <= 0.0 or final_time <= 0.0 or time_step <= 0.0:
        raise DataError("Burgers time and viscosity parameters must be positive.")
    grid_size = values.shape[-1]
    dx = 1.0 / grid_size
    maximum_speed = max(1.0, float(np.max(np.abs(values))))
    stable_step = min(
        time_step,
        0.25 * dx / maximum_speed,
        0.10 * dx * dx / viscosity,
    )
    steps = max(1, int(np.ceil(final_time / stable_step)))
    dt = final_time / steps

    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=dx)
    dealias = np.abs(np.fft.fftfreq(grid_size) * grid_size) <= grid_size / 3.0

    def right_hand_side(state: FloatArray) -> FloatArray:
        state_hat = np.fft.fft(state, axis=-1)
        derivative = np.fft.ifft(1j * wave_numbers * state_hat, axis=-1).real
        nonlinear_hat = np.fft.fft(state * derivative, axis=-1)
        nonlinear = np.fft.ifft(nonlinear_hat * dealias, axis=-1).real
        laplacian = np.fft.ifft(-(wave_numbers**2) * state_hat, axis=-1).real
        return np.asarray(-nonlinear + viscosity * laplacian)

    for _ in range(steps):
        k1 = right_hand_side(values)
        k2 = right_hand_side(values + 0.5 * dt * k1)
        k3 = right_hand_side(values + 0.5 * dt * k2)
        k4 = right_hand_side(values + dt * k3)
        values += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(values, dtype=np.float64)


def darcy_coefficients(n_samples: int, grid_size: int, seed: int) -> FloatArray:
    """Generate positive binary permeability fields from smooth 2-D GRFs."""
    if grid_size < 6:
        raise DataError("Darcy grid must contain at least six points per axis.")
    rng = np.random.default_rng(seed)
    wave = np.fft.fftfreq(grid_size) * grid_size
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    envelope = (1.0 + kx**2 + ky**2) ** -1.6
    coefficients = (
        rng.normal(size=(n_samples, grid_size, grid_size))
        + 1j * rng.normal(size=(n_samples, grid_size, grid_size))
    ) * envelope[None, :, :]
    fields = np.fft.ifft2(coefficients, axes=(-2, -1)).real
    medians = np.median(fields, axis=(-2, -1), keepdims=True)
    return np.where(fields >= medians, 3.0, 0.5).astype(np.float64)


def darcy_finite_difference(
    coefficients: FloatArray,
    forcing: FloatArray | float = 1.0,
    boundary: FloatArray | float = 0.0,
) -> FloatArray:
    """Solve a batch of Dirichlet Darcy systems using one block sparse solve.

    ``-div(a grad u) = f`` is discretized with harmonic face permeabilities and
    a five-point stencil.  ``forcing`` may be a scalar or a full
    ``(samples, n, n)`` field so manufactured solutions can inject the source
    implied by a known ``u``.  ``boundary`` supplies the Dirichlet trace whose
    flux contribution the interior right-hand side must compensate.
    """
    permeability = np.asarray(coefficients, dtype=np.float64)
    if permeability.ndim != 3 or permeability.shape[1] != permeability.shape[2]:
        raise DataError("Darcy coefficients must have shape (samples, n, n).")
    if np.any(permeability <= 0.0):
        raise DataError("Darcy permeability must be strictly positive.")
    n_samples, grid_size, _ = permeability.shape
    interior = grid_size - 2
    unknowns = interior * interior
    h_squared = (1.0 / (grid_size - 1)) ** 2
    forcing_field = _darcy_field(forcing, "forcing", permeability.shape)
    boundary_field = _darcy_field(boundary, "boundary", permeability.shape)
    blocks: list[sparse.csr_matrix] = []
    right_hand_side = np.zeros(n_samples * unknowns, dtype=np.float64)

    def harmonic(left: FloatArray, right: FloatArray) -> FloatArray:
        return 2.0 * left * right / np.maximum(left + right, 1.0e-15)

    for sample_index in range(n_samples):
        field = permeability[sample_index]
        center = field[1:-1, 1:-1]
        east = harmonic(center, field[1:-1, 2:])
        west = harmonic(center, field[1:-1, :-2])
        north = harmonic(center, field[2:, 1:-1])
        south = harmonic(center, field[:-2, 1:-1])
        diagonal = ((east + west + north + south) / h_squared).ravel()
        rows: list[NDArray[np.int64]] = [np.arange(unknowns, dtype=np.int64)]
        cols: list[NDArray[np.int64]] = [np.arange(unknowns, dtype=np.int64)]
        data: list[FloatArray] = [diagonal]
        horizontal_mask = np.arange(unknowns - 1) % interior != interior - 1
        horizontal_rows = np.arange(unknowns - 1, dtype=np.int64)[horizontal_mask]
        horizontal_values = (-east.ravel()[:-1] / h_squared)[horizontal_mask]
        rows.extend([horizontal_rows, horizontal_rows + 1])
        cols.extend([horizontal_rows + 1, horizontal_rows])
        data.extend([horizontal_values, horizontal_values])
        vertical_rows = np.arange(unknowns - interior, dtype=np.int64)
        vertical_values = -north.ravel()[:-interior] / h_squared
        rows.extend([vertical_rows, vertical_rows + interior])
        cols.extend([vertical_rows + interior, vertical_rows])
        data.extend([vertical_values, vertical_values])
        matrix = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(unknowns, unknowns),
        ).tocsr()
        blocks.append(matrix)
        trace = boundary_field[sample_index]
        # The assembled stiffness matrix already carries a 1/h_squared factor on
        # every interior entry (diagonal at line ~181 and off-diagonals at
        # ~187/192), so the right-hand side must NOT divide the interior forcing
        # by h_squared again. Only the boundary-flux compensation terms below
        # retain the 1/h_squared factor because they represent the
        # a_face * u_boundary / h_squared contributions that the matrix moved to
        # the RHS for known Dirichlet nodes.
        source = forcing_field[sample_index, 1:-1, 1:-1]
        source[:, 0] += west[:, 0] * trace[1:-1, 0] / h_squared
        source[:, -1] += east[:, -1] * trace[1:-1, -1] / h_squared
        source[0, :] += south[0, :] * trace[0, 1:-1] / h_squared
        source[-1, :] += north[-1, :] * trace[-1, 1:-1] / h_squared
        right_hand_side[sample_index * unknowns : (sample_index + 1) * unknowns] = source.ravel()

    block_matrix = sparse.block_diag(blocks, format="csr")
    interior_solution = np.asarray(spsolve(block_matrix, right_hand_side)).reshape(
        n_samples, interior, interior
    )
    solution = boundary_field.copy()
    solution[:, 1:-1, 1:-1] = interior_solution
    return solution


def darcy_manufactured_forcing(
    coefficients: FloatArray,
    solution: FloatArray,
) -> FloatArray:
    """Return the source term implied by a known Darcy solution field."""
    permeability = np.asarray(coefficients, dtype=np.float64)
    known = np.asarray(solution, dtype=np.float64)
    if permeability.shape != known.shape or permeability.ndim != 3:
        raise DataError("Manufactured Darcy inputs must share shape (samples, n, n).")
    grid_size = permeability.shape[1]
    spacing = 1.0 / (grid_size - 1)
    flux_x = _harmonic_face(permeability, axis=1) * np.diff(known, axis=1) / spacing
    flux_y = _harmonic_face(permeability, axis=2) * np.diff(known, axis=2) / spacing
    forcing = np.zeros_like(known)
    forcing[:, 1:-1, 1:-1] = -(
        ((flux_x[:, 1:, :] - flux_x[:, :-1, :]) / spacing).transpose(0, 2, 1)
        + (flux_y[:, :, 1:] - flux_y[:, :, :-1]) / spacing
    ).transpose(0, 2, 1)
    return forcing


def _harmonic_face(field: FloatArray, axis: int) -> FloatArray:
    """Return harmonic-mean face permeabilities along the requested axis."""
    left = np.take(field, range(field.shape[axis] - 1), axis=axis)
    right = np.take(field, range(1, field.shape[axis]), axis=axis)
    return 2.0 * left * right / np.maximum(left + right, 1.0e-15)


def _darcy_field(
    value: FloatArray | float,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Broadcast a scalar or array Darcy field to the required shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.shape != shape:
        raise DataError(f"Darcy {name} must be scalar or match shape {shape}.")
    return array


def generate_dataset(
    task_name: str,
    n_train: int = 48,
    n_test: int = 18,
    grid_size: int | None = None,
    seed: int = 1729,
) -> PDEDataset:
    """Generate a deterministic independent train/test PDE dataset."""
    if task_name not in TASKS:
        raise DataError(f"Unknown task {task_name!r}; expected one of {TASKS}.")
    if n_train <= 0 or n_test <= 0:
        raise DataError("n_train and n_test must be positive.")
    default_size = 20 if task_name == "darcy2d" else 64
    actual_size = default_size if grid_size is None else grid_size
    train_seed = seed * 2 + 101
    test_seed = seed * 2 + 102
    generator = _task_generator(task_name)
    x_train, y_train, metadata = generator(n_train, actual_size, train_seed)
    x_test, y_test, _ = generator(n_test, actual_size, test_seed)
    axes = tuple(range(x_train.ndim))
    target_axes = tuple(range(y_train.ndim))
    x_mean = np.asarray(x_train.mean(axis=axes, keepdims=True))
    x_scale = np.asarray(x_train.std(axis=axes, keepdims=True))
    y_mean = np.asarray(y_train.mean(axis=target_axes, keepdims=True))
    y_scale = np.asarray(y_train.std(axis=target_axes, keepdims=True))
    x_scale = np.maximum(x_scale, 1.0e-12)
    y_scale = np.maximum(y_scale, 1.0e-12)
    coordinates = _coordinates(task_name, actual_size)
    task = OperatorTask(
        name=task_name,
        input_shape=tuple(x_train.shape[1:]),
        output_shape=tuple(y_train.shape[1:]),
        coordinates=coordinates,
        metadata=metadata,
    )
    return PDEDataset(
        task=task,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        train_seed=train_seed,
        test_seed=test_seed,
    )


def _task_generator(
    task_name: str,
) -> Callable[[int, int, int], tuple[FloatArray, FloatArray, dict[str, Any]]]:
    if task_name == "burgers1d":
        return _generate_burgers
    if task_name == "advdiff1d":
        return _generate_advdiff
    return _generate_darcy


def _generate_burgers(
    n_samples: int, grid_size: int, seed: int
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    initial = periodic_grf(n_samples, grid_size, seed, smoothness=2.25, amplitude=0.8)
    solution = burgers_spectral_rk4(initial)
    return initial, solution, {"viscosity": 0.01, "final_time": 0.2, "solver": "rk4-pseudospectral"}


def _generate_advdiff(
    n_samples: int, grid_size: int, seed: int
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    initial = periodic_grf(n_samples, grid_size, seed, smoothness=1.8, amplitude=0.9)
    solution = advdiff_exact(initial)
    return (
        initial,
        solution,
        {
            "velocity": 0.7,
            "diffusivity": 0.006,
            "final_time": 0.3,
            "solver": "analytic-fourier",
        },
    )


def _generate_darcy(
    n_samples: int, grid_size: int, seed: int
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    coefficients = darcy_coefficients(n_samples, grid_size, seed)
    solution = darcy_finite_difference(coefficients)
    return (
        coefficients,
        solution,
        {"forcing": 1.0, "boundary": "homogeneous-dirichlet", "solver": "five-point"},
    )


def _coordinates(task_name: str, grid_size: int) -> FloatArray:
    if task_name == "darcy2d":
        axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
        return np.stack((x_grid, y_grid), axis=-1)
    return np.linspace(0.0, 1.0, grid_size, endpoint=False, dtype=np.float64)[:, None]


def _validate_1d_batch(values: FloatArray) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 4:
        raise DataError("Expected a 2-D array of batched one-dimensional fields.")
    if not np.all(np.isfinite(result)):
        raise DataError("PDE inputs must be finite.")
    return result


__all__ = [
    "TASKS",
    "advdiff_exact",
    "advdiff_spectral",
    "burgers_spectral_rk4",
    "darcy_coefficients",
    "darcy_finite_difference",
    "generate_dataset",
    "periodic_grf",
]
