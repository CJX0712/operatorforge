"""Configuration loading and immutable quality gates.

Every field can be overridden with an ``ENV_OPF_*`` environment variable.
The frozen gates are: G1 ratio <= 0.90 and effect margin > 0.5 sigma sum;
G2 super-resolution ratio <= 1.05 versus FNO; G3 gradient <= 1e-6,
analytic PDE <= 1e-8, Parseval <= 1e-12; G4 deterministic core metrics.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, get_type_hints

from core.errors import ConfigurationError

GATE_THRESHOLDS: dict[str, float] = {
    "g1_ratio": 0.90,
    "g1_effect_sigma_fraction": 0.50,
    "g2_fno_tolerance": 1.05,
    "g3_gradient_max_rel_error": 1.0e-6,
    "g3_advdiff_max_error": 1.0e-8,
    "g3_parseval_max_error": 1.0e-12,
    "g4_determinism_atol": 0.0,
}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated benchmark configuration with environment overrides."""

    seed: int = 1729
    seeds: tuple[int, ...] = (17, 29, 43)
    n_train: int = 48
    n_test: int = 18
    grid_size_1d: int = 64
    grid_size_2d: int = 20
    superresolution_grid: int = 256
    modes: int = 10
    width: int = 24
    depth: int = 3
    epochs: int = 24
    batch_size: int = 16
    learning_rate: float = 0.01
    ridge: float = 1.0e-5
    output_path: str = "benchmark.json"

    _ENV_PREFIX: ClassVar[str] = "ENV_OPF_"

    @classmethod
    def from_env(cls, **overrides: Any) -> BenchmarkConfig:
        """Create a configuration from defaults, environment, and explicit values."""
        values: dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for field_name in cls.__dataclass_fields__:
            if field_name.startswith("_"):
                continue
            env_name = f"{cls._ENV_PREFIX}{field_name.upper()}"
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            values[field_name] = _coerce(raw_value, type_hints[field_name], env_name)
        values.update(overrides)
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate ranges required for numerically meaningful benchmarks."""
        if not self.seeds:
            raise ConfigurationError("At least one benchmark seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError("Benchmark seeds must be unique.")
        positive_fields = {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "grid_size_1d": self.grid_size_1d,
            "grid_size_2d": self.grid_size_2d,
            "modes": self.modes,
            "width": self.width,
            "depth": self.depth,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ConfigurationError(f"Positive configuration required for: {', '.join(invalid)}")
        if self.grid_size_1d < 16 or self.grid_size_2d < 8:
            raise ConfigurationError("Grids are too small for spectral evaluation.")
        if self.modes >= self.grid_size_1d // 2 + 1:
            raise ConfigurationError("modes must be smaller than the one-sided spectrum.")
        if self.learning_rate <= 0.0 or self.ridge < 0.0:
            raise ConfigurationError("learning_rate must be positive and ridge non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result


def _coerce(raw_value: str, target_type: Any, env_name: str) -> Any:
    try:
        if target_type is int:
            return int(raw_value)
        if target_type is float:
            return float(raw_value)
        if target_type == tuple[int, ...]:
            return tuple(int(item.strip()) for item in raw_value.split(",") if item.strip())
        return raw_value
    except ValueError as exc:
        raise ConfigurationError(f"Invalid value for {env_name}: {raw_value!r}") from exc


BenchmarkConfig().validate()

__all__ = ["GATE_THRESHOLDS", "BenchmarkConfig"]
