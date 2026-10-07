"""Tests for core configuration, types, interfaces, errors, and registry.

Author: 晨星
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from core.config import GATE_THRESHOLDS, BenchmarkConfig
from core.errors import (
    ConfigurationError,
    DataError,
    EvaluationError,
    ModelError,
    OperatorForgeError,
    TrainingError,
)
from core.interfaces import OperatorModel
from core.seed import current_seed, set_all
from core.types import BenchmarkReport, OperatorTask, PDEDataset
from operators import registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSeedContract:
    """Verify all documented random streams are reproducible."""

    def test_set_all_replays_python_numpy_and_generator_streams(self) -> None:
        set_all(90210)
        first = (random.random(), np.random.random(4), np.random.default_rng(90210).random(4))

        returned = set_all(90210)
        second = (random.random(), np.random.random(4), returned.random(4))

        assert first[0] == second[0]
        assert np.array_equal(first[1], second[1])
        assert np.array_equal(first[2], second[2])
        assert current_seed() == 90210
        assert os.environ["PYTHONHASHSEED"] == "90210"

    @pytest.mark.parametrize("invalid", [-1, 1.5, "3", None])
    def test_set_all_rejects_invalid_seeds(self, invalid: object) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            set_all(invalid)  # type: ignore[arg-type]


class TestConfigurationContract:
    """Exercise defaults, environment overrides, and validation failures."""

    def test_gate_thresholds_are_frozen_to_preregistered_values(self) -> None:
        assert GATE_THRESHOLDS == {
            "g1_ratio": 0.90,
            "g1_effect_sigma_fraction": 0.50,
            "g2_fno_tolerance": 1.05,
            "g3_gradient_max_rel_error": 1.0e-6,
            "g3_advdiff_max_error": 1.0e-8,
            "g3_parseval_max_error": 1.0e-12,
            "g4_determinism_atol": 0.0,
        }

    def test_from_env_coerces_scalars_and_seed_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV_OPF_N_TRAIN", "12")
        monkeypatch.setenv("ENV_OPF_LEARNING_RATE", "0.025")
        monkeypatch.setenv("ENV_OPF_SEEDS", "3, 5,8")

        config = BenchmarkConfig.from_env(output_path="qa.json")

        assert config.n_train == 12
        assert config.learning_rate == 0.025
        assert config.seeds == (3, 5, 8)
        assert config.to_dict()["seeds"] == [3, 5, 8]
        assert config.output_path == "qa.json"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"seeds": ()}, "At least one"),
            ({"seeds": (1, 1)}, "unique"),
            ({"n_train": 0}, "Positive configuration"),
            ({"grid_size_1d": 8}, "too small"),
            ({"modes": 33, "grid_size_1d": 64}, "one-sided spectrum"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"ridge": -1.0}, "ridge"),
        ],
    )
    def test_invalid_configurations_raise_e500(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ConfigurationError, match=message) as error:
            BenchmarkConfig.from_env(**kwargs)
        assert str(error.value).startswith("E500:")

    def test_invalid_environment_value_raises_e500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV_OPF_WIDTH", "wide")

        with pytest.raises(ConfigurationError, match="ENV_OPF_WIDTH"):
            BenchmarkConfig.from_env()


class TestErrorRegistry:
    """Prove stable codes and fail-fast duplicate detection."""

    @pytest.mark.parametrize(
        ("error_type", "code"),
        [
            (DataError, "E100"),
            (ModelError, "E200"),
            (TrainingError, "E300"),
            (EvaluationError, "E400"),
            (ConfigurationError, "E500"),
        ],
    )
    def test_error_types_have_codes_and_nonempty_messages(
        self, error_type: type[OperatorForgeError], code: str
    ) -> None:
        error = error_type("  ")

        assert error.code == code
        assert error.message.strip()
        assert str(error).startswith(f"{code}: ")

    @pytest.mark.parametrize(
        ("declaration", "expected"),
        [
            (
                "type('ConflictingCode', (OperatorForgeError,), {'code': 'E100'})",
                "Duplicate error code E100",
            ),
            (
                "type('DataError', (OperatorForgeError,), {'code': 'E901'})",
                "Duplicate error class name DataError",
            ),
        ],
    )
    def test_duplicate_error_code_or_name_fails_in_isolated_process(
        self, declaration: str, expected: str
    ) -> None:
        script = (
            "from core.errors import OperatorForgeError, validate_error_registry; "
            f"{declaration}; validate_error_registry()"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        assert completed.returncode != 0
        assert expected in completed.stderr


class TestTypesAndRegistry:
    """Check train-only normalization and operator factory behavior."""

    def test_dataset_normalization_uses_supplied_train_statistics_only(self) -> None:
        x_train = np.array([[1.0, 3.0], [5.0, 7.0]])
        y_train = 2.0 * x_train
        x_test = np.array([[101.0, 103.0]])
        y_test = 2.0 * x_test
        task = OperatorTask("toy", (2,), (2,), np.arange(2.0)[:, None])
        dataset = PDEDataset(
            task=task,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_mean=x_train.mean(keepdims=True),
            x_scale=x_train.std(keepdims=True),
            y_mean=y_train.mean(keepdims=True),
            y_scale=y_train.std(keepdims=True),
            train_seed=1,
            test_seed=2,
        )

        normalized_train_x, normalized_train_y = dataset.normalized_train()
        normalized_test_x, normalized_test_y = dataset.normalized_test()

        assert abs(float(normalized_train_x.mean())) < 1.0e-15
        assert abs(float(normalized_train_y.mean())) < 1.0e-15
        assert float(normalized_test_x.mean()) > 10.0
        assert float(normalized_test_y.mean()) > 10.0

    def test_dataset_rejects_seed_or_sample_leakage_contract_violations(self) -> None:
        task = OperatorTask("toy", (2,), (2,), np.arange(2.0)[:, None])
        common = {
            "task": task,
            "x_train": np.zeros((2, 2)),
            "y_train": np.zeros((2, 2)),
            "x_test": np.zeros((1, 2)),
            "y_test": np.zeros((1, 2)),
            "x_mean": np.zeros((1, 1)),
            "x_scale": np.ones((1, 1)),
            "y_mean": np.zeros((1, 1)),
            "y_scale": np.ones((1, 1)),
        }
        with pytest.raises(ValueError, match="different"):
            PDEDataset(**common, train_seed=4, test_seed=4)
        with pytest.raises(ValueError, match="Training input"):
            PDEDataset(**{**common, "y_train": np.zeros((3, 2))}, train_seed=4, test_seed=5)

    def test_public_report_is_json_compatible(self) -> None:
        report = BenchmarkReport(
            project="OperatorForge",
            author="晨星",
            configuration={"seed": 1},
            results=[],
            aggregates=[],
            gates={},
            ablations=[],
            failure_cases=[],
            elapsed_sec=0.0,
        )
        assert report.to_dict() == asdict(report)

    def test_registry_lists_and_constructs_protocol_conforming_models(self) -> None:
        names = registry.available_operators()
        assert names == tuple(sorted(names))
        assert {"cnn", "coarse", "deeponet", "fno", "mlp", "opfuse", "rbf"} <= set(names)
        assert isinstance(
            registry.create_operator(" FNO ", modes=3, width=4, depth=1), OperatorModel
        )

    def test_registry_rejects_unknown_duplicate_invalid_and_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
        with pytest.raises(ModelError, match="Unknown operator"):
            registry.create_operator("not-real")
        with pytest.raises(ModelError, match="Invalid parameters"):
            registry.create_operator("fno", impossible=True)
        with pytest.raises(ModelError, match="cannot be empty"):
            registry.register_operator(" ", lambda: None, lambda: True)  # type: ignore[arg-type]
        with pytest.raises(ModelError, match="already registered"):
            registry.register_operator("FNO", lambda: None, lambda: True)  # type: ignore[arg-type]

        factory = registry._REGISTRY["rbf"][0]
        monkeypatch.setitem(registry._REGISTRY, "rbf", (factory, lambda: False))
        assert "rbf" not in registry.available_operators()
        with pytest.raises(ModelError, match="unavailable"):
            registry.create_operator("rbf")
