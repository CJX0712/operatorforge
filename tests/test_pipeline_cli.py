"""Integration tests for benchmark orchestration and CLI entry points.

Author: 晨星
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import cli
import operators.registry as operator_registry
import pipeline.benchmark as benchmark_module
from core.config import BenchmarkConfig
from core.types import BenchmarkReport
from pipeline.benchmark import (
    BenchmarkPipeline,
    deterministic_signature,
    reports_are_deterministic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_config(output_path: str = "qa-benchmark.json") -> BenchmarkConfig:
    return BenchmarkConfig.from_env(
        seed=23,
        seeds=(3,),
        n_train=5,
        n_test=2,
        grid_size_1d=16,
        grid_size_2d=8,
        superresolution_grid=32,
        modes=3,
        width=4,
        depth=1,
        ridge=1.0e-4,
        output_path=output_path,
    )


class TestBenchmarkPipeline:
    """Drive the real end-to-end pipeline at a compact deterministic scale."""

    def test_small_benchmark_runs_all_models_tasks_ablations_and_gates(self) -> None:
        report = BenchmarkPipeline(_small_config()).benchmark()

        assert report.project == "OperatorForge"
        assert report.author == "晨星"
        assert len(report.results) == 3 * 7
        assert len(report.aggregates) == 3 * 7
        assert {item["arm"] for item in report.ablations} == {
            "full",
            "no_gate",
            "no_markov",
            "no_highfreq",
        }
        assert set(report.gates) == {
            "G1_performance",
            "G2_superresolution",
            "G3_invariants",
            "G4_determinism",
        }
        assert report.gates["G3_invariants"]["passed"] is True
        assert report.elapsed_sec >= 0.0

    def test_save_requires_report_then_writes_utf8_json(self, tmp_path: Path) -> None:
        output = tmp_path / "结果.json"
        pipeline = BenchmarkPipeline(_small_config(str(output)))
        with pytest.raises(RuntimeError, match=r"benchmark\(\)"):
            pipeline.save()

        report = pipeline.benchmark()
        saved = pipeline.save()
        loaded = json.loads(saved.read_text(encoding="utf-8"))

        assert saved == output
        assert loaded["author"] == "晨星"
        assert loaded["project"] == report.project

    def test_deterministic_signature_ignores_elapsed_timing(self) -> None:
        first_results = [{"model_name": "x", "relative_l2": 0.1, "elapsed_sec": 1.0}]
        second_results = [{"model_name": "x", "relative_l2": 0.1, "elapsed_sec": 999.0}]
        aggregates = [{"model_name": "x", "relative_l2_mean": 0.1}]
        assert deterministic_signature(first_results, aggregates) == deterministic_signature(
            second_results, aggregates
        )

        base = {
            "project": "OperatorForge",
            "author": "晨星",
            "configuration": {},
            "aggregates": aggregates,
            "gates": {},
            "ablations": [],
            "failure_cases": [],
        }
        first = BenchmarkReport(**base, results=first_results, elapsed_sec=1.0)
        second = BenchmarkReport(**base, results=second_results, elapsed_sec=999.0)
        assert reports_are_deterministic(first, second)

    def test_missing_optional_backend_is_marked_skipped_without_fake_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = operator_registry._REGISTRY["rbf"][0]
        monkeypatch.setitem(operator_registry._REGISTRY, "rbf", (factory, lambda: False))
        monkeypatch.setattr(benchmark_module, "available_rbf", lambda: False, raising=False)
        monkeypatch.setattr(benchmark_module, "TASKS", ("advdiff1d",))
        monkeypatch.setattr(benchmark_module, "BASELINE_NAMES", ("rbf",))
        monkeypatch.setattr(benchmark_module, "SUPERRESOLUTION_NAMES", ())
        monkeypatch.setattr(
            BenchmarkPipeline, "_evaluate_gates", lambda self, results, aggregates: {}
        )
        monkeypatch.setattr(
            benchmark_module,
            "_derive_failure_cases",
            lambda results, aggregates, ablations, gates: [],
        )

        report = BenchmarkPipeline(_small_config()).benchmark()

        skipped = [
            item
            for item in report.results
            if item.get("model_name") == "rbf" and item.get("status") == "skipped"
        ]
        assert len(skipped) == 1
        assert skipped[0].get("relative_l2") is None
        assert "reason" in skipped[0] and skipped[0]["reason"]

    def test_unknown_internal_model_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark model"):
            BenchmarkPipeline(_small_config())._create_model("missing", "advdiff1d", 1)


class TestCommandLine:
    """Verify every CLI subcommand plus readable argument failures."""

    def test_list_subcommand_exits_zero_and_prints_models(self, capsys) -> None:
        exit_code = cli.main(["list"])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "fno" in output.splitlines()
        assert "opfuse" in output.splitlines()

    def test_benchmark_subcommand_exits_zero_and_writes_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ) -> None:
        report = BenchmarkReport(
            project="OperatorForge",
            author="晨星",
            configuration={},
            results=[],
            aggregates=[],
            gates={"G1_performance": {"passed": True}},
            ablations=[],
            failure_cases=[],
            elapsed_sec=0.01,
        )

        class FakePipeline:
            def __init__(self, config: BenchmarkConfig) -> None:
                self.config = config

            def benchmark(self) -> BenchmarkReport:
                return report

            def save(self, path: Path) -> Path:
                path.write_text("{}\n", encoding="utf-8")
                return path

        output = tmp_path / "cli-result.json"
        monkeypatch.setattr(cli, "BenchmarkPipeline", FakePipeline)

        exit_code = cli.main(
            [
                "benchmark",
                "--output",
                str(output),
                "--n-train",
                "5",
                "--n-test",
                "2",
                "--width",
                "4",
                "--depth",
                "1",
                "--modes",
                "3",
                "--seeds",
                "7",
            ]
        )
        summary = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert output.exists()
        assert summary["gates"] == {"G1_performance": True}

    def test_invalid_subcommand_returns_nonzero_with_readable_error(self) -> None:
        completed = subprocess.run(
            [sys.executable, "cli.py", "not-a-command"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert completed.returncode != 0
        assert "invalid choice" in completed.stderr
        assert "benchmark" in completed.stderr and "list" in completed.stderr

    def test_real_list_entrypoint_smoke(self) -> None:
        completed = subprocess.run(
            [sys.executable, "cli.py", "list"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert completed.returncode == 0
        assert "opfuse" in completed.stdout.splitlines()
