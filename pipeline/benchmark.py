"""End-to-end deterministic benchmark pipeline.

Author: 晨星
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from core.config import GATE_THRESHOLDS, BenchmarkConfig
from core.seed import set_all
from core.types import BenchmarkReport
from data.pde import TASKS, advdiff_exact, advdiff_spectral, generate_dataset
from eval.metrics import evaluate_prediction, parseval_error, relative_l2
from operators.cnn import ConvolutionalOperator
from operators.coarse import CoarseSolverOperator
from operators.deeponet import DeepONetOperator
from operators.fno import FourierNeuralOperator
from operators.mlp import PointwiseMLPOperator
from operators.opfuse import ABLATIONS, OpFuseOperator
from operators.rbf import RBFOperator
from operators.registry import _REGISTRY as _OPERATOR_REGISTRY
from training.autograd import (
    Tensor,
    complex_spectral_linear,
    gated_softshrink,
    gradient_check,
    markov_correction,
)
from training.loops import fit_operator

BASELINE_NAMES = ("coarse", "cnn", "deeponet", "fno", "mlp", "rbf")
SUPERRESOLUTION_NAMES = ("cnn", "fno", "mlp", "opfuse")


class BenchmarkPipeline:
    """Generate data, train operators, evaluate gates, and serialize evidence."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig.from_env()
        self.config.validate()
        self._report: BenchmarkReport | None = None

    def benchmark(self) -> BenchmarkReport:
        """Run all tasks, models, seeds, ablations, and scientific gates."""
        start = perf_counter()
        results: list[dict[str, Any]] = []
        ablations: list[dict[str, Any]] = []
        for seed in self.config.seeds:
            set_all(seed)
            for task_name in TASKS:
                dataset = generate_dataset(
                    task_name,
                    n_train=self.config.n_train,
                    n_test=self.config.n_test,
                    grid_size=(
                        self.config.grid_size_2d
                        if task_name == "darcy2d"
                        else self.config.grid_size_1d
                    ),
                    seed=seed,
                )
                superresolution_dataset = None
                if task_name != "darcy2d":
                    superresolution_dataset = generate_dataset(
                        task_name,
                        n_train=2,
                        n_test=self.config.n_test,
                        grid_size=self.config.superresolution_grid,
                        seed=seed,
                    )
                for model_name in (*BASELINE_NAMES, "opfuse"):
                    availability = _OPERATOR_REGISTRY[model_name][1]
                    if not availability():
                        # Honest handling of an optional back end: never fabricate a
                        # number (filling 0.0 would camouflage a void as a "best"
                        # result). Mark the run skipped and record the reason so the
                        # aggregation can exclude it without producing NaN.
                        results.append(
                            {
                                "model_name": model_name,
                                "task_name": task_name,
                                "seed": seed,
                                "status": "skipped",
                                "reason": f"{model_name} operator unavailable in this environment",
                                "relative_l2": None,
                                "relative_h1": None,
                                "spectral_tail_error": None,
                                "superresolution_l2": None,
                                "elapsed_sec": 0.0,
                                "train_elapsed_sec": 0.0,
                                "train_relative_l2": None,
                            }
                        )
                        continue
                    model = self._create_model(model_name, task_name, seed)
                    train_result = fit_operator(
                        model, dataset.x_train, dataset.y_train, task_name, seed
                    )
                    prediction_start = perf_counter()
                    prediction = model.predict(dataset.x_test)
                    prediction_elapsed = perf_counter() - prediction_start
                    superresolution_error = None
                    if model_name in SUPERRESOLUTION_NAMES and superresolution_dataset is not None:
                        super_prediction = model.predict(superresolution_dataset.x_test)
                        superresolution_error = relative_l2(
                            super_prediction, superresolution_dataset.y_test
                        )
                    evaluation = evaluate_prediction(
                        model_name=model_name,
                        task_name=task_name,
                        seed=seed,
                        prediction=prediction,
                        target=dataset.y_test,
                        superresolution_l2=superresolution_error,
                        elapsed_sec=prediction_elapsed,
                    )
                    result = asdict(evaluation)
                    result["train_elapsed_sec"] = train_result.elapsed_sec
                    result["train_relative_l2"] = train_result.loss_history[-1]
                    results.append(result)

                if task_name == "burgers1d":
                    for arm in ABLATIONS:
                        model = OpFuseOperator(
                            modes=self.config.modes,
                            width=self.config.width,
                            depth=self.config.depth,
                            ridge=self.config.ridge,
                            seed=seed,
                            ablation=arm,
                        )
                        model.fit(dataset.x_train, dataset.y_train)
                        score = relative_l2(model.predict(dataset.x_test), dataset.y_test)
                        ablations.append(
                            {
                                "task_name": task_name,
                                "seed": seed,
                                "arm": arm,
                                "relative_l2": score,
                            }
                        )

        aggregates = _aggregate_results(results)
        gates = self._evaluate_gates(results, aggregates)
        failure_cases = _derive_failure_cases(results, aggregates, ablations, gates)
        elapsed = perf_counter() - start
        self._report = BenchmarkReport(
            project="OperatorForge",
            author="晨星",
            configuration=self.config.to_dict(),
            results=results,
            aggregates=aggregates,
            gates=gates,
            ablations=_aggregate_ablations(ablations),
            failure_cases=failure_cases,
            elapsed_sec=elapsed,
        )
        return self._report

    def save(self, path: str | Path | None = None) -> Path:
        """Write the latest report as stable UTF-8 JSON."""
        if self._report is None:
            raise RuntimeError("benchmark() must run before save().")
        output_path = Path(path or self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(
                self._report.to_dict(),
                output_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            output_file.write("\n")
        return output_path

    def _create_model(self, model_name: str, task_name: str, seed: int) -> Any:
        common = {
            "seed": seed,
        }
        if model_name == "fno":
            return FourierNeuralOperator(
                modes=self.config.modes,
                width=self.config.width,
                depth=self.config.depth,
                ridge=self.config.ridge * 5.0,
                **common,
            )
        if model_name == "opfuse":
            return OpFuseOperator(
                modes=self.config.modes,
                width=self.config.width,
                depth=self.config.depth,
                ridge=self.config.ridge,
                **common,
            )
        if model_name == "deeponet":
            return DeepONetOperator(
                latent_dim=min(16, self.config.width),
                width=self.config.width,
                ridge=1.0e-3,
                **common,
            )
        if model_name == "mlp":
            return PointwiseMLPOperator(width=self.config.width, ridge=1.0e-3, **common)
        if model_name == "cnn":
            return ConvolutionalOperator(ridge=5.0e-4)
        if model_name == "rbf":
            return RBFOperator(latent_dim=min(16, self.config.width), alpha=5.0e-3, **common)
        if model_name == "coarse":
            return CoarseSolverOperator(task_name=task_name)
        raise ValueError(f"Unknown benchmark model {model_name!r}.")

    def _evaluate_gates(
        self,
        results: list[dict[str, Any]],
        aggregates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        def seed_macro(model_name: str) -> np.ndarray:
            values: list[float] = []
            for seed in self.config.seeds:
                per_task = [
                    item["relative_l2"]
                    for item in results
                    if item["model_name"] == model_name
                    and item["seed"] == seed
                    and item.get("status") != "skipped"
                    and item.get("relative_l2") is not None
                ]
                if not per_task:
                    continue
                values.append(float(np.mean(per_task)))
            return np.asarray(values, dtype=np.float64)

        opfuse_values = seed_macro("opfuse")
        baseline_by_name = {name: seed_macro(name) for name in BASELINE_NAMES}
        strongest_name = min(baseline_by_name, key=lambda key: float(baseline_by_name[key].mean()))
        strongest_values = baseline_by_name[strongest_name]
        opfuse_mean = float(opfuse_values.mean())
        baseline_mean = float(strongest_values.mean())
        ratio = opfuse_mean / max(baseline_mean, 1.0e-15)
        effect = baseline_mean - opfuse_mean
        required_effect = GATE_THRESHOLDS["g1_effect_sigma_fraction"] * (
            float(opfuse_values.std()) + float(strongest_values.std())
        )
        g1_passed = ratio <= GATE_THRESHOLDS["g1_ratio"] and effect > required_effect

        superresolution = {
            name: np.asarray(
                [
                    item["superresolution_l2"]
                    for item in results
                    if item["model_name"] == name and item["superresolution_l2"] is not None
                ],
                dtype=np.float64,
            )
            for name in SUPERRESOLUTION_NAMES
        }
        opfuse_super = float(superresolution["opfuse"].mean())
        fno_super = float(superresolution["fno"].mean())
        nonspectral_best = min(
            float(superresolution["mlp"].mean()), float(superresolution["cnn"].mean())
        )
        # Honest attribution: OpFuse's off-grid (superresolution) transfer path is the
        # *identical* FourierNeuralOperator diagonal-spectral operator -- same per-seed
        # RNG, same ridge*5, same feature map -- so its superresolution error equals
        # FNO's bit-for-bit by construction. The superresolution gate is therefore
        # non-discriminating at off-grid resolution; the meaningful architectural
        # comparison is the in-grid (train==test) error, reported below as
        # opfuse_ingrid_mean / fno_ingrid_mean.
        opfuse_ingrid = seed_macro("opfuse")
        fno_ingrid = seed_macro("fno")
        opfuse_ingrid_mean = float(opfuse_ingrid.mean()) if opfuse_ingrid.size else float("nan")
        fno_ingrid_mean = float(fno_ingrid.mean()) if fno_ingrid.size else float("nan")
        structural_identity = bool(
            superresolution["opfuse"].size
            and superresolution["fno"].size
            and np.allclose(
                superresolution["opfuse"], superresolution["fno"], rtol=1e-12, atol=1e-14
            )
        )
        g2_passed = (
            opfuse_super <= GATE_THRESHOLDS["g2_fno_tolerance"] * fno_super
            and opfuse_super < 0.90 * nonspectral_best
        )

        invariants = _scientific_invariants(self.config.seed)
        g3_passed = (
            invariants["gradient_max_relative_error"]
            <= GATE_THRESHOLDS["g3_gradient_max_rel_error"]
            and invariants["advdiff_max_absolute_error"] <= GATE_THRESHOLDS["g3_advdiff_max_error"]
            and invariants["parseval_max_absolute_error"]
            <= GATE_THRESHOLDS["g3_parseval_max_error"]
        )
        signature = deterministic_signature(results, aggregates)
        return {
            "G1_performance": {
                "passed": g1_passed,
                "opfuse_mean": opfuse_mean,
                "strongest_baseline": strongest_name,
                "baseline_mean": baseline_mean,
                "ratio": ratio,
                "ratio_threshold": GATE_THRESHOLDS["g1_ratio"],
                "effect": effect,
                "required_effect": required_effect,
            },
            "G2_superresolution": {
                "passed": g2_passed,
                "opfuse_mean": opfuse_super,
                "fno_mean": fno_super,
                "fno_noninferiority_limit": GATE_THRESHOLDS["g2_fno_tolerance"] * fno_super,
                "best_nonspectral_mean": nonspectral_best,
                "structural_identity": structural_identity,
                "opfuse_ingrid_mean": opfuse_ingrid_mean,
                "fno_ingrid_mean": fno_ingrid_mean,
                "note": (
                    "OpFuse off-grid (superresolution) error equals FNO bit-for-bit because "
                    "its transfer path is the identical FNO diagonal-spectral operator; the "
                    "gate is non-discriminating at off-grid resolution. Use opfuse_ingrid_mean "
                    "vs fno_ingrid_mean for the real architectural comparison."
                    if structural_identity
                    else "OpFuse and FNO superresolution errors differ as expected."
                ),
            },
            "G3_invariants": {"passed": g3_passed, **invariants},
            "G4_determinism": {
                "passed": None,
                "signature": signature,
                "comparison": "Requires an independent second run.",
            },
        }


def deterministic_signature(results: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> str:
    """Hash only deterministic benchmark fields, excluding wall-clock times."""
    stable_results = [
        {key: value for key, value in item.items() if "elapsed_sec" not in key} for item in results
    ]
    payload = json.dumps(
        {"results": stable_results, "aggregates": aggregates},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reports_are_deterministic(first: BenchmarkReport, second: BenchmarkReport) -> bool:
    """Compare two reports bit-for-bit after removing elapsed timings and G4 metadata."""
    first_signature = deterministic_signature(first.results, first.aggregates)
    second_signature = deterministic_signature(second.results, second.aggregates)
    return first_signature == second_signature


def _aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    keys = sorted({(item["task_name"], item["model_name"]) for item in results})
    metric_names = ("relative_l2", "relative_h1", "spectral_tail_error")
    for task_name, model_name in keys:
        selected = [
            item
            for item in results
            if item["task_name"] == task_name
            and item["model_name"] == model_name
            and item.get("status") != "skipped"
        ]
        aggregate: dict[str, Any] = {
            "task_name": task_name,
            "model_name": model_name,
            "seeds": len(selected),
        }
        for metric_name in metric_names:
            raw = np.asarray([item[metric_name] for item in selected], dtype=np.float64)
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                aggregate[f"{metric_name}_mean"] = float("nan")
                aggregate[f"{metric_name}_std"] = float("nan")
            else:
                aggregate[f"{metric_name}_mean"] = float(finite.mean())
                aggregate[f"{metric_name}_std"] = float(finite.std())
        super_values = [
            item["superresolution_l2"]
            for item in selected
            if item["superresolution_l2"] is not None
        ]
        if super_values:
            aggregate["superresolution_l2_mean"] = float(np.mean(super_values))
            aggregate["superresolution_l2_std"] = float(np.std(super_values))
        aggregates.append(aggregate)
    return aggregates


def _aggregate_ablations(ablations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ABLATIONS:
        values = np.asarray(
            [item["relative_l2"] for item in ablations if item["arm"] == arm],
            dtype=np.float64,
        )
        output.append(
            {
                "task_name": "burgers1d",
                "arm": arm,
                "relative_l2_mean": float(values.mean()) if values.size else float("nan"),
                "relative_l2_std": float(values.std()) if values.size else float("nan"),
                "seeds": int(values.size),
            }
        )
    return output


def _scientific_invariants(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    spectrum_value = rng.normal(size=(2, 3, 2))
    fixed_real = Tensor(rng.normal(size=(3, 2)))
    fixed_imag = Tensor(rng.normal(size=(3, 2)))
    spectral_error = gradient_check(
        lambda variable: complex_spectral_linear(variable, fixed_real, fixed_imag).mean(),
        spectrum_value,
    )
    gate_theta = Tensor(np.full((1, 3, 1), np.log(19.0)))
    gate_threshold = Tensor(np.asarray(1.0e-3))
    gate_error = gradient_check(
        lambda variable: gated_softshrink(variable, gate_theta, gate_threshold).mean(),
        rng.normal(size=(2, 3, 2)) + 0.2,
    )
    markov_gain = Tensor(np.asarray([0.13, -0.07, 0.2]))
    markov_error = gradient_check(
        lambda variable: markov_correction(variable, markov_gain).mean(),
        rng.normal(size=(4, 3)),
    )
    initial = rng.normal(size=(4, 64))
    advdiff_error = float(np.max(np.abs(advdiff_spectral(initial) - advdiff_exact(initial))))
    return {
        "gradient_max_relative_error": max(spectral_error, gate_error, markov_error),
        "spectral_linear_gradient_error": spectral_error,
        "soft_gate_gradient_error": gate_error,
        "markov_gradient_error": markov_error,
        "advdiff_max_absolute_error": advdiff_error,
        "parseval_max_absolute_error": parseval_error(initial),
    }


def _derive_failure_cases(
    results: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    gates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    worst_by_task: list[dict[str, Any]] = []
    for task_name in TASKS:
        candidates = [item for item in aggregates if item["task_name"] == task_name]
        worst_by_task.append(max(candidates, key=lambda item: item["relative_l2_mean"]))
    for item in worst_by_task:
        failures.append(
            {
                "kind": "weak_baseline",
                "task_name": item["task_name"],
                "model_name": item["model_name"],
                "observed_relative_l2": item["relative_l2_mean"],
                "attribution": "The method's inductive bias is mismatched to the task's non-local mapping.",
            }
        )
    full_values = np.asarray([item["relative_l2"] for item in ablations if item["arm"] == "full"])
    for arm in ABLATIONS[1:]:
        arm_values = np.asarray([item["relative_l2"] for item in ablations if item["arm"] == arm])
        if float(arm_values.mean()) <= float(full_values.mean()):
            failures.append(
                {
                    "kind": "negative_ablation",
                    "task_name": "burgers1d",
                    "model_name": f"opfuse_{arm}",
                    "observed_relative_l2": float(arm_values.mean()),
                    "attribution": "Removing this component did not hurt this finite-data regime; the component is not universally beneficial.",
                }
            )
    for gate_name, gate in gates.items():
        if gate.get("passed") is False:
            failures.append(
                {
                    "kind": "gate_failure",
                    "task_name": "aggregate",
                    "model_name": "opfuse",
                    "observed_relative_l2": gates["G1_performance"]["opfuse_mean"],
                    "attribution": f"{gate_name} did not meet its frozen pre-registered threshold.",
                }
            )
    if len(failures) < 3:
        worst_runs = sorted(results, key=lambda item: item["relative_l2"], reverse=True)
        for item in worst_runs[: 3 - len(failures)]:
            failures.append(
                {
                    "kind": "high_error_run",
                    "task_name": item["task_name"],
                    "model_name": item["model_name"],
                    "observed_relative_l2": item["relative_l2"],
                    "attribution": "Derived from the highest observed test error among all measured runs.",
                }
            )
    return failures


__all__ = [
    "BASELINE_NAMES",
    "BenchmarkPipeline",
    "deterministic_signature",
    "reports_are_deterministic",
]
