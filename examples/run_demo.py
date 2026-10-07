"""Run the complete fast OperatorForge demo twice for determinism.

Author: 晨星
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BenchmarkConfig  # noqa: E402
from pipeline.benchmark import BenchmarkPipeline, reports_are_deterministic  # noqa: E402


def main() -> int:
    """Produce benchmark.json and verify a second run bit-for-bit."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output_path = PROJECT_ROOT / "benchmark.json"
    repeat_path = PROJECT_ROOT / "benchmark_repeat.json"
    config = BenchmarkConfig.from_env(
        seeds=(17, 29, 43),
        n_train=24,
        n_test=8,
        grid_size_1d=64,
        grid_size_2d=16,
        superresolution_grid=256,
        modes=6,
        width=12,
        depth=2,
        ridge=2.0e-5,
        output_path=str(output_path),
    )
    first_pipeline = BenchmarkPipeline(config)
    first_report = first_pipeline.benchmark()
    first_pipeline.save(output_path)

    second_pipeline = BenchmarkPipeline(config)
    second_report = second_pipeline.benchmark()
    deterministic = reports_are_deterministic(first_report, second_report)
    first_report.gates["G4_determinism"]["passed"] = deterministic
    first_report.gates["G4_determinism"]["comparison"] = (
        "Independent second run matched bit-for-bit after excluding elapsed_sec."
        if deterministic
        else "Independent second run produced different core metrics."
    )
    first_pipeline.save(output_path)
    second_pipeline.save(repeat_path)
    print(
        json.dumps(
            {
                "author": "晨星",
                "benchmark": str(output_path),
                "deterministic": deterministic,
                "elapsed_sec": first_report.elapsed_sec + second_report.elapsed_sec,
                "gates": {name: gate["passed"] for name, gate in first_report.gates.items()},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if deterministic else 2


if __name__ == "__main__":
    raise SystemExit(main())
