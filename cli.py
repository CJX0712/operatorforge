"""Command-line entry point for OperatorForge.

Author: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.config import BenchmarkConfig
from operators.registry import available_operators
from pipeline.benchmark import BenchmarkPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="operatorforge",
        description="Deterministic NumPy neural-operator benchmark by 晨星.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available operator implementations.")
    benchmark_parser = subparsers.add_parser("benchmark", help="Run the reproducible benchmark.")
    benchmark_parser.add_argument(
        "--output", default="benchmark.json", help="UTF-8 JSON output path."
    )
    benchmark_parser.add_argument("--n-train", type=int, default=48)
    benchmark_parser.add_argument("--n-test", type=int, default=18)
    benchmark_parser.add_argument("--width", type=int, default=24)
    benchmark_parser.add_argument("--depth", type=int, default=3)
    benchmark_parser.add_argument("--modes", type=int, default=10)
    benchmark_parser.add_argument("--seeds", default="17,29,43")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return a process exit status."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    options = parser.parse_args(arguments)
    if options.command == "list":
        print("\n".join(available_operators()))
        return 0
    seeds = tuple(int(item.strip()) for item in options.seeds.split(",") if item.strip())
    config = BenchmarkConfig.from_env(
        n_train=options.n_train,
        n_test=options.n_test,
        width=options.width,
        depth=options.depth,
        modes=options.modes,
        seeds=seeds,
        output_path=options.output,
    )
    pipeline = BenchmarkPipeline(config)
    report = pipeline.benchmark()
    output_path = pipeline.save(Path(options.output))
    summary = {
        "output": str(output_path),
        "elapsed_sec": report.elapsed_sec,
        "gates": {name: gate["passed"] for name, gate in report.gates.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
