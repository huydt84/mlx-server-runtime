"""Parse benchmark leaf commands delegated by the native MLX Air CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

from mlx_benchmark.runner import run_benchmark

_GATEWAY_EXECUTABLE_ENV = "MLX_AIR_GATEWAY_EXECUTABLE"
_BENCHMARK_EXECUTION_FAILURE = 50


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-air bench",
        description="Run MLX Air benchmarks and inspect benchmark results.",
    )
    actions = parser.add_subparsers(dest="action", required=True)

    run = actions.add_parser("run", help="run a benchmark suite")
    run.add_argument("--suite", required=True, help="benchmark suite name")
    run.add_argument("--focus", help="optional benchmark focus name")
    run.add_argument("--benchmark-config", metavar="PATH", help="benchmark TOML file")
    run.add_argument(
        "--profile",
        choices=("none", "representative", "all"),
        default="none",
        help="profiling selection (default: none)",
    )
    run.add_argument(
        "--server-mode",
        choices=("self-launched", "external"),
        default="self-launched",
        help="server ownership mode (default: self-launched)",
    )
    run.add_argument("--base-url", metavar="URL", help="external server base URL")
    run.add_argument("--output-dir", metavar="PATH", help="artifact output directory")

    diagnose = actions.add_parser(
        "diagnose", help="collect diagnostics for an existing result"
    )
    diagnose.add_argument("--result", required=True, metavar="PATH")
    workload = diagnose.add_mutually_exclusive_group(required=True)
    workload.add_argument("--workload-family", metavar="NAME")
    workload.add_argument("--all", action="store_true")

    calibrate = actions.add_parser(
        "calibrate", help="calibrate benchmark repetition counts"
    )
    calibrate.add_argument("--suite", required=True)
    calibrate.add_argument("--focus")
    calibrate.add_argument("--benchmark-config", metavar="PATH")
    calibrate.add_argument(
        "--repetitions", required=True, type=_positive_integer, metavar="N"
    )
    return parser


def _validate_run_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> None:
    if arguments.action != "run":
        return
    if arguments.server_mode == "external" and arguments.base_url is None:
        parser.error("--base-url is required with --server-mode external")
    if arguments.server_mode == "self-launched" and arguments.base_url is not None:
        parser.error("--base-url is not valid with --server-mode self-launched")
    if arguments.suite != "smoke":
        parser.error("only the built-in --suite smoke workload is currently supported")
    if arguments.focus is not None:
        parser.error("--focus is not available until benchmark configuration support")
    if arguments.benchmark_config is not None:
        parser.error(
            "--benchmark-config is not available until benchmark configuration support"
        )
    if arguments.profile != "none":
        parser.error("timed workloads currently require --profile none")


def _validate_gateway(parser: argparse.ArgumentParser) -> Path:
    value = os.environ.get(_GATEWAY_EXECUTABLE_ENV)
    if value is None:
        parser.error(f"{_GATEWAY_EXECUTABLE_ENV} is not set")
    gateway = Path(value)
    if not gateway.is_absolute():
        parser.error(f"{_GATEWAY_EXECUTABLE_ENV} must be an absolute path")
    if not gateway.is_file():
        parser.error(f"gateway executable does not exist: {gateway}")
    return gateway


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one benchmark action delegated by ``mlx-air``.

    Args:
        argv: Optional leaf command arguments. Process arguments are used when
            omitted.

    Returns:
        The benchmark command exit status.
    """
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    _validate_run_arguments(parser, arguments)
    gateway = _validate_gateway(parser)
    if arguments.action == "run":
        return run_benchmark(arguments, gateway)
    print("error: benchmark execution is not available in this build", file=sys.stderr)
    return _BENCHMARK_EXECUTION_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
