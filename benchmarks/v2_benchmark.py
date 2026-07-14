#!/usr/bin/env python3
"""Run the native-v2 user-impact benchmark and compare optimization snapshots.

Fair latency and throughput measurements run with every profiler disabled.
Whole-pipeline/Metal and model-graph profiling use separate diagnostic server
processes so synchronization and capture overhead cannot contaminate benchmark
rows.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import http.client
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from mlx_lm.utils import load_tokenizer  # noqa: E402
from mlx_worker.native_mlx.bootstrap import resolve_model_path  # noqa: E402
from mlx_worker.native_mlx.pipeline_profile import (  # noqa: E402
    PipelineEvent,
    write_pipeline_artifacts,
)


DEFAULT_MODELS = (
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/LFM2.5-8B-A1B-MLX-4bit",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/gemma-3-270m-it-qat-8bit",
)
DEFAULT_CONFIGURATIONS = ("serial-radix", "overlap-radix")
CONFIGURATION_ENV = {
    "serial-radix": {
        "MLX_RUNTIME_NATIVE_EXECUTION_MODE": "serial",
        "MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY": "radix",
    },
    "overlap-radix": {
        "MLX_RUNTIME_NATIVE_EXECUTION_MODE": "overlap",
        "MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY": "radix",
    },
    "overlap-block-hash": {
        "MLX_RUNTIME_NATIVE_EXECUTION_MODE": "overlap",
        "MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY": "block-hash",
    },
}
PRESETS = {
    "smoke": {"warmups": 0, "samples": 1, "order_rounds": 1},
    "standard": {"warmups": 1, "samples": 5, "order_rounds": 2},
    "optimization": {"warmups": 2, "samples": 20, "order_rounds": 2},
}
P95_MIN_SAMPLES = 20
REQUIRED_PIPELINE_COMPONENTS = {
    "runtime",
    "transport",
    "scheduler",
    "executor",
    "cache",
    "model",
    "sampling",
    "mlx",
    "streaming",
    "gateway",
}


@dataclass(frozen=True)
class RequestSpec:
    """One deterministic public API request in a benchmark scenario."""

    name: str
    messages: tuple[dict[str, str], ...]
    max_tokens: int
    stream: bool
    prompt_tokens: int

    def payload(self, model: str) -> dict[str, Any]:
        """Return an OpenAI-compatible request payload."""

        payload: dict[str, Any] = {
            "model": model,
            "messages": list(self.messages),
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": self.stream,
        }
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        return payload


@dataclass(frozen=True)
class Scenario:
    """A named user-visible workload with a fixed concurrency shape."""

    name: str
    description: str
    requests: tuple[RequestSpec, ...]
    concurrency: int
    prime: RequestSpec | None = None


@dataclass(frozen=True)
class RequestSample:
    """One measured public gateway response."""

    scenario: str
    request: str
    round: int
    status: int
    ttft_ms: float | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    decode_tokens_per_second: float | None
    end_to_end_tokens_per_second: float | None
    text_sha256: str
    request_id: str | None
    started_monotonic_ns: int
    finished_monotonic_ns: int
    error: str | None = None


@dataclass
class ScenarioAccumulator:
    """Raw measurements collected across rotated configuration rounds."""

    samples: list[RequestSample] = field(default_factory=list)
    workload_wall_ms: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class Gateway(AbstractContextManager["Gateway"]):
    """Start and stop one isolated native-v2 gateway process."""

    def __init__(
        self,
        *,
        binary: Path,
        config: Path,
        port: int,
        log_path: Path,
        env: dict[str, str],
        launch_timeout: int,
    ) -> None:
        self.binary = binary
        self.config = config
        self.port = port
        self.log_path = log_path
        self.env = env
        self.launch_timeout = launch_timeout
        self.process: subprocess.Popen[Any] | None = None
        self._log: Any = None

    def __enter__(self) -> Gateway:
        if not _is_port_free(self.port):
            raise RuntimeError(f"benchmark port {self.port} is already in use")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        process_env = os.environ.copy()
        process_env.update(self.env)
        process_env["MLX_RUNTIME_CONFIG"] = str(self.config)
        self.process = subprocess.Popen(
            [str(self.binary)],
            cwd=REPO_ROOT,
            env=process_env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_gateway(self.process, self.port, self.launch_timeout, self.log_path)
        return self

    def __exit__(self, *exc_info: object) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                process.wait(timeout=30)
        if self._log is not None:
            self._log.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested benchmark command."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        comparison = compare_results(
            _read_json(args.baseline),
            _read_json(args.candidate),
            max_regression_ratio=args.max_regression_pct / 100.0,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(comparison, indent=2) + "\n")
        args.output_markdown.write_text(render_comparison(comparison))
        print(f"v2_benchmark_comparison={args.output_json}")
        print(f"v2_benchmark_comparison_report={args.output_markdown}")
        print(f"v2_benchmark_optimization_passed={int(comparison['passed'])}")
        return 0 if comparison["passed"] else 1
    return run_benchmark(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser(
        "run", help="Run fair performance workloads and separate diagnostic profiles."
    )
    run.add_argument("--model", dest="models", action="append")
    run.add_argument(
        "--configuration",
        dest="configurations",
        action="append",
        choices=tuple(CONFIGURATION_ENV),
        help=(
            "Repeat to select runtime setups. Defaults to serial-radix and "
            "overlap-radix."
        ),
    )
    run.add_argument("--preset", choices=tuple(PRESETS), default="optimization")
    run.add_argument("--warmups", type=int)
    run.add_argument("--samples", type=int)
    run.add_argument("--order-rounds", type=int)
    run.add_argument("--order-seed", type=int, default=42)
    run.add_argument(
        "--profile",
        choices=("none", "system", "graph", "all"),
        default="all",
        help="Profiles are diagnostic and always run outside timed measurements.",
    )
    run.add_argument(
        "--metal",
        action="store_true",
        help="Add a bounded Metal .gputrace to each system profile.",
    )
    run.add_argument("--baseline", type=Path)
    run.add_argument("--max-regression-pct", type=float, default=2.0)
    run.add_argument("--label", default="candidate")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--port", type=int, default=18400)
    run.add_argument("--launch-timeout", type=int, default=300)
    run.add_argument("--no-build", action="store_true")
    run.add_argument("--text-cache-budget-bytes", type=int, default=536_870_912)

    compare = subcommands.add_parser(
        "compare", help="Compare two independently captured results.json files."
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--max-regression-pct", type=float, default=2.0)
    compare.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/v2-comparison.json"),
    )
    compare.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("benchmarks/results/v2-comparison.md"),
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> int:
    """Run all selected models and configurations and write durable evidence."""

    preset = PRESETS[args.preset]
    warmups = preset["warmups"] if args.warmups is None else args.warmups
    samples = preset["samples"] if args.samples is None else args.samples
    order_rounds = (
        preset["order_rounds"] if args.order_rounds is None else args.order_rounds
    )
    if warmups < 0 or samples <= 0 or order_rounds <= 0:
        raise SystemExit(
            "warmups must be non-negative; samples/rounds must be positive"
        )
    if args.max_regression_pct < 0:
        raise SystemExit("max regression must be non-negative")
    if args.metal and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise SystemExit("--metal requires MTL_CAPTURE_ENABLED=1 before startup")

    models = tuple(args.models or DEFAULT_MODELS)
    configurations = tuple(args.configurations or DEFAULT_CONFIGURATIONS)
    output_dir = _output_dir(args.output_dir, args.label)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = REPO_ROOT / "target" / "release" / "mlx_runtime_gateway"
    if not args.no_build:
        _log("building release gateway outside measured regions")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "mlx_runtime_gateway"],
            cwd=REPO_ROOT,
            check=True,
        )
    if not binary.is_file():
        raise SystemExit(f"gateway binary does not exist: {binary}")

    manifest = {
        "label": args.label,
        "preset": args.preset,
        "models": list(models),
        "configurations": list(configurations),
        "warmups_per_scenario_per_round": warmups,
        "target_samples_per_scenario": samples,
        "order_rounds": order_rounds,
        "order_seed": args.order_seed,
        "profile": args.profile,
        "metal_capture": args.metal,
        "temperature": 0.0,
        "top_p": 1.0,
        "text_cache_budget_bytes": args.text_cache_budget_bytes,
        "metric_direction": {
            "ttft_ms": "lower_is_better",
            "latency_ms": "lower_is_better",
            "decode_tokens_per_second": "higher_is_better",
            "completion_tokens_per_second": "higher_is_better",
            "requests_per_second": "higher_is_better",
        },
        "profilers_excluded_from_performance_rows": True,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": _source_metadata(),
        "host": _host_metadata(),
        "manifest": manifest,
        "runs": [],
        "failures": [],
    }

    for model_index, model in enumerate(models):
        _log(f"model {model_index + 1}/{len(models)}: {model}")
        try:
            model_path = resolve_model_path(model)
            scenarios = build_scenarios(model_path)
            manifest.setdefault("scenarios_by_model", {})[model] = [
                {
                    "name": scenario.name,
                    "description": scenario.description,
                    "concurrency": scenario.concurrency,
                    "request_shapes": [
                        {
                            "prompt_tokens": request.prompt_tokens,
                            "max_tokens": request.max_tokens,
                            "stream": request.stream,
                            "request_sha256": _request_fingerprint(request),
                        }
                        for request in scenario.requests
                    ],
                }
                for scenario in scenarios
            ]
            model_runs = _benchmark_model(
                model=model,
                model_index=model_index,
                scenarios=scenarios,
                configurations=configurations,
                binary=binary,
                output_dir=output_dir,
                port=args.port,
                warmups=warmups,
                target_samples=samples,
                order_rounds=order_rounds,
                order_seed=args.order_seed,
                launch_timeout=args.launch_timeout,
                text_cache_budget_bytes=args.text_cache_budget_bytes,
                profile=args.profile,
                metal=args.metal,
            )
            result["runs"].extend(model_runs)
            result["failures"].extend(
                failure for run in model_runs for failure in run.get("failures", [])
            )
        except Exception as exc:
            failure = {
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            result["failures"].append(failure)
            _log(f"model failed: {failure['error']}")

    manifest["workload_fingerprint"] = _workload_fingerprint(manifest)
    result_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(render_report(result))
    comparison: dict[str, Any] | None = None
    if args.baseline is not None:
        comparison = compare_results(
            _read_json(args.baseline),
            result,
            max_regression_ratio=args.max_regression_pct / 100.0,
        )
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        (output_dir / "comparison.md").write_text(render_comparison(comparison))

    passed = not result["failures"] and (
        comparison is None or bool(comparison["passed"])
    )
    print(f"v2_benchmark_results={result_path}")
    print(f"v2_benchmark_report={report_path}")
    print(f"v2_benchmark_models={len(models)}")
    print(f"v2_benchmark_configurations={len(configurations)}")
    print(f"v2_benchmark_validation_ok={int(passed)}")
    return 0 if passed else 1


def build_scenarios(model_path: Path) -> tuple[Scenario, ...]:
    """Build token-checked short, long, cache, concurrency, and mixed workloads."""

    tokenizer = load_tokenizer(model_path)

    def token_count(messages: Sequence[dict[str, str]]) -> int:
        values = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
        return len(values)

    short = "Explain why low first-token latency matters to an interactive user."
    medium = _prompt_at_least_tokens(token_count, 512, "medium")
    long = _prompt_at_least_tokens(token_count, 2048, "long")
    shared_system = _prompt_at_least_tokens(token_count, 512, "shared-system")

    def request(
        name: str,
        content: str,
        *,
        max_tokens: int,
        stream: bool = True,
        system: str | None = None,
    ) -> RequestSpec:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return RequestSpec(
            name=name,
            messages=tuple(messages),
            max_tokens=max_tokens,
            stream=stream,
            prompt_tokens=token_count(messages),
        )

    interactive = request("interactive", short, max_tokens=32)
    nonstream = request("nonstream", short, max_tokens=32, stream=False)
    long_prefill = request(
        "long-prefill", long + "\nSummarize the key points.", max_tokens=16
    )
    decode = request(
        "decode",
        "Write a numbered list and continue until the token limit.",
        max_tokens=128,
    )
    shared = tuple(
        request(
            f"shared-{index}",
            f"Answer variant {index} in one paragraph.",
            max_tokens=32,
            system=shared_system,
        )
        for index in range(4)
    )
    concurrency4 = tuple(
        request(
            f"concurrent-{index}",
            f"{short} Request variant {index}.",
            max_tokens=64,
        )
        for index in range(4)
    )
    concurrency8 = tuple(
        request(
            f"pressure-{index}",
            f"{short} Pressure request {index}.",
            max_tokens=32,
        )
        for index in range(8)
    )
    mixed = (
        request("mixed-short-0", short, max_tokens=64),
        request("mixed-short-1", short + " Use an analogy.", max_tokens=64),
        request("mixed-short-2", short + " Use bullets.", max_tokens=64),
        request("mixed-short-3", short + " Be concise.", max_tokens=64),
        request("mixed-medium-0", medium, max_tokens=32),
        request("mixed-medium-1", medium + " medium branch", max_tokens=32),
        request("mixed-long", long, max_tokens=16),
        request("mixed-decode", "Continue a numbered sequence.", max_tokens=128),
    )
    return (
        Scenario(
            "interactive_short_stream",
            "Single interactive streaming request; emphasizes TTFT and token cadence.",
            (interactive,),
            1,
        ),
        Scenario(
            "short_nonstream",
            "Single non-streaming request; measures complete-response latency.",
            (nonstream,),
            1,
        ),
        Scenario(
            "long_prefill_stream",
            "Long prompt with short output; emphasizes prefill and TTFT.",
            (long_prefill,),
            1,
        ),
        Scenario(
            "sustained_decode_stream",
            "Short prompt with long output; emphasizes decode throughput.",
            (decode,),
            1,
        ),
        Scenario(
            "shared_prefix_warm_concurrency4",
            "Four concurrent requests after priming a shared system prefix.",
            shared,
            4,
            prime=shared[0],
        ),
        Scenario(
            "short_concurrency4",
            "Four simultaneous interactive requests.",
            concurrency4,
            4,
        ),
        Scenario(
            "short_concurrency8",
            "Eight simultaneous short requests for queue and throughput pressure.",
            concurrency8,
            8,
        ),
        Scenario(
            "mixed_prefill_decode_concurrency8",
            "Eight simultaneous short, medium, long-prefill, and long-decode requests.",
            mixed,
            8,
        ),
    )


def _benchmark_model(
    *,
    model: str,
    model_index: int,
    scenarios: tuple[Scenario, ...],
    configurations: tuple[str, ...],
    binary: Path,
    output_dir: Path,
    port: int,
    warmups: int,
    target_samples: int,
    order_rounds: int,
    order_seed: int,
    launch_timeout: int,
    text_cache_budget_bytes: int,
    profile: str,
    metal: bool,
) -> list[dict[str, Any]]:
    model_dir = output_dir / "models" / _slug(model)
    accumulators = {
        configuration: {scenario.name: ScenarioAccumulator() for scenario in scenarios}
        for configuration in configurations
    }
    samples_by_round = _split_samples(target_samples, order_rounds)
    for round_index, round_samples in enumerate(samples_by_round, start=1):
        config_order = list(configurations)
        if round_index % 2 == 0:
            config_order.reverse()
        _log(
            f"{model}: performance round {round_index}/{order_rounds}; "
            f"configuration order={','.join(config_order)}"
        )
        for config_index, configuration in enumerate(config_order):
            run_dir = model_dir / configuration / "performance" / f"round-{round_index}"
            runtime_config = _write_runtime_config(
                model=model,
                port=port,
                output_dir=run_dir,
                socket_tag=f"{model_index}-{config_index}-{round_index}",
            )
            env = _gateway_env(
                configuration,
                text_cache_budget_bytes=text_cache_budget_bytes,
            )
            with Gateway(
                binary=binary,
                config=runtime_config,
                port=port,
                log_path=run_dir / "gateway.log",
                env=env,
                launch_timeout=launch_timeout,
            ):
                ordered_scenarios = list(scenarios)
                random.Random(order_seed + model_index * 10_000 + round_index).shuffle(
                    ordered_scenarios
                )
                for scenario in ordered_scenarios:
                    _log(
                        f"{model} {configuration} round={round_index} "
                        f"scenario={scenario.name} target_samples={round_samples}"
                    )
                    _prime_and_warmup(port, model, scenario, warmups=warmups)
                    observed, wall_ms = _measure_scenario(
                        port,
                        model,
                        scenario,
                        target_samples=round_samples,
                        round_index=round_index,
                    )
                    accumulator = accumulators[configuration][scenario.name]
                    accumulator.samples.extend(observed)
                    accumulator.workload_wall_ms.extend(wall_ms)
                    accumulator.metrics = _metrics(port)
                    _assert_configuration_metrics(configuration, accumulator.metrics)

    runs: list[dict[str, Any]] = []
    for configuration in configurations:
        config_dir = model_dir / configuration
        scenario_rows = []
        run_failures: list[dict[str, str]] = []
        for scenario in scenarios:
            accumulator = accumulators[configuration][scenario.name]
            row = summarize_scenario(
                scenario,
                accumulator.samples,
                accumulator.workload_wall_ms,
                accumulator.metrics,
            )
            scenario_rows.append(row)
            if row["errors"]:
                run_failures.append(
                    {
                        "model": model,
                        "configuration": configuration,
                        "scenario": scenario.name,
                        "error": f"{row['errors']} measured request(s) failed",
                    }
                )
        profiles: dict[str, Any] = {}
        if profile in {"system", "all"}:
            try:
                profiles["system"] = _run_system_profile(
                    model=model,
                    configuration=configuration,
                    scenario=next(
                        item for item in scenarios if item.name == "long_prefill_stream"
                    ),
                    binary=binary,
                    output_dir=config_dir / "system-profile",
                    port=port,
                    launch_timeout=launch_timeout,
                    text_cache_budget_bytes=text_cache_budget_bytes,
                    metal=metal,
                    socket_tag=f"{model_index}-{configuration}-system",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                profiles["system"] = {"error": error}
                run_failures.append(
                    {
                        "model": model,
                        "configuration": configuration,
                        "scenario": "system-profile",
                        "error": error,
                    }
                )
        if profile in {"graph", "all"}:
            try:
                profiles["graph"] = _run_graph_profile(
                    model=model,
                    configuration=configuration,
                    scenarios=tuple(
                        item
                        for item in scenarios
                        if item.name
                        in {"long_prefill_stream", "sustained_decode_stream"}
                    ),
                    binary=binary,
                    output_dir=config_dir / "graph-profile",
                    port=port,
                    launch_timeout=launch_timeout,
                    text_cache_budget_bytes=text_cache_budget_bytes,
                    socket_tag=f"{model_index}-{configuration}-graph",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                profiles["graph"] = {"error": error}
                run_failures.append(
                    {
                        "model": model,
                        "configuration": configuration,
                        "scenario": "graph-profile",
                        "error": error,
                    }
                )
        run = {
            "model": model,
            "configuration": configuration,
            "configuration_environment": CONFIGURATION_ENV[configuration],
            "scenarios": scenario_rows,
            "profiles": profiles,
            "failures": run_failures,
        }
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "result.json").write_text(json.dumps(run, indent=2) + "\n")
        runs.append(run)
    return runs


def _prime_and_warmup(
    port: int, model: str, scenario: Scenario, *, warmups: int
) -> None:
    if scenario.prime is not None:
        _request(port, model, scenario.name, scenario.prime, round_index=0)
    for _ in range(warmups):
        _execute_request_batch(port, model, scenario, round_index=0)


def _measure_scenario(
    port: int,
    model: str,
    scenario: Scenario,
    *,
    target_samples: int,
    round_index: int,
) -> tuple[list[RequestSample], list[float]]:
    observed: list[RequestSample] = []
    wall_ms: list[float] = []
    while len(observed) < target_samples:
        started = time.perf_counter()
        batch = _execute_request_batch(port, model, scenario, round_index=round_index)
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        observed.extend(batch)
    return observed, wall_ms


def _execute_request_batch(
    port: int, model: str, scenario: Scenario, *, round_index: int
) -> list[RequestSample]:
    if scenario.concurrency <= 1:
        return [
            _request(port, model, scenario.name, request, round_index=round_index)
            for request in scenario.requests
        ]
    with ThreadPoolExecutor(max_workers=scenario.concurrency) as pool:
        return list(
            pool.map(
                lambda request: _request(
                    port, model, scenario.name, request, round_index=round_index
                ),
                scenario.requests,
            )
        )


def _request(
    port: int,
    model: str,
    scenario: str,
    request: RequestSpec,
    *,
    round_index: int,
) -> RequestSample:
    payload = request.payload(model)
    encoded = json.dumps(payload).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.perf_counter()
    started_ns = time.perf_counter_ns()
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=encoded,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if request.stream else "application/json",
            },
        )
        response = connection.getresponse()
        ttft_ms: float | None = None
        text = ""
        prompt_tokens = request.prompt_tokens
        completion_tokens = 0
        request_id: str | None = None
        if request.stream:
            while raw := response.fp.readline():
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                raw_id = event.get("id")
                if raw_id:
                    request_id = str(raw_id).removeprefix("chatcmpl-")
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content") or ""
                    if delta and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                    text += delta
                usage = event.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                completion_tokens = int(
                    usage.get("completion_tokens", completion_tokens)
                )
        else:
            body = json.loads(response.read())
            raw_id = body.get("id")
            if raw_id:
                request_id = str(raw_id).removeprefix("chatcmpl-")
            if response.status == 200:
                text = str(body["choices"][0]["message"]["content"])
                usage = body.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                completion_tokens = int(usage.get("completion_tokens", 0))
        latency_ms = (time.perf_counter() - started) * 1000.0
        finished_ns = time.perf_counter_ns()
        decode_tps = None
        if ttft_ms is not None and completion_tokens > 1 and latency_ms > ttft_ms:
            decode_tps = (completion_tokens - 1) / ((latency_ms - ttft_ms) / 1000.0)
        e2e_tps = (
            completion_tokens / (latency_ms / 1000.0)
            if completion_tokens > 0 and latency_ms > 0
            else None
        )
        return RequestSample(
            scenario=scenario,
            request=request.name,
            round=round_index,
            status=response.status,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            decode_tokens_per_second=decode_tps,
            end_to_end_tokens_per_second=e2e_tps,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            request_id=request_id,
            started_monotonic_ns=started_ns,
            finished_monotonic_ns=finished_ns,
            error=None if response.status == 200 else f"HTTP {response.status}",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        finished_ns = time.perf_counter_ns()
        return RequestSample(
            scenario=scenario,
            request=request.name,
            round=round_index,
            status=0,
            ttft_ms=None,
            latency_ms=latency_ms,
            prompt_tokens=request.prompt_tokens,
            completion_tokens=0,
            decode_tokens_per_second=None,
            end_to_end_tokens_per_second=None,
            text_sha256=hashlib.sha256(b"").hexdigest(),
            request_id=None,
            started_monotonic_ns=started_ns,
            finished_monotonic_ns=finished_ns,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        connection.close()


def summarize_scenario(
    scenario: Scenario,
    samples: Sequence[RequestSample],
    workload_wall_ms: Sequence[float],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Aggregate one scenario without hiding raw request observations."""

    good = [sample for sample in samples if sample.status == 200]
    latency = [sample.latency_ms for sample in good]
    ttft = [sample.ttft_ms for sample in good if sample.ttft_ms is not None]
    decode_tps = [
        sample.decode_tokens_per_second
        for sample in good
        if sample.decode_tokens_per_second is not None
    ]
    e2e_tps = [
        sample.end_to_end_tokens_per_second
        for sample in good
        if sample.end_to_end_tokens_per_second is not None
    ]
    total_wall_s = max(sum(workload_wall_ms) / 1000.0, 1e-9)
    completion_total = sum(sample.completion_tokens for sample in good)
    batch_completion_tps: list[float] = []
    batch_request_tps: list[float] = []
    requests_per_batch = max(1, len(scenario.requests))
    for batch_index, wall_ms in enumerate(workload_wall_ms):
        batch = samples[
            batch_index * requests_per_batch : (batch_index + 1) * requests_per_batch
        ]
        batch_good = [sample for sample in batch if sample.status == 200]
        wall_s = max(wall_ms / 1000.0, 1e-9)
        batch_completion_tps.append(
            sum(sample.completion_tokens for sample in batch_good) / wall_s
        )
        batch_request_tps.append(len(batch_good) / wall_s)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "concurrency": scenario.concurrency,
        "samples": len(good),
        "errors": len(samples) - len(good),
        "error_rate": (len(samples) - len(good)) / max(1, len(samples)),
        "prompt_tokens_mean": _mean([float(sample.prompt_tokens) for sample in good]),
        "completion_tokens_mean": _mean(
            [float(sample.completion_tokens) for sample in good]
        ),
        "ttft_mean_ms": _mean(ttft),
        "ttft_p50_ms": _percentile(ttft, 0.50),
        "ttft_p95_ms": _percentile(ttft, 0.95, min_samples=P95_MIN_SAMPLES),
        "latency_mean_ms": _mean(latency),
        "latency_p50_ms": _percentile(latency, 0.50),
        "latency_p95_ms": _percentile(latency, 0.95, min_samples=P95_MIN_SAMPLES),
        "decode_tokens_per_second_mean": _mean(decode_tps),
        "decode_tokens_per_second_p50": _percentile(decode_tps, 0.50),
        "end_to_end_tokens_per_second_mean": _mean(e2e_tps),
        "completion_tokens_per_second": completion_total / total_wall_s,
        "requests_per_second": len(good) / total_wall_s,
        "completion_tokens_per_second_samples": batch_completion_tps,
        "requests_per_second_samples": batch_request_tps,
        "workload_wall_ms": list(workload_wall_ms),
        "runtime_metrics": metrics,
        "raw_samples": [asdict(sample) for sample in samples],
    }


def _run_system_profile(
    *,
    model: str,
    configuration: str,
    scenario: Scenario,
    binary: Path,
    output_dir: Path,
    port: int,
    launch_timeout: int,
    text_cache_budget_bytes: int,
    metal: bool,
    socket_tag: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = _write_runtime_config(
        model=model,
        port=port,
        output_dir=output_dir,
        socket_tag=socket_tag,
    )
    run_id = f"v2-system-{int(time.time())}-{_slug(model)}-{configuration}"
    env = _gateway_env(
        configuration,
        text_cache_budget_bytes=text_cache_budget_bytes,
        extra={
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE": "1",
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR": str(output_dir),
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID": run_id,
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD": scenario.name,
            "MLX_RUNTIME_NATIVE_METAL_CAPTURE": "1" if metal else "0",
        },
    )
    request = scenario.requests[0]
    with Gateway(
        binary=binary,
        config=runtime_config,
        port=port,
        log_path=output_dir / "gateway.log",
        env=env,
        launch_timeout=launch_timeout,
    ):
        sample = _request(port, model, scenario.name, request, round_index=0)
        if sample.status != 200 or sample.request_id is None:
            raise RuntimeError(f"system profile request failed: {sample.error}")
        events_path = output_dir / "pipeline-events.jsonl"
        deadline = time.monotonic() + 30
        events: list[PipelineEvent] = []
        while time.monotonic() < deadline:
            if events_path.is_file():
                events = _load_pipeline_events(events_path)
                if any(
                    event.request_id == sample.request_id
                    and event.component == "runtime"
                    and event.stage == "terminal"
                    for event in events
                ):
                    break
            time.sleep(0.1)
        request_worker_events = [
            event for event in events if event.request_id == sample.request_id
        ]
        if not request_worker_events:
            raise RuntimeError("pipeline profiler did not write request events")
        if not any(
            event.component == "runtime" and event.stage == "terminal"
            for event in request_worker_events
        ):
            raise RuntimeError("pipeline profiler did not write a terminal event")
        worker_start = min(request_worker_events, key=lambda event: event.monotonic_ns)
        gateway_offset_us = max(
            0,
            worker_start.offset_us
            - (worker_start.monotonic_ns - sample.started_monotonic_ns) // 1_000,
        )
        events.append(
            PipelineEvent(
                schema_version=1,
                run_id=run_id,
                request_id=sample.request_id,
                backend="native-mlx",
                model=model,
                workload=scenario.name,
                component="gateway",
                stage="http_round_trip",
                monotonic_ns=sample.started_monotonic_ns,
                offset_us=gateway_offset_us,
                duration_us=(sample.finished_monotonic_ns - sample.started_monotonic_ns)
                // 1_000,
                prompt_tokens=sample.prompt_tokens,
                completion_tokens=sample.completion_tokens,
                state="completed",
                details={"stream": request.stream, "benchmark_latency": False},
            )
        )
        request_events = [
            event for event in events if event.request_id == sample.request_id
        ]
        write_pipeline_artifacts(request_events, output_dir)
    present = {event.component for event in request_events}
    missing = sorted(REQUIRED_PIPELINE_COMPONENTS - present)
    stage_totals: dict[str, float] = {}
    for event in request_events:
        key = f"{event.component}.{event.stage}"
        stage_totals[key] = stage_totals.get(key, 0.0) + event.duration_us / 1000.0
    metal_path = output_dir / "pipeline.gputrace"
    if metal and not metal_path.is_file():
        raise RuntimeError("Metal capture requested but pipeline.gputrace is missing")
    payload = {
        "diagnostic_only": True,
        "excluded_from_performance_rows": True,
        "scenario": scenario.name,
        "request_id": sample.request_id,
        "pipeline_components": sorted(present),
        "missing_pipeline_components": missing,
        "stage_totals_ms": stage_totals,
        "pipeline_events": str(output_dir / "pipeline-events.jsonl"),
        "pipeline_trace": str(output_dir / "pipeline-trace.json"),
        "pipeline_report": str(output_dir / "pipeline-report.md"),
        "metal_capture": str(metal_path) if metal else None,
    }
    (output_dir / "system-profile.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    if missing:
        raise RuntimeError(f"system profile missing components: {', '.join(missing)}")
    return payload


def _run_graph_profile(
    *,
    model: str,
    configuration: str,
    scenarios: tuple[Scenario, ...],
    binary: Path,
    output_dir: Path,
    port: int,
    launch_timeout: int,
    text_cache_budget_bytes: int,
    socket_tag: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = _write_runtime_config(
        model=model,
        port=port,
        output_dir=output_dir,
        socket_tag=socket_tag,
    )
    env = _gateway_env(
        configuration,
        text_cache_budget_bytes=text_cache_budget_bytes,
        extra={"MLX_RUNTIME_NATIVE_GRAPH_PROFILE": "1"},
    )
    with Gateway(
        binary=binary,
        config=runtime_config,
        port=port,
        log_path=output_dir / "gateway.log",
        env=env,
        launch_timeout=launch_timeout,
    ):
        workloads = []
        for scenario in scenarios:
            source = scenario.requests[0]
            profile_request = RequestSpec(
                name=f"graph-{source.name}",
                messages=source.messages,
                max_tokens=1 if "prefill" in scenario.name else 8,
                stream=True,
                prompt_tokens=source.prompt_tokens,
            )
            sample = _request(
                port,
                model,
                scenario.name,
                profile_request,
                round_index=0,
            )
            if sample.status != 200:
                raise RuntimeError(
                    f"graph profile workload {scenario.name} failed: {sample.error}"
                )
            metrics = _metrics(port, include_all=True)
            graph_metrics = {
                key: value
                for key, value in metrics.items()
                if key.startswith(
                    (
                        "mlx_model_graph_",
                        "mlx_executor_stage_latency_by_backend_ms",
                    )
                )
            }
            if not any(key.startswith("mlx_model_graph_") for key in graph_metrics):
                raise RuntimeError("graph profile exported no model_graph metrics")
            workloads.append(
                {
                    "scenario": scenario.name,
                    "prompt_tokens": sample.prompt_tokens,
                    "completion_tokens": sample.completion_tokens,
                    "profile_max_tokens": profile_request.max_tokens,
                    "metrics": graph_metrics,
                }
            )
    payload = {
        "diagnostic_only": True,
        "excluded_from_performance_rows": True,
        "scenario": "prefill_and_decode",
        "requests": len(workloads),
        "workloads": workloads,
        "gateway_log": str(output_dir / "gateway.log"),
    }
    path = output_dir / "graph-profile.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {**payload, "artifact": str(path)}


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_regression_ratio: float,
) -> dict[str, Any]:
    """Compare equivalent runs with explicit metric direction and confidence bounds."""

    if max_regression_ratio < 0:
        raise ValueError("max regression ratio must be non-negative")
    left_manifest = baseline["manifest"]
    right_manifest = candidate["manifest"]
    comparable_keys = (
        "models",
        "configurations",
        "warmups_per_scenario_per_round",
        "target_samples_per_scenario",
        "order_rounds",
        "order_seed",
        "temperature",
        "top_p",
        "text_cache_budget_bytes",
    )
    mismatches = [
        key
        for key in comparable_keys
        if left_manifest.get(key) != right_manifest.get(key)
    ]
    if left_manifest.get("workload_fingerprint") != right_manifest.get(
        "workload_fingerprint"
    ):
        mismatches.append("workload_fingerprint")
    if mismatches:
        raise ValueError(
            "benchmark manifests are not comparable: "
            + ", ".join(sorted(set(mismatches)))
        )

    baseline_rows = _scenario_index(baseline)
    candidate_rows = _scenario_index(candidate)
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("benchmark model/configuration/scenario rows differ")
    rows: list[dict[str, Any]] = []
    regressions: list[str] = []
    parity_failures: list[str] = []
    metrics = (
        ("latency_ms", "latency_mean_ms", "lower"),
        ("ttft_ms", "ttft_mean_ms", "lower"),
        (
            "decode_tokens_per_second",
            "decode_tokens_per_second_mean",
            "higher",
        ),
        (
            "end_to_end_tokens_per_second",
            "end_to_end_tokens_per_second_mean",
            "higher",
        ),
        (
            "completion_tokens_per_second_samples",
            "completion_tokens_per_second",
            "higher",
        ),
        ("requests_per_second_samples", "requests_per_second", "higher"),
    )
    for key in baseline_rows:
        left = baseline_rows[key]
        right = candidate_rows[key]
        parity = _parity_signature(left) == _parity_signature(right)
        if not parity:
            parity_failures.append("/".join(key))
        for raw_name, summary_name, direction in metrics:
            left_value = left.get(summary_name)
            right_value = right.get(summary_name)
            if left_value is None or right_value is None:
                continue
            left_raw = _raw_metric(left, raw_name)
            right_raw = _raw_metric(right, raw_name)
            if math.isclose(float(left_value), 0.0, abs_tol=1e-12):
                continue
            change = (float(right_value) - float(left_value)) / float(left_value)
            ci_low, ci_high = _relative_mean_confidence_interval(left_raw, right_raw)
            if direction == "lower":
                regressed = ci_low > max_regression_ratio
            else:
                regressed = ci_high < -max_regression_ratio
            verdict = _change_verdict(change, direction, max_regression_ratio)
            row = {
                "model": key[0],
                "configuration": key[1],
                "scenario": key[2],
                "metric": summary_name,
                "direction": f"{direction}_is_better",
                "baseline": left_value,
                "candidate": right_value,
                "relative_change": change,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "verdict": verdict,
                "regression": regressed,
                "explicit_change": _describe_change(
                    float(left_value), float(right_value), direction
                ),
            }
            rows.append(row)
            if regressed:
                regressions.append(
                    f"{'/'.join(key)} {summary_name}: {row['explicit_change']}; "
                    f"95% interval [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]"
                )
    input_failures = [
        *baseline.get("failures", []),
        *candidate.get("failures", []),
    ]
    passed = not regressions and not parity_failures and not input_failures
    return {
        "schema_version": 1,
        "baseline": baseline.get("source", {}),
        "candidate": candidate.get("source", {}),
        "max_regression_pct": max_regression_ratio * 100.0,
        "passed": passed,
        "parity_passed": not parity_failures,
        "parity_failures": parity_failures,
        "regressions": regressions,
        "input_failures": input_failures,
        "rows": rows,
    }


def render_report(result: dict[str, Any]) -> str:
    """Render the user-facing absolute performance and profiling report."""

    manifest = result["manifest"]
    lines = [
        "# Native v2 Ultimate Benchmark",
        "",
        "> Fair performance rows were measured with pipeline, Metal, and model-graph profiling disabled.",
        "",
        f"- Label: `{manifest['label']}`",
        f"- Source commit: `{result['source'].get('git_commit', 'unknown')}`",
        f"- Dirty source tree: `{result['source'].get('git_dirty', 'unknown')}`",
        f"- Preset: `{manifest['preset']}`",
        f"- Models: {len(manifest['models'])}",
        f"- Configurations: {', '.join(manifest['configurations'])}",
        f"- Target samples per scenario: {manifest['target_samples_per_scenario']}",
        f"- Rotated order rounds: {manifest['order_rounds']}",
        "",
        "## Metric direction",
        "",
        "- TTFT and latency: lower is better.",
        "- Decode, completion, and request throughput: higher is better.",
        "- A profiler duration is diagnostic evidence, not a fair benchmark result.",
        "",
        "## User-visible performance",
        "",
        "| model | configuration | scenario | samples | errors | prompt tokens mean | completion tokens mean | TTFT mean ms (lower) | TTFT p50 ms (lower) | TTFT p95 ms (lower) | latency mean ms (lower) | latency p50 ms (lower) | latency p95 ms (lower) | decode tok/s mean (higher) | aggregate completion tok/s (higher) | requests/s (higher) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        for row in run["scenarios"]:
            lines.append(
                "| {model} | {configuration} | {scenario} | {samples} | {errors} | {prompt} | {completion} | {ttft_mean} | {ttft_p50} | {ttft_p95} | {latency_mean} | {latency_p50} | {latency_p95} | {decode_tps} | {aggregate_tps} | {request_tps} |".format(
                    model=run["model"],
                    configuration=run["configuration"],
                    scenario=row["scenario"],
                    samples=row["samples"],
                    errors=row["errors"],
                    prompt=_fmt(row["prompt_tokens_mean"]),
                    completion=_fmt(row["completion_tokens_mean"]),
                    ttft_mean=_fmt(row["ttft_mean_ms"]),
                    ttft_p50=_fmt(row["ttft_p50_ms"]),
                    ttft_p95=_fmt(row["ttft_p95_ms"]),
                    latency_mean=_fmt(row["latency_mean_ms"]),
                    latency_p50=_fmt(row["latency_p50_ms"]),
                    latency_p95=_fmt(row["latency_p95_ms"]),
                    decode_tps=_fmt(row["decode_tokens_per_second_mean"]),
                    aggregate_tps=_fmt(row["completion_tokens_per_second"]),
                    request_tps=_fmt(row["requests_per_second"]),
                )
            )
    lines.extend(
        [
            "",
            "## Diagnostic profiles",
            "",
            "| model | configuration | profile | scenario | artifact |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for run in result["runs"]:
        system_profile = run["profiles"].get("system")
        if system_profile and "scenario" in system_profile:
            lines.append(
                f"| {run['model']} | {run['configuration']} | whole pipeline/system | {system_profile['scenario']} | `{system_profile['pipeline_report']}` |"
            )
        graph_profile = run["profiles"].get("graph")
        if graph_profile and "scenario" in graph_profile:
            lines.append(
                f"| {run['model']} | {run['configuration']} | model inference graph | {graph_profile['scenario']} | `{graph_profile['artifact']}` |"
            )
    if result["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{item['model']}`: {item['error']}" for item in result["failures"]
        )
    lines.append("")
    return "\n".join(lines)


def render_comparison(result: dict[str, Any]) -> str:
    """Render an explicit baseline-to-candidate optimization verdict."""

    lines = [
        "# Native v2 Optimization Comparison",
        "",
        f"- Overall gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Output/token parity: **{'PASS' if result['parity_passed'] else 'FAIL'}**",
        f"- Allowed regression: {result['max_regression_pct']:.2f}%",
        "",
        "Latency and TTFT are reverse metrics: lower is better. Throughput is higher-is-better.",
        "",
        "| model | configuration | scenario | metric | direction | baseline | candidate | explicit result | 95% relative interval | verdict |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {model} | {configuration} | {scenario} | {metric} | {direction} | {baseline:.3f} | {candidate:.3f} | {explicit_change} | [{low:.2f}%, {high:.2f}%] | {verdict} |".format(
                **row,
                low=row["ci95_low"] * 100.0,
                high=row["ci95_high"] * 100.0,
            )
        )
    if result["parity_failures"]:
        lines.extend(["", "## Parity failures", ""])
        lines.extend(f"- {item}" for item in result["parity_failures"])
    if result["regressions"]:
        lines.extend(["", "## Regressions", ""])
        lines.extend(f"- {item}" for item in result["regressions"])
    if result.get("input_failures"):
        lines.extend(["", "## Input benchmark failures", ""])
        lines.extend(
            f"- {item.get('model', 'unknown')}: {item.get('error', item)}"
            for item in result["input_failures"]
        )
    lines.append("")
    return "\n".join(lines)


def _write_runtime_config(
    *, model: str, port: int, output_dir: Path, socket_tag: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = (REPO_ROOT / "config" / "runtime.toml").read_text()
    text = _replace_config_value(source, "port", str(port))
    text = _replace_config_value(text, "backend", "native-mlx")
    text = _replace_config_value(text, "model", model)
    socket_path = f"/tmp/mlx-runtime-v2-benchmark-{_slug(socket_tag)}.sock"
    text = _replace_config_value(text, "ipc_path", socket_path)
    target = output_dir / "runtime.toml"
    target.write_text(text)
    return target


def _replace_config_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    rendered = value if value.isdigit() else json.dumps(value)
    updated, count = pattern.subn(rf"\g<1>{rendered}", text)
    if count == 0:
        raise ValueError(f"runtime config has no {key} key")
    return updated


def _gateway_env(
    configuration: str,
    *,
    text_cache_budget_bytes: int,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env = {
        **CONFIGURATION_ENV[configuration],
        "MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES": str(text_cache_budget_bytes),
        "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE": "0",
        "MLX_RUNTIME_NATIVE_GRAPH_PROFILE": "0",
        "MLX_RUNTIME_NATIVE_METAL_CAPTURE": "0",
    }
    if extra:
        env.update(extra)
    return env


def _wait_for_gateway(
    process: subprocess.Popen[Any], port: int, timeout: int, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"gateway exited during startup:\n{tail}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"gateway readiness timed out after {timeout}s; log={log_path}")


def _is_port_free(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return False
    except OSError:
        return True


def _metrics(port: int, *, include_all: bool = False) -> dict[str, float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request("GET", "/metrics")
    response = connection.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"metrics returned HTTP {response.status}")
    wanted = (
        "mlx_scheduler_",
        "mlx_executor_",
        "mlx_prefix_",
        "mlx_radix_",
        "mlx_native_execution_mode",
        "mlx_worker_memory_bytes",
        "mlx_model_graph_",
    )
    rows: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, raw = line.rpartition(" ")
        if not separator or (not include_all and not key.startswith(wanted)):
            continue
        try:
            rows[key] = float(raw)
        except ValueError:
            continue
    return rows


def _assert_configuration_metrics(
    configuration: str, metrics: dict[str, float]
) -> None:
    expected_mode = CONFIGURATION_ENV[configuration][
        "MLX_RUNTIME_NATIVE_EXECUTION_MODE"
    ]
    matched = any(
        key.startswith("mlx_native_execution_mode")
        and f'mode="{expected_mode}"' in key
        and value == 1.0
        for key, value in metrics.items()
    )
    if not matched:
        observed = sorted(
            key for key in metrics if key.startswith("mlx_native_execution_mode")
        )
        raise RuntimeError(
            f"configuration {configuration} expected execution mode "
            f"{expected_mode}; observed={observed}"
        )


def _load_pipeline_events(path: Path) -> list[PipelineEvent]:
    if not path.is_file():
        raise RuntimeError(f"pipeline event artifact is missing: {path}")
    allowed = set(PipelineEvent.__dataclass_fields__)
    return [
        PipelineEvent(
            **{key: value for key, value in json.loads(line).items() if key in allowed}
        )
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _prompt_at_least_tokens(
    token_count: Callable[[Sequence[dict[str, str]]], int],
    target: int,
    label: str,
) -> str:
    words = [f"{label}-{index}" for index in range(max(target, 64) * 2)]
    low = 1
    high = len(words)
    while low < high:
        middle = (low + high) // 2
        content = " ".join(words[:middle])
        count = token_count([{"role": "user", "content": content}])
        if count < target:
            low = middle + 1
        else:
            high = middle
    return " ".join(words[:low])


def _split_samples(samples: int, rounds: int) -> list[int]:
    base, remainder = divmod(samples, rounds)
    return [base + (1 if index < remainder else 0) for index in range(rounds)]


def _scenario_index(
    result: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (run["model"], run["configuration"], scenario["scenario"]): scenario
        for run in result["runs"]
        for scenario in run["scenarios"]
    }


def _parity_signature(row: dict[str, Any]) -> list[tuple[str, int, int, str]]:
    return sorted(
        (
            str(sample["request"]),
            int(sample["prompt_tokens"]),
            int(sample["completion_tokens"]),
            str(sample["text_sha256"]),
        )
        for sample in row["raw_samples"]
        if int(sample["status"]) == 200
    )


def _raw_metric(row: dict[str, Any], name: str) -> list[float]:
    if name.endswith("_samples"):
        return [float(value) for value in row.get(name, [])]
    values = []
    for sample in row["raw_samples"]:
        value = sample.get(name)
        if sample["status"] == 200 and value is not None:
            values.append(float(value))
    return values


def _relative_mean_confidence_interval(
    baseline: Sequence[float], candidate: Sequence[float]
) -> tuple[float, float]:
    if not baseline or not candidate:
        return (math.nan, math.nan)
    baseline_mean = statistics.mean(baseline)
    candidate_mean = statistics.mean(candidate)
    if baseline_mean == 0:
        return (math.nan, math.nan)
    relative_mean = (candidate_mean - baseline_mean) / baseline_mean
    if len(baseline) < 2 or len(candidate) < 2:
        return relative_mean, relative_mean
    standard_error = math.sqrt(
        statistics.variance(baseline) / len(baseline)
        + statistics.variance(candidate) / len(candidate)
    )
    critical = _student_t_critical_95(min(len(baseline), len(candidate)) - 1)
    margin = critical * standard_error / baseline_mean
    return relative_mean - margin, relative_mean + margin


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    table = (
        (1, 12.706),
        (2, 4.303),
        (3, 3.182),
        (4, 2.776),
        (5, 2.571),
        (6, 2.447),
        (7, 2.365),
        (8, 2.306),
        (9, 2.262),
        (10, 2.228),
        (12, 2.179),
        (15, 2.131),
        (20, 2.086),
        (30, 2.042),
        (60, 2.000),
        (120, 1.980),
    )
    for upper, critical in table:
        if degrees_of_freedom <= upper:
            return critical
    return 1.960


def _describe_change(baseline: float, candidate: float, direction: str) -> str:
    delta = candidate - baseline
    magnitude = abs(delta)
    relative = abs(delta / baseline) * 100.0 if baseline else 0.0
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        return f"{baseline:.3f} -> {candidate:.3f}; unchanged"
    movement = "lower" if delta < 0 else "higher"
    better = (direction == "lower" and delta < 0) or (
        direction == "higher" and delta > 0
    )
    outcome = "better" if better else "worse"
    return (
        f"{baseline:.3f} -> {candidate:.3f}; {magnitude:.3f} {movement} "
        f"({relative:.2f}% {outcome})"
    )


def _change_verdict(change: float, direction: str, tolerance: float) -> str:
    if abs(change) <= tolerance:
        return "within_noise"
    if direction == "lower":
        return "better" if change < 0 else "worse"
    return "better" if change > 0 else "worse"


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _percentile(
    values: Sequence[float], quantile: float, *, min_samples: int = 1
) -> float | None:
    if len(values) < min_samples:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def _workload_fingerprint(manifest: dict[str, Any]) -> str:
    comparable = {
        key: value
        for key, value in manifest.items()
        if key not in {"label", "profile", "metal_capture", "workload_fingerprint"}
    }
    return hashlib.sha256(
        json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _request_fingerprint(request: RequestSpec) -> str:
    payload = {
        "name": request.name,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
        "prompt_tokens": request.prompt_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_metadata() -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"])
    dirty = bool(_command_output(["git", "status", "--porcelain"]))
    branch = _command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def _host_metadata() -> dict[str, Any]:
    packages = {}
    for package in ("mlx", "mlx-lm"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "processor": _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": _command_output(["sysctl", "-n", "hw.memsize"]),
        "packages": packages,
    }


def _command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _output_dir(path: Path | None, label: str) -> Path:
    if path is not None:
        return path if path.is_absolute() else REPO_ROOT / path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "benchmarks" / "results" / "v2" / f"{stamp}-{_slug(label)}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
