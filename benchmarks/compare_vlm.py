#!/usr/bin/env python3
"""Benchmark raw ``mlx-vlm``, ``mlx_vlm.server``, and this project for VLM."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import http.client
import json
import random
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, TextIO
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"

DEFAULT_VLM_MODELS = [
    # "mlx-community/LFM2-VL-1.6B-4bit",
    # "mlx-community/gemma-4-e2b-it-8bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
]
DEFAULT_CONCURRENCY_LEVELS = (1, 2, 4)
TEMPERATURE = 0.0
TOP_P = 1.0
TOKEN_DIFF_WARN_FRACTION = 0.05

VLM_SYSTEM_PREFIX = (
    "You are expert visual analyst. Describe images precisely, separate "
    "observation from inference, keep summaries grounded in visible content."
)
VLM_SHARED_SUFFIXES = [
    "Give one-sentence summary first, then 3 precise bullets.",
    "Call out most visually distinctive detail and any uncertainty.",
    "Focus on colors, layout, and object relationships.",
]
VLM_LONG_BASES = [
    (
        "Prepare structured image summary suitable for benchmark dataset. "
        "Cover scene composition, salient objects, relative positions, colors, "
        "texture cues, and likely content type such as photo, illustration, or chart."
    ),
    (
        "Explain how multimodal assistant should reason about these images without "
        "hallucinating. Mention what is directly visible, what remains ambiguous, and "
        "what concise answer format best fits content."
    ),
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from benchmarks.vlm_fixtures import (  # noqa: E402
    ImageMetadata,
    VlmFixture,
    collect_image_metadata,
    prepare_fixtures,
)
from mlx_worker.benchmarking import (  # noqa: E402
    BenchmarkResult,
    BenchmarkRun,
    P95_MIN_SAMPLES,
    P99_MIN_SAMPLES,
    VlmCancellationResult,
    VlmComparisonRow,
    VlmConcurrencyResult,
    VlmFixtureReportRow,
    VlmScenarioRun,
    VlmStreamingBackendResult,
    calculate_decode_tokens_per_second,
    calculate_end_to_end_tokens_per_second,
    calculate_overhead,
    calculate_overhead_percent,
    calculate_per_token_latency_ms,
    mean,
    now_utc_iso,
    percentile,
    write_vlm_report,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


@dataclass(frozen=True)
class VlmStreamResult:
    ttft_ms: float | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    text: str
    backend: str = "http"
    fixture_name: str = "compat-case"
    fixture_category: str = "single_image"
    prompt_preview: str = ""
    prompt_text_source: str = "http request messages"
    max_tokens: int = 0
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    image_count: int = 0
    total_image_pixels: int | None = None
    total_megapixels: float | None = None
    widths_summary: str = "not_available"
    heights_summary: str = "not_available"
    formats_summary: str = "not_available"
    total_file_size_bytes: int | None = None
    image_load_ms: float | None = None
    image_decode_ms: float | None = None
    image_preprocess_ms: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    sse_chunk_count: int | None = None
    sse_chunk_interval_mean_ms: float | None = None
    sse_chunk_interval_p50_ms: float | None = None
    sse_chunk_interval_p95_ms: float | None = None
    parse_errors: int = 0
    stream_completed_normally: bool | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class VlmPromptCase:
    name: str
    messages: list[dict[str, Any]]
    image_paths: tuple[str, ...]
    prompt_tokens_estimate: int
    category: str = "single_image"
    prompt_text: str = ""
    image_metadata: tuple[ImageMetadata, ...] = ()

    @property
    def image_count(self) -> int:
        return len(self.image_paths)

    @property
    def total_image_pixels(self) -> int | None:
        pixels = [m.pixels for m in self.image_metadata if m.pixels is not None]
        if len(pixels) != len(self.image_metadata):
            return None
        return sum(pixels)

    @property
    def total_megapixels(self) -> float | None:
        total_pixels = self.total_image_pixels
        if total_pixels is None:
            return None
        return total_pixels / 1_000_000.0

    @property
    def total_file_size_bytes(self) -> int:
        return sum(m.file_size_bytes for m in self.image_metadata)

    @property
    def widths_summary(self) -> str:
        return _dimension_summary([m.width for m in self.image_metadata])

    @property
    def heights_summary(self) -> str:
        return _dimension_summary([m.height for m in self.image_metadata])

    @property
    def formats_summary(self) -> str:
        values = [m.format or "unknown" for m in self.image_metadata]
        return ",".join(dict.fromkeys(values))


@dataclass(frozen=True)
class NopSampler:
    pass


@dataclass(frozen=True)
class RunningService:
    backend_name: str
    base_url: str
    model: str
    readiness_url: str | None


class _ProgressTracker:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: TextIO | None = None,
        use_tqdm: bool | None = None,
    ) -> None:
        self.label = label
        self.total = max(total, 0)
        self.stream = stream or sys.stderr
        self.count = 0
        if use_tqdm is None:
            use_tqdm = (
                _tqdm is not None and getattr(self.stream, "isatty", lambda: False)()
            )
        self._bar = (
            _tqdm(
                total=self.total, desc=label, unit="step", leave=False, file=self.stream
            )
            if use_tqdm and self.total > 0
            else None
        )
        if self._bar is None:
            if self.total == 0:
                _log_event(f"{self.label}: nothing to do", stream=self.stream)
            else:
                _log_event(f"{self.label}: 0/{self.total}", stream=self.stream)

    def advance(self, detail: str) -> None:
        if self.total == 0:
            return
        self.count += 1
        if self._bar is not None:
            self._bar.set_postfix_str(detail, refresh=False)
            self._bar.update(1)
            return
        percent = (self.count / self.total) * 100.0
        _log_event(
            f"{self.label}: {self.count}/{self.total} ({percent:.0f}%) - {detail}",
            stream=self.stream,
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            return
        if self.total > 0:
            _log_event(f"{self.label}: completed", stream=self.stream)


def _log_event(message: str, *, stream: TextIO | None = None) -> None:
    target = stream or sys.stderr
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=target, flush=True)


def _image_prompt_for_fixture(name: str) -> str:
    lowered = name.lower()
    if "fish" in lowered:
        return (
            "Describe this image. Is it photo or illustration? Summarize main subject, "
            "dominant colors, and overall style."
        )
    if "fruit" in lowered:
        return (
            "Describe this image of food or produce. Name visible items, colors, and "
            "arrangement on table."
        )
    if "lake" in lowered:
        return "Summarize this landscape image. Describe terrain, water, sky, and overall mood."
    return f"Describe this image ({name}) in detail and summarize what is visible."


def _estimate_vlm_prompt_tokens(prompt_text: str, image_count: int) -> int:
    return max(32, (len(prompt_text) // 4) + (image_count * 48))


def _make_vlm_case(
    name: str,
    category: str,
    prompt_text: str,
    image_paths: tuple[str, ...],
) -> VlmPromptCase:
    metadata = tuple(collect_image_metadata(Path(path)) for path in image_paths)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_path}}
        for image_path in image_paths
    )
    return VlmPromptCase(
        name=name,
        category=category,
        prompt_text=prompt_text,
        messages=[{"role": "user", "content": content}],
        image_paths=image_paths,
        prompt_tokens_estimate=_estimate_vlm_prompt_tokens(
            prompt_text, len(image_paths)
        ),
        image_metadata=metadata,
    )


def _build_long_vlm_prompt(image_labels: list[str]) -> str:
    details = " ".join(VLM_LONG_BASES)
    labels = ", ".join(image_labels)
    return (
        f"{VLM_SYSTEM_PREFIX}\n"
        f"You are given image set: {labels}. {details} Give short summary, then "
        "detailed comparison, then final one-line caption for each image."
    )


def _build_vlm_cases(fixtures: list[VlmFixture]) -> list[VlmPromptCase]:
    normalized: list[tuple[str, str]] = []
    for fixture in fixtures:
        assert fixture.image_path is not None, "fixture image_path must be set"
        normalized.append((fixture.name, str(fixture.image_path)))
    if not normalized:
        return []

    cases: list[VlmPromptCase] = []
    for index, (name, image_path) in enumerate(normalized[:3], start=1):
        base_prompt = _image_prompt_for_fixture(name)
        cases.append(
            _make_vlm_case(
                f"single-{index}-{name}", "single_image", base_prompt, (image_path,)
            )
        )
        suffix = VLM_SHARED_SUFFIXES[(index - 1) % len(VLM_SHARED_SUFFIXES)]
        cases.append(
            _make_vlm_case(
                f"prefix-{index}-{name}",
                "prefix_single_image",
                f"{VLM_SYSTEM_PREFIX}\n{base_prompt} {suffix}",
                (image_path,),
            )
        )

    if len(normalized) >= 2:
        pair_names = [normalized[0][0], normalized[1][0]]
        pair_paths = (normalized[0][1], normalized[1][1])
        cases.append(
            _make_vlm_case(
                f"compare-{pair_names[0]}-{pair_names[1]}",
                "two_image_compare",
                "You will receive two images. Describe each image briefly, then compare "
                "their subjects, style, and likely use case.",
                pair_paths,
            )
        )

    if len(normalized) >= 3:
        image_labels = [name for name, _ in normalized[:3]]
        image_paths = tuple(path for _, path in normalized[:3])
        cases.append(
            _make_vlm_case(
                "multi-image-summary",
                "multi_image_summary",
                "You will receive three images. Give one-line summary for each image, then "
                "identify which one looks most like natural photo, which one looks most "
                "synthetic or illustrative, and which one contains densest visual detail.",
                image_paths,
            )
        )
        cases.append(
            _make_vlm_case(
                "long-multi-image-analysis",
                "long_multi_image_analysis",
                _build_long_vlm_prompt(image_labels),
                image_paths,
            )
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm", action="store_true", help="Accepted for CLI symmetry.")
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--backend", dest="backends", action="append")
    parser.add_argument("--all-backends", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--benchmark-mode",
        choices=("smoke", "normal", "stable"),
        default="smoke",
    )
    parser.add_argument("--warmup-runs-per-fixture", type=int)
    parser.add_argument("--measured-runs-per-fixture", type=int)
    parser.add_argument("--warmup-trials", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument(
        "--scenario",
        choices=("baseline", "streaming", "cancellation", "concurrency", "all"),
        default="baseline",
    )
    parser.add_argument("--fixtures", action="append")
    parser.add_argument("--fixture-category", action="append")
    parser.add_argument("--concurrency-levels", default="1,2,4")
    parser.add_argument("--backend-order", default="raw,server,project")
    parser.add_argument("--randomize-backend-order", action="store_true")
    parser.add_argument("--backend-order-seed", type=int)
    parser.add_argument("--order-rounds", type=int, default=1)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "phase_9_vlm_report.md",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--project-port", type=int, default=8000)
    parser.add_argument("--server-port", type=int, default=8001)
    parser.add_argument("--launch-timeout", type=int, default=90)
    parser.add_argument("--readiness-timeout", type=int, default=90)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--cancellation-delay-ms", type=int, default=300)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--skip-project", action="store_true")
    args = parser.parse_args(argv)

    if args.report_path is not None:
        args.output_md = args.report_path
    if not args.output_md.is_absolute():
        args.output_md = REPO_ROOT / args.output_md
    if args.output_json is not None and not args.output_json.is_absolute():
        args.output_json = REPO_ROOT / args.output_json

    warmup_runs, measured_runs = _resolve_run_counts(args)
    backends = _resolve_backends(args)
    concurrency_levels = _parse_concurrency_levels(args.concurrency_levels)
    scenario_names = _resolve_scenario_names(args.scenario)

    image_dir = Path(tempfile.mkdtemp(prefix="mlx-vlm-benchmark-fixtures-"))
    _log_event(f"generating VLM fixtures under {image_dir}")
    fixtures = prepare_fixtures(image_dir)
    prompt_cases = _filter_prompt_cases(
        _build_vlm_cases(fixtures),
        fixtures=args.fixtures,
        categories=args.fixture_category,
    )
    _log_event(
        f"fixture set ready: {len(prompt_cases)} case(s) - "
        + ", ".join(case.name for case in prompt_cases)
    )

    models = args.models or DEFAULT_VLM_MODELS
    if not models:
        _log_event("ERROR: no VLM models configured")
        return 1
    if not prompt_cases:
        _log_event("ERROR: no VLM fixtures selected")
        return 1

    runs: list[BenchmarkRun] = []
    for model_index, model_name in enumerate(models, start=1):
        _log_event(f"[model {model_index}/{len(models)}] benchmarking {model_name}")
        scenario_runs = tuple(
            _run_vlm_scenario(
                scenario_name,
                model_name,
                prompt_cases,
                args=args,
                selected_backends=backends,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                concurrency_levels=concurrency_levels,
            )
            for scenario_name in scenario_names
        )
        baseline_scenario = next(
            (scenario for scenario in scenario_runs if scenario.scenario == "baseline"),
            None,
        )

        runs.append(
            BenchmarkRun(
                model=model_name,
                prompt="VLM fixture suite: "
                + ", ".join(case.name for case in prompt_cases),
                max_tokens=args.max_tokens,
                results=baseline_scenario.results if baseline_scenario else (),
                generated_at=now_utc_iso(),
                benchmark_mode=args.benchmark_mode,
                warmup_runs_per_fixture=warmup_runs,
                measured_runs_per_fixture=measured_runs,
                scenario=args.scenario,
                metadata={"scenario_runs": scenario_runs},
            )
        )

    write_vlm_report(args.output_md, runs)
    _log_event(f"report_written={args.output_md}")
    print(f"report_written={args.output_md}")
    if args.output_json is not None:
        _write_json_report(args.output_json, runs)
        _log_event(f"json_report_written={args.output_json}")
        print(f"json_report_written={args.output_json}")
    return 0


def _resolve_run_counts(args: argparse.Namespace) -> tuple[int, int]:
    defaults = {
        "smoke": (0, 1),
        "normal": (1, 3),
        "stable": (1, 5),
    }
    warmup_runs, measured_runs = defaults[args.benchmark_mode]
    if args.warmup_runs_per_fixture is not None:
        warmup_runs = args.warmup_runs_per_fixture
    if args.measured_runs_per_fixture is not None:
        measured_runs = args.measured_runs_per_fixture
    if args.warmup_trials is not None:
        warmup_runs = args.warmup_trials
    if args.trials is not None:
        measured_runs = args.trials
    return (warmup_runs, measured_runs)


def _resolve_backends(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_backends or not args.backends:
        backends = ["raw", "server", "project"]
    else:
        normalized: list[str] = []
        for backend in args.backends:
            lowered = backend.lower().strip()
            if lowered in {"raw", "raw mlx-vlm"}:
                normalized.append("raw")
            elif lowered in {"server", "mlx_vlm.server"}:
                normalized.append("server")
            elif lowered in {"project", "this project"}:
                normalized.append("project")
        backends = list(dict.fromkeys(normalized))
    if args.skip_raw and "raw" in backends:
        backends.remove("raw")
    if args.skip_server and "server" in backends:
        backends.remove("server")
    if args.skip_project and "project" in backends:
        backends.remove("project")
    return tuple(backends)


def _resolve_scenario_names(selected: str) -> tuple[str, ...]:
    if selected == "all":
        return ("baseline", "streaming", "cancellation", "concurrency")
    return (selected,)


def _parse_concurrency_levels(text: str) -> tuple[int, ...]:
    levels = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not levels:
        return DEFAULT_CONCURRENCY_LEVELS
    return tuple(dict.fromkeys(level for level in levels if level > 0))


def _filter_prompt_cases(
    prompt_cases: list[VlmPromptCase],
    *,
    fixtures: list[str] | None,
    categories: list[str] | None,
) -> list[VlmPromptCase]:
    selected = prompt_cases
    if fixtures:
        fixture_set = {name.strip() for name in fixtures if name.strip()}
        selected = [case for case in selected if case.name in fixture_set]
    if categories:
        category_set = {name.strip() for name in categories if name.strip()}
        selected = [case for case in selected if case.category in category_set]
    return selected


def _run_vlm_scenario(
    scenario_name: str,
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    *,
    args: argparse.Namespace,
    selected_backends: tuple[str, ...],
    warmup_runs: int,
    measured_runs: int,
    concurrency_levels: tuple[int, ...],
) -> VlmScenarioRun:
    scenario_backends = _scenario_backends(scenario_name, selected_backends)
    backend_orders = _build_backend_orders(
        scenario_backends,
        explicit_order=args.backend_order,
        randomize=args.randomize_backend_order,
        seed=args.backend_order_seed,
        order_rounds=args.order_rounds,
        scenario_name=scenario_name,
    )
    started_at = now_utc_iso()
    order_round_details = tuple(
        {
            "order_round_index": index,
            "backend_order": tuple(_backend_display_name(name) for name in order),
        }
        for index, order in enumerate(backend_orders, start=1)
    )
    primary_backend_order = (
        tuple(_backend_display_name(name) for name in backend_orders[0])
        if backend_orders
        else ()
    )

    baseline_measurements: dict[str, list[VlmStreamResult]] = {}
    baseline_load_times: dict[str, list[float]] = {}
    streaming_measurements: dict[str, list[VlmStreamResult]] = {}
    streaming_unavailable: dict[str, tuple[str, ...]] = {}
    cancellation_results: list[VlmCancellationResult] = []
    concurrency_measurements: dict[tuple[str, int], list[VlmStreamResult]] = {}
    concurrency_wall_times: dict[tuple[str, int], list[float]] = {}

    for order in backend_orders:
        _log_event(
            f"[{model_name}] scenario={scenario_name} backend_order={','.join(order)}"
        )
        for backend in order:
            if scenario_name == "baseline":
                result, measurements = _run_baseline_backend(
                    backend,
                    model_name,
                    prompt_cases,
                    args.max_tokens,
                    warmup_runs,
                    measured_runs,
                    args,
                )
                baseline_measurements.setdefault(result.backend, []).extend(
                    measurements
                )
                if result.vlm_load_time_ms is not None:
                    baseline_load_times.setdefault(result.backend, []).append(
                        result.vlm_load_time_ms
                    )
                _log_event(_result_summary(model_name, result))
                continue

            if scenario_name == "streaming":
                measurements, unavailable_notes = _run_streaming_backend(
                    backend,
                    model_name,
                    prompt_cases,
                    args.max_tokens,
                    measured_runs,
                    args,
                )
                display_name = _backend_display_name(backend)
                if measurements is not None:
                    streaming_measurements.setdefault(display_name, []).extend(
                        measurements
                    )
                if unavailable_notes:
                    streaming_unavailable[display_name] = unavailable_notes
                continue

            if scenario_name == "cancellation":
                cancellation_results.append(
                    _benchmark_vlm_project_cancellation(
                        model_name,
                        prompt_cases,
                        args.max_tokens,
                        args.project_port,
                        args.launch_timeout,
                        args.readiness_timeout,
                        args.timeout_seconds,
                        args.cancellation_delay_ms,
                    )
                )
                continue

            if scenario_name == "concurrency":
                groups = _run_concurrency_backend(
                    backend,
                    model_name,
                    prompt_cases,
                    args.max_tokens,
                    measured_runs,
                    args,
                    concurrency_levels,
                )
                for concurrency, measurements, wall_clock_ms in groups:
                    key = (_backend_display_name(backend), concurrency)
                    concurrency_measurements.setdefault(key, []).extend(measurements)
                    concurrency_wall_times.setdefault(key, []).append(wall_clock_ms)

    ended_at = now_utc_iso()
    expected_samples = _expected_samples_for_scenario(
        scenario_name,
        prompt_cases,
        measured_runs,
        order_rounds=len(backend_orders),
    )

    if scenario_name == "baseline":
        results = tuple(
            _reduce_vlm_measurements(
                backend_name,
                measurements,
                load_time_ms=mean(baseline_load_times.get(backend_name, [])),
            )
            for backend_name, measurements in sorted(baseline_measurements.items())
        )
        fixture_rows = _build_fixture_rows(
            [sample for samples in baseline_measurements.values() for sample in samples]
        )
        comparison_rows = _build_baseline_comparison_rows(
            results,
            benchmark_mode=args.benchmark_mode,
            scenario_name=scenario_name,
            fixture_names=tuple(case.name for case in prompt_cases),
            max_tokens=args.max_tokens,
        )
        fairness_warnings = _collect_fairness_warnings(results, fixture_rows)
        interpretation = _baseline_interpretation(results, comparison_rows)
        return VlmScenarioRun(
            scenario=scenario_name,
            benchmark_mode=args.benchmark_mode,
            started_at=started_at,
            ended_at=ended_at,
            fixture_count=len(prompt_cases),
            fixture_names=tuple(case.name for case in prompt_cases),
            warmup_runs_per_fixture=warmup_runs,
            measured_runs_per_fixture=measured_runs,
            expected_measured_samples_per_backend=expected_samples,
            backend_order=primary_backend_order,
            order_rounds=len(backend_orders),
            order_randomized=args.randomize_backend_order,
            backend_order_seed=args.backend_order_seed,
            aggregated_across_order_rounds=len(backend_orders) > 1,
            order_round_details=order_round_details,
            results=results,
            fixture_rows=fixture_rows,
            comparison_rows=comparison_rows,
            fairness_warnings=fairness_warnings,
            interpretation=interpretation,
        )

    if scenario_name == "streaming":
        comparison_rows = _build_streaming_comparison_rows(
            streaming_measurements,
            benchmark_mode=args.benchmark_mode,
            scenario_name=scenario_name,
            fixture_names=tuple(
                case.name for case in prompt_cases[: min(3, len(prompt_cases))]
            ),
            max_tokens=args.max_tokens,
        )
        streaming_rows = _reduce_streaming_results(
            streaming_measurements, streaming_unavailable
        )
        return VlmScenarioRun(
            scenario=scenario_name,
            benchmark_mode=args.benchmark_mode,
            started_at=started_at,
            ended_at=ended_at,
            fixture_count=min(3, len(prompt_cases)),
            fixture_names=tuple(
                case.name for case in prompt_cases[: min(3, len(prompt_cases))]
            ),
            warmup_runs_per_fixture=0,
            measured_runs_per_fixture=measured_runs,
            expected_measured_samples_per_backend=expected_samples,
            backend_order=primary_backend_order,
            order_rounds=len(backend_orders),
            order_randomized=args.randomize_backend_order,
            backend_order_seed=args.backend_order_seed,
            aggregated_across_order_rounds=len(backend_orders) > 1,
            order_round_details=order_round_details,
            streaming_rows=streaming_rows,
            comparison_rows=comparison_rows,
            fairness_warnings=_collect_streaming_warnings(
                streaming_rows, comparison_rows
            ),
            interpretation=(
                "Streaming scenario is TTFT/chunk behavior only.",
                "Raw non-streaming TTFT is not reused here.",
            ),
        )

    if scenario_name == "cancellation":
        cancellation_row = _aggregate_cancellation_results(cancellation_results)
        return VlmScenarioRun(
            scenario=scenario_name,
            benchmark_mode=args.benchmark_mode,
            started_at=started_at,
            ended_at=ended_at,
            fixture_count=1,
            fixture_names=(prompt_cases[0].name,) if prompt_cases else (),
            warmup_runs_per_fixture=0,
            measured_runs_per_fixture=len(cancellation_results),
            expected_measured_samples_per_backend=len(cancellation_results),
            backend_order=primary_backend_order,
            order_rounds=len(backend_orders),
            order_randomized=args.randomize_backend_order,
            backend_order_seed=args.backend_order_seed,
            aggregated_across_order_rounds=len(backend_orders) > 1,
            order_round_details=order_round_details,
            cancellation_rows=(cancellation_row,) if cancellation_row else (),
            fairness_warnings=(),
            interpretation=(
                "Cancellation scenario reports cancel-and-recovery behavior only.",
            ),
        )

    concurrency_rows = tuple(
        _reduce_concurrency_scenario(
            backend_name,
            concurrency,
            measurements,
            wall_clock_duration_ms=sum(
                concurrency_wall_times[(backend_name, concurrency)]
            ),
        )
        for (backend_name, concurrency), measurements in sorted(
            concurrency_measurements.items(), key=lambda item: (item[0][0], item[0][1])
        )
    )
    return VlmScenarioRun(
        scenario=scenario_name,
        benchmark_mode=args.benchmark_mode,
        started_at=started_at,
        ended_at=ended_at,
        fixture_count=min(3, len(prompt_cases)),
        fixture_names=tuple(
            case.name for case in prompt_cases[: min(3, len(prompt_cases))]
        ),
        warmup_runs_per_fixture=0,
        measured_runs_per_fixture=measured_runs,
        expected_measured_samples_per_backend=expected_samples,
        backend_order=primary_backend_order,
        order_rounds=len(backend_orders),
        order_randomized=args.randomize_backend_order,
        backend_order_seed=args.backend_order_seed,
        aggregated_across_order_rounds=len(backend_orders) > 1,
        order_round_details=order_round_details,
        concurrency_rows=concurrency_rows,
        fairness_warnings=tuple(
            dict.fromkeys(w for row in concurrency_rows for w in row.warnings)
        ),
        interpretation=(
            "Concurrency scenario isolates local queueing/backpressure behavior.",
            "Do not use concurrency rows as headline single-request latency.",
        ),
    )


def _scenario_backends(
    scenario_name: str, selected_backends: tuple[str, ...]
) -> tuple[str, ...]:
    allowed = {
        "baseline": {"raw", "server", "project"},
        "streaming": {"raw", "server", "project"},
        "cancellation": {"project"},
        "concurrency": {"server", "project"},
    }[scenario_name]
    return tuple(backend for backend in selected_backends if backend in allowed)


def _build_backend_orders(
    available_backends: tuple[str, ...],
    *,
    explicit_order: str,
    randomize: bool,
    seed: int | None,
    order_rounds: int,
    scenario_name: str,
) -> tuple[tuple[str, ...], ...]:
    if order_rounds < 1:
        raise ValueError("order_rounds must be >= 1")
    explicit = [
        part.strip().lower() for part in explicit_order.split(",") if part.strip()
    ]
    order = tuple(backend for backend in explicit if backend in available_backends)
    if tuple(dict.fromkeys(order)) != order:
        raise ValueError("backend_order contains duplicates")
    missing = [backend for backend in available_backends if backend not in order]
    order = tuple(list(order) + missing)
    if randomize and len(order) > 1:
        rng = random.Random(f"{seed}:{scenario_name}")
        mutable = list(order)
        rng.shuffle(mutable)
        order = tuple(mutable)
    if not order:
        return ((),)
    rounds: list[tuple[str, ...]] = []
    for round_index in range(order_rounds):
        offset = round_index % len(order)
        rounds.append(order[offset:] + order[:offset])
    return tuple(rounds)


def _backend_display_name(backend: str) -> str:
    return {
        "raw": "raw mlx-vlm",
        "server": "mlx_vlm.server",
        "project": "this project",
    }.get(backend, backend)


def _expected_samples_for_scenario(
    scenario_name: str,
    prompt_cases: list[VlmPromptCase],
    measured_runs: int,
    *,
    order_rounds: int,
) -> int:
    case_count = len(prompt_cases)
    if scenario_name in {"streaming", "concurrency"}:
        case_count = min(3, len(prompt_cases))
    if scenario_name == "cancellation":
        case_count = 1
    return measured_runs * case_count * max(order_rounds, 1)


def _benchmark_raw_mlx_vlm(
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[BenchmarkResult, list[VlmStreamResult]]:
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm import load as vlm_load

    load_start = time.perf_counter()
    try:
        model, processor = vlm_load(model_name)
    except Exception as exc:
        load_time_ms = (time.perf_counter() - load_start) * 1000.0
        measurements = [
            _failed_vlm_stream_result("raw mlx-vlm", case, max_tokens, exc)
            for _ in range(max(measured_runs, 1))
            for case in prompt_cases
        ]
        return (
            _reduce_vlm_measurements(
                "raw mlx-vlm", measurements, load_time_ms=load_time_ms
            ),
            measurements,
        )
    _ = vlm_generate
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    _log_event(f"[raw mlx-vlm] model load: {load_time_ms:.1f} ms")

    warmup_tracker = _ProgressTracker(
        "raw mlx-vlm warmup", warmup_runs * len(prompt_cases)
    )
    for run_index in range(1, warmup_runs + 1):
        for case in prompt_cases:
            _raw_vlm_generate_once(model, processor, case, max_tokens)
            warmup_tracker.advance(f"warmup {run_index}/{warmup_runs}, {case.name}")
    warmup_tracker.close()

    measurements: list[VlmStreamResult] = []
    measured_tracker = _ProgressTracker(
        "raw mlx-vlm measured", measured_runs * len(prompt_cases)
    )
    for run_index in range(1, measured_runs + 1):
        for case in prompt_cases:
            try:
                measurements.append(
                    _raw_vlm_generate_once(model, processor, case, max_tokens)
                )
            except Exception as exc:
                measurements.append(
                    _failed_vlm_stream_result("raw mlx-vlm", case, max_tokens, exc)
                )
                _log_event(
                    f"[raw mlx-vlm] request failed for {case.name}: {type(exc).__name__}: {exc}"
                )
            measured_tracker.advance(f"run {run_index}/{measured_runs}, {case.name}")
    measured_tracker.close()
    return (
        _reduce_vlm_measurements(
            "raw mlx-vlm", measurements, load_time_ms=load_time_ms
        ),
        measurements,
    )


def _raw_vlm_generate_once(
    model: Any,
    processor: Any,
    case: VlmPromptCase,
    max_tokens: int,
) -> VlmStreamResult:
    from mlx_vlm import generate as vlm_generate

    prompt_str = _build_vlm_prompt(model, processor, case)
    image_load_ms, image_decode_ms, image_preprocess_ms = _measure_image_file_inputs(
        case
    )
    start = time.perf_counter()
    result = vlm_generate(
        model,
        processor,
        prompt_str,
        image=list(case.image_paths),
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        verbose=False,
    )
    end = time.perf_counter()
    text = result.text if hasattr(result, "text") else str(result)
    completion_tokens = int(getattr(result, "generation_tokens", 0) or 0)
    prompt_tokens = int(
        getattr(result, "prompt_tokens", 0) or case.prompt_tokens_estimate
    )
    latency_ms = (end - start) * 1000.0
    return _build_result(
        "raw mlx-vlm",
        case,
        ttft_ms=None,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        text=text,
        prompt_preview=_truncate(prompt_str),
        prompt_text_source="raw chat template",
        max_tokens=max_tokens,
        image_load_ms=image_load_ms,
        image_decode_ms=image_decode_ms,
        image_preprocess_ms=image_preprocess_ms,
        finish_reason=getattr(result, "finish_reason", None) or "stop",
        stream_completed_normally=True,
        notes=("raw non-streaming generation does not expose real TTFT",),
    )


def _benchmark_raw_mlx_vlm_streaming(
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
) -> tuple[list[VlmStreamResult] | None, tuple[str, ...]]:
    from mlx_vlm import load as vlm_load

    raw_stream_generate = _discover_raw_stream_generate()
    if raw_stream_generate is None:
        return (
            None,
            ("raw mlx-vlm streaming API was not available; raw TTFT is not reported",),
        )
    model, processor = vlm_load(model_name)
    scenario_cases = prompt_cases[: min(3, len(prompt_cases))]
    measurements: list[VlmStreamResult] = []
    tracker = _ProgressTracker(
        "raw mlx-vlm streaming", measured_runs * len(scenario_cases)
    )
    for run_index in range(1, measured_runs + 1):
        for case in scenario_cases:
            try:
                measurements.append(
                    _raw_vlm_stream_once(
                        raw_stream_generate,
                        model,
                        processor,
                        case,
                        max_tokens,
                    )
                )
            except Exception as exc:
                measurements.append(
                    _failed_vlm_stream_result("raw mlx-vlm", case, max_tokens, exc)
                )
            tracker.advance(f"run {run_index}/{measured_runs}, {case.name}")
    tracker.close()
    return (measurements, ())


def _discover_raw_stream_generate() -> Any | None:
    try:
        import mlx_vlm

        if hasattr(mlx_vlm, "stream_generate"):
            return mlx_vlm.stream_generate
    except Exception:
        return None
    return None


def _raw_vlm_stream_once(
    raw_stream_generate: Any,
    model: Any,
    processor: Any,
    case: VlmPromptCase,
    max_tokens: int,
) -> VlmStreamResult:
    prompt_str = _build_vlm_prompt(model, processor, case)
    image_load_ms, image_decode_ms, image_preprocess_ms = _measure_image_file_inputs(
        case
    )
    start = time.perf_counter()
    first_chunk_at: float | None = None
    previous_chunk_at: float | None = None
    chunk_intervals_ms: list[float] = []
    text_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    chunk_count = 0
    for event in raw_stream_generate(
        model,
        processor,
        prompt_str,
        image=list(case.image_paths),
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        verbose=False,
    ):
        segment = getattr(event, "text", "") or ""
        if segment:
            now = time.perf_counter()
            if first_chunk_at is None:
                first_chunk_at = now
            if previous_chunk_at is not None:
                chunk_intervals_ms.append((now - previous_chunk_at) * 1000.0)
            previous_chunk_at = now
            text_parts.append(segment)
            chunk_count += 1
        prompt_tokens = getattr(event, "prompt_tokens", prompt_tokens)
        completion_tokens = getattr(event, "generation_tokens", completion_tokens)
        finish_reason = getattr(event, "finish_reason", finish_reason)
    end = time.perf_counter()
    if first_chunk_at is None:
        first_chunk_at = end
    final_text = "".join(text_parts)
    if completion_tokens is None:
        completion_tokens = max(1, len(final_text) // 4) if final_text.strip() else 0
    if prompt_tokens is None:
        prompt_tokens = case.prompt_tokens_estimate
    return _build_result(
        "raw mlx-vlm",
        case,
        ttft_ms=(first_chunk_at - start) * 1000.0,
        latency_ms=(end - start) * 1000.0,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        text=final_text,
        prompt_preview=_truncate(prompt_str),
        prompt_text_source="raw chat template",
        max_tokens=max_tokens,
        image_load_ms=image_load_ms,
        image_decode_ms=image_decode_ms,
        image_preprocess_ms=image_preprocess_ms,
        finish_reason=finish_reason,
        sse_chunk_count=chunk_count,
        sse_chunk_interval_mean_ms=mean(chunk_intervals_ms),
        sse_chunk_interval_p50_ms=percentile(chunk_intervals_ms, 50),
        sse_chunk_interval_p95_ms=percentile(
            chunk_intervals_ms, 95, min_samples=P95_MIN_SAMPLES
        ),
        stream_completed_normally=finish_reason is not None or bool(text_parts),
    )


def _build_vlm_prompt(model: Any, processor: Any, case: VlmPromptCase) -> str:
    try:
        from mlx_vlm.prompt_utils import apply_chat_template, get_chat_template
        from mlx_vlm.utils import load_config

        model_path = getattr(model, "model_path", None)
        config = load_config(str(model_path)) if model_path else None
        if config is not None:
            rendered = apply_chat_template(
                processor,
                config,
                case.messages,
                add_generation_prompt=True,
                num_images=len(case.image_paths),
            )
            if isinstance(rendered, str):
                return rendered
            inner = [rendered] if isinstance(rendered, dict) else list(rendered)
            return get_chat_template(processor, inner, add_generation_prompt=True)
    except Exception:
        pass
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            case.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return _flatten_messages(case.messages) + "\nassistant:"


def _benchmark_vlm_server(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    warmup_runs: int,
    measured_runs: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
) -> tuple[BenchmarkResult, list[VlmStreamResult]]:
    command_variants = [
        [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            model,
        ],
    ]
    with _launch_vlm_http_service(
        "mlx_vlm.server",
        command_variants,
        f"http://127.0.0.1:{port}",
        model,
        probe_case=prompt_cases[0],
        launch_timeout=launch_timeout,
        readiness_timeout=readiness_timeout,
        request_timeout=request_timeout,
    ) as service:
        return _run_http_baseline(
            service,
            prompt_cases,
            max_tokens,
            warmup_runs,
            measured_runs,
            request_timeout,
        )


def _benchmark_vlm_project(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    warmup_runs: int,
    measured_runs: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
) -> tuple[BenchmarkResult, list[VlmStreamResult]]:
    with tempfile.TemporaryDirectory(prefix="mlx-vlm-benchmark-project-") as tmpdir_str:
        config_path = _prepare_project_config(
            model,
            port,
            config_dir=Path(tmpdir_str),
            vlm_model=model,
        )
        readiness_url = f"/models/{quote(model, safe='')}/ready"
        with _launch_vlm_http_service(
            "this project",
            [["cargo", "run", "--release", "-p", "mlx_runtime_gateway"]],
            f"http://127.0.0.1:{port}",
            model,
            probe_case=prompt_cases[0],
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=launch_timeout,
            readiness_timeout=readiness_timeout,
            request_timeout=request_timeout,
        ) as service:
            return _run_http_baseline(
                service,
                prompt_cases,
                max_tokens,
                warmup_runs,
                measured_runs,
                request_timeout,
            )


def _run_baseline_backend(
    backend: str,
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    args: argparse.Namespace,
) -> tuple[BenchmarkResult, list[VlmStreamResult]]:
    if backend == "raw":
        _log_event(f"[{model_name}] raw mlx-vlm baseline")
        return _benchmark_raw_mlx_vlm(
            model_name,
            prompt_cases,
            max_tokens,
            warmup_runs,
            measured_runs,
        )
    if backend == "server":
        _log_event(f"[{model_name}] mlx_vlm.server baseline")
        return _benchmark_vlm_server(
            model_name,
            prompt_cases,
            max_tokens,
            args.server_port,
            warmup_runs,
            measured_runs,
            args.launch_timeout,
            args.readiness_timeout,
            args.timeout_seconds,
        )
    _log_event(f"[{model_name}] this project baseline")
    return _benchmark_vlm_project(
        model_name,
        prompt_cases,
        max_tokens,
        args.project_port,
        warmup_runs,
        measured_runs,
        args.launch_timeout,
        args.readiness_timeout,
        args.timeout_seconds,
    )


def _run_streaming_backend(
    backend: str,
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
    args: argparse.Namespace,
) -> tuple[list[VlmStreamResult] | None, tuple[str, ...]]:
    if backend == "raw":
        return _benchmark_raw_mlx_vlm_streaming(
            model_name,
            prompt_cases,
            max_tokens,
            measured_runs,
        )
    if backend == "server":
        command_variants = [
            [
                sys.executable,
                "-m",
                "mlx_vlm.server",
                "--model",
                model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.server_port),
            ]
        ]
        with _launch_vlm_http_service(
            "mlx_vlm.server",
            command_variants,
            f"http://127.0.0.1:{args.server_port}",
            model_name,
            probe_case=prompt_cases[0],
            launch_timeout=args.launch_timeout,
            readiness_timeout=args.readiness_timeout,
            request_timeout=args.timeout_seconds,
        ) as service:
            return (
                _run_http_streaming_measurements(
                    service,
                    prompt_cases,
                    max_tokens,
                    measured_runs,
                    args.timeout_seconds,
                ),
                (),
            )
    with tempfile.TemporaryDirectory(
        prefix="mlx-vlm-benchmark-project-stream-"
    ) as tmpdir_str:
        config_path = _prepare_project_config(
            model_name,
            args.project_port,
            config_dir=Path(tmpdir_str),
            vlm_model=model_name,
        )
        readiness_url = f"/models/{quote(model_name, safe='')}/ready"
        with _launch_vlm_http_service(
            "this project",
            [["cargo", "run", "--release", "-p", "mlx_runtime_gateway"]],
            f"http://127.0.0.1:{args.project_port}",
            model_name,
            probe_case=prompt_cases[0],
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=args.launch_timeout,
            readiness_timeout=args.readiness_timeout,
            request_timeout=args.timeout_seconds,
        ) as service:
            return (
                _run_http_streaming_measurements(
                    service,
                    prompt_cases,
                    max_tokens,
                    measured_runs,
                    args.timeout_seconds,
                ),
                (),
            )


def _run_concurrency_backend(
    backend: str,
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
    args: argparse.Namespace,
    concurrency_levels: tuple[int, ...],
) -> list[tuple[int, list[VlmStreamResult], float]]:
    if backend == "server":
        command_variants = [
            [
                sys.executable,
                "-m",
                "mlx_vlm.server",
                "--model",
                model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.server_port),
            ]
        ]
        with _launch_vlm_http_service(
            "mlx_vlm.server",
            command_variants,
            f"http://127.0.0.1:{args.server_port}",
            model_name,
            probe_case=prompt_cases[0],
            launch_timeout=args.launch_timeout,
            readiness_timeout=args.readiness_timeout,
            request_timeout=args.timeout_seconds,
        ) as service:
            return _run_http_concurrency_measurements(
                service,
                prompt_cases,
                max_tokens,
                measured_runs,
                args.timeout_seconds,
                concurrency_levels,
            )
    with tempfile.TemporaryDirectory(
        prefix="mlx-vlm-benchmark-project-concurrency-"
    ) as tmpdir_str:
        config_path = _prepare_project_config(
            model_name,
            args.project_port,
            config_dir=Path(tmpdir_str),
            vlm_model=model_name,
        )
        readiness_url = f"/models/{quote(model_name, safe='')}/ready"
        with _launch_vlm_http_service(
            "this project",
            [["cargo", "run", "--release", "-p", "mlx_runtime_gateway"]],
            f"http://127.0.0.1:{args.project_port}",
            model_name,
            probe_case=prompt_cases[0],
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=args.launch_timeout,
            readiness_timeout=args.readiness_timeout,
            request_timeout=args.timeout_seconds,
        ) as service:
            return _run_http_concurrency_measurements(
                service,
                prompt_cases,
                max_tokens,
                measured_runs,
                args.timeout_seconds,
                concurrency_levels,
            )


def _run_http_baseline(
    service: RunningService,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    request_timeout: int,
) -> tuple[BenchmarkResult, list[VlmStreamResult]]:
    warmup_tracker = _ProgressTracker(
        f"{service.backend_name} warmup", warmup_runs * len(prompt_cases)
    )
    for run_index in range(1, warmup_runs + 1):
        for case in prompt_cases:
            _request_vlm_non_streaming_completion(
                service.backend_name,
                service.base_url,
                service.model,
                case,
                max_tokens,
                request_timeout=request_timeout,
            )
            warmup_tracker.advance(f"warmup {run_index}/{warmup_runs}, {case.name}")
    warmup_tracker.close()

    measurements: list[VlmStreamResult] = []
    measured_tracker = _ProgressTracker(
        f"{service.backend_name} measured", measured_runs * len(prompt_cases)
    )
    for run_index in range(1, measured_runs + 1):
        for case in prompt_cases:
            try:
                measurements.append(
                    _request_vlm_non_streaming_completion(
                        service.backend_name,
                        service.base_url,
                        service.model,
                        case,
                        max_tokens,
                        request_timeout=request_timeout,
                    )
                )
            except Exception as exc:
                measurements.append(
                    _failed_vlm_stream_result(
                        service.backend_name, case, max_tokens, exc
                    )
                )
                _log_event(
                    f"[{service.backend_name}] request failed for {case.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
            measured_tracker.advance(f"run {run_index}/{measured_runs}, {case.name}")
    measured_tracker.close()
    return (_reduce_vlm_measurements(service.backend_name, measurements), measurements)


def _benchmark_vlm_http_service(
    backend_name: str,
    command_variants: list[list[str]],
    base_url: str,
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    warmup_trials: int,
    trials: int,
    *,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    readiness_url: str | None = None,
    launch_timeout: int = 300,
    readiness_timeout: int = 300,
    request_timeout: int = 120,
) -> BenchmarkResult:
    """Backward-compatible wrapper retained for existing tests."""

    try:
        with _launch_vlm_http_service(
            backend_name,
            command_variants,
            base_url,
            model,
            probe_case=prompt_cases[0],
            cwd=cwd,
            extra_env=extra_env,
            readiness_url=readiness_url,
            launch_timeout=launch_timeout,
            readiness_timeout=readiness_timeout,
            request_timeout=request_timeout,
        ) as service:
            result, _measurements = _run_http_baseline(
                service,
                prompt_cases,
                max_tokens,
                warmup_trials,
                trials,
                request_timeout,
            )
            return result
    except Exception as exc:
        measurements = [
            _failed_vlm_stream_result(backend_name, case, max_tokens, exc)
            for _ in range(max(trials, 1))
            for case in prompt_cases
        ]
        return _reduce_vlm_measurements(backend_name, measurements)


def _benchmark_vlm_server_streaming(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    measured_runs: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
) -> list[VlmStreamingBackendResult]:
    command_variants = [
        [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
    ]
    with _launch_vlm_http_service(
        "mlx_vlm.server",
        command_variants,
        f"http://127.0.0.1:{port}",
        model,
        probe_case=prompt_cases[0],
        launch_timeout=launch_timeout,
        readiness_timeout=readiness_timeout,
        request_timeout=request_timeout,
    ) as service:
        return _run_http_streaming_scenario(
            service, prompt_cases, max_tokens, measured_runs, request_timeout
        )


def _benchmark_vlm_project_streaming(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    measured_runs: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
) -> list[VlmStreamingBackendResult]:
    with tempfile.TemporaryDirectory(
        prefix="mlx-vlm-benchmark-project-stream-"
    ) as tmpdir_str:
        config_path = _prepare_project_config(
            model,
            port,
            config_dir=Path(tmpdir_str),
            vlm_model=model,
        )
        readiness_url = f"/models/{quote(model, safe='')}/ready"
        with _launch_vlm_http_service(
            "this project",
            [["cargo", "run", "--release", "-p", "mlx_runtime_gateway"]],
            f"http://127.0.0.1:{port}",
            model,
            probe_case=prompt_cases[0],
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=launch_timeout,
            readiness_timeout=readiness_timeout,
            request_timeout=request_timeout,
        ) as service:
            return _run_http_streaming_scenario(
                service, prompt_cases, max_tokens, measured_runs, request_timeout
            )


def _run_http_streaming_scenario(
    service: RunningService,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
    request_timeout: int,
) -> list[VlmStreamingBackendResult]:
    return [
        _reduce_streaming_scenario(
            service.backend_name,
            _run_http_streaming_measurements(
                service,
                prompt_cases,
                max_tokens,
                measured_runs,
                request_timeout,
            ),
        )
    ]


def _run_http_streaming_measurements(
    service: RunningService,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
    request_timeout: int,
) -> list[VlmStreamResult]:
    scenario_cases = prompt_cases[: min(3, len(prompt_cases))]
    streaming_measurements: list[VlmStreamResult] = []
    tracker = _ProgressTracker(
        f"{service.backend_name} streaming scenario",
        measured_runs * len(scenario_cases),
    )
    for run_index in range(1, measured_runs + 1):
        for case in scenario_cases:
            streaming_measurements.append(
                _request_vlm_streaming_completion(
                    service.backend_name,
                    service.base_url,
                    service.model,
                    case,
                    max_tokens,
                    request_timeout=request_timeout,
                )
            )
            tracker.advance(f"stream {run_index}/{measured_runs}, {case.name}")
    tracker.close()
    return streaming_measurements


def _benchmark_vlm_project_cancellation(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
    cancellation_delay_ms: int,
) -> VlmCancellationResult:
    case = next(
        (
            case
            for case in prompt_cases
            if case.category in {"multi_image_summary", "long_multi_image_analysis"}
        ),
        prompt_cases[0],
    )
    with tempfile.TemporaryDirectory(
        prefix="mlx-vlm-benchmark-project-cancel-"
    ) as tmpdir_str:
        config_path = _prepare_project_config(
            model,
            port,
            config_dir=Path(tmpdir_str),
            vlm_model=model,
        )
        readiness_url = f"/models/{quote(model, safe='')}/ready"
        with _launch_vlm_http_service(
            "this project",
            [["cargo", "run", "--release", "-p", "mlx_runtime_gateway"]],
            f"http://127.0.0.1:{port}",
            model,
            probe_case=prompt_cases[0],
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=launch_timeout,
            readiness_timeout=readiness_timeout,
            request_timeout=request_timeout,
        ) as service:
            cancel_errors = 0
            follow_up_success = 0
            follow_up_error = 0
            follow_up_latencies: list[float] = []
            try:
                _cancel_streaming_request(
                    service.base_url,
                    service.model,
                    case,
                    max_tokens,
                    request_timeout,
                    cancellation_delay_ms,
                )
            except Exception:
                cancel_errors += 1
            try:
                follow_up = _request_vlm_streaming_completion(
                    service.backend_name,
                    service.base_url,
                    service.model,
                    case,
                    max_tokens,
                    request_timeout=request_timeout,
                )
                if follow_up.latency_ms is not None:
                    follow_up_latencies.append(follow_up.latency_ms)
                follow_up_success += 1
            except Exception:
                follow_up_error += 1
            worker_health = _check_project_health(service.base_url, readiness_url)
            warnings: list[str] = []
            if cancel_errors:
                warnings.append("cancellation stream did not close cleanly")
            if follow_up_error:
                warnings.append("follow-up VLM request failed after cancellation")
            return VlmCancellationResult(
                backend="this project",
                cancellation_attempts=1,
                server_observed_cancellations=None,
                cancellation_errors=cancel_errors,
                follow_up_success_count=follow_up_success,
                follow_up_error_count=follow_up_error,
                follow_up_latency_mean_ms=mean(follow_up_latencies),
                worker_health_after_cancel=worker_health,
                warnings=tuple(warnings),
            )


def _benchmark_vlm_server_concurrency(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    measured_runs: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
    concurrency_levels: tuple[int, ...],
) -> list[VlmConcurrencyResult]:
    command_variants = [
        [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
    ]
    with _launch_vlm_http_service(
        "mlx_vlm.server",
        command_variants,
        f"http://127.0.0.1:{port}",
        model,
        probe_case=prompt_cases[0],
        launch_timeout=launch_timeout,
        readiness_timeout=readiness_timeout,
        request_timeout=request_timeout,
    ) as service:
        return _run_http_concurrency_scenario(
            service,
            prompt_cases,
            max_tokens,
            measured_runs,
            request_timeout,
            concurrency_levels,
        )


def _benchmark_vlm_project_concurrency(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    measured_runs: int,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
    concurrency_levels: tuple[int, ...],
) -> list[VlmConcurrencyResult]:
    with tempfile.TemporaryDirectory(
        prefix="mlx-vlm-benchmark-project-concurrency-"
    ) as tmpdir_str:
        config_path = _prepare_project_config(
            model,
            port,
            config_dir=Path(tmpdir_str),
            vlm_model=model,
        )
        readiness_url = f"/models/{quote(model, safe='')}/ready"
        with _launch_vlm_http_service(
            "this project",
            [["cargo", "run", "--release", "-p", "mlx_runtime_gateway"]],
            f"http://127.0.0.1:{port}",
            model,
            probe_case=prompt_cases[0],
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=launch_timeout,
            readiness_timeout=readiness_timeout,
            request_timeout=request_timeout,
        ) as service:
            return _run_http_concurrency_scenario(
                service,
                prompt_cases,
                max_tokens,
                measured_runs,
                request_timeout,
                concurrency_levels,
            )


def _run_http_concurrency_scenario(
    service: RunningService,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
    request_timeout: int,
    concurrency_levels: tuple[int, ...],
) -> list[VlmConcurrencyResult]:
    return [
        _reduce_concurrency_scenario(
            service.backend_name,
            concurrency,
            measurements,
            wall_clock_duration_ms=wall_clock_duration_ms,
        )
        for concurrency, measurements, wall_clock_duration_ms in _run_http_concurrency_measurements(
            service,
            prompt_cases,
            max_tokens,
            measured_runs,
            request_timeout,
            concurrency_levels,
        )
    ]


def _run_http_concurrency_measurements(
    service: RunningService,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    measured_runs: int,
    request_timeout: int,
    concurrency_levels: tuple[int, ...],
) -> list[tuple[int, list[VlmStreamResult], float]]:
    results: list[tuple[int, list[VlmStreamResult], float]] = []
    scenario_cases = prompt_cases[: min(3, len(prompt_cases))]
    for concurrency in concurrency_levels:
        requests = [
            scenario_cases[index % len(scenario_cases)]
            for index in range(concurrency * measured_runs)
        ]
        start = time.perf_counter()
        measurements: list[VlmStreamResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _request_vlm_streaming_completion,
                    service.backend_name,
                    service.base_url,
                    service.model,
                    case,
                    max_tokens,
                    request_timeout=request_timeout,
                )
                for case in requests
            ]
            for future in as_completed(futures):
                try:
                    measurements.append(future.result())
                except Exception as exc:
                    measurements.append(
                        _failed_vlm_stream_result(
                            service.backend_name,
                            requests[0],
                            max_tokens,
                            exc,
                        )
                    )
        end = time.perf_counter()
        results.append((concurrency, measurements, (end - start) * 1000.0))
    return results


@contextmanager
def _launch_vlm_http_service(
    backend_name: str,
    command_variants: list[list[str]],
    base_url: str,
    model: str,
    *,
    probe_case: VlmPromptCase,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    readiness_url: str | None = None,
    launch_timeout: int,
    readiness_timeout: int,
    request_timeout: int,
) -> Iterator[RunningService]:
    with tempfile.TemporaryDirectory(prefix="mlx-vlm-benchmark-") as tmpdir:
        stdout_path = Path(tmpdir) / f"{backend_name.replace(' ', '_')}.stdout.log"
        stderr_path = Path(tmpdir) / f"{backend_name.replace(' ', '_')}.stderr.log"
        port = _extract_port(base_url)
        last_error: str | None = None
        for variant_index, command in enumerate(command_variants, start=1):
            _log_event(
                f"[{backend_name}] launch attempt {variant_index}/{len(command_variants)}: {' '.join(command)}"
            )
            if not _is_port_free("127.0.0.1", port):
                last_error = f"port {port} already in use before launch"
                continue
            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                env=_merged_env(extra_env),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            try:
                if not _wait_for_process_port(
                    process,
                    "127.0.0.1",
                    port,
                    timeout_s=launch_timeout,
                    label=backend_name,
                ):
                    last_error = (
                        _read_process_failure(stderr_path, stdout_path)
                        or "timed out waiting for port"
                    )
                    continue
                _wait_for_vlm_service_ready(
                    base_url,
                    readiness_url,
                    model,
                    probe_case,
                    timeout_s=readiness_timeout,
                    request_timeout=request_timeout,
                    label=backend_name,
                )
                yield RunningService(
                    backend_name=backend_name,
                    base_url=base_url,
                    model=model,
                    readiness_url=readiness_url,
                )
                return
            finally:
                _terminate_process_group(process)
                stdout_file.close()
                stderr_file.close()
        raise RuntimeError(
            f"{backend_name} benchmark failed after trying all launch variants: {last_error}"
        )


def _request_vlm_streaming_completion(
    backend_name: str,
    base_url: str,
    model: str,
    case: VlmPromptCase,
    max_tokens: int,
    *,
    request_timeout: int,
) -> VlmStreamResult:
    payload = {
        "model": model,
        "messages": case.messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_delta_at: float | None = None
    final_text_parts: list[str] = []
    completion_tokens = 0
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    parse_errors = 0
    chunk_times_ms: list[float] = []
    previous_chunk_at: float | None = None
    with urlopen(request, timeout=request_timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                payload_data = json.loads(data)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            usage = payload_data.get("usage")
            if usage and isinstance(usage, dict):
                usage_completion = usage.get("completion_tokens")
                if usage_completion is not None:
                    completion_tokens = int(usage_completion)
                usage_prompt = usage.get("prompt_tokens")
                if usage_prompt is not None:
                    prompt_tokens = int(usage_prompt)
            choices = payload_data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or choice.get("message", {}).get("content")
                or ""
            )
            if content:
                now = time.perf_counter()
                if first_delta_at is None:
                    first_delta_at = now
                if previous_chunk_at is not None:
                    chunk_times_ms.append((now - previous_chunk_at) * 1000.0)
                previous_chunk_at = now
                final_text_parts.append(content)
            finish_reason = choice.get("finish_reason") or finish_reason
    end = time.perf_counter()
    final_text = "".join(final_text_parts)
    if first_delta_at is None:
        first_delta_at = end
    notes: list[str] = []
    if prompt_tokens is None:
        notes.append(
            "streaming usage.prompt_tokens unavailable; token-normalized comparisons disabled"
        )
    validated_completion_tokens = _validated_completion_tokens(
        completion_tokens if completion_tokens > 0 else None,
        max_tokens=max_tokens,
        notes=notes,
        backend_name=backend_name,
    )
    if validated_completion_tokens is None:
        notes.append(
            "streaming usage.completion_tokens unavailable; token-normalized comparisons disabled"
        )
    return _build_result(
        backend_name,
        case,
        ttft_ms=(first_delta_at - start) * 1000.0,
        latency_ms=(end - start) * 1000.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=validated_completion_tokens,
        text=final_text,
        prompt_preview=_truncate(_flatten_messages(case.messages)),
        prompt_text_source="http request messages",
        max_tokens=max_tokens,
        finish_reason=finish_reason,
        parse_errors=parse_errors,
        sse_chunk_count=len(final_text_parts) if final_text_parts else 0,
        sse_chunk_interval_mean_ms=mean(chunk_times_ms),
        sse_chunk_interval_p50_ms=percentile(chunk_times_ms, 50),
        sse_chunk_interval_p95_ms=percentile(
            chunk_times_ms, 95, min_samples=P95_MIN_SAMPLES
        ),
        stream_completed_normally=finish_reason is not None or bool(final_text_parts),
        notes=tuple(notes),
    )


def _request_vlm_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    prompt_tokens_estimate: int | None = None,
    image_count: int = 1,
    request_timeout: int = 120,
) -> VlmStreamResult:
    """Backward-compatible wrapper for older tests and callers."""

    case = _compat_case_from_messages(
        messages,
        prompt_tokens_estimate=prompt_tokens_estimate,
        image_count=image_count,
    )
    return _request_vlm_streaming_completion(
        "http",
        base_url,
        model,
        case,
        max_tokens,
        request_timeout=request_timeout,
    )


def _request_vlm_non_streaming_completion(
    backend_name: str,
    base_url: str,
    model: str,
    case: VlmPromptCase,
    max_tokens: int,
    *,
    request_timeout: int,
) -> VlmStreamResult:
    payload = {
        "model": model,
        "messages": case.messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urlopen(request, timeout=request_timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    end = time.perf_counter()
    usage = body.get("usage") or {}
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content") or ""
    notes: list[str] = []
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is not None:
        prompt_tokens = int(prompt_tokens)
    else:
        notes.append(
            "non-streaming usage.prompt_tokens unavailable; token-normalized comparisons disabled"
        )
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None:
        completion_tokens = int(completion_tokens)
    completion_tokens = _validated_completion_tokens(
        completion_tokens,
        max_tokens=max_tokens,
        notes=notes,
        backend_name=backend_name,
    )
    if completion_tokens is None:
        notes.append(
            "non-streaming usage.completion_tokens unavailable; token-normalized comparisons disabled"
        )
    return _build_result(
        backend_name,
        case,
        ttft_ms=None,
        latency_ms=(end - start) * 1000.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        text=text,
        prompt_preview=_truncate(_flatten_messages(case.messages)),
        prompt_text_source="http request messages",
        max_tokens=max_tokens,
        finish_reason=choice.get("finish_reason"),
        stream_completed_normally=True,
        notes=tuple(notes),
    )


def _validated_completion_tokens(
    completion_tokens: int | None,
    *,
    max_tokens: int,
    notes: list[str],
    backend_name: str,
) -> int | None:
    if completion_tokens is None:
        return None
    if completion_tokens < 0:
        notes.append(
            f"{backend_name} reported invalid negative completion token count; ignored"
        )
        return None
    if completion_tokens > max_tokens:
        notes.append(
            f"{backend_name} reported completion_tokens greater than max_tokens; ignored"
        )
        return None
    return completion_tokens


def _cancel_streaming_request(
    base_url: str,
    model: str,
    case: VlmPromptCase,
    max_tokens: int,
    request_timeout: int,
    cancellation_delay_ms: int,
) -> None:
    payload = {
        "model": model,
        "messages": case.messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
    }
    port = _extract_port(base_url)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=request_timeout)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    deadline = time.monotonic() + (cancellation_delay_ms / 1000.0)
    try:
        while time.monotonic() < deadline:
            line = response.fp.readline()  # type: ignore[union-attr]
            if not line:
                break
            if b"data:" not in line:
                continue
            if b"[DONE]" in line:
                break
            if b'"content"' in line or b'"reasoning"' in line:
                break
    finally:
        connection.close()


def _check_project_health(base_url: str, readiness_url: str | None) -> str | None:
    if not readiness_url:
        return None
    try:
        with urlopen(Request(f"{base_url}{readiness_url}"), timeout=10) as response:
            return "ready" if response.status < 500 else f"http_{response.status}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def _wait_for_vlm_service_ready(
    base_url: str,
    readiness_url: str | None,
    model: str,
    probe_case: VlmPromptCase | list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int = 8,
    *,
    timeout_s: int,
    request_timeout: int = 120,
    label: str,
) -> None:
    if probe_case is None:
        probe_case = messages or []
    if not isinstance(probe_case, VlmPromptCase):
        probe_case = _compat_case_from_messages(probe_case, image_count=1)
    deadline = time.monotonic() + timeout_s
    next_heartbeat = time.monotonic()
    last_error: str | None = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            _log_event(f"[{label}] waiting for service readiness")
            next_heartbeat = now + 5
        try:
            if readiness_url:
                with urlopen(
                    Request(f"{base_url}{readiness_url}"), timeout=10
                ) as response:
                    if response.status < 500:
                        _log_event(f"[{label}] readiness endpoint accepted requests")
                        return
                    last_error = f"readiness endpoint returned HTTP {response.status}"
            else:
                _request_vlm_completion(
                    base_url,
                    model,
                    probe_case.messages,
                    max_tokens,
                    prompt_tokens_estimate=probe_case.prompt_tokens_estimate,
                    image_count=probe_case.image_count,
                    request_timeout=request_timeout,
                )
                _log_event(f"[{label}] streaming endpoint accepted requests")
                return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"service did not become ready: {last_error}")


def _wait_for_process_port(
    process: subprocess.Popen[Any],
    host: str,
    port: int,
    *,
    timeout_s: int,
    label: str,
) -> bool:
    deadline = time.monotonic() + timeout_s
    next_heartbeat = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            _log_event(f"[{label}] waiting for port {host}:{port} to open")
            next_heartbeat = now + 5
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                _log_event(f"[{label}] port {host}:{port} is open")
                return True
        except OSError:
            time.sleep(1)
    return False


def _is_port_free(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return False
    except (OSError, ConnectionRefusedError):
        return True


def _extract_port(base_url: str) -> int:
    return int(base_url.rsplit(":", 1)[1])


def _merged_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return env


def _read_process_failure(stderr_path: Path, stdout_path: Path) -> str | None:
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    if stderr:
        return stderr.splitlines()[-1]
    if stdout:
        return stdout.splitlines()[-1]
    return None


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        pgid = os.getpgid(process.pid)
    except (OSError, ProcessLookupError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        process.wait(timeout=30)


def _prepare_project_config(
    model: str,
    port: int,
    config_dir: Path | None = None,
    *,
    vlm_model: str | None = None,
) -> Path:
    source = REPO_ROOT / "config" / "runtime.toml"
    if config_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="mlx-runtime-config-"))
        ipc_root = temp_dir
    else:
        ipc_root = config_dir
        temp_dir = config_dir / "config"
        temp_dir.mkdir(exist_ok=True)
    target = temp_dir / "runtime.toml"
    text = source.read_text(encoding="utf-8")
    if not vlm_model:
        text = _replace_config_value(text, "model", model)
    text = _replace_config_value(text, "port", str(port))
    text = _replace_config_value(text, "ipc_path", str(ipc_root / "m.sock"))
    if vlm_model:
        text = _set_vlm_config(text, vlm_model)
    target.write_text(text, encoding="utf-8")
    return target


def _set_vlm_config(text: str, vlm_model: str) -> str:
    lines: list[str] = []
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lstrip("#").strip().startswith("vlm_model ="):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f'{indent}vlm_model = "{vlm_model}"')
            found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f'\nvlm_model = "{vlm_model}"')
    return "\n".join(lines) + "\n"


def _replace_config_value(text: str, key: str, value: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key} = "):
            prefix = line.split("=", 1)[0].rstrip()
            if value.isdigit():
                lines.append(f"{prefix} = {value}")
            else:
                lines.append(f'{prefix} = "{value}"')
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _failed_vlm_stream_result(*args: Any) -> VlmStreamResult:
    if len(args) == 2:
        backend = "http"
        case, exc = args
        max_tokens = 0
    elif len(args) == 4:
        backend, case, max_tokens, exc = args
    else:
        raise TypeError("_failed_vlm_stream_result expects 2 or 4 arguments")
    return _build_result(
        backend,
        case,
        ttft_ms=None,
        latency_ms=None,
        prompt_tokens=case.prompt_tokens_estimate,
        completion_tokens=None,
        text="",
        prompt_preview=_truncate(_flatten_messages(case.messages)),
        prompt_text_source="http request messages"
        if backend != "raw mlx-vlm"
        else "raw chat template",
        max_tokens=max_tokens,
        error=f"{type(exc).__name__}: {exc}",
        stream_completed_normally=False,
    )


def _build_result(
    backend: str,
    case: VlmPromptCase,
    *,
    ttft_ms: float | None,
    latency_ms: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    text: str,
    prompt_preview: str,
    prompt_text_source: str,
    max_tokens: int,
    image_load_ms: float | None = None,
    image_decode_ms: float | None = None,
    image_preprocess_ms: float | None = None,
    finish_reason: str | None = None,
    error: str | None = None,
    notes: tuple[str, ...] = (),
    sse_chunk_count: int | None = None,
    sse_chunk_interval_mean_ms: float | None = None,
    sse_chunk_interval_p50_ms: float | None = None,
    sse_chunk_interval_p95_ms: float | None = None,
    parse_errors: int = 0,
    stream_completed_normally: bool | None = None,
) -> VlmStreamResult:
    return VlmStreamResult(
        backend=backend,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        text=text,
        fixture_name=case.name,
        fixture_category=case.category,
        prompt_preview=prompt_preview,
        prompt_text_source=prompt_text_source,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        image_count=case.image_count,
        total_image_pixels=case.total_image_pixels,
        total_megapixels=case.total_megapixels,
        widths_summary=case.widths_summary,
        heights_summary=case.heights_summary,
        formats_summary=case.formats_summary,
        total_file_size_bytes=case.total_file_size_bytes,
        image_load_ms=image_load_ms,
        image_decode_ms=image_decode_ms,
        image_preprocess_ms=image_preprocess_ms,
        finish_reason=finish_reason,
        error=error,
        notes=tuple(note for note in notes if note),
        sse_chunk_count=sse_chunk_count,
        sse_chunk_interval_mean_ms=sse_chunk_interval_mean_ms,
        sse_chunk_interval_p50_ms=sse_chunk_interval_p50_ms,
        sse_chunk_interval_p95_ms=sse_chunk_interval_p95_ms,
        parse_errors=parse_errors,
        stream_completed_normally=stream_completed_normally,
    )


def _reduce_vlm_measurements(
    backend: str,
    measurements: Iterable[VlmStreamResult],
    *,
    load_time_ms: float | None = None,
) -> BenchmarkResult:
    measurements = list(measurements)
    if not measurements:
        raise ValueError(f"{backend} produced no VLM benchmark measurements")
    successful = [sample for sample in measurements if sample.succeeded]
    errors = len(measurements) - len(successful)

    ttft_values = [
        sample.ttft_ms for sample in successful if sample.ttft_ms is not None
    ]
    latency_values = [
        sample.latency_ms for sample in successful if sample.latency_ms is not None
    ]
    prompt_token_values = [
        float(sample.prompt_tokens)
        for sample in successful
        if sample.prompt_tokens is not None
    ]
    completion_token_values = [
        float(sample.completion_tokens)
        for sample in successful
        if sample.completion_tokens is not None
    ]
    image_count_values = [float(sample.image_count) for sample in successful]
    image_load_values = [
        sample.image_load_ms
        for sample in successful
        if sample.image_load_ms is not None
    ]
    image_decode_values = [
        sample.image_decode_ms
        for sample in successful
        if sample.image_decode_ms is not None
    ]
    image_preprocess_values = [
        sample.image_preprocess_ms
        for sample in successful
        if sample.image_preprocess_ms is not None
    ]

    decode_time_values: list[float] = []
    decode_tps_values: list[float] = []
    e2e_tps_values: list[float] = []
    warnings: list[str] = []
    notes = tuple(
        dict.fromkeys(note for sample in measurements for note in sample.notes)
    )
    for sample in successful:
        if sample.latency_ms is None:
            continue
        if sample.ttft_ms is not None and sample.ttft_ms > sample.latency_ms:
            warnings.append("sample had TTFT greater than total latency")
        completion_tokens = float(sample.completion_tokens or 0)
        e2e_tps = calculate_end_to_end_tokens_per_second(
            completion_tokens, sample.latency_ms
        )
        if e2e_tps is not None:
            e2e_tps_values.append(e2e_tps)
        if sample.ttft_ms is None:
            continue
        decode_time_ms = sample.latency_ms - sample.ttft_ms
        if decode_time_ms > 0:
            decode_time_values.append(decode_time_ms)
        decode_tps = calculate_decode_tokens_per_second(
            completion_tokens, decode_time_ms
        )
        if decode_tps is not None:
            decode_tps_values.append(decode_tps)

    if errors:
        warnings.append(f"{errors} measured request(s) failed")
    if len(latency_values) < P95_MIN_SAMPLES:
        warnings.append(
            f"latency_p95_ms unavailable with only {len(latency_values)} successful sample(s)"
        )
    if len(latency_values) < P99_MIN_SAMPLES:
        warnings.append(
            f"latency_p99_ms unavailable with only {len(latency_values)} successful sample(s)"
        )
    if len(ttft_values) < P95_MIN_SAMPLES:
        warnings.append(
            f"ttft_p95_ms unavailable with only {len(ttft_values)} successful sample(s)"
        )
    if len(ttft_values) < P99_MIN_SAMPLES:
        warnings.append(
            f"ttft_p99_ms unavailable with only {len(ttft_values)} successful sample(s)"
        )
    if not image_preprocess_values:
        warnings.append("image_preprocess_ms unavailable for this backend")

    completion_mean = mean(completion_token_values)
    latency_mean_ms = mean(latency_values)
    return BenchmarkResult(
        backend=backend,
        samples=len(successful),
        errors=errors,
        error_rate=(errors / len(measurements)) if measurements else 0.0,
        ttft_mean_ms=mean(ttft_values),
        ttft_p50_ms=percentile(ttft_values, 50),
        ttft_p95_ms=percentile(ttft_values, 95, min_samples=P95_MIN_SAMPLES),
        ttft_p99_ms=percentile(ttft_values, 99, min_samples=P99_MIN_SAMPLES),
        latency_mean_ms=latency_mean_ms,
        latency_p50_ms=percentile(latency_values, 50),
        latency_p95_ms=percentile(latency_values, 95, min_samples=P95_MIN_SAMPLES),
        latency_p99_ms=percentile(latency_values, 99, min_samples=P99_MIN_SAMPLES),
        prompt_tokens_mean=mean(prompt_token_values),
        completion_tokens_mean=completion_mean,
        completion_tokens_p50=percentile(completion_token_values, 50),
        total_tokens_mean=mean(
            [p + c for p, c in zip(prompt_token_values, completion_token_values)]
        ),
        decode_time_mean_ms=mean(decode_time_values),
        latency_per_completion_token_ms=calculate_per_token_latency_ms(
            latency_mean_ms, completion_mean
        ),
        decode_time_per_completion_token_ms=calculate_per_token_latency_ms(
            mean(decode_time_values), completion_mean
        ),
        latency_p50_per_completion_token_ms=calculate_per_token_latency_ms(
            percentile(latency_values, 50),
            percentile(completion_token_values, 50),
        ),
        decode_tokens_per_second_mean=mean(decode_tps_values),
        decode_tokens_per_second_p50=percentile(decode_tps_values, 50),
        end_to_end_tokens_per_second_mean=mean(e2e_tps_values),
        end_to_end_tokens_per_second_p50=percentile(e2e_tps_values, 50),
        image_preprocess_latency_ms_mean=mean(image_preprocess_values),
        image_preprocess_latency_ms_p50=percentile(image_preprocess_values, 50),
        image_preprocess_latency_ms_p95=percentile(
            image_preprocess_values,
            95,
            min_samples=P95_MIN_SAMPLES,
        ),
        image_count_mean=mean(image_count_values),
        vlm_load_time_ms=load_time_ms,
        image_load_ms_mean=mean(image_load_values),
        image_load_ms_p50=percentile(image_load_values, 50),
        image_load_ms_p95=percentile(
            image_load_values, 95, min_samples=P95_MIN_SAMPLES
        ),
        image_decode_ms_mean=mean(image_decode_values),
        image_decode_ms_p50=percentile(image_decode_values, 50),
        image_decode_ms_p95=percentile(
            image_decode_values,
            95,
            min_samples=P95_MIN_SAMPLES,
        ),
        notes=notes,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _build_fixture_rows(
    measurements: list[VlmStreamResult],
) -> tuple[VlmFixtureReportRow, ...]:
    by_backend_fixture: dict[tuple[str, str], list[VlmStreamResult]] = {}
    for sample in measurements:
        by_backend_fixture.setdefault((sample.backend, sample.fixture_name), []).append(
            sample
        )

    rows: list[VlmFixtureReportRow] = []
    for (backend_name, fixture_name), samples in sorted(by_backend_fixture.items()):
        successful = [sample for sample in samples if sample.succeeded]
        errors = len(samples) - len(successful)
        first = samples[0]
        ttft_values = [
            sample.ttft_ms for sample in successful if sample.ttft_ms is not None
        ]
        latency_values = [
            sample.latency_ms for sample in successful if sample.latency_ms is not None
        ]
        prompt_values = [
            float(sample.prompt_tokens)
            for sample in successful
            if sample.prompt_tokens is not None
        ]
        completion_values = [
            float(sample.completion_tokens)
            for sample in successful
            if sample.completion_tokens is not None
        ]
        image_preprocess_values = [
            sample.image_preprocess_ms
            for sample in successful
            if sample.image_preprocess_ms is not None
        ]
        image_load_values = [
            sample.image_load_ms
            for sample in successful
            if sample.image_load_ms is not None
        ]
        image_decode_values = [
            sample.image_decode_ms
            for sample in successful
            if sample.image_decode_ms is not None
        ]
        decode_tps_values: list[float] = []
        e2e_tps_values: list[float] = []
        stop_reasons = [
            sample.finish_reason for sample in successful if sample.finish_reason
        ]
        for sample in successful:
            if sample.ttft_ms is None or sample.latency_ms is None:
                continue
            decode_time_ms = sample.latency_ms - sample.ttft_ms
            completion_tokens = float(sample.completion_tokens or 0)
            decode_tps = calculate_decode_tokens_per_second(
                completion_tokens, decode_time_ms
            )
            if decode_tps is not None:
                decode_tps_values.append(decode_tps)
            e2e_tps = calculate_end_to_end_tokens_per_second(
                completion_tokens, sample.latency_ms
            )
            if e2e_tps is not None:
                e2e_tps_values.append(e2e_tps)
        warnings: list[str] = []
        if len(successful) < P95_MIN_SAMPLES:
            warnings.append("fixture sample count too low for p95")
        if not image_preprocess_values:
            warnings.append("image_preprocess_ms not_available")
        latency_mean_ms = mean(latency_values)
        rows.append(
            VlmFixtureReportRow(
                backend=backend_name,
                fixture_name=fixture_name,
                fixture_category=first.fixture_category,
                image_count=first.image_count,
                total_image_pixels=first.total_image_pixels,
                total_megapixels=first.total_megapixels,
                widths_summary=first.widths_summary,
                heights_summary=first.heights_summary,
                formats_summary=first.formats_summary,
                total_file_size_bytes=first.total_file_size_bytes,
                prompt_preview=first.prompt_preview,
                prompt_text_source=first.prompt_text_source,
                prompt_tokens_mean=mean(prompt_values),
                completion_tokens_mean=mean(completion_values),
                total_tokens_mean=mean(
                    [p + c for p, c in zip(prompt_values, completion_values)]
                ),
                ttft_mean_ms=mean(ttft_values),
                ttft_p50_ms=percentile(ttft_values, 50),
                latency_mean_ms=latency_mean_ms,
                latency_p50_ms=percentile(latency_values, 50),
                latency_p95_ms=percentile(
                    latency_values, 95, min_samples=P95_MIN_SAMPLES
                ),
                image_load_ms_mean=mean(image_load_values),
                image_decode_ms_mean=mean(image_decode_values),
                image_preprocess_ms_mean=mean(image_preprocess_values),
                image_preprocess_ms_p50=percentile(image_preprocess_values, 50),
                image_preprocess_ms_p95=percentile(
                    image_preprocess_values, 95, min_samples=P95_MIN_SAMPLES
                ),
                decode_tps_mean=mean(decode_tps_values),
                e2e_tps_mean=mean(e2e_tps_values),
                latency_per_completion_token_ms=calculate_per_token_latency_ms(
                    latency_mean_ms,
                    mean(completion_values),
                ),
                latency_per_image_ms=(latency_mean_ms / first.image_count)
                if latency_mean_ms is not None and first.image_count > 0
                else None,
                latency_per_megapixel_ms=(latency_mean_ms / first.total_megapixels)
                if latency_mean_ms is not None
                and first.total_megapixels not in (None, 0)
                else None,
                samples=len(successful),
                errors=errors,
                error_rate=(errors / len(samples)) if samples else 0.0,
                max_tokens=first.max_tokens,
                temperature=first.temperature,
                top_p=first.top_p,
                stop_reason_summary=",".join(dict.fromkeys(stop_reasons)) or None,
                warnings=tuple(warnings),
            )
        )
    return tuple(rows)


def _reduce_streaming_scenario(
    backend: str,
    measurements: list[VlmStreamResult],
) -> VlmStreamingBackendResult:
    successful = [sample for sample in measurements if sample.succeeded]
    errors = len(measurements) - len(successful)
    ttft_values = [
        sample.ttft_ms for sample in successful if sample.ttft_ms is not None
    ]
    latency_values = [
        sample.latency_ms for sample in successful if sample.latency_ms is not None
    ]
    completion_values = [
        float(sample.completion_tokens)
        for sample in successful
        if sample.completion_tokens is not None
    ]
    prompt_values = [
        float(sample.prompt_tokens)
        for sample in successful
        if sample.prompt_tokens is not None
    ]
    chunk_count_values = [
        float(sample.sse_chunk_count)
        for sample in successful
        if sample.sse_chunk_count is not None
    ]
    interval_mean_values = [
        sample.sse_chunk_interval_mean_ms
        for sample in successful
        if sample.sse_chunk_interval_mean_ms is not None
    ]
    finish_reasons = [sample.finish_reason or "not_available" for sample in successful]
    parse_errors = sum(sample.parse_errors for sample in measurements)
    return VlmStreamingBackendResult(
        backend=backend,
        mode="streaming",
        streaming_available=True,
        samples=len(successful),
        errors=errors,
        error_rate=(errors / len(measurements)) if measurements else 0.0,
        ttft_mean_ms=mean(ttft_values),
        ttft_p50_ms=percentile(ttft_values, 50),
        ttft_p95_ms=percentile(ttft_values, 95, min_samples=P95_MIN_SAMPLES),
        latency_mean_ms=mean(latency_values),
        latency_p50_ms=percentile(latency_values, 50),
        latency_p95_ms=percentile(latency_values, 95, min_samples=P95_MIN_SAMPLES),
        completion_tokens_mean=mean(completion_values),
        prompt_tokens_mean=mean(prompt_values),
        sse_chunk_count_mean=mean(chunk_count_values)
        if backend != "raw mlx-vlm"
        else None,
        raw_chunk_count_mean=mean(chunk_count_values)
        if backend == "raw mlx-vlm"
        else None,
        sse_chunk_interval_mean_ms=mean(interval_mean_values)
        if backend != "raw mlx-vlm"
        else None,
        sse_chunk_interval_p50_ms=percentile(interval_mean_values, 50)
        if backend != "raw mlx-vlm"
        else None,
        sse_chunk_interval_p95_ms=percentile(
            interval_mean_values,
            95,
            min_samples=P95_MIN_SAMPLES,
        )
        if backend != "raw mlx-vlm"
        else None,
        chunk_interval_mean_ms=mean(interval_mean_values),
        chunk_interval_p50_ms=percentile(interval_mean_values, 50),
        chunk_interval_p95_ms=percentile(
            interval_mean_values,
            95,
            min_samples=P95_MIN_SAMPLES,
        ),
        stream_completed_normally=all(
            sample.stream_completed_normally for sample in successful
        )
        if successful
        else None,
        finish_reason_distribution=",".join(dict.fromkeys(finish_reasons)) or None,
        parse_errors=parse_errors,
        notes=tuple(
            dict.fromkeys(note for sample in measurements for note in sample.notes)
        ),
        warnings=(),
    )


def _reduce_streaming_results(
    measurements_by_backend: dict[str, list[VlmStreamResult]],
    unavailable_notes: dict[str, tuple[str, ...]],
) -> tuple[VlmStreamingBackendResult, ...]:
    rows: list[VlmStreamingBackendResult] = []
    for backend_name, measurements in sorted(measurements_by_backend.items()):
        rows.append(_reduce_streaming_scenario(backend_name, measurements))
    for backend_name, notes in sorted(unavailable_notes.items()):
        if backend_name in measurements_by_backend:
            continue
        rows.append(
            VlmStreamingBackendResult(
                backend=backend_name,
                mode="streaming",
                streaming_available=False,
                samples=0,
                errors=0,
                error_rate=0.0,
                ttft_mean_ms=None,
                ttft_p50_ms=None,
                ttft_p95_ms=None,
                latency_mean_ms=None,
                latency_p50_ms=None,
                latency_p95_ms=None,
                completion_tokens_mean=None,
                prompt_tokens_mean=None,
                sse_chunk_count_mean=None,
                raw_chunk_count_mean=None,
                sse_chunk_interval_mean_ms=None,
                sse_chunk_interval_p50_ms=None,
                sse_chunk_interval_p95_ms=None,
                chunk_interval_mean_ms=None,
                chunk_interval_p50_ms=None,
                chunk_interval_p95_ms=None,
                stream_completed_normally=None,
                finish_reason_distribution=None,
                parse_errors=0,
                notes=notes,
                warnings=notes,
            )
        )
    return tuple(rows)


def _reduce_concurrency_scenario(
    backend: str,
    concurrency: int,
    measurements: list[VlmStreamResult],
    *,
    wall_clock_duration_ms: float,
) -> VlmConcurrencyResult:
    successful = [sample for sample in measurements if sample.succeeded]
    errors = len(measurements) - len(successful)
    ttft_values = [
        sample.ttft_ms for sample in successful if sample.ttft_ms is not None
    ]
    latency_values = [
        sample.latency_ms for sample in successful if sample.latency_ms is not None
    ]
    total_completion_tokens = sum(
        sample.completion_tokens or 0 for sample in successful
    )
    warnings: list[str] = []
    if concurrency > 4:
        warnings.append(
            "higher concurrency can trigger local memory pressure on MacBook-class host"
        )
    warnings.append(
        "concurrency results are local behavior checks, not production capacity"
    )
    return VlmConcurrencyResult(
        backend=backend,
        concurrency=concurrency,
        completed_requests=len(successful),
        errors=errors,
        error_rate=(errors / len(measurements)) if measurements else 0.0,
        wall_clock_duration_ms=wall_clock_duration_ms,
        requests_per_second=(len(successful) / (wall_clock_duration_ms / 1000.0))
        if wall_clock_duration_ms > 0
        else None,
        completion_tokens_per_second=(
            total_completion_tokens / (wall_clock_duration_ms / 1000.0)
        )
        if wall_clock_duration_ms > 0
        else None,
        ttft_mean_ms=mean(ttft_values),
        ttft_p50_ms=percentile(ttft_values, 50),
        latency_mean_ms=mean(latency_values),
        latency_p50_ms=percentile(latency_values, 50),
        latency_p95_ms=percentile(latency_values, 95, min_samples=P95_MIN_SAMPLES),
        queue_wait_mean_ms=None,
        max_queue_depth=None,
        warnings=tuple(warnings),
    )


def _collect_fairness_warnings(
    backend_results: list[BenchmarkResult],
    fixture_rows: tuple[VlmFixtureReportRow, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    raw = next(
        (result for result in backend_results if result.backend == "raw mlx-vlm"), None
    )
    if raw is not None:
        for result in backend_results:
            if result.backend == raw.backend:
                continue
            prompt_delta = _relative_delta(
                result.prompt_tokens_mean, raw.prompt_tokens_mean
            )
            completion_delta = _relative_delta(
                result.completion_tokens_mean,
                raw.completion_tokens_mean,
            )
            if prompt_delta is not None and prompt_delta > TOKEN_DIFF_WARN_FRACTION:
                warnings.append(
                    f"{result.backend} prompt_tokens_mean differs from raw mlx-vlm by {prompt_delta * 100.0:.1f}%; raw-vs-HTTP stays direct-call reference only"
                )
            if (
                completion_delta is not None
                and completion_delta > TOKEN_DIFF_WARN_FRACTION
            ):
                warnings.append(
                    f"{result.backend} completion_tokens_mean differs from raw mlx-vlm by {completion_delta * 100.0:.1f}%; prefer HTTP-vs-HTTP headline comparison"
                )
    rows_by_fixture_backend = {
        (row.fixture_name, row.backend): row for row in fixture_rows
    }
    for raw_row in [row for row in fixture_rows if row.backend == "raw mlx-vlm"]:
        for backend_name in ("mlx_vlm.server", "this project"):
            other = rows_by_fixture_backend.get((raw_row.fixture_name, backend_name))
            if other is None:
                continue
            prompt_delta = _relative_delta(
                other.prompt_tokens_mean, raw_row.prompt_tokens_mean
            )
            completion_delta = _relative_delta(
                other.completion_tokens_mean,
                raw_row.completion_tokens_mean,
            )
            if prompt_delta is not None and prompt_delta > TOKEN_DIFF_WARN_FRACTION:
                warnings.append(
                    f"fixture {raw_row.fixture_name}: {backend_name} prompt token count differs from raw by {prompt_delta * 100.0:.1f}%"
                )
            if (
                completion_delta is not None
                and completion_delta > TOKEN_DIFF_WARN_FRACTION
            ):
                warnings.append(
                    f"fixture {raw_row.fixture_name}: {backend_name} completion token count differs from raw by {completion_delta * 100.0:.1f}%"
                )
    return tuple(dict.fromkeys(warnings))


def _build_baseline_comparison_rows(
    backend_results: tuple[BenchmarkResult, ...],
    *,
    benchmark_mode: str,
    scenario_name: str,
    fixture_names: tuple[str, ...],
    max_tokens: int,
) -> tuple[VlmComparisonRow, ...]:
    results_by_backend = {result.backend: result for result in backend_results}
    rows: list[VlmComparisonRow] = []
    server = results_by_backend.get("mlx_vlm.server")
    project = results_by_backend.get("this project")
    if server is not None and project is not None:
        rows.append(
            _comparison_row_from_results(
                server,
                project,
                scenario_name=scenario_name,
                benchmark_mode=benchmark_mode,
                fixture_names=fixture_names,
                max_tokens=max_tokens,
                fairness_level="api_fairness",
                comparison_kind="http_fair_comparison",
            )
        )
    raw = results_by_backend.get("raw mlx-vlm")
    if raw is not None:
        for backend_name in ("mlx_vlm.server", "this project"):
            other = results_by_backend.get(backend_name)
            if other is None:
                continue
            rows.append(
                _comparison_row_from_results(
                    raw,
                    other,
                    scenario_name=scenario_name,
                    benchmark_mode=benchmark_mode,
                    fixture_names=fixture_names,
                    max_tokens=max_tokens,
                    fairness_level="semantic_fairness",
                    comparison_kind="direct_call_reference",
                )
            )
    return tuple(rows)


def _build_streaming_comparison_rows(
    measurements_by_backend: dict[str, list[VlmStreamResult]],
    *,
    benchmark_mode: str,
    scenario_name: str,
    fixture_names: tuple[str, ...],
    max_tokens: int,
) -> tuple[VlmComparisonRow, ...]:
    result_by_backend = {
        backend_name: _reduce_vlm_measurements(backend_name, measurements)
        for backend_name, measurements in measurements_by_backend.items()
    }
    return _build_baseline_comparison_rows(
        tuple(result_by_backend.values()),
        benchmark_mode=benchmark_mode,
        scenario_name=scenario_name,
        fixture_names=fixture_names,
        max_tokens=max_tokens,
    )


def _comparison_row_from_results(
    left: BenchmarkResult,
    right: BenchmarkResult,
    *,
    scenario_name: str,
    benchmark_mode: str,
    fixture_names: tuple[str, ...],
    max_tokens: int,
    fairness_level: str,
    comparison_kind: str,
) -> VlmComparisonRow:
    prompt_delta = _relative_delta(right.prompt_tokens_mean, left.prompt_tokens_mean)
    completion_delta = _relative_delta(
        right.completion_tokens_mean,
        left.completion_tokens_mean,
    )
    token_equivalent = None
    if prompt_delta is not None and completion_delta is not None:
        token_equivalent = (
            prompt_delta <= TOKEN_DIFF_WARN_FRACTION
            and completion_delta <= TOKEN_DIFF_WARN_FRACTION
        )
    reasons: list[str] = []
    same_backend_category = _backend_category(left.backend) == _backend_category(
        right.backend
    )
    if not same_backend_category and token_equivalent is not True:
        reasons.append("backend category differs without proven token equivalence")
    if comparison_kind == "direct_call_reference":
        reasons.append("raw direct-call reference is not strict serving headline without model-input parity")
    if prompt_delta is None:
        reasons.append("prompt token counts unavailable")
    elif prompt_delta > TOKEN_DIFF_WARN_FRACTION:
        reasons.append("prompt token mismatch > 5%")
    if completion_delta is None:
        reasons.append("completion token counts unavailable")
    elif completion_delta > TOKEN_DIFF_WARN_FRACTION:
        reasons.append("completion token mismatch > 5%")
    if abs(left.error_rate - right.error_rate) > 0.05:
        reasons.append("error rates are not comparable")
    if min(left.samples, right.samples) < P95_MIN_SAMPLES:
        reasons.append("measured sample count is too low for strong tail claims")
    headline_eligible = not reasons
    return VlmComparisonRow(
        scenario=scenario_name,
        backend_a=left.backend,
        backend_b=right.backend,
        fairness_level=fairness_level,
        comparison_kind=comparison_kind,
        same_model=True,
        same_scenario=True,
        same_benchmark_mode=True,
        same_fixture_set=bool(fixture_names),
        same_max_tokens=True,
        same_temperature=True,
        same_top_p=True,
        same_streaming_mode=True,
        same_backend_category=same_backend_category,
        prompt_tokens_mean_delta_pct=(
            prompt_delta * 100.0 if prompt_delta is not None else None
        ),
        completion_tokens_mean_delta_pct=(
            completion_delta * 100.0 if completion_delta is not None else None
        ),
        token_equivalent=token_equivalent,
        error_rates_comparable=abs(left.error_rate - right.error_rate) <= 0.05,
        sufficient_samples=min(left.samples, right.samples) >= P95_MIN_SAMPLES,
        headline_eligible=headline_eligible,
        reasons_not_headline_eligible=tuple(reasons),
        latency_mean_delta_ms=calculate_overhead(
            right.latency_mean_ms, left.latency_mean_ms
        ),
        latency_p50_delta_ms=calculate_overhead(
            right.latency_p50_ms, left.latency_p50_ms
        ),
        latency_p95_delta_ms=calculate_overhead(
            right.latency_p95_ms, left.latency_p95_ms
        ),
        ttft_mean_delta_ms=calculate_overhead(right.ttft_mean_ms, left.ttft_mean_ms),
        decode_tps_delta_pct=calculate_overhead_percent(
            right.decode_tokens_per_second_mean,
            left.decode_tokens_per_second_mean,
        ),
        e2e_tps_delta_pct=calculate_overhead_percent(
            right.end_to_end_tokens_per_second_mean,
            left.end_to_end_tokens_per_second_mean,
        ),
    )


def _backend_category(backend_name: str) -> str:
    if backend_name == "raw mlx-vlm":
        return "direct_call"
    return "http"


def _collect_streaming_warnings(
    streaming_rows: tuple[VlmStreamingBackendResult, ...],
    comparison_rows: tuple[VlmComparisonRow, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for row in streaming_rows:
        warnings.extend(row.warnings)
        if not row.streaming_available:
            warnings.extend(row.notes)
    for row in comparison_rows:
        if not row.headline_eligible:
            warnings.append(
                f"{row.backend_a} vs {row.backend_b} is not headline-eligible for streaming scenario: {'; '.join(row.reasons_not_headline_eligible)}"
            )
    return tuple(dict.fromkeys(warnings))


def _aggregate_cancellation_results(
    results: list[VlmCancellationResult],
) -> VlmCancellationResult | None:
    if not results:
        return None
    warnings = tuple(dict.fromkeys(w for row in results for w in row.warnings))
    return VlmCancellationResult(
        backend=results[0].backend,
        cancellation_attempts=sum(row.cancellation_attempts for row in results),
        server_observed_cancellations=None,
        cancellation_errors=sum(row.cancellation_errors for row in results),
        follow_up_success_count=sum(row.follow_up_success_count for row in results),
        follow_up_error_count=sum(row.follow_up_error_count for row in results),
        follow_up_latency_mean_ms=mean(
            [
                row.follow_up_latency_mean_ms
                for row in results
                if row.follow_up_latency_mean_ms is not None
            ]
        ),
        worker_health_after_cancel=results[-1].worker_health_after_cancel,
        warnings=warnings,
    )


def _baseline_interpretation(
    results: tuple[BenchmarkResult, ...],
    comparison_rows: tuple[VlmComparisonRow, ...],
) -> tuple[str, ...]:
    interpretation = [
        "Baseline scenario is only scenario for headline latency comparison.",
        "Raw mlx-vlm remains direct-call reference unless model-input parity is proven.",
    ]
    for row in comparison_rows:
        if row.comparison_kind == "http_fair_comparison" and row.headline_eligible:
            interpretation.append(
                "HTTP-vs-HTTP comparison is headline-eligible for token-equivalent serving claims."
            )
            break
    return tuple(dict.fromkeys(interpretation))


def _measure_image_file_inputs(
    case: VlmPromptCase,
) -> tuple[float | None, float | None, float | None]:
    if not case.image_paths:
        return (0.0, 0.0, 0.0)
    load_start = time.perf_counter()
    for image_path in case.image_paths:
        Path(image_path).read_bytes()
    load_end = time.perf_counter()
    for image_path in case.image_paths:
        collect_image_metadata(Path(image_path))
    end = time.perf_counter()
    image_load_ms = (load_end - load_start) * 1000.0
    image_decode_ms = (end - load_end) * 1000.0
    return (image_load_ms, image_decode_ms, image_load_ms + image_decode_ms)


def _compat_case_from_messages(
    messages: list[dict[str, Any]],
    *,
    prompt_tokens_estimate: int | None = None,
    image_count: int,
) -> VlmPromptCase:
    image_paths = _extract_image_paths_from_messages(messages)
    prompt_text = _flatten_messages(messages)
    estimated_tokens = prompt_tokens_estimate
    if estimated_tokens is None:
        if not prompt_text and not image_paths:
            estimated_tokens = 256
        else:
            estimated_tokens = _estimate_vlm_prompt_tokens(
                prompt_text,
                image_count or len(image_paths),
            )
    metadata = tuple(
        collect_image_metadata(Path(path))
        for path in image_paths
        if Path(path).exists()
    )
    return VlmPromptCase(
        name="compat-case",
        category=(
            "single_image"
            if (image_count or len(image_paths)) <= 1
            else "multi_image_summary"
        ),
        prompt_text=prompt_text,
        messages=messages,
        image_paths=image_paths,
        prompt_tokens_estimate=estimated_tokens,
        image_metadata=metadata,
    )


def _extract_image_paths_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, ...]:
    image_paths: list[str] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "image_url":
                continue
            image_url = item.get("image_url", {}).get("url")
            if isinstance(image_url, str):
                image_paths.append(image_url)
    return tuple(image_paths)


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            rendered_parts: list[str] = []
            for item in content:
                if item.get("type") == "text":
                    rendered_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    rendered_parts.append("[image]")
            rendered = " ".join(rendered_parts)
        else:
            rendered = str(content)
        parts.append(f"{role}: {rendered}")
    return "\n".join(parts)


def _truncate(text: str, max_chars: int = 180) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _dimension_summary(values: list[int | None]) -> str:
    available = [value for value in values if value is not None]
    if not available:
        return "not_available"
    if len(available) == 1:
        return str(available[0])
    return f"min={min(available)},max={max(available)},mean={sum(available) / len(available):.1f}"


def _relative_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return abs(value - baseline) / baseline


def _write_json_report(path: Path, runs: list[BenchmarkRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": [asdict(run) for run in runs]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _result_summary(model_name: str, result: BenchmarkResult) -> str:
    latency_text = (
        f"{result.latency_mean_ms:.1f} ms"
        if result.latency_mean_ms is not None
        else "n/a"
    )
    ttft_text = (
        f"{result.ttft_mean_ms:.1f} ms" if result.ttft_mean_ms is not None else "n/a"
    )
    return (
        f"[VLM model {model_name}] {result.backend} done: "
        f"latency_mean={latency_text}, ttft_mean={ttft_text}, "
        f"samples={result.samples}, errors={result.errors}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
