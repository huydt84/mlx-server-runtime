#!/usr/bin/env python3
"""Benchmark raw ``mlx-vlm``, ``mlx_vlm.server``, and this project for VLM."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"

DEFAULT_VLM_MODELS = [
    "mlx-community/LFM2.5-VL-1.6B-4bit",
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "mlx-community/Qwen3.5-2B-4bit"
]

VLM_SYSTEM_PREFIX = (
    "You are an expert visual analyst. Describe images precisely, separate "
    "observation from inference, and keep summaries grounded in visible content."
)
VLM_SHARED_SUFFIXES = [
    "Give a one-sentence summary first, then 3 precise bullets.",
    "Call out the most visually distinctive detail and any uncertainty.",
    "Focus on colors, layout, and object relationships.",
]
VLM_LONG_BASES = [
    (
        "Prepare a structured image summary suitable for a benchmark dataset. "
        "Cover scene composition, salient objects, relative positions, colors, "
        "texture cues, and likely content type such as photo, illustration, or chart."
    ),
    (
        "Explain how a multimodal assistant should reason about these images without "
        "hallucinating. Mention what is directly visible, what remains ambiguous, and "
        "what concise answer format best fits the content."
    ),
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from benchmarks.vlm_fixtures import VlmFixture, prepare_fixtures  # noqa: E402
from mlx_worker.benchmarking import (  # noqa: E402
    BenchmarkResult,
    BenchmarkRun,
    P95_MIN_SAMPLES,
    P99_MIN_SAMPLES,
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


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VlmStreamResult:
    """Captured latency and token data for one VLM request.

    Carries VLM-specific fields in addition to the standard timing and
    token data shared with text benchmarks.
    """

    ttft_ms: float | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    text: str
    finish_reason: str | None = None
    image_count: int = 0
    image_preprocess_ms: float | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """True when the sample should be included in aggregates."""
        return self.error is None


@dataclass(frozen=True)
class NopSampler:
    """Placeholder sampler for backends that do not use an MLX sampler."""

    pass


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


class _ProgressTracker:
    """Track benchmark loop progress on stderr without touching timed output."""

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
                _tqdm is not None
                and getattr(self.stream, "isatty", lambda: False)()
            )
        self._bar = (
            _tqdm(
                total=self.total,
                desc=label,
                unit="step",
                leave=False,
                file=self.stream,
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
        """Advance one step and surface the current benchmark case."""
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
        """Close the progress display cleanly."""
        if self._bar is not None:
            self._bar.close()
            return
        if self.total > 0:
            _log_event(f"{self.label}: completed", stream=self.stream)


def _log_event(message: str, *, stream: TextIO | None = None) -> None:
    """Write a timestamped status line to stderr."""
    target = stream or sys.stderr
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=target, flush=True)


# ---------------------------------------------------------------------------
# Prompt case construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VlmPromptCase:
    """One VLM prompt case for all compared backends."""

    name: str
    messages: list[dict[str, Any]]
    image_paths: tuple[str, ...]
    prompt_tokens_estimate: int


def _image_prompt_for_fixture(name: str) -> str:
    lowered = name.lower()
    if "fish" in lowered:
        return (
            "Describe this image. Is it a photo or illustration? Summarize the "
            "main subject, dominant colors, and overall style."
        )
    if "fruit" in lowered:
        return (
            "Describe this image of food or produce. Name visible items, colors, "
            "and the arrangement on the table."
        )
    if "lake" in lowered:
        return (
            "Summarize this landscape image. Describe the terrain, water, sky, and "
            "the overall mood of the scene."
        )
    return f"Describe this image ({name}) in detail and summarize what is visible."


def _estimate_vlm_prompt_tokens(prompt_text: str, image_count: int) -> int:
    return max(32, (len(prompt_text) // 4) + (image_count * 48))


def _make_vlm_case(
    name: str,
    prompt_text: str,
    image_paths: tuple[str, ...],
) -> VlmPromptCase:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_path}}
        for image_path in image_paths
    )
    return VlmPromptCase(
        name=name,
        messages=[{"role": "user", "content": content}],
        image_paths=image_paths,
        prompt_tokens_estimate=_estimate_vlm_prompt_tokens(
            prompt_text, len(image_paths)
        ),
    )


def _build_long_vlm_prompt(image_labels: list[str]) -> str:
    details = " ".join(VLM_LONG_BASES)
    labels = ", ".join(image_labels)
    return (
        f"{VLM_SYSTEM_PREFIX}\n"
        f"You are given image set: {labels}. "
        f"{details} Give a short summary, then a detailed comparison, then a final "
        "one-line caption for each image."
    )


def _build_vlm_cases(
    fixtures: list[VlmFixture],
) -> list[VlmPromptCase]:
    """Convert fixtures into mixed VLM prompt cases.

    Mirrors `compare.py` intent: short prompts, shared-prefix variants, and a
    longer multi-image prompt so all backends see a more representative mix.
    """
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
            _make_vlm_case(f"single-{index}-{name}", base_prompt, (image_path,))
        )
        suffix = VLM_SHARED_SUFFIXES[(index - 1) % len(VLM_SHARED_SUFFIXES)]
        cases.append(
            _make_vlm_case(
                f"prefix-{index}-{name}",
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
                "You will receive three images. Give a one-line summary for each image, "
                "then identify which one looks most like a natural photo, which one looks "
                "most synthetic or illustrative, and which one contains the densest visual detail.",
                image_paths,
            )
        )
        cases.append(
            _make_vlm_case(
                "long-multi-image-analysis",
                _build_long_vlm_prompt(image_labels),
                image_paths,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum completion tokens per request.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT
        / "benchmarks"
        / "results"
        / "phase_9_vlm_report.md",
    )
    parser.add_argument("--project-port", type=int, default=8000)
    parser.add_argument("--server-port", type=int, default=8001)
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=90,
        help="Seconds to wait for a backend process to open its port.",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=int,
        default=90,
        help="Seconds to wait for a backend readiness signal.",
    )
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw mlx-vlm backend (useful when only testing HTTP backends).",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip mlx_vlm.server backend.",
    )
    parser.add_argument(
        "--skip-project",
        action="store_true",
        help="Skip this project backend.",
    )
    args = parser.parse_args(argv)
    if not args.report_path.is_absolute():
        args.report_path = REPO_ROOT / args.report_path

    image_dir = Path(
        tempfile.mkdtemp(prefix="mlx-vlm-benchmark-fixtures-")
    )
    _log_event(f"generating VLM fixtures under {image_dir}")
    fixtures = prepare_fixtures(image_dir)
    _log_event(
        f"fixture set ready: {len(fixtures)} case(s) — "
        + ", ".join(f.name for f in fixtures)
    )

    models = args.models or DEFAULT_VLM_MODELS
    prompt_cases = _build_vlm_cases(fixtures)

    if not models:
        _log_event("ERROR: no VLM models provided and no defaults work")
        return 1

    runs: list[BenchmarkRun] = []
    for model_index, model_name in enumerate(models, start=1):
        _log_event(
            f"[model {model_index}/{len(models)}] benchmarking {model_name}"
        )

        # Collect per-backend results for this model.
        backend_results: list[BenchmarkResult] = []

        if not args.skip_raw:
            _log_event(f"[{model_name}] raw mlx-vlm backend")
            raw_result = _benchmark_raw_mlx_vlm(
                model_name,
                prompt_cases,
                args.max_tokens,
                args.warmup_trials,
                args.trials,
            )
            _log_event(_result_summary(model_name, raw_result))
            backend_results.append(raw_result)

        if not args.skip_server:
            _log_event(f"[{model_name}] mlx_vlm.server backend")
            server_result = _benchmark_vlm_server(
                model_name,
                prompt_cases,
                args.max_tokens,
                args.server_port,
                args.warmup_trials,
                args.trials,
                args.launch_timeout,
                args.readiness_timeout,
            )
            _log_event(_result_summary(model_name, server_result))
            backend_results.append(server_result)

        if not args.skip_project:
            _log_event(f"[{model_name}] this project backend")
            project_result = _benchmark_vlm_project(
                model_name,
                prompt_cases,
                args.max_tokens,
                args.project_port,
                args.warmup_trials,
                args.trials,
                args.launch_timeout,
                args.readiness_timeout,
            )
            _log_event(_result_summary(model_name, project_result))
            backend_results.append(project_result)

        if not backend_results:
            _log_event(f"ERROR: no backends benchmarked for {model_name}")
            continue

        runs.append(
            BenchmarkRun(
                model=model_name,
                prompt=
                "VLM fixture suite: "
                + ", ".join(case.name for case in prompt_cases),
                max_tokens=args.max_tokens,
                generated_at=now_utc_iso(),
                results=tuple(backend_results),
            )
        )

    write_vlm_report(args.report_path, runs)
    _log_event(f"report_written={args.report_path}")
    print(f"report_written={args.report_path}")
    return 0


# ---------------------------------------------------------------------------
# Backend: raw mlx-vlm
# ---------------------------------------------------------------------------


def _benchmark_raw_mlx_vlm(
    model_name: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    warmup_trials: int,
    trials: int,
) -> BenchmarkResult:
    """Benchmark raw ``mlx_vlm.generate`` / ``mlx_vlm.stream_generate``."""
    from mlx_vlm import load as vlm_load, generate as vlm_generate

    load_start = time.perf_counter()
    try:
        model, processor = vlm_load(model_name)
    except Exception as exc:
        load_time_ms = (time.perf_counter() - load_start) * 1000.0
        _log_event(
            f"[raw mlx-vlm] model load failed after {load_time_ms:.1f} ms: "
            f"{type(exc).__name__}: {exc}"
        )
        measurements = [
            _failed_vlm_stream_result(pc, exc)
            for _ in range(max(trials, 1))
            for pc in prompt_cases
        ]
        return _reduce_vlm_measurements(
            "raw mlx-vlm",
            measurements,
            load_time_ms=load_time_ms,
            summarize_distribution=True,
        )

    _vlm_model = model
    _vlm_processor = processor
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    _log_event(f"[raw mlx-vlm] model load: {load_time_ms:.1f} ms")

    # Warmup
    warmup_total = warmup_trials * len(prompt_cases)
    warmup_tracker = _ProgressTracker("raw mlx-vlm warmup", warmup_total)
    for warmup_index in range(1, warmup_trials + 1):
        for pc in prompt_cases:
            _raw_vlm_generate_once(
                model,
                processor,
                pc,
                max_tokens,
            )
            warmup_tracker.advance(
                f"warmup {warmup_index}/{warmup_trials}, {pc.name}"
            )
    warmup_tracker.close()

    # Measured trials
    trial_total = trials * len(prompt_cases)
    trial_tracker = _ProgressTracker("raw mlx-vlm measured", trial_total)
    measurements: list[VlmStreamResult] = []
    for trial_index in range(1, trials + 1):
        for pc in prompt_cases:
            try:
                measurements.append(
                    _raw_vlm_generate_once(
                        model,
                        processor,
                        pc,
                        max_tokens,
                    )
                )
            except Exception as exc:
                measurements.append(
                    _failed_vlm_stream_result(pc, exc)
                )
                _log_event(
                    f"[raw mlx-vlm] request failed for {pc.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
            trial_tracker.advance(
                f"trial {trial_index}/{trials}, {pc.name}"
            )
    trial_tracker.close()
    return _reduce_vlm_measurements(
        "raw mlx-vlm", measurements, load_time_ms=load_time_ms,
        summarize_distribution=True,
    )


def _raw_vlm_generate_once(
    model: Any,
    processor: Any,
    pc: VlmPromptCase,
    max_tokens: int,
) -> VlmStreamResult:
    """Run a single non-streaming VLM generation."""
    start = time.perf_counter()
    from mlx_vlm import generate as vlm_generate

    prompt_str = _build_vlm_prompt(model, processor, pc)
    image_list: list[str] = list(pc.image_paths) if pc.image_paths else None  # type: ignore[assignment]

    result = vlm_generate(
        model,
        processor,
        prompt_str,
        image=image_list,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        verbose=False,
    )
    end = time.perf_counter()

    text = result.text if hasattr(result, "text") else str(result)
    finish_reason = getattr(result, "finish_reason", None) or "stop"
    prompt_tokens = int(getattr(result, "prompt_tokens", 0))
    completion_tokens = int(getattr(result, "generation_tokens", 0))

    # Approximate TTFT for non-streaming: use half the latency.
    latency_ms = (end - start) * 1000.0
    ttft_ms = latency_ms / 2.0

    return VlmStreamResult(
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens or pc.prompt_tokens_estimate,
        completion_tokens=completion_tokens,
        text=text,
        image_count=len(pc.image_paths),
        finish_reason=finish_reason,
    )


def _build_vlm_prompt(
    model: Any, processor: Any, pc: VlmPromptCase
) -> str:
    """Build the prompt string for mlx-vlm from a prompt case."""
    try:
        from mlx_vlm.prompt_utils import (
            apply_chat_template,
            get_chat_template,
        )
        from mlx_vlm.utils import load_config

        model_path = getattr(model, "model_path", None)
        config = load_config(str(model_path)) if model_path else None
        if config is not None:
            result = apply_chat_template(
                processor,
                config,
                pc.messages,
                add_generation_prompt=True,
                num_images=len(pc.image_paths),
            )
            if isinstance(result, str):
                return result
            inner = [result] if isinstance(result, dict) else list(result)
            return get_chat_template(
                processor, inner, add_generation_prompt=True
            )
    except Exception:
        pass

    # Fallback: processor.apply_chat_template
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            pc.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Last resort: naive string
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in pc.messages
    ) + "\nassistant:"


# ---------------------------------------------------------------------------
# Backend: mlx_vlm.server (HTTP)
# ---------------------------------------------------------------------------


def _benchmark_vlm_server(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    warmup_trials: int,
    trials: int,
    launch_timeout: int,
    readiness_timeout: int,
) -> BenchmarkResult:
    """Benchmark ``mlx_vlm.server`` via its HTTP endpoint."""
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
    return _benchmark_vlm_http_service(
        "mlx_vlm.server",
        command_variants,
        f"http://127.0.0.1:{port}",
        model,
        prompt_cases,
        max_tokens,
        warmup_trials,
        trials,
        launch_timeout=launch_timeout,
        readiness_timeout=readiness_timeout,
    )


# ---------------------------------------------------------------------------
# Backend: this project (HTTP)
# ---------------------------------------------------------------------------


def _benchmark_vlm_project(
    model: str,
    prompt_cases: list[VlmPromptCase],
    max_tokens: int,
    port: int,
    warmup_trials: int,
    trials: int,
    launch_timeout: int,
    readiness_timeout: int,
) -> BenchmarkResult:
    """Benchmark this project via its HTTP endpoint."""
    with tempfile.TemporaryDirectory(prefix="mlx-vlm-benchmark-project-") as tmpdir_str:
        tmpdir_path = Path(tmpdir_str)
        config_path = _prepare_project_config(
            model,
            port,
            config_dir=tmpdir_path,
            vlm_model=model,
        )
        command_variants = [
            [
                "cargo",
                "run",
                "--release",
                "-p",
                "mlx_runtime_gateway",
            ]
        ]
        # Use VLM-specific readiness endpoint (same pattern as Phase 8
        # host-validation script) so the benchmark waits until the VLM
        # model has finished loading and warming up.
        readiness_url = f"/models/{quote(model, safe='')}/ready"
        return _benchmark_vlm_http_service(
            "this project",
            command_variants,
            f"http://127.0.0.1:{port}",
            model,
            prompt_cases,
            max_tokens,
            warmup_trials,
            trials,
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url=readiness_url,
            launch_timeout=launch_timeout,
            readiness_timeout=readiness_timeout,
        )


# ---------------------------------------------------------------------------
# Shared HTTP service benchmark (VLM variant)
# ---------------------------------------------------------------------------


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
) -> BenchmarkResult:
    """Benchmark a VLM HTTP service by launching it as a subprocess."""
    with tempfile.TemporaryDirectory(prefix="mlx-vlm-benchmark-") as tmpdir:
        stdout_path = Path(tmpdir) / f"{backend_name.replace(' ', '_')}.stdout.log"
        stderr_path = Path(tmpdir) / f"{backend_name.replace(' ', '_')}.stderr.log"
        last_error: str | None = None
        port = _extract_port(base_url)

        for variant_index, command in enumerate(command_variants, start=1):
            _log_event(
                f"[{backend_name}] launch attempt {variant_index}/"
                f"{len(command_variants)}: {' '.join(command)}"
            )
            if not _is_port_free("127.0.0.1", port):
                last_error = f"port {port} already in use before launch"
                _log_event(f"[{backend_name}] {last_error}")
                continue

            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd) if cwd is not None else None,
                    env=_merged_env(extra_env),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
            except BaseException:
                stdout_file.close()
                stderr_file.close()
                raise

            try:
                if not _wait_for_process_port(
                    process,
                    "127.0.0.1",
                    port,
                    timeout_s=launch_timeout,
                    label=backend_name,
                ):
                    last_error = _read_process_failure(
                        stderr_path, stdout_path
                    ) or f"timed out waiting for {backend_name} to open its port"
                    _log_event(f"[{backend_name}] launch failed: {last_error}")
                    continue

                # Wait for readiness.
                try:
                    _wait_for_vlm_service_ready(
                        base_url,
                        readiness_url,
                        model,
                        prompt_cases[0].messages,
                        max_tokens,
                        timeout_s=readiness_timeout,
                        label=backend_name,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    _log_event(
                        f"[{backend_name}] readiness failed: {last_error}"
                    )
                    continue

                # Warmup
                warmup_total = warmup_trials * len(prompt_cases)
                warmup_tracker = _ProgressTracker(
                    f"{backend_name} warmup", warmup_total
                )
                for warmup_index in range(1, warmup_trials + 1):
                    for pc in prompt_cases:
                        _request_vlm_completion(
                            base_url,
                            model,
                            pc.messages,
                            max_tokens,
                            prompt_tokens_estimate=pc.prompt_tokens_estimate,
                            image_count=len(pc.image_paths),
                        )
                        warmup_tracker.advance(
                            f"warmup {warmup_index}/{warmup_trials}, {pc.name}"
                        )
                warmup_tracker.close()

                # Measured trials
                trial_total = trials * len(prompt_cases)
                trial_tracker = _ProgressTracker(
                    f"{backend_name} measured", trial_total
                )
                measurements: list[VlmStreamResult] = []
                for trial_index in range(1, trials + 1):
                    for pc in prompt_cases:
                        try:
                            measurements.append(
                                _request_vlm_completion(
                                    base_url,
                                    model,
                                    pc.messages,
                                    max_tokens,
                                    prompt_tokens_estimate=pc.prompt_tokens_estimate,
                                    image_count=len(pc.image_paths),
                                )
                            )
                        except Exception as exc:
                            measurements.append(
                                _failed_vlm_stream_result(pc, exc)
                            )
                            _log_event(
                                f"[{backend_name}] request failed for "
                                f"{pc.name}: {type(exc).__name__}: {exc}"
                            )
                        trial_tracker.advance(
                            f"trial {trial_index}/{trials}, {pc.name}"
                        )
                trial_tracker.close()
                return _reduce_vlm_measurements(
                    backend_name,
                    measurements,
                    summarize_distribution=True,
                )
            except Exception as exc:
                last_error = str(exc)
                _log_event(
                    f"[{backend_name}] benchmark attempt failed: {last_error}"
                )
            finally:
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
                stdout_file.close()
                stderr_file.close()

        failure_exc = RuntimeError(
            f"{backend_name} benchmark failed after trying all launch variants: "
            f"{last_error}"
        )
        measurements = [
            _failed_vlm_stream_result(pc, failure_exc)
            for _ in range(max(trials, 1))
            for pc in prompt_cases
        ]
        return _reduce_vlm_measurements(
            backend_name,
            measurements,
            summarize_distribution=True,
        )


# ---------------------------------------------------------------------------
# HTTP request helpers for VLM
# ---------------------------------------------------------------------------


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
    """Send a VLM chat request and capture streaming response.

    When the upstream server omits ``usage`` from the SSE stream the
    completion token count is estimated from the response text length
    so benchmark throughput metrics remain usable.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
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

    with urlopen(request, timeout=request_timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            payload_data = json.loads(data)
            usage = payload_data.get("usage")
            if usage and isinstance(usage, dict):
                completion_tokens = int(
                    usage.get("completion_tokens", completion_tokens)
                )
                if prompt_tokens is None:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
            choice = payload_data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or choice.get("message", {}).get("content")
                or ""
            )
            if content:
                final_text_parts.append(content)
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()

    end = time.perf_counter()
    final_text = "".join(final_text_parts)
    if first_delta_at is None:
        first_delta_at = end
    latency_ms = (end - start) * 1000.0
    ttft_ms = (first_delta_at - start) * 1000.0

    # Fallback: estimate tokens from text when server omits usage.
    if not completion_tokens and final_text.strip():
        # Rough estimate: ~4 chars per token for VLM text output.
        completion_tokens = max(1, len(final_text) // 4)

    return VlmStreamResult(
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens or prompt_tokens_estimate or 256,
        completion_tokens=completion_tokens,
        text=final_text,
        image_count=image_count,
        error=None,
    )


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def _wait_for_vlm_service_ready(
    base_url: str,
    readiness_url: str | None,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    timeout_s: int,
    label: str,
) -> None:
    """Wait until the VLM HTTP service accepts requests."""
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    now = time.monotonic()
    next_heartbeat = now
    while now < deadline:
        if now >= next_heartbeat:
            _log_event(f"[{label}] waiting for service readiness")
            next_heartbeat = now + 5
        try:
            if readiness_url:
                with urlopen(
                    Request(f"{base_url}{readiness_url}"), timeout=10
                ) as resp:
                    if resp.status < 500:
                        _log_event(
                            f"[{label}] readiness endpoint accepted requests"
                        )
                        return
                    last_error = (
                        f"readiness endpoint returned HTTP {resp.status}"
                    )
            else:
                _request_vlm_completion(
                    base_url,
                    model,
                    messages,
                    max_tokens,
                )
                _log_event(
                    f"[{label}] streaming endpoint accepted requests"
                )
                return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
        now = time.monotonic()
    raise RuntimeError(
        f"service did not become ready: {last_error}"
    )


# ---------------------------------------------------------------------------
# Port / process helpers
# ---------------------------------------------------------------------------


def _wait_for_process_port(
    process: subprocess.Popen[Any],
    host: str,
    port: int,
    *,
    timeout_s: int,
    label: str,
) -> bool:
    deadline = time.monotonic() + timeout_s
    now = time.monotonic()
    next_heartbeat = now
    while now < deadline:
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
            now = time.monotonic()
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


def _read_process_failure(
    stderr_path: Path, stdout_path: Path
) -> str | None:
    stderr = stderr_path.read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    stdout = stdout_path.read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    if stderr:
        return stderr.splitlines()[-1]
    if stdout:
        return stdout.splitlines()[-1]
    return None


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
    # Overwrite text model only in text-only mode.
    # When vlm_model is set, preserve the original text model so the
    # gateway's text worker can start correctly.
    if not vlm_model:
        text = _replace_config_value(text, "model", model)
    text = _replace_config_value(text, "port", str(port))
    text = _replace_config_value(
        text,
        "ipc_path",
        str(ipc_root / "m.sock"),
    )
    if vlm_model:
        text = _set_vlm_config(text, vlm_model)
    target.write_text(text, encoding="utf-8")
    return target


def _set_vlm_config(text: str, vlm_model: str) -> str:
    """Uncomment and set ``vlm_model`` in the generated config.

    Searches for a line containing ``vlm_model`` (commented or not),
    uncomments it, and sets the value.  Appends a new entry if none
    exists.
    """
    lines: list[str] = []
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        # Match commented or uncommented vlm_model lines.
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
    lines = []
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


# ---------------------------------------------------------------------------
# Measurement reduction
# ---------------------------------------------------------------------------


def _failed_vlm_stream_result(
    pc: VlmPromptCase, exc: BaseException
) -> VlmStreamResult:
    """Build a failed sample excluded from aggregates."""
    return VlmStreamResult(
        ttft_ms=None,
        latency_ms=None,
        prompt_tokens=pc.prompt_tokens_estimate,
        completion_tokens=None,
        text="",
        image_count=len(pc.image_paths),
        error=f"{type(exc).__name__}: {exc}",
    )


def _reduce_vlm_measurements(
    backend: str,
    measurements: Iterable[VlmStreamResult],
    *,
    load_time_ms: float | None = None,
    summarize_distribution: bool = False,
) -> BenchmarkResult:
    """Aggregate VLM measurements into a BenchmarkResult."""
    measurements = list(measurements)
    if not measurements:
        raise ValueError(
            f"{backend} produced no VLM benchmark measurements"
        )

    successful = [s for s in measurements if s.succeeded]
    errors = len(measurements) - len(successful)

    ttft_values = [
        s.ttft_ms for s in successful if s.ttft_ms is not None
    ]
    latency_values = [
        s.latency_ms for s in successful if s.latency_ms is not None
    ]
    prompt_token_values = [
        float(s.prompt_tokens)
        for s in successful
        if s.prompt_tokens is not None
    ]
    completion_token_values = [
        float(s.completion_tokens)
        for s in successful
        if s.completion_tokens is not None
    ]
    image_preprocess_values = [
        s.image_preprocess_ms
        for s in successful
        if s.image_preprocess_ms is not None
    ]
    image_count_values = [
        float(s.image_count) for s in successful
    ]

    decode_time_values: list[float] = []
    decode_tps_values: list[float] = []
    e2e_tps_values: list[float] = []
    ttft_greater_than_latency = 0
    decode_tps_unavailable = 0
    for s in successful:
        if s.ttft_ms is None or s.latency_ms is None:
            continue
        if s.ttft_ms > s.latency_ms:
            ttft_greater_than_latency += 1
        decode_time_ms = s.latency_ms - s.ttft_ms
        if decode_time_ms > 0:
            decode_time_values.append(decode_time_ms)
        ct = float(s.completion_tokens or 0)
        d_tps = calculate_decode_tokens_per_second(ct, decode_time_ms)
        if d_tps is None:
            decode_tps_unavailable += 1
        else:
            decode_tps_values.append(d_tps)
        e2e_tps = calculate_end_to_end_tokens_per_second(
            ct, s.latency_ms
        )
        if e2e_tps is not None:
            e2e_tps_values.append(e2e_tps)

    warnings: list[str] = []
    notes = tuple(
        dict.fromkeys(
            note for sample in measurements for note in sample.notes
        )
    )
    if errors:
        warnings.append(f"{errors} measured request(s) failed")
    if ttft_greater_than_latency:
        warnings.append(
            f"{ttft_greater_than_latency} sample(s) had TTFT greater "
            "than total latency"
        )
    if decode_tps_unavailable:
        warnings.append(
            "decode tokens/sec could not be computed for "
            f"{decode_tps_unavailable} sample(s)"
        )

    latency_mean_ms = mean(latency_values)
    latency_p50_ms = percentile(latency_values, 50)
    ttft_mean_ms = mean(ttft_values)
    ttft_p50_ms = percentile(ttft_values, 50)
    if (
        latency_mean_ms is not None
        and latency_p50_ms is not None
        and latency_mean_ms > 0
        and abs(latency_mean_ms - latency_p50_ms) / latency_mean_ms
        > 0.25
    ):
        warnings.append(
            "latency_p50_ms differs materially from latency_mean_ms; "
            "distribution may be skewed"
        )
    if len(latency_values) < P95_MIN_SAMPLES:
        warnings.append(
            f"latency_p95_ms unavailable with only "
            f"{len(latency_values)} successful sample(s)"
        )
    if len(latency_values) < P99_MIN_SAMPLES:
        warnings.append(
            f"latency_p99_ms unavailable with only "
            f"{len(latency_values)} successful sample(s)"
        )
    if not successful:
        warnings.append(
            "no successful measured requests; aggregate metrics unavailable"
        )

    return BenchmarkResult(
        backend=backend,
        samples=len(successful),
        errors=errors,
        error_rate=(
            (errors / len(measurements)) if measurements else 0.0
        ),
        ttft_mean_ms=ttft_mean_ms,
        ttft_p50_ms=ttft_p50_ms,
        ttft_p95_ms=percentile(
            ttft_values, 95, min_samples=P95_MIN_SAMPLES
        ),
        ttft_p99_ms=percentile(
            ttft_values, 99, min_samples=P99_MIN_SAMPLES
        ),
        latency_mean_ms=latency_mean_ms,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=percentile(
            latency_values, 95, min_samples=P95_MIN_SAMPLES
        ),
        latency_p99_ms=percentile(
            latency_values, 99, min_samples=P99_MIN_SAMPLES
        ),
        prompt_tokens_mean=mean(prompt_token_values),
        completion_tokens_mean=mean(completion_token_values),
        completion_tokens_p50=percentile(completion_token_values, 50),
        total_tokens_mean=mean(
            [
                p + c
                for p, c in zip(prompt_token_values, completion_token_values)
                if p is not None and c is not None
            ]
        ),
        decode_time_mean_ms=mean(decode_time_values),
        latency_per_completion_token_ms=calculate_per_token_latency_ms(
            latency_mean_ms, mean(completion_token_values)
        ),
        decode_time_per_completion_token_ms=calculate_per_token_latency_ms(
            mean(decode_time_values), mean(completion_token_values)
        ),
        latency_p50_per_completion_token_ms=calculate_per_token_latency_ms(
            latency_p50_ms, percentile(completion_token_values, 50)
        ),
        decode_tokens_per_second_mean=mean(decode_tps_values),
        decode_tokens_per_second_p50=percentile(decode_tps_values, 50),
        end_to_end_tokens_per_second_mean=mean(e2e_tps_values),
        end_to_end_tokens_per_second_p50=percentile(e2e_tps_values, 50),
        image_preprocess_latency_ms_mean=mean(image_preprocess_values),
        image_count_mean=mean(image_count_values),
        vlm_load_time_ms=load_time_ms,
        notes=notes if summarize_distribution else notes,
        warnings=tuple(dict.fromkeys(warnings)),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _result_summary(
    model_name: str, result: BenchmarkResult
) -> str:
    """Render a compact per-backend result line."""
    output_tps = 0.0
    if (
        result.latency_mean_ms is not None
        and result.latency_mean_ms > 0
        and result.completion_tokens_mean is not None
    ):
        output_tps = result.completion_tokens_mean / (
            result.latency_mean_ms / 1000.0
        )
    latency_text = (
        f"{result.latency_mean_ms:.1f} ms"
        if result.latency_mean_ms is not None
        else "n/a"
    )
    ttft_text = (
        f"{result.ttft_mean_ms:.1f} ms"
        if result.ttft_mean_ms is not None
        else "n/a"
    )
    return (
        f"[VLM model {model_name}] {result.backend} done: "
        f"latency_mean={latency_text}, "
        f"ttft_mean={ttft_text}, "
        f"samples={result.samples}, errors={result.errors}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
