#!/usr/bin/env python3
"""Benchmark raw `mlx-lm`, `mlx_lm.server`, and this project."""

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
from typing import Any, Iterable, TextIO
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULT_MODELS = [
    "mlx-community/LFM2.5-8B-A1B-MLX-4bit",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/gemma-3-270m-it-qat-8bit",
]
DEFAULT_PROMPT = "Say hello in one short sentence."
SHORT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing briefly.",
    "Write a haiku about programming.",
    "What is 2+2?",
    "Summarize the theory of relativity.",
    "Define recursion in programming.",
    "What is the speed of light?",
    "What is the boiling point of water?",
]
LONG_BASES = [
    (
        "Explain the development of transformer models, including self-attention, "
        "multi-head attention, scaling laws, inference tradeoffs, and deployment."
    ),
    (
        "Describe how modern GPU and Metal-based inference stacks schedule prompt "
        "processing, decode, KV cache management, batching, and memory movement."
    ),
]
SYSTEM_PREFIX = (
    "You are an expert AI assistant with deep knowledge in science, engineering, "
    "history, and mathematics. Give precise, structured answers."
)
SHARED_STARTERS = [
    "User: Explain neural networks.\nAssistant:",
    "User: How do transformers work?\nAssistant:",
]
SHARED_SUFFIXES = [
    "Compare them to classical methods.",
    "What are the main limitations?",
    "Give a practical example.",
    "Explain the training objective.",
    "Discuss recent advances.",
    "What are common misconceptions?",
    "How is inference optimized?",
    "What should a beginner read first?",
]

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from mlx_worker.benchmarking import (  # noqa: E402
    BenchmarkResult,
    BenchmarkRun,
    P95_MIN_SAMPLES,
    P99_MIN_SAMPLES,
    calculate_decode_tokens_per_second,
    calculate_end_to_end_tokens_per_second,
    mean,
    now_utc_iso,
    percentile,
    write_report_suite,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - benchmark still works without tqdm
    _tqdm = None


@dataclass(frozen=True)
class StreamResult:
    """Captured latency and token data for a streaming request."""

    ttft_ms: float | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    text: str
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Return True when the sample should be included in aggregates."""

        return self.error is None


@dataclass(frozen=True)
class PromptCase:
    """One prompt workload case for all compared backends."""

    name: str
    messages: list[dict[str, str]]
    prompt_input: str | list[int]
    prompt_tokens: int


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
    """Write a timestamped benchmark status line to stderr."""

    target = stream or sys.stderr
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=target, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt to benchmark. Repeat for multiple prompts. Defaults to a built-in suite.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=8,
        help="Limit built-in prompt suite size. Use 0 for all built-in prompts.",
    )
    parser.add_argument(
        "--include-long-prompts",
        action="store_true",
        help="Include long-prefill prompts in the built-in suite.",
    )
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--long-prompt-multiplier", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "phase_6_report.md",
    )
    parser.add_argument("--project-port", type=int, default=8000)
    parser.add_argument("--server-port", type=int, default=8001)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.report_path.is_absolute():
        args.report_path = REPO_ROOT / args.report_path

    from mlx_lm import load

    models = args.models or DEFAULT_MODELS
    runs: list[BenchmarkRun] = []
    _log_event(
        "benchmark start: "
        f"{len(models)} model(s), warmup_trials={args.warmup_trials}, "
        f"trials={args.trials}, prompt_limit={args.prompt_limit}, "
        f"include_long_prompts={args.include_long_prompts}, max_tokens={args.max_tokens}"
    )

    for model_index, model_name in enumerate(models, start=1):
        _log_event(f"[model {model_index}/{len(models)}] loading {model_name}")
        model, tokenizer = load(model_name)
        prompt_cases = _build_prompt_cases(
            tokenizer,
            args.prompts
            or _default_prompt_suite(
                tokenizer,
                include_long=args.include_long_prompts,
                prompt_limit=args.prompt_limit,
                prefill_step_size=args.prefill_step_size,
                long_prompt_multiplier=args.long_prompt_multiplier,
            ),
        )
        _log_event(
            f"[model {model_index}/{len(models)}] prompt suite ready: "
            f"{len(prompt_cases)} case(s), "
            f"{sum(case.prompt_tokens for case in prompt_cases)} prompt tokens total"
        )

        raw_result = _benchmark_raw_mlx_lm(
            model,
            tokenizer,
            prompt_cases,
            args.max_tokens,
            args.warmup_trials,
            args.trials,
        )
        _log_event(_result_summary(model_name, raw_result))
        server_result = _benchmark_mlx_lm_server(
            model_name,
            prompt_cases,
            tokenizer,
            args.max_tokens,
            args.server_port,
            args.warmup_trials,
            args.trials,
        )
        _log_event(_result_summary(model_name, server_result))
        project_result = _benchmark_project(
            model_name,
            prompt_cases,
            tokenizer,
            args.max_tokens,
            args.project_port,
            args.warmup_trials,
            args.trials,
        )
        _log_event(_result_summary(model_name, project_result))

        runs.append(
            BenchmarkRun(
                model=model_name,
                prompt=_prompt_summary(prompt_cases),
                max_tokens=args.max_tokens,
                generated_at=now_utc_iso(),
                results=(raw_result, server_result, project_result),
            )
        )

    write_report_suite(args.report_path, runs)
    _log_event(f"report_written={args.report_path}")
    print(f"report_written={args.report_path}")
    return 0


def _benchmark_raw_mlx_lm(
    model: Any,
    tokenizer: Any,
    prompt_cases: list[PromptCase],
    max_tokens: int,
    warmup_trials: int,
    trials: int,
) -> BenchmarkResult:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0, top_p=1.0)
    warmup_total = warmup_trials * len(prompt_cases)
    warmup_tracker = _ProgressTracker("raw mlx-lm warmup", warmup_total)
    for warmup_index in range(1, warmup_trials + 1):
        for prompt_case in prompt_cases:
            _ = _stream_generate_once(
                lambda prompt_input=prompt_case.prompt_input: stream_generate(
                    model,
                    tokenizer,
                    prompt_input,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ),
                tokenizer,
                prompt_case.prompt_tokens,
                max_tokens,
            )
            warmup_tracker.advance(
                _progress_detail(
                    prompt_case, phase_index=warmup_index, phase_total=warmup_trials
                )
            )
    warmup_tracker.close()

    trial_total = trials * len(prompt_cases)
    trial_tracker = _ProgressTracker("raw mlx-lm measured trials", trial_total)
    measurements: list[StreamResult] = []
    for trial_index in range(1, trials + 1):
        for prompt_case in prompt_cases:
            try:
                measurements.append(
                    _stream_generate_once(
                        lambda prompt_input=prompt_case.prompt_input: stream_generate(
                            model,
                            tokenizer,
                            prompt_input,
                            max_tokens=max_tokens,
                            sampler=sampler,
                        ),
                        tokenizer,
                        prompt_case.prompt_tokens,
                        max_tokens,
                    )
                )
            except Exception as exc:
                measurements.append(
                    _failed_stream_result(prompt_case.prompt_tokens, exc)
                )
                _log_event(
                    f"[raw mlx-lm] measured request failed for {prompt_case.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
            trial_tracker.advance(
                _progress_detail(
                    prompt_case, phase_index=trial_index, phase_total=trials
                )
            )
    trial_tracker.close()
    return _reduce_measurements("raw mlx-lm", measurements, summarize_distribution=True)


def _benchmark_mlx_lm_server(
    model: str,
    prompt_cases: list[PromptCase],
    tokenizer: Any,
    max_tokens: int,
    port: int,
    warmup_trials: int,
    trials: int,
) -> BenchmarkResult:
    command_variants = [
        [
            sys.executable,
            "-m",
            "mlx_lm.server",
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
            "mlx_lm.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            model,
        ],
    ]
    return _benchmark_http_service(
        "mlx_lm.server",
        command_variants,
        f"http://127.0.0.1:{port}",
        model,
        prompt_cases,
        tokenizer,
        max_tokens,
        warmup_trials,
        trials,
    )


def _benchmark_project(
    model: str,
    prompt_cases: list[PromptCase],
    tokenizer: Any,
    max_tokens: int,
    port: int,
    warmup_trials: int,
    trials: int,
) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="mlx-benchmark-project-") as tmpdir_str:
        tmpdir_path = Path(tmpdir_str)
        config_path = _prepare_project_config(model, port, config_dir=tmpdir_path)
        command_variants = [
            [
                "cargo",
                "run",
                "--release",
                "-p",
                "mlx_runtime_gateway",
            ]
        ]
        return _benchmark_http_service(
            "this project",
            command_variants,
            f"http://127.0.0.1:{port}",
            model,
            prompt_cases,
            tokenizer,
            max_tokens,
            warmup_trials,
            trials,
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
        )


def _benchmark_http_service(
    backend_name: str,
    command_variants: list[list[str]],
    base_url: str,
    model: str,
    prompt_cases: list[PromptCase],
    tokenizer: Any,
    max_tokens: int,
    warmup_trials: int,
    trials: int,
    *,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    readiness_url: str | None = None,
) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="mlx-benchmark-") as tmpdir:
        stdout_path = Path(tmpdir) / f"{backend_name.replace(' ', '_')}.stdout.log"
        stderr_path = Path(tmpdir) / f"{backend_name.replace(' ', '_')}.stderr.log"
        last_error: str | None = None
        port = _extract_port(base_url)

        for variant_index, command in enumerate(command_variants, start=1):
            _log_event(
                f"[{backend_name}] launch attempt {variant_index}/{len(command_variants)}: "
                + " ".join(command)
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
                    timeout_s=300,
                    label=backend_name,
                ):
                    last_error = _read_process_failure(stderr_path, stdout_path) or (
                        f"timed out waiting for {backend_name} to open its port"
                    )
                    _log_event(f"[{backend_name}] launch failed: {last_error}")
                    continue

                readiness_case = prompt_cases[0]
                _wait_for_service_ready(
                    base_url,
                    readiness_url,
                    model,
                    readiness_case.messages,
                    max_tokens,
                    timeout_s=300,
                    label=backend_name,
                )
                warmup_total = warmup_trials * len(prompt_cases)
                warmup_tracker = _ProgressTracker(
                    f"{backend_name} warmup", warmup_total
                )
                for warmup_index in range(1, warmup_trials + 1):
                    for prompt_case in prompt_cases:
                        _request_completion(
                            base_url,
                            model,
                            prompt_case.messages,
                            max_tokens,
                            tokenizer,
                            prompt_case.prompt_tokens,
                        )
                        warmup_tracker.advance(
                            _progress_detail(
                                prompt_case,
                                phase_index=warmup_index,
                                phase_total=warmup_trials,
                            )
                        )
                warmup_tracker.close()

                trial_total = trials * len(prompt_cases)
                trial_tracker = _ProgressTracker(
                    f"{backend_name} measured trials", trial_total
                )
                measurements: list[StreamResult] = []
                for trial_index in range(1, trials + 1):
                    for prompt_case in prompt_cases:
                        try:
                            measurements.append(
                                _request_completion(
                                    base_url,
                                    model,
                                    prompt_case.messages,
                                    max_tokens,
                                    tokenizer,
                                    prompt_case.prompt_tokens,
                                )
                            )
                        except Exception as exc:
                            measurements.append(
                                _failed_stream_result(prompt_case.prompt_tokens, exc)
                            )
                            _log_event(
                                f"[{backend_name}] measured request failed for "
                                f"{prompt_case.name}: {type(exc).__name__}: {exc}"
                            )
                        trial_tracker.advance(
                            _progress_detail(
                                prompt_case,
                                phase_index=trial_index,
                                phase_total=trials,
                            )
                        )
                trial_tracker.close()
                return _reduce_measurements(
                    backend_name, measurements, summarize_distribution=True
                )
            except Exception as exc:
                last_error = str(exc)
                _log_event(f"[{backend_name}] benchmark attempt failed: {last_error}")
            finally:
                pgid = os.getpgid(process.pid)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                    process.wait(timeout=30)
                stdout_file.close()
                stderr_file.close()

        raise RuntimeError(
            f"{backend_name} benchmark failed after trying all launch variants: {last_error}"
        )


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


def _wait_for_service_ready(
    base_url: str,
    readiness_url: str | None,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    timeout_s: int,
    label: str,
) -> None:
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
                with urlopen(Request(f"{base_url}{readiness_url}"), timeout=10) as resp:
                    if resp.status < 500:
                        _log_event(f"[{label}] readiness endpoint accepted requests")
                        return
                    last_error = f"readiness endpoint returned HTTP {resp.status}"
            else:
                _request_completion(
                    base_url,
                    model,
                    messages,
                    max_tokens,
                    tokenizer=None,
                    prompt_tokens=None,
                )
                _log_event(f"[{label}] streaming endpoint accepted requests")
                return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
        now = time.monotonic()
    raise RuntimeError(f"service did not become ready: {last_error}")


def _request_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    tokenizer: Any | None,
    prompt_tokens: int | None,
) -> StreamResult:
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

    with urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            usage = payload.get("usage")
            if usage and isinstance(usage, dict):
                completion_tokens = int(
                    usage.get("completion_tokens", completion_tokens)
                )
                if prompt_tokens is None:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
            choice = payload["choices"][0]
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
    if prompt_tokens is None and tokenizer is not None:
        prompt_tokens = len(_tokenize_prompt(tokenizer, messages))
    if completion_tokens == 0 and tokenizer is not None:
        completion_tokens = len(tokenizer.encode(final_text, add_special_tokens=False))

    return StreamResult(
        ttft_ms=(first_delta_at - start) * 1000.0,
        latency_ms=(end - start) * 1000.0,
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens,
        text=final_text,
    )


def _stream_generate_once(
    factory, tokenizer, prompt_tokens: int, max_tokens: int
) -> StreamResult:
    start = time.perf_counter()
    first_delta_at: float | None = None
    text_parts: list[str] = []
    prompt_count = prompt_tokens
    completion_count = 0

    for response in factory():
        if first_delta_at is None:
            first_delta_at = time.perf_counter()
        text_parts.append(getattr(response, "text", ""))
        prompt_count = int(getattr(response, "prompt_tokens", prompt_count))
        completion_count = int(getattr(response, "generation_tokens", completion_count))

    end = time.perf_counter()
    final_text = "".join(text_parts)
    if first_delta_at is None:
        first_delta_at = end
    if completion_count == 0:
        completion_count = len(tokenizer.encode(final_text, add_special_tokens=False))

    return StreamResult(
        ttft_ms=(first_delta_at - start) * 1000.0,
        latency_ms=(end - start) * 1000.0,
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        text=final_text,
    )


def _failed_stream_result(prompt_tokens: int, exc: BaseException) -> StreamResult:
    """Build a failed measured sample that is excluded from aggregates."""

    return StreamResult(
        ttft_ms=None,
        latency_ms=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=None,
        text="",
        error=f"{type(exc).__name__}: {exc}",
    )


def _reduce_measurements(
    backend: str,
    measurements: Iterable[StreamResult],
    *,
    summarize_distribution: bool = False,
) -> BenchmarkResult:
    measurements = list(measurements)
    if not measurements:
        raise ValueError(f"{backend} produced no benchmark measurements")

    successful = [sample for sample in measurements if sample.succeeded]
    errors = len(measurements) - len(successful)
    ttft_values = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
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
    total_token_values = [
        float(sample.prompt_tokens + sample.completion_tokens)
        for sample in successful
        if sample.prompt_tokens is not None and sample.completion_tokens is not None
    ]

    decode_time_values: list[float] = []
    decode_tps_values: list[float] = []
    e2e_tps_values: list[float] = []
    ttft_greater_than_latency = 0
    decode_tps_unavailable = 0
    for sample in successful:
        if sample.ttft_ms is None or sample.latency_ms is None:
            continue
        if sample.ttft_ms > sample.latency_ms:
            ttft_greater_than_latency += 1
        decode_time_ms = sample.latency_ms - sample.ttft_ms
        if decode_time_ms > 0:
            decode_time_values.append(decode_time_ms)
        completion_tokens = float(sample.completion_tokens or 0)
        decode_tps = calculate_decode_tokens_per_second(completion_tokens, decode_time_ms)
        if decode_tps is None:
            decode_tps_unavailable += 1
        else:
            decode_tps_values.append(decode_tps)
        e2e_tps = calculate_end_to_end_tokens_per_second(
            completion_tokens, sample.latency_ms
        )
        if e2e_tps is not None:
            e2e_tps_values.append(e2e_tps)

    warnings: list[str] = []
    notes = tuple(dict.fromkeys(note for sample in measurements for note in sample.notes))
    if errors:
        warnings.append(f"{errors} measured request(s) failed")
    if ttft_greater_than_latency:
        warnings.append(
            f"{ttft_greater_than_latency} sample(s) had TTFT greater than total latency"
        )
    if decode_tps_unavailable:
        warnings.append(
            f"decode tokens/sec could not be computed for {decode_tps_unavailable} sample(s)"
        )

    latency_mean_ms = mean(latency_values)
    latency_p50_ms = percentile(latency_values, 50)
    ttft_mean_ms = mean(ttft_values)
    ttft_p50_ms = percentile(ttft_values, 50)
    if (
        latency_mean_ms is not None
        and latency_p50_ms is not None
        and latency_mean_ms > 0
        and abs(latency_mean_ms - latency_p50_ms) / latency_mean_ms > 0.25
    ):
        warnings.append(
            "latency_p50_ms differs materially from latency_mean_ms; distribution may be skewed"
        )
    if len(latency_values) < P95_MIN_SAMPLES:
        warnings.append(
            f"latency_p95_ms and ttft_p95_ms unavailable with only {len(latency_values)} successful sample(s)"
        )
    if len(latency_values) < P99_MIN_SAMPLES:
        warnings.append(
            f"latency_p99_ms and ttft_p99_ms unavailable with only {len(latency_values)} successful sample(s)"
        )
    if not successful:
        warnings.append("no successful measured requests; aggregate metrics are unavailable")

    return BenchmarkResult(
        backend=backend,
        samples=len(successful),
        errors=errors,
        error_rate=(errors / len(measurements)) if measurements else 0.0,
        ttft_mean_ms=ttft_mean_ms,
        ttft_p50_ms=ttft_p50_ms,
        ttft_p95_ms=percentile(ttft_values, 95, min_samples=P95_MIN_SAMPLES),
        ttft_p99_ms=percentile(ttft_values, 99, min_samples=P99_MIN_SAMPLES),
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
        total_tokens_mean=mean(total_token_values),
        decode_time_mean_ms=mean(decode_time_values),
        decode_tokens_per_second_mean=mean(decode_tps_values),
        decode_tokens_per_second_p50=percentile(decode_tps_values, 50),
        end_to_end_tokens_per_second_mean=mean(e2e_tps_values),
        end_to_end_tokens_per_second_p50=percentile(e2e_tps_values, 50),
        notes=notes if summarize_distribution else notes,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _progress_detail(
    prompt_case: PromptCase, *, phase_index: int, phase_total: int
) -> str:
    """Render one concise progress label for warmup or measured trials."""

    return (
        f"trial {phase_index}/{phase_total}, "
        f"{prompt_case.name}, prompt_tokens={prompt_case.prompt_tokens}"
    )


def _result_summary(model_name: str, result: BenchmarkResult) -> str:
    """Render a compact per-backend result line for long benchmark runs."""

    output_tps = 0.0
    if (
        result.latency_mean_ms is not None
        and result.latency_mean_ms > 0
        and result.completion_tokens_mean is not None
    ):
        output_tps = result.completion_tokens_mean / (result.latency_mean_ms / 1000.0)
    latency_text = (
        f"{result.latency_mean_ms:.1f} ms"
        if result.latency_mean_ms is not None
        else "n/a"
    )
    ttft_text = f"{result.ttft_mean_ms:.1f} ms" if result.ttft_mean_ms is not None else "n/a"
    completion_text = (
        f"{result.completion_tokens_mean:.1f}"
        if result.completion_tokens_mean is not None
        else "n/a"
    )
    return (
        f"[model {model_name}] {result.backend} done: "
        f"latency_mean={latency_text}, "
        f"ttft_mean={ttft_text}, "
        f"completion_tokens_mean={completion_text}, "
        f"output_tps_mean={output_tps:.1f}, "
        f"samples={result.samples}, errors={result.errors}"
    )


def _percentile(values: Iterable[float], p: float) -> float:
    result = percentile(list(values), p)
    return 0.0 if result is None else result


def _default_prompt_suite(
    tokenizer: Any,
    *,
    include_long: bool,
    prompt_limit: int,
    prefill_step_size: int,
    long_prompt_multiplier: int,
) -> list[str]:
    prompts = [
        *SHORT_PROMPTS[:4],
        *make_shared_prefix_prompts()[:2],
        *make_partial_prefix_prompts()[:2],
        *SHORT_PROMPTS[4:],
        *make_shared_prefix_prompts()[2:],
        *make_partial_prefix_prompts()[2:],
    ]
    if include_long:
        prompts.extend(
            make_long_prompts(tokenizer, prefill_step_size, long_prompt_multiplier)
        )
    if prompt_limit > 0:
        return prompts[:prompt_limit]
    return prompts


def make_long_prompts(
    tokenizer: Any, prefill_step_size: int, multiplier: int
) -> list[str]:
    """Build long prompts sized relative to configured prefill step."""

    target = max(2 * prefill_step_size + 32, prefill_step_size * multiplier)
    prompts = []
    for base in LONG_BASES:
        text = base
        while len(tokenizer.encode(text)) < target:
            text += " " + base
        prompts.append(text)
    return prompts


def make_shared_prefix_prompts() -> list[str]:
    """Build prompts with exact shared prefixes."""

    prompts = []
    for starter in SHARED_STARTERS:
        prefix = f"{SYSTEM_PREFIX}\n{starter}"
        for suffix in SHARED_SUFFIXES:
            prompts.append(f"{prefix} {suffix}")
    return prompts


def make_partial_prefix_prompts() -> list[str]:
    """Build prompts with partially shared conversation history."""

    prompts = []
    base_histories = [
        [
            "User: Summarize neural networks.",
            "Assistant: Neural networks are layered function approximators.",
            "User: Explain backpropagation.",
        ],
        [
            "User: Explain batching in inference servers.",
            "Assistant: Batching groups requests to improve throughput.",
            "User: Describe prefix caching.",
        ],
    ]
    suffixes = [
        "Give a concrete example.",
        "What are the main tradeoffs?",
        "What goes wrong under bursty traffic?",
        "How would you benchmark this?",
    ]
    for history in base_histories:
        for keep in range(1, len(history) + 1):
            prefix = "\n".join([SYSTEM_PREFIX, *history[:keep]])
            for suffix in suffixes:
                prompts.append(f"{prefix}\nUser: {suffix}\nAssistant:")
    return prompts


def _build_prompt_cases(tokenizer: Any, prompts: list[str]) -> list[PromptCase]:
    if not prompts:
        prompts = [DEFAULT_PROMPT]
    cases = []
    for index, prompt in enumerate(prompts, start=1):
        messages = [{"role": "user", "content": prompt}]
        prompt_input = _build_prompt_input(tokenizer, messages)
        prompt_tokens = _count_prompt_tokens(tokenizer, prompt_input)
        cases.append(
            PromptCase(
                name=f"prompt-{index}",
                messages=messages,
                prompt_input=prompt_input,
                prompt_tokens=prompt_tokens,
            )
        )
    return cases


def _count_prompt_tokens(tokenizer: Any, prompt_input: str | list[int]) -> int:
    if isinstance(prompt_input, list):
        return len(prompt_input)
    tokens = tokenizer.encode(prompt_input, add_special_tokens=False)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return len(tokens)


def _prompt_summary(prompt_cases: list[PromptCase]) -> str:
    if len(prompt_cases) == 1:
        return prompt_cases[0].messages[-1]["content"]
    total_prompt_tokens = sum(case.prompt_tokens for case in prompt_cases)
    return f"prompt suite: {len(prompt_cases)} cases, {total_prompt_tokens} prompt tokens total"


def _build_prompt_input(
    tokenizer: Any, messages: list[dict[str, str]]
) -> str | list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return list(tokens)

    return (
        "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        )
        + "\nAssistant:"
    )


def _tokenize_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    prompt_input = _build_prompt_input(tokenizer, messages)
    if isinstance(prompt_input, list):
        return prompt_input
    tokens = tokenizer.encode(prompt_input, add_special_tokens=False)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return list(tokens)


def _merged_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return env


def _prepare_project_config(
    model: str, port: int, config_dir: Path | None = None
) -> Path:
    source = REPO_ROOT / "config" / "runtime.toml"
    if config_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="mlx-runtime-config-"))
    else:
        temp_dir = config_dir / "config"
        temp_dir.mkdir(exist_ok=True)
    target = temp_dir / "runtime.toml"
    text = source.read_text(encoding="utf-8")
    text = _replace_config_value(text, "model", model)
    text = _replace_config_value(text, "port", str(port))
    text = _replace_config_value(
        text,
        "ipc_path",
        str(temp_dir / "mlx-runtime.sock"),
    )
    target.write_text(text, encoding="utf-8")
    return target


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


def _extract_port(base_url: str) -> int:
    return int(base_url.rsplit(":", 1)[1])


def _read_process_failure(stderr_path: Path, stdout_path: Path) -> str | None:
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    if stderr:
        return stderr.splitlines()[-1]
    if stdout:
        return stdout.splitlines()[-1]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
