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
from typing import Any, Iterable
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_PROMPT = "Say hello in one short sentence."

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from mlx_worker.benchmarking import (  # noqa: E402
    BenchmarkResult,
    BenchmarkRun,
    now_utc_iso,
    write_report,
)


@dataclass(frozen=True)
class StreamResult:
    """Captured latency and token data for a streaming request."""

    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    text: str
    notes: tuple[str, ...] = ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16)
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

    from mlx_lm import load

    model, tokenizer = load(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    prompt_input = _build_prompt_input(tokenizer, messages)
    prompt_tokens = len(_tokenize_prompt(tokenizer, messages))

    raw_result = _benchmark_raw_mlx_lm(
        model,
        tokenizer,
        prompt_input,
        prompt_tokens,
        args.max_tokens,
        args.warmup_trials,
        args.trials,
    )
    server_result = _benchmark_mlx_lm_server(
        args.model,
        messages,
        tokenizer,
        prompt_tokens,
        args.max_tokens,
        args.server_port,
        args.warmup_trials,
        args.trials,
    )
    project_result = _benchmark_project(
        args.model,
        messages,
        tokenizer,
        prompt_tokens,
        args.max_tokens,
        args.project_port,
        args.warmup_trials,
        args.trials,
    )

    run = BenchmarkRun(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        generated_at=now_utc_iso(),
        results=(raw_result, server_result, project_result),
    )
    write_report(args.report_path, run)
    print(f"report_written={args.report_path}")
    return 0


def _benchmark_raw_mlx_lm(
    model: Any,
    tokenizer: Any,
    prompt_input: str | list[int],
    prompt_tokens: int,
    max_tokens: int,
    warmup_trials: int,
    trials: int,
) -> BenchmarkResult:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0, top_p=1.0)
    for _ in range(warmup_trials):
        _ = _stream_generate_once(
            lambda: stream_generate(
                model,
                tokenizer,
                prompt_input,
                max_tokens=max_tokens,
                sampler=sampler,
            ),
            tokenizer,
            prompt_tokens,
            max_tokens,
        )

    measurements = [
        _stream_generate_once(
            lambda: stream_generate(
                model,
                tokenizer,
                prompt_input,
                max_tokens=max_tokens,
                sampler=sampler,
            ),
            tokenizer,
            prompt_tokens,
            max_tokens,
        )
        for _ in range(trials)
    ]
    return _reduce_measurements("raw mlx-lm", measurements)


def _benchmark_mlx_lm_server(
    model: str,
    messages: list[dict[str, str]],
    tokenizer: Any,
    prompt_tokens: int,
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
        messages,
        tokenizer,
        prompt_tokens,
        max_tokens,
        warmup_trials,
        trials,
    )


def _benchmark_project(
    model: str,
    messages: list[dict[str, str]],
    tokenizer: Any,
    prompt_tokens: int,
    max_tokens: int,
    port: int,
    warmup_trials: int,
    trials: int,
) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="mlx-benchmark-project-") as tmpdir_str:
        tmpdir_path = Path(tmpdir_str)
        config_path = _prepare_project_config(port, config_dir=tmpdir_path)
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
            messages,
            tokenizer,
            prompt_tokens,
            max_tokens,
            warmup_trials,
            trials,
            cwd=REPO_ROOT,
            extra_env={"MLX_RUNTIME_CONFIG": str(config_path)},
            readiness_url="/health",
        )


def _benchmark_http_service(
    backend_name: str,
    command_variants: list[list[str]],
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    tokenizer: Any,
    prompt_tokens: int,
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

        for command in command_variants:
            if not _is_port_free("127.0.0.1", port):
                last_error = f"port {port} already in use before launch"
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
                ):
                    last_error = _read_process_failure(stderr_path, stdout_path) or (
                        f"timed out waiting for {backend_name} to open its port"
                    )
                    continue

                _wait_for_service_ready(
                    base_url, readiness_url, model, messages, max_tokens, timeout_s=300
                )
                for _ in range(warmup_trials):
                    _request_completion(
                        base_url, model, messages, max_tokens, tokenizer, prompt_tokens
                    )

                measurements = [
                    _request_completion(
                        base_url, model, messages, max_tokens, tokenizer, prompt_tokens
                    )
                    for _ in range(trials)
                ]
                return _reduce_measurements(backend_name, measurements)
            except Exception as exc:
                last_error = str(exc)
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
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
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


def _wait_for_service_ready(
    base_url: str,
    readiness_url: str | None,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            if readiness_url:
                with urlopen(Request(f"{base_url}{readiness_url}"), timeout=10) as resp:
                    if resp.status < 500:
                        return
            else:
                _request_completion(
                    base_url,
                    model,
                    messages,
                    max_tokens,
                    tokenizer=None,
                    prompt_tokens=None,
                )
                return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2)
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
                delta.get("content") or choice.get("message", {}).get("content") or ""
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


def _reduce_measurements(
    backend: str, measurements: Iterable[StreamResult]
) -> BenchmarkResult:
    measurements = list(measurements)
    if not measurements:
        raise ValueError(f"{backend} produced no benchmark measurements")

    ttft = sum(sample.ttft_ms for sample in measurements) / len(measurements)
    latency = sum(sample.latency_ms for sample in measurements) / len(measurements)
    prompt_tokens = measurements[-1].prompt_tokens
    completion_tokens = measurements[-1].completion_tokens
    notes = tuple(
        dict.fromkeys(note for sample in measurements for note in sample.notes)
    )
    return BenchmarkResult(
        backend=backend,
        ttft_ms=ttft,
        latency_ms=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        notes=notes,
    )


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


def _prepare_project_config(port: int, config_dir: Path | None = None) -> Path:
    source = REPO_ROOT / "config" / "runtime.toml"
    if config_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="mlx-runtime-config-"))
    else:
        temp_dir = config_dir / "config"
        temp_dir.mkdir(exist_ok=True)
    target = temp_dir / "runtime.toml"
    text = source.read_text(encoding="utf-8")
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
