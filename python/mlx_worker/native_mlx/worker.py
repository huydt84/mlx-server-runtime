"""Native MLX worker transport and composition root."""

from __future__ import annotations

import inspect
import select
import sys
import time
from collections.abc import Callable
from socket import socket
from typing import Any

from ..config import WorkerConfig
from ..ipc import (
    BenchmarkResetRequest,
    BenchmarkResetState,
    CancelRequest,
    ModelLoadProgress,
    ModelStatus,
    TranscriptionRequest,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_command,
    encode_bootstrap_message,
    encode_event,
)
from ..tool_calling import request_error_code, resolve_tool_format
from .bootstrap import (
    NativeBootstrapFailure,
    build_native_artifacts,
    detect_native_architecture,
)
from .diagnostics import (
    NativeParityResult,
    NativePrefillDecodeParityResult,
    build_prompt_fingerprint,
    compare_native_prefill_decode_to_mlx_lm,
    compare_native_to_mlx_lm,
    trace_native_debug_to_mlx_lm,
)
from .interfaces import NativeRuntime
from .memory import KVMemorySettings
from .pipeline_profile import PipelineProfiler
from .runtime import NativeRuntime as NativeRequestRuntime
from .scheduler import NativeContinuousScheduler

_RECV_BYTES = 1 << 20

__all__ = [
    "NativeParityResult",
    "NativePrefillDecodeParityResult",
    "apply_mlx_cache_limit",
    "build_prompt_fingerprint",
    "compare_native_prefill_decode_to_mlx_lm",
    "compare_native_to_mlx_lm",
    "create_native_worker",
    "detect_native_architecture",
    "run_native_worker",
    "trace_native_debug_to_mlx_lm",
]


def run_native_worker(
    client: socket,
    config: WorkerConfig,
    *,
    native_worker_factory: Callable[..., NativeRuntime] | None = None,
) -> int:
    """Run native bootstrap and transport commands to the request runtime."""

    started = _now()
    if getattr(config, "vlm_model", None):
        _emit_failure(
            client,
            config.model,
            started,
            NativeBootstrapFailure(
                _model_error(
                    "VLM_UNSUPPORTED",
                    f"native-mlx does not serve vision-language models, so "
                    f"vlm_model {config.vlm_model!r} cannot be loaded",
                    "backend_selection",
                    "invalid_configuration",
                    str(config.vlm_model),
                )
            ),
        )
        return 1
    stage_callback = _stage_emitter(client, config.model, started)
    try:
        stage_callback("architecture_detection", "verifying")
        factory = native_worker_factory or create_native_worker
        if "stage_callback" in inspect.signature(factory).parameters:
            runtime = factory(config, stage_callback=stage_callback)
        else:
            runtime = factory(config)
        stage_callback("deterministic_warmup", "warming_up")
        runtime.warmup()
    except NativeBootstrapFailure as exc:
        _emit_failure(client, config.model, started, exc)
        return 1
    except Exception as exc:
        failure = NativeBootstrapFailure(
            _model_error(
                "NATIVE_STARTUP_FAILED",
                str(exc),
                "native_executor_construction",
                "supported_class_bug",
                config.model,
            )
        )
        _emit_failure(client, config.model, started, failure)
        return 1

    loaded_at = _now()
    client.sendall(
        encode_bootstrap_message(
            ModelStatus(
                model=config.model,
                revision=config.model,
                state="ready",
                ready=True,
                servable=True,
                progress=ModelLoadProgress(current_phase="deterministic_warmup"),
                device="mps",
                dtype="float16",
                loaded_at=loaded_at,
                started_loading_at=started,
                last_transition_at=loaded_at,
                last_error=None,
                warmup_passed=True,
                last_warmup_at=loaded_at,
                last_warmup_latency_ms=runtime.last_warmup_latency_ms,
            )
        )
    )
    client.sendall(encode_bootstrap_message(WorkerReady()))

    from ..transcription import TranscriptionService

    transcription = TranscriptionService(getattr(config, "transcription_model", None))
    buffer = bytearray()
    while True:
        raw = _read_line(client, buffer, block=runtime.idle())
        while raw is not None:
            if not raw:
                runtime.close()
                return 0
            try:
                decode_started_ns = time.perf_counter_ns()
                command = decode_command(raw)
                if command is None:
                    raise ValueError("unsupported worker command")
                if isinstance(command, CancelRequest):
                    runtime.cancel(command.request_id)
                elif isinstance(command, TranscriptionRequest):
                    client.sendall(encode_event(transcription.handle(command)))
                elif isinstance(command, BenchmarkResetRequest):
                    if not runtime.idle():
                        raise RuntimeError("benchmark reset requires an idle scheduler")
                    cache_state = runtime.benchmark_reset(
                        clear_cache=command.clear_cache,
                        reset_counters=command.reset_counters,
                    )
                    client.sendall(
                        encode_event(
                            BenchmarkResetState(
                                request_id=command.request_id,
                                scheduler_idle=True,
                                cache_state=cache_state,
                            )
                        )
                    )
                else:
                    record_transport = getattr(runtime, "record_transport", None)
                    if callable(record_transport):
                        record_transport(
                            command.request_id,
                            "ipc_decode",
                            started_ns=decode_started_ns,
                        )
                    runtime.submit(command)
            except Exception as exc:
                client.sendall(
                    encode_event(
                        WorkerCommandError(
                            code=(
                                "BENCHMARK_BUSY"
                                if isinstance(
                                    locals().get("command"), BenchmarkResetRequest
                                )
                                else request_error_code(exc)
                            ),
                            request_id=getattr(command, "request_id", "unknown")
                            if "command" in locals()
                            else "unknown",
                            message=str(exc),
                        )
                    )
                )
            raw = _read_line(client, buffer, block=False)
        if not runtime.idle():
            for event in runtime.tick():
                send_started_ns = time.perf_counter_ns()
                client.sendall(encode_event(event.payload))
                request_id = getattr(event.payload, "request_id", None)
                record_transport = getattr(runtime, "record_transport", None)
                if request_id is not None and callable(record_transport):
                    record_transport(
                        request_id,
                        "ipc_encode_send",
                        started_ns=send_started_ns,
                    )
                    flush_profile = getattr(runtime, "flush_profile", None)
                    if callable(flush_profile):
                        flush_profile()


def apply_mlx_cache_limit(limit_bytes: int) -> None:
    """Bound the MLX buffer cache that keeps freed device buffers for reuse.

    Batch and sequence shapes change every step, so the default cache, sized to
    the whole device memory limit, fills with buffers that are never reused
    and pushes the rest of the host into swap.

    Args:
        limit_bytes: Maximum bytes of freed buffers MLX may retain.
    """

    import mlx.core as mx

    mx.set_cache_limit(limit_bytes)


def create_native_worker(
    config: WorkerConfig,
    *,
    stage_callback: Callable[[str, str], None] | None = None,
    **_: object,
) -> NativeRuntime:
    """Compose bootstrap, shared executor, scheduler, and request runtime."""

    apply_mlx_cache_limit(config.native_mlx_cache_limit_bytes)
    prefill_rows = getattr(
        config, "text_prompt_concurrency", getattr(config, "prompt_concurrency", 4)
    )
    prefill_chunk = getattr(
        config, "text_prefill_chunk_size", getattr(config, "prefill_chunk_size", 256)
    )
    memory = KVMemorySettings(
        utilization=getattr(config, "memory_utilization", 0.5),
        kv_cache_bytes=getattr(config, "kv_cache_bytes", None),
        max_total_tokens_per_request=getattr(
            config, "max_total_tokens_per_request", 65_536
        ),
        max_total_tokens_explicit=getattr(config, "max_total_tokens_explicit", False),
        prefill_tokens=prefill_rows * prefill_chunk,
        prefill_rows=prefill_rows,
        buffer_cache_limit_bytes=config.native_mlx_cache_limit_bytes,
    )
    artifacts = build_native_artifacts(
        config.model,
        stage_callback,
        memory=memory,
        cache_max_entries=(
            config.text_cache_max_entries
            if getattr(config, "text_cache_max_entries_explicit", False)
            else None
        ),
        kv_page_size=config.native_kv_page_size,
        execution_backend_id=config.native_execution_backend,
        execution_mode=getattr(config, "native_execution_mode", "overlap"),
        prefix_cache_strategy=config.native_prefix_cache_strategy,
        graph_profile=config.native_graph_profile,
        drafter=getattr(config, "drafter", None),
    )
    artifacts.executor.load(artifacts.options)
    if artifacts.kv_memory is not None:
        print(
            f"native-mlx memory {artifacts.kv_memory.describe()} "
            f"max_total_tokens_per_request={artifacts.max_total_tokens} "
            f"drafter={getattr(config, 'drafter', None) or 'none'}",
            file=sys.stderr,
            flush=True,
        )
    profiler = PipelineProfiler.from_environment(config.model)
    scheduler = NativeContinuousScheduler(
        artifacts.executor,
        artifacts.cache_coordinator,
        prefill_batch_size=getattr(
            config,
            "text_prompt_concurrency",
            getattr(config, "prompt_concurrency", 4),
        ),
        prefill_step_size=getattr(
            config,
            "text_prefill_chunk_size",
            getattr(config, "prefill_chunk_size", 256),
        ),
        scheduling_policy=getattr(config, "native_scheduling_policy", "fcfs"),
        execution_mode=getattr(config, "native_execution_mode", "overlap"),
        terminal_token_ids=artifacts.eos_token_ids,
        profiler=profiler,
    )
    tool_format = resolve_tool_format(
        artifacts.tokenizer, getattr(config, "tool_call_parser", "auto")
    )
    print(
        f"native-mlx tool_call_parser={tool_format.name if tool_format else 'none'}",
        file=sys.stderr,
        flush=True,
    )
    return NativeRequestRuntime(
        scheduler,
        model_ref=config.model,
        prompt_tokenizer=artifacts.tokenizer,
        decode_target=artifacts.decode_target,
        eos_token_ids=artifacts.eos_token_ids,
        profiler=profiler,
        max_total_tokens=artifacts.max_total_tokens,
        warmup_prompt_lengths=_warmup_prompt_lengths(artifacts.executor),
        tool_format=tool_format,
    )


def _warmup_prompt_lengths(executor: Any) -> tuple[int, ...]:
    """Return warmup prompt lengths that reach every paged-decode kernel variant."""

    from .paged_decode import warmup_key_counts

    kv_heads = getattr(getattr(executor, "cache_backend", None), "num_kv_heads", 0)
    return warmup_key_counts(int(kv_heads)) if kv_heads else ()


def _read_line(client: socket, buffer: bytearray, *, block: bool) -> bytes | None:
    """Read one newline-delimited frame.

    Returns ``None`` when a non-blocking read has no complete frame yet, and
    ``b""`` on EOF. Bytes of a partial frame stay in ``buffer``.
    """

    scanned = 0
    while True:
        newline = buffer.find(b"\n", scanned)
        if newline >= 0:
            line = bytes(buffer[: newline + 1])
            del buffer[: newline + 1]
            return line
        scanned = len(buffer)
        if not block and not select.select([client], [], [], 0)[0]:
            return None
        chunk = client.recv(_RECV_BYTES)
        if not chunk:
            return b""
        buffer.extend(chunk)
        block = False


def _stage_emitter(
    client: socket,
    model: str,
    started: int,
) -> Callable[[str, str], None]:
    def emit(stage: str, state: str) -> None:
        client.sendall(
            encode_bootstrap_message(
                ModelStatus(
                    model=model,
                    revision=model,
                    state=state,  # type: ignore[arg-type]
                    ready=False,
                    servable=False,
                    progress=ModelLoadProgress(current_phase=stage),
                    device=None,
                    dtype=None,
                    loaded_at=None,
                    started_loading_at=started,
                    last_transition_at=_now(),
                    last_error=None,
                    warmup_passed=False,
                    last_warmup_at=None,
                    last_warmup_latency_ms=None,
                )
            )
        )

    return emit


def _emit_failure(
    client: socket,
    model: str,
    started: int,
    failure: NativeBootstrapFailure,
) -> None:
    client.sendall(
        encode_bootstrap_message(
            ModelStatus(
                model=model,
                revision=model,
                state="failed",
                ready=False,
                servable=False,
                progress=None,
                device=None,
                dtype=None,
                loaded_at=None,
                started_loading_at=started,
                last_transition_at=_now(),
                last_error=failure.error,
                warmup_passed=False,
                last_warmup_at=None,
                last_warmup_latency_ms=None,
            )
        )
    )
    client.sendall(
        encode_bootstrap_message(
            WorkerError(failure.error.message, error=failure.error)
        )
    )


def _model_error(
    code: str,
    message: str,
    stage: str,
    category: str,
    detail: str,
):
    from ..ipc import ModelError

    return ModelError(
        code=code,
        message=f"{message}. Start with --backend v1 to serve through mlx-lm.",
        at=_now(),
        backend="native-mlx",
        stage=stage,
        category=category,
        detail=detail,
    )


def _now() -> int:
    return int(time.time())
