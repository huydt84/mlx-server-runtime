"""Native MLX worker transport and composition root."""

from __future__ import annotations

import inspect
import select
import time
from socket import socket
from typing import Callable

from ..config import WorkerConfig
from ..ipc import (
    CancelRequest,
    ModelLoadProgress,
    ModelStatus,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_command,
    encode_bootstrap_message,
    encode_event,
)
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
from .pipeline_profile import PipelineProfiler
from .runtime import NativeRuntime as NativeRequestRuntime
from .scheduler import NativeContinuousScheduler

__all__ = [
    "NativeParityResult",
    "NativePrefillDecodeParityResult",
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
                            code="INVALID_REQUEST",
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


def create_native_worker(
    config: WorkerConfig,
    *,
    stage_callback: Callable[[str, str], None] | None = None,
    **_: object,
) -> NativeRuntime:
    """Compose bootstrap, shared executor, scheduler, and request runtime."""

    artifacts = build_native_artifacts(
        config.model,
        stage_callback,
        cache_budget_bytes=config.text_cache_budget_bytes,
        cache_max_entries=config.text_cache_max_entries,
        kv_page_size=config.native_kv_page_size,
        prefix_cache_strategy=config.native_prefix_cache_strategy,
        graph_profile=config.native_graph_profile,
    )
    artifacts.executor.load(artifacts.options)
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
        profiler=profiler,
    )
    return NativeRequestRuntime(
        scheduler,
        model_ref=config.model,
        prompt_tokenizer=artifacts.tokenizer,
        decode_target=artifacts.decode_target,
        eos_token_ids=artifacts.eos_token_ids,
        profiler=profiler,
    )


def _read_line(client: socket, buffer: bytearray, *, block: bool) -> bytes | None:
    newline = buffer.find(b"\n")
    if newline >= 0:
        line = bytes(buffer[: newline + 1])
        del buffer[: newline + 1]
        return line
    if not block and not select.select([client], [], [], 0)[0]:
        return None
    chunk = client.recv(4096)
    if not chunk:
        return b""
    buffer.extend(chunk)
    return _read_line(client, buffer, block=False)


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
        message=f"{message}. Default v1 backend remains available.",
        at=_now(),
        backend="native-mlx",
        stage=stage,
        category=category,
        detail=detail,
    )


def _now() -> int:
    return int(time.time())
