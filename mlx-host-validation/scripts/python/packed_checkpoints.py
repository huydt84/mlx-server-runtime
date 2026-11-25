"""Host-only checks of LookAtMyQuant packed checkpoints."""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_worker.native_mlx.bootstrap import (
    BootstrapArtifacts,
    NativeBootstrapFailure,
    build_finalized_token_ids,
    build_native_artifacts,
)
from mlx_worker.native_mlx.interfaces import (
    ForwardBatch,
    ForwardMode,
    SamplingParams,
    SchedulableRequest,
)
from mlx_worker.native_mlx.memory import KVMemorySettings
from mlx_worker.native_mlx.scheduler import NativeContinuousScheduler

_QUESTION = "What is the capital of France? Answer in one word."
_CODE = "Write a Python function that returns the n-th Fibonacci number."
_LONG_TOKENS = 1500
_BUFFER_CACHE_BYTES = 256 << 20
_MAX_MEAN_KL = 0.01
_MIN_ARGMAX_AGREEMENT = 0.9


def main() -> None:
    """Run every check and print one key per result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--drafter", default="")
    parser.add_argument("--kv-cache-mib", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    mx.set_cache_limit(_BUFFER_CACHE_BYTES)
    memory = KVMemorySettings(utilization=0.9, kv_cache_bytes=args.kv_cache_mib << 20)

    artifacts = build_native_artifacts(args.checkpoint, memory=memory)
    _check_step_shapes(artifacts)
    _check_short_steps(artifacts)
    plain = _check_answer(artifacts, args.tokens)
    _check_window_pages(artifacts)
    _check_layout_rejection(artifacts.architecture.model_path)
    if args.drafter:
        weights_bytes = mx.get_active_memory()
        del artifacts
        gc.collect()
        mx.clear_cache()
        print(f"released_mib={(weights_bytes - mx.get_active_memory()) >> 20}")
        artifacts = build_native_artifacts(
            args.checkpoint, memory=memory, drafter=args.drafter
        )
        _check_speculation(artifacts, plain, args.tokens)
    else:
        print("speculation_skipped=1")
    print("packed_checkpoints_ok=1")


def _prompt(artifacts: BootstrapArtifacts, text: str) -> tuple[int, ...]:
    return tuple(
        build_finalized_token_ids(
            artifacts.architecture.model_path, [{"role": "user", "content": text}]
        )
    )


def _check_step_shapes(artifacts: BootstrapArtifacts) -> None:
    """Compare one forward with the same tokens split over several steps."""

    tokens = _prompt(artifacts, _CODE)
    whole = _forward(artifacts, [tokens])
    split = [tokens[:5], tokens[5:6], tokens[6:8], tokens[8:]]
    chunked = _forward(artifacts, split)
    log_whole = whole - mx.logsumexp(whole, axis=-1, keepdims=True)
    log_chunked = chunked - mx.logsumexp(chunked, axis=-1, keepdims=True)
    kl = (mx.exp(log_whole) * (log_whole - log_chunked)).sum(axis=-1).mean().item()
    same = (whole.argmax(axis=-1) == chunked.argmax(axis=-1)).astype(mx.float32)
    agreement = same.mean().item()
    print(f"step_shapes_mean_kl={kl:.6f}")
    print(f"step_shapes_argmax_agreement={agreement:.4f}")
    _require(
        kl <= _MAX_MEAN_KL and agreement >= _MIN_ARGMAX_AGREEMENT,
        "logits depend on how a prompt is split into steps",
    )
    print("step_shapes_ok=1")


def _check_short_steps(artifacts: BootstrapArtifacts) -> None:
    """Check that short steps compute exactly what one-token steps do."""

    tokens = _prompt(artifacts, _CODE)[:16]
    single = _forward(artifacts, [(token,) for token in tokens])
    short = _forward(artifacts, [tokens[:5], tokens[5:8], tokens[8:16]])
    _require(
        bool(mx.array_equal(single, short).item()),
        "short steps differ from one-token steps",
    )
    print("short_steps_ok=1")


def _forward(artifacts: BootstrapArtifacts, chunks: list[tuple[int, ...]]) -> mx.array:
    executor = artifacts.executor
    backend = executor.cache_backend
    handle = backend.create("steps")
    cache = backend.get(handle, "steps")
    outputs = []
    for chunk in chunks:
        reservation = backend.reserve_batch((cache,), (len(chunk),))
        batch = ForwardBatch(
            ForwardMode.PREFILL,
            (len(chunk),),
            (cache.size(),),
            "causal",
            executor.attention_backend.contexts(reservation, ForwardMode.PREFILL),
        )
        logits = executor.model(mx.array([chunk]), None, batch)[0]
        mx.eval(logits)
        reservation.commit()
        outputs.append(logits.astype(mx.float32))
    backend.release(handle)
    return mx.concatenate(outputs)


def _check_answer(artifacts: BootstrapArtifacts, tokens: int) -> list[int]:
    """Check a known answer and return the code prompt's plain tokens."""

    answer = _generate(artifacts, _prompt(artifacts, _QUESTION), max(tokens, 160))
    text = artifacts.tokenizer.decode(answer)
    print(f"answer_tokens={len(answer)}")
    _require("Paris" in text, f"unexpected answer {text!r}")
    print("answer_ok=1")
    return _generate(artifacts, _prompt(artifacts, _CODE), tokens)


def _check_window_pages(artifacts: BootstrapArtifacts) -> None:
    """Check that a long prompt keeps only its window in the windowed pool."""

    backend: Any = artifacts.executor.cache_backend
    tokens = _prompt(artifacts, _CODE)
    long = (tokens * (_LONG_TOKENS // len(tokens) + 1))[:_LONG_TOKENS]
    peak = 0

    def record() -> None:
        nonlocal peak
        peak = max(peak, backend.metrics()["window_used_pages"])

    _generate(artifacts, long, 4, after_tick=record)
    limit = backend.window_pages_for(len(long) + 4)
    print(f"window_peak_pages={peak}")
    print(f"window_limit_pages={limit}")
    _require(0 < peak <= limit, "a request holds more than its window")
    print("window_pages_ok=1")


def _check_layout_rejection(model_path: Path) -> None:
    """Check that the unpacked checkpoint layout fails before loading."""

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (target / name).write_bytes((model_path / name).read_bytes())
        try:
            build_native_artifacts(str(target))
        except NativeBootstrapFailure as exc:
            print(f"other_layout_code={exc.error.code}")
            _require(
                exc.error.code == "INVALID_WEIGHT_ARTIFACTS"
                and "LookAtMyQuant" in exc.error.message,
                f"unexpected error {exc.error.code}: {exc.error.message}",
            )
        else:
            raise SystemExit("a checkpoint without a pack manifest was accepted")
    print("other_layout_rejected_ok=1")


def _check_speculation(
    artifacts: BootstrapArtifacts, plain: list[int], tokens: int
) -> None:
    """Check that a speculative run returns the plain tokens."""

    controller = artifacts.executor.speculator.controller
    verified = kept = steps = 0
    record = controller.record

    def counting(
        confidences: Sequence[float], accepted: int, elapsed_ms: float
    ) -> None:
        nonlocal verified, kept, steps
        verified += len(confidences)
        kept += accepted
        steps += 1
        record(confidences, accepted, elapsed_ms)

    controller.record = counting
    speculative = _generate(artifacts, _prompt(artifacts, _CODE), tokens)
    print(f"speculative_steps={steps}")
    print(f"verified_tokens={verified}")
    print(f"kept_tokens={kept}")
    _require(verified > 0, "no step verified proposed tokens")
    _require(speculative == plain, "speculation changed the tokens")
    print("speculation_ok=1")


def _generate(
    artifacts: BootstrapArtifacts,
    prompt: tuple[int, ...],
    max_tokens: int,
    *,
    after_tick: Any = None,
) -> list[int]:
    scheduler = NativeContinuousScheduler(
        artifacts.executor,
        artifacts.cache_coordinator,
        execution_mode=artifacts.executor.execution_mode,
        terminal_token_ids=artifacts.eos_token_ids,
    )
    scheduler.submit(
        SchedulableRequest(
            request_id="request",
            prompt_token_ids=prompt,
            sampling=SamplingParams(),
            enqueued_at=time.perf_counter(),
            max_tokens=max_tokens,
        )
    )
    tokens: list[int] = []
    finished = False
    while not scheduler.idle():
        for event in scheduler.tick():
            if event.kind == "execution_error":
                raise SystemExit(f"execution failed: {event.message}")
            if event.kind != "token" or finished:
                continue
            tokens.append(int(event.token_id))
            finished = (
                len(tokens) >= max_tokens or tokens[-1] in artifacts.eos_token_ids
            )
            if finished:
                scheduler.finish("request")
        if after_tick is not None:
            after_tick()
    scheduler.close()
    return tokens


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    main()
