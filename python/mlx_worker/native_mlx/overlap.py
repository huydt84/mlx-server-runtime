"""MLX stream capability probes used by native-v2 validation.

The probe in this module is deliberately limited to independent synthetic
arrays.  A successful result proves that MLX can order a dependency across two
GPU streams; it does not prove that the native model and paged-cache graph can
be moved to a second stream.  The serving path therefore remains serial until
the real-model host gate supplies that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import mlx.core as mx


@dataclass(frozen=True)
class MlxStreamProbeResult:
    """Result of the bounded cross-stream dependency probe."""

    supported: bool
    elapsed_ms: int
    detail: str


def probe_cross_stream_dependency() -> MlxStreamProbeResult:
    """Check stream-local dispatch and dependency ordering without model state."""

    started = time.perf_counter()
    if not mx.metal.is_available():
        return MlxStreamProbeResult(
            supported=False,
            elapsed_ms=_elapsed_ms(started),
            detail="MLX Metal is unavailable",
        )
    try:
        first_stream = mx.new_stream(mx.gpu)
        second_stream = mx.new_stream(mx.gpu)
        with mx.stream(first_stream):
            source = mx.arange(32, dtype=mx.float32) * 2
            mx.async_eval(source)
        with mx.stream(second_stream):
            dependent = source + 1
            mx.async_eval(dependent)
        mx.eval(dependent)
        values = dependent.tolist()
        if values != [float(index * 2 + 1) for index in range(32)]:
            return MlxStreamProbeResult(
                supported=False,
                elapsed_ms=_elapsed_ms(started),
                detail="cross-stream dependency produced incorrect values",
            )
    except Exception as exc:  # pragma: no cover - hardware/API dependent
        return MlxStreamProbeResult(
            supported=False,
            elapsed_ms=_elapsed_ms(started),
            detail=f"{type(exc).__name__}: {exc}",
        )
    return MlxStreamProbeResult(
        supported=True,
        elapsed_ms=_elapsed_ms(started),
        detail="synthetic dependency completed; native model/cache evidence required",
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


__all__ = ["MlxStreamProbeResult", "probe_cross_stream_dependency"]
