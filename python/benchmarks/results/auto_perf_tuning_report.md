# Phase 6 Benchmark Report

- generated_at: 2026-06-15T13:24:52+00:00
- model: mlx-community/Qwen2.5-7B-Instruct-4bit
- max_tokens: 8
- prompt: Say hello in one short sentence.

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 723.3 | 772.7 | 36 | 3 | +0.0 | +0.0 | - |
| mlx_lm.server | 108.3 | 148.3 | 36 | 2 | -615.0 | -624.4 | - |
| this project | 236.0 | 286.7 | 36 | 2 | -487.3 | -486.0 | - |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.
