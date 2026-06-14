# Phase 6 Benchmark Report

- generated_at: 2026-06-14T12:32:46+00:00
- model: mlx-community/Qwen2.5-7B-Instruct-4bit
- max_tokens: 8
- prompt: Say hello in one short sentence.

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 773.0 | 822.4 | 36 | 3 | +0.0 | +0.0 | - |
| mlx_lm.server | 136.5 | 176.9 | 36 | 2 | -636.5 | -645.5 | - |
| this project | 367.4 | 416.5 | 36 | 2 | -405.6 | -406.0 | - |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.
