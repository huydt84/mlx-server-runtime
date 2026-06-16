# Phase 6 Benchmark Report

## Model: mlx-community/LFM2.5-8B-A1B-MLX-4bit

- generated_at: 2026-06-16T11:55:54+00:00
- model: mlx-community/LFM2.5-8B-A1B-MLX-4bit
- max_tokens: 256
- prompt: prompt suite: 16 cases, 520 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 110.2 | 1485.1 | 49 | 256 | +0.0 | +0.0 | samples=32; latency_p50_ms=1599.5; latency_p95_ms=1645.3; ttft_p50_ms=110.1; ttft_p95_ms=133.1 |
| mlx_lm.server | 131.9 | 1765.9 | 49 | 255 | +21.7 | +280.8 | samples=32; latency_p50_ms=1901.4; latency_p95_ms=1950.6; ttft_p50_ms=141.0; ttft_p95_ms=154.5 |
| this project | 112.9 | 1497.1 | 49 | 256 | +2.7 | +12.0 | samples=32; latency_p50_ms=1602.1; latency_p95_ms=1673.9; ttft_p50_ms=115.2; ttft_p95_ms=134.3 |

## Overhead Summary

mlx_lm.server was 280.8 ms slower than raw mlx-lm on total latency and 21.7 ms slower on TTFT.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Qwen3-4B-Instruct-2507-4bit

- generated_at: 2026-06-16T12:01:45+00:00
- model: mlx-community/Qwen3-4B-Instruct-2507-4bit
- max_tokens: 256
- prompt: prompt suite: 16 cases, 500 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 168.8 | 2325.1 | 47 | 256 | +0.0 | +0.0 | samples=32; latency_p50_ms=3030.5; latency_p95_ms=3144.0; ttft_p50_ms=161.1; ttft_p95_ms=213.9 |
| mlx_lm.server | 152.0 | 2530.9 | 47 | 256 | -16.7 | +205.8 | samples=32; latency_p50_ms=3312.3; latency_p95_ms=3479.1; ttft_p50_ms=151.0; ttft_p95_ms=164.1 |
| this project | 167.5 | 2313.4 | 47 | 256 | -1.3 | -11.7 | samples=32; latency_p50_ms=3017.2; latency_p95_ms=3162.1; ttft_p50_ms=169.3; ttft_p95_ms=214.1 |

## Overhead Summary

mlx_lm.server was 205.8 ms slower than raw mlx-lm on total latency and -16.7 ms slower on TTFT.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/gemma-3-270m-it-qat-8bit

- generated_at: 2026-06-16T12:02:37+00:00
- model: mlx-community/gemma-3-270m-it-qat-8bit
- max_tokens: 256
- prompt: prompt suite: 16 cases, 527 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 126.1 | 268.8 | 49 | 168 | +0.0 | +0.0 | samples=32; latency_p50_ms=212.5; latency_p95_ms=589.5; ttft_p50_ms=125.1; ttft_p95_ms=133.4 |
| mlx_lm.server | 140.5 | 374.1 | 49 | 163 | +14.4 | +105.3 | samples=32; latency_p50_ms=276.5; latency_p95_ms=902.6; ttft_p50_ms=139.9; ttft_p95_ms=148.7 |
| this project | 133.5 | 285.3 | 49 | 167 | +7.4 | +16.5 | samples=32; latency_p50_ms=232.4; latency_p95_ms=609.5; ttft_p50_ms=132.5; ttft_p95_ms=148.4 |

## Overhead Summary

mlx_lm.server was 105.3 ms slower than raw mlx-lm on total latency and 14.4 ms slower on TTFT.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.
