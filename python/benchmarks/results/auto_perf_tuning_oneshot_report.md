# Phase 6 Benchmark Report

## Model: mlx-community/Qwen2.5-7B-Instruct-4bit

- generated_at: 2026-06-16T00:52:36+00:00
- model: mlx-community/Qwen2.5-7B-Instruct-4bit
- max_tokens: 8
- prompt: prompt suite: 8 cases, 423 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 330.3 | 467.2 | 73 | 8 | +0.0 | +0.0 | samples=8; latency_p50_ms=444.9; latency_p95_ms=701.5; ttft_p50_ms=309.1; ttft_p95_ms=560.8 |
| mlx_lm.server | 202.9 | 335.5 | 73 | 8 | -127.5 | -131.6 | samples=8; latency_p50_ms=315.6; latency_p95_ms=447.2; ttft_p50_ms=182.8; ttft_p95_ms=314.1 |
| this project | 272.0 | 412.1 | 73 | 8 | -58.3 | -55.1 | samples=8; latency_p50_ms=410.2; latency_p95_ms=457.3; ttft_p50_ms=270.4; ttft_p95_ms=315.7 |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Qwen3-8B-4bit

- generated_at: 2026-06-16T00:52:54+00:00
- model: mlx-community/Qwen3-8B-4bit
- max_tokens: 8
- prompt: prompt suite: 8 cases, 255 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 339.1 | 481.2 | 52 | 8 | +0.0 | +0.0 | samples=8; latency_p50_ms=405.1; latency_p95_ms=957.4; ttft_p50_ms=262.6; ttft_p95_ms=815.1 |
| mlx_lm.server | 220.9 | 344.7 | 52 | 7 | -118.2 | -136.5 | samples=8; latency_p50_ms=343.9; latency_p95_ms=402.8; ttft_p50_ms=220.8; ttft_p95_ms=278.3 |
| this project | 220.4 | 368.3 | 52 | 8 | -118.7 | -112.9 | samples=8; latency_p50_ms=370.8; latency_p95_ms=411.3; ttft_p50_ms=220.3; ttft_p95_ms=264.1 |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-bf16

- generated_at: 2026-06-16T00:53:25+00:00
- model: mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-bf16
- max_tokens: 8
- prompt: prompt suite: 8 cases, 343 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 428.0 | 684.6 | 63 | 8 | +0.0 | +0.0 | samples=8; latency_p50_ms=463.1; latency_p95_ms=1569.1; ttft_p50_ms=207.0; ttft_p95_ms=1313.7 |
| mlx_lm.server | 209.8 | 470.1 | 63 | 8 | -218.2 | -214.4 | samples=8; latency_p50_ms=462.1; latency_p95_ms=529.5; ttft_p50_ms=205.9; ttft_p95_ms=268.6 |
| this project | 198.8 | 460.1 | 63 | 8 | -229.3 | -224.4 | samples=8; latency_p50_ms=461.6; latency_p95_ms=471.1; ttft_p50_ms=197.8; ttft_p95_ms=210.1 |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Qwen3.5-9B-4bit

- generated_at: 2026-06-16T00:55:02+00:00
- model: mlx-community/Qwen3.5-9B-4bit
- max_tokens: 8
- prompt: prompt suite: 8 cases, 281 prompt tokens total

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 2091.2 | 2248.5 | 57 | 8 | +0.0 | +0.0 | samples=8; latency_p50_ms=478.1; latency_p95_ms=9804.2; ttft_p50_ms=322.3; ttft_p95_ms=9643.6 |
| mlx_lm.server | 310.4 | 465.7 | 57 | 8 | -1780.8 | -1782.8 | samples=8; latency_p50_ms=473.3; latency_p95_ms=547.7; ttft_p50_ms=318.4; ttft_p95_ms=392.3 |
| this project | 288.7 | 456.5 | 57 | 8 | -1802.4 | -1792.0 | samples=8; latency_p50_ms=456.8; latency_p95_ms=507.1; ttft_p50_ms=289.9; ttft_p95_ms=337.0 |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.
