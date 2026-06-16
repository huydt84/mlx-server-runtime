# Phase 6 Benchmark Report

## Model: mlx-community/Qwen2.5-7B-Instruct-4bit

- generated_at: 2026-06-15T13:44:34+00:00
- model: mlx-community/Qwen2.5-7B-Instruct-4bit
- max_tokens: 4
- prompt: Say hello in one short sentence.

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 829.2 | 879.1 | 36 | 3 | +0.0 | +0.0 | - |
| mlx_lm.server | 106.2 | 146.3 | 36 | 2 | -722.9 | -732.8 | - |
| this project | 239.6 | 293.0 | 36 | 2 | -589.5 | -586.1 | - |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Qwen3-8B-4bit

- generated_at: 2026-06-15T13:44:44+00:00
- model: mlx-community/Qwen3-8B-4bit
- max_tokens: 4
- prompt: Say hello in one short sentence.

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 1202.9 | 1272.8 | 15 | 4 | +0.0 | +0.0 | - |
| mlx_lm.server | 168.0 | 168.0 | 15 | 0 | -1034.9 | -1104.8 | - |
| this project | 184.9 | 256.9 | 15 | 4 | -1018.0 | -1015.8 | - |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-bf16

- generated_at: 2026-06-15T13:45:05+00:00
- model: mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-bf16
- max_tokens: 4
- prompt: Say hello in one short sentence.

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 1999.4 | 2084.0 | 26 | 3 | +0.0 | +0.0 | - |
| mlx_lm.server | 143.8 | 223.4 | 26 | 2 | -1855.6 | -1860.6 | - |
| this project | 207.8 | 296.0 | 26 | 2 | -1791.6 | -1788.0 | - |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.


## Model: mlx-community/Qwen3.5-9B-4bit

- generated_at: 2026-06-15T13:45:32+00:00
- model: mlx-community/Qwen3.5-9B-4bit
- max_tokens: 4
- prompt: Say hello in one short sentence.

## Results

| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw mlx-lm | 13957.4 | 14036.5 | 17 | 4 | +0.0 | +0.0 | - |
| mlx_lm.server | 247.7 | 247.7 | 17 | 0 | -13709.7 | -13788.8 | - |
| this project | 232.5 | 319.7 | 17 | 4 | -13724.8 | -13716.8 | - |

## Overhead Summary

No backend exceeded the raw mlx-lm baseline in measured latency.

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.
