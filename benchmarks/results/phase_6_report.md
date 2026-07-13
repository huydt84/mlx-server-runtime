# Phase 6 Benchmark Report

## Model: mlx-community/LFM2.5-8B-A1B-MLX-4bit

## Benchmark Configuration

- generated_at: 2026-07-13T00:58:38+00:00
- model: mlx-community/LFM2.5-8B-A1B-MLX-4bit
- max_tokens: 8
- prompt_suite: prompt suite: 8 cases, 264 prompt tokens total

## Metric Definitions

- `samples`: successful measured requests included in aggregate statistics.
- `errors`: measured requests that failed and were excluded from latency and token aggregates.
- `error_rate = errors / (samples + errors)`.
- `ttft_*`: time from request start until the first generated token arrives.
- `latency_*`: end-to-end request time from request start until the final token or final response.
- `decode_time_mean_ms = latency_mean_ms - ttft_mean_ms`.
- `decode_tokens_per_second = completion_tokens / (decode_time_ms / 1000)` when decode time is positive.
- `end_to_end_tokens_per_second = completion_tokens / (latency_ms / 1000)` when latency is positive.
- `latency_per_completion_token_ms = latency_mean_ms / completion_tokens_per_request_mean`.
- `decode_time_per_completion_token_ms = decode_time_mean_ms / completion_tokens_per_request_mean`.
- `*_p95` is reported only when at least 20 successful samples exist; `*_p99` requires at least 100 successful samples.

## Raw Per-Backend Metrics

| backend | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_p99_ms | prompt_tokens_per_request_mean | completion_tokens_per_request_mean | total_tokens_per_request_mean | decode_time_mean_ms | latency_per_completion_token_ms | decode_time_per_completion_token_ms | latency_p50_per_completion_token_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mlx-lm | 8 | 0 | 0.0% | 110.8 | 122.0 | - | - | 155.9 | 167.2 | - | - | 33.0 | 8.0 | 41.0 | 45.1 | 19.5 | 5.6 | 20.9 |
| mlx_lm.server | 8 | 0 | 0.0% | 115.3 | 120.2 | - | - | 156.3 | 161.4 | - | - | 33.0 | 7.0 | 40.0 | 41.0 | 22.3 | 5.9 | 23.1 |
| this project | 8 | 0 | 0.0% | 111.7 | 113.1 | - | - | 163.1 | 164.4 | - | - | 33.0 | 8.0 | 41.0 | 51.4 | 20.4 | 6.4 | 20.6 |

## Throughput Metrics

| backend | decode_tokens_per_second_mean | decode_tokens_per_second_p50 | end_to_end_tokens_per_second_mean | end_to_end_tokens_per_second_p50 |
| --- | ---: | ---: | ---: | ---: |
| raw mlx-lm | 177.3 | 177.1 | 52.1 | 47.8 |
| mlx_lm.server | 170.6 | 170.6 | 46.3 | 43.9 |
| this project | 157.0 | 157.1 | 49.6 | 49.0 |

## Overhead Vs Raw MLX-LM

| backend | ttft_mean_overhead_ms | latency_mean_overhead_ms | ttft_p50_overhead_ms | latency_p50_overhead_ms | ttft_mean_overhead_percent | latency_mean_overhead_percent | decode_tps_delta_percent | e2e_tps_delta_percent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mlx-lm | +0.0 | +0.0 | +0.0 | +0.0 | +0.0% | +0.0% | +0.0% | +0.0% |
| mlx_lm.server | +4.5 | +0.4 | -1.8 | -5.8 | +4.1% | +0.3% | -3.8% | -11.2% |
| this project | +1.0 | +7.2 | -8.9 | -2.8 | +0.9% | +4.6% | -11.4% | -4.8% |

## Notes / Warnings

- suite warning: mlx_lm.server completion_tokens_per_request_mean differs from raw mlx-lm by 12.5%; prefer normalized per-token metrics over raw latency comparisons
- raw mlx-lm: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- raw mlx-lm: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)
- mlx_lm.server: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- mlx_lm.server: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)
- this project: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- this project: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.

## Overhead Summary

this project was 7.2 ms slower than raw mlx-lm on mean latency and 1.0 ms slower on mean TTFT.


## Model: mlx-community/Qwen3-4B-Instruct-2507-4bit

## Benchmark Configuration

- generated_at: 2026-07-13T00:58:49+00:00
- model: mlx-community/Qwen3-4B-Instruct-2507-4bit
- max_tokens: 8
- prompt_suite: prompt suite: 8 cases, 255 prompt tokens total

## Metric Definitions

- `samples`: successful measured requests included in aggregate statistics.
- `errors`: measured requests that failed and were excluded from latency and token aggregates.
- `error_rate = errors / (samples + errors)`.
- `ttft_*`: time from request start until the first generated token arrives.
- `latency_*`: end-to-end request time from request start until the final token or final response.
- `decode_time_mean_ms = latency_mean_ms - ttft_mean_ms`.
- `decode_tokens_per_second = completion_tokens / (decode_time_ms / 1000)` when decode time is positive.
- `end_to_end_tokens_per_second = completion_tokens / (latency_ms / 1000)` when latency is positive.
- `latency_per_completion_token_ms = latency_mean_ms / completion_tokens_per_request_mean`.
- `decode_time_per_completion_token_ms = decode_time_mean_ms / completion_tokens_per_request_mean`.
- `*_p95` is reported only when at least 20 successful samples exist; `*_p99` requires at least 100 successful samples.

## Raw Per-Backend Metrics

| backend | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_p99_ms | prompt_tokens_per_request_mean | completion_tokens_per_request_mean | total_tokens_per_request_mean | decode_time_mean_ms | latency_per_completion_token_ms | decode_time_per_completion_token_ms | latency_p50_per_completion_token_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mlx-lm | 8 | 0 | 0.0% | 157.0 | 171.5 | - | - | 243.5 | 258.4 | - | - | 31.9 | 8.0 | 39.9 | 86.5 | 30.4 | 10.8 | 32.3 |
| mlx_lm.server | 8 | 0 | 0.0% | 147.0 | 149.6 | - | - | 235.4 | 237.8 | - | - | 31.9 | 7.9 | 39.8 | 88.4 | 29.9 | 11.2 | 29.7 |
| this project | 8 | 0 | 0.0% | 150.7 | 154.7 | - | - | 247.8 | 253.9 | - | - | 31.9 | 7.9 | 39.8 | 97.2 | 31.5 | 12.3 | 31.7 |

## Throughput Metrics

| backend | decode_tokens_per_second_mean | decode_tokens_per_second_p50 | end_to_end_tokens_per_second_mean | end_to_end_tokens_per_second_p50 |
| --- | ---: | ---: | ---: | ---: |
| raw mlx-lm | 92.5 | 92.5 | 33.2 | 31.0 |
| mlx_lm.server | 89.1 | 90.4 | 33.9 | 33.6 |
| this project | 81.2 | 81.7 | 32.0 | 29.8 |

## Overhead Vs Raw MLX-LM

| backend | ttft_mean_overhead_ms | latency_mean_overhead_ms | ttft_p50_overhead_ms | latency_p50_overhead_ms | ttft_mean_overhead_percent | latency_mean_overhead_percent | decode_tps_delta_percent | e2e_tps_delta_percent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mlx-lm | +0.0 | +0.0 | +0.0 | +0.0 | +0.0% | +0.0% | +0.0% | +0.0% |
| mlx_lm.server | -10.0 | -8.1 | -22.0 | -20.6 | -6.4% | -3.3% | -3.7% | +2.1% |
| this project | -6.3 | +4.3 | -16.8 | -4.5 | -4.0% | +1.8% | -12.2% | -3.4% |

## Notes / Warnings

- raw mlx-lm: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- raw mlx-lm: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)
- mlx_lm.server: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- mlx_lm.server: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)
- this project: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- this project: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.

## Overhead Summary

this project was 4.3 ms slower than raw mlx-lm on mean latency and -6.3 ms slower on mean TTFT.


## Model: mlx-community/gemma-3-270m-it-qat-8bit

## Benchmark Configuration

- generated_at: 2026-07-13T00:58:59+00:00
- model: mlx-community/gemma-3-270m-it-qat-8bit
- max_tokens: 8
- prompt_suite: prompt suite: 8 cases, 269 prompt tokens total

## Metric Definitions

- `samples`: successful measured requests included in aggregate statistics.
- `errors`: measured requests that failed and were excluded from latency and token aggregates.
- `error_rate = errors / (samples + errors)`.
- `ttft_*`: time from request start until the first generated token arrives.
- `latency_*`: end-to-end request time from request start until the final token or final response.
- `decode_time_mean_ms = latency_mean_ms - ttft_mean_ms`.
- `decode_tokens_per_second = completion_tokens / (decode_time_ms / 1000)` when decode time is positive.
- `end_to_end_tokens_per_second = completion_tokens / (latency_ms / 1000)` when latency is positive.
- `latency_per_completion_token_ms = latency_mean_ms / completion_tokens_per_request_mean`.
- `decode_time_per_completion_token_ms = decode_time_mean_ms / completion_tokens_per_request_mean`.
- `*_p95` is reported only when at least 20 successful samples exist; `*_p99` requires at least 100 successful samples.

## Raw Per-Backend Metrics

| backend | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_p99_ms | prompt_tokens_per_request_mean | completion_tokens_per_request_mean | total_tokens_per_request_mean | decode_time_mean_ms | latency_per_completion_token_ms | decode_time_per_completion_token_ms | latency_p50_per_completion_token_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mlx-lm | 8 | 0 | 0.0% | 129.2 | 126.7 | - | - | 150.9 | 148.3 | - | - | 33.6 | 8.0 | 41.6 | 21.7 | 18.9 | 2.7 | 18.5 |
| mlx_lm.server | 8 | 0 | 0.0% | 130.0 | 130.8 | - | - | 155.5 | 157.0 | - | - | 33.6 | 8.0 | 41.6 | 25.5 | 19.4 | 3.2 | 19.6 |
| this project | 8 | 0 | 0.0% | 126.2 | 126.4 | - | - | 157.5 | 158.0 | - | - | 33.6 | 8.0 | 41.6 | 31.3 | 19.7 | 3.9 | 19.7 |

## Throughput Metrics

| backend | decode_tokens_per_second_mean | decode_tokens_per_second_p50 | end_to_end_tokens_per_second_mean | end_to_end_tokens_per_second_p50 |
| --- | ---: | ---: | ---: | ---: |
| raw mlx-lm | 369.2 | 371.1 | 53.2 | 54.0 |
| mlx_lm.server | 315.7 | 304.8 | 51.5 | 51.0 |
| this project | 262.3 | 250.8 | 50.9 | 50.7 |

## Overhead Vs Raw MLX-LM

| backend | ttft_mean_overhead_ms | latency_mean_overhead_ms | ttft_p50_overhead_ms | latency_p50_overhead_ms | ttft_mean_overhead_percent | latency_mean_overhead_percent | decode_tps_delta_percent | e2e_tps_delta_percent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mlx-lm | +0.0 | +0.0 | +0.0 | +0.0 | +0.0% | +0.0% | +0.0% | +0.0% |
| mlx_lm.server | +0.8 | +4.5 | +4.1 | +8.7 | +0.6% | +3.0% | -14.5% | -3.1% |
| this project | -3.0 | +6.5 | -0.4 | +9.7 | -2.3% | +4.3% | -29.0% | -4.2% |

## Notes / Warnings

- raw mlx-lm: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- raw mlx-lm: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)
- mlx_lm.server: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- mlx_lm.server: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)
- this project: latency_p95_ms and ttft_p95_ms unavailable with only 8 successful sample(s)
- this project: latency_p99_ms and ttft_p99_ms unavailable with only 8 successful sample(s)

## Observability / Control

- raw mlx-lm: direct execution path with no HTTP serving surface.
- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.
- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.

## Overhead Summary

this project was 6.5 ms slower than raw mlx-lm on mean latency and -3.0 ms slower on mean TTFT.
