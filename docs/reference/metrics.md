# Metrics Reference

All Prometheus metrics exposed at `GET /metrics` (configurable via `telemetry.metrics_path`).

## Counters

| Metric | Description |
|--------|-------------|
| `mlx_requests_total` | Total observed requests |
| `mlx_requests_failed_total` | Total failed requests |
| `mlx_requests_cancelled_total` | Total cancelled requests |
| `mlx_queue_rejected_total` | Total requests rejected by queue limits |
| `mlx_worker_restarts_total` | Worker process restarts |
| `mlx_prompt_tokens_total` | Total prompt tokens observed |
| `mlx_completion_tokens_total` | Total completion tokens observed |
| `mlx_ipc_messages_sent_total` | Total IPC messages sent from gateway to worker |
| `mlx_ipc_messages_received_total` | Total IPC messages received by gateway from worker |

## Gauges

| Metric | Description |
|--------|-------------|
| `mlx_requests_active` | Current active requests |
| `mlx_queue_depth` | Current queued requests waiting for a slot |
| `mlx_worker_up` | 1 if worker is ready and connected, 0 otherwise |
| `mlx_decode_tokens_per_second` | Latest decode throughput |
| `mlx_prefill_tokens_per_second` | Latest prefill throughput |
| `mlx_ipc_roundtrip_latency_ms` | Latest IPC roundtrip latency |
| `mlx_worker_memory_bytes` | Latest worker memory estimate |
| `mlx_kv_cache_bytes` | Latest KV cache memory estimate |

## Histograms

| Metric | Description | Buckets (ms) |
|--------|-------------|--------------|
| `mlx_ttft_ms` | Time to first token | 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000 |
| `mlx_request_latency_ms` | End-to-end request latency | Same as above |

Each histogram exposes:

- `<name>_bucket{le="<upper>"}` — cumulative count per bucket
- `<name>_bucket{le="+Inf"}` — total count
- `<name>_sum` — sum of observed values
- `<name>_count` — count of observed values
