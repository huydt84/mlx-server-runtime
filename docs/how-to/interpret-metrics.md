# How to Interpret Metrics

Prometheus metrics are exposed at `GET /metrics` (configurable path).

## Key Metrics

### Request Volume

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_requests_total` | counter | Cumulative requests received |
| `mlx_requests_active` | gauge | Currently processing |
| `mlx_requests_failed_total` | counter | Requests that ended with error |
| `mlx_requests_cancelled_total` | counter | Requests cancelled (client disconnect) |

**If `mlx_requests_active` stays high**, increase `max_active_requests` or scale up.

### Queue Pressure

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_queue_depth` | gauge | Requests waiting for a worker slot |
| `mlx_queue_rejected_total` | counter | 429 responses due to full queue |

**If `mlx_queue_rejected_total` increases**, raise `max_pending_requests` or add capacity.

### Worker Health

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_worker_up` | gauge | 1 if worker is ready, 0 otherwise |
| `mlx_worker_restarts_total` | counter | Worker process restarts |

**If `mlx_worker_up` is 0**, the worker crashed or failed to start. Check logs.

### Latency

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_ttft_ms_bucket` | histogram | Time to first token distribution |
| `mlx_request_latency_ms_bucket` | histogram | End-to-end request latency distribution |

Use the histograms to track p50/p90/p99 latencies.

### Token Throughput

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_prompt_tokens_total` | counter | Cumulative prompt tokens processed |
| `mlx_completion_tokens_total` | counter | Cumulative generated tokens |
| `mlx_decode_tokens_per_second` | gauge | Latest decode throughput |
| `mlx_prefill_tokens_per_second` | gauge | Latest prefill throughput |

### IPC

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_ipc_messages_sent_total` | counter | Messages sent from Rust to Python |
| `mlx_ipc_messages_received_total` | counter | Messages received by Rust from Python |
| `mlx_ipc_roundtrip_latency_ms` | gauge | Latest IPC roundtrip latency |

High `mlx_ipc_roundtrip_latency_ms` may indicate large messages or worker congestion.

### Memory

| Metric | Type | Meaning |
|--------|------|---------|
| `mlx_worker_memory_bytes` | gauge | Worker process memory estimate |
| `mlx_kv_cache_bytes` | gauge | KV cache memory estimate |

Monitor for memory growth across sustained load.

## PromQL Examples

```promql
# Request rate (req/s)
rate(mlx_requests_total[1m])

# p99 TTFT (ms)
histogram_quantile(0.99, rate(mlx_ttft_ms_bucket[5m]))

# Queue saturation
mlx_queue_depth / mlx_queue_rejected_total

# Error rate
rate(mlx_requests_failed_total[5m]) / rate(mlx_requests_total[5m]) * 100
```
