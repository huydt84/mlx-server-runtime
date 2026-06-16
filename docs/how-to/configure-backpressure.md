# How to Configure Backpressure

Tune request admission and queue limits in `config/runtime.toml` under the `[limits]` section.

## Available Limits

```toml
[limits]
# Maximum requests waiting in queue before returning 429
max_pending_requests = 64

# Maximum concurrent requests being processed by the worker
max_active_requests = 16

# Maximum prompt tokens per request, returns 400 if exceeded
max_prompt_tokens = 32768

# Maximum generated tokens per request
max_completion_tokens = 4096

# Maximum prompt + completion tokens per request
max_total_tokens_per_request = 65536

# Seconds a request will wait in queue before timing out (429)
request_timeout_seconds = 300
```

## How Admission Works

1. Request arrives, parsed, validated against token limits.
2. If `active < max_active_requests`, it proceeds immediately.
3. If active is at limit, the request enters the waiting queue.
4. If `waiting >= max_pending_requests`, returns **HTTP 429**.
5. If the queue wait exceeds `request_timeout_seconds`, returns **HTTP 429** with `QUEUE_TIMEOUT`.
6. If the worker is down, returns **HTTP 503**.

## Suggested Tuning

| Workload Pattern | `max_pending` | `max_active` | Rationale |
|-----------------|---------------|--------------|-----------|
| Light, low concurrency | 16 | 4 | Few clients, small models |
| Heavy, bursty | 128 | 32 | Many concurrent requests |
| Streaming-heavy | 64 | 16 | Each stream holds a slot longer |
| Batch-optimized | 64 | 64 | Worker batches internally |

## Token Limits

Set `max_prompt_tokens` and `max_completion_tokens` based on your model's context window. For a model with 32K context, `max_total_tokens_per_request` should not exceed 32768.

Requests exceeding any token limit return **HTTP 400** with `INVALID_REQUEST`.
