# Architecture

## Why Rust + Python?

The decision to use two languages rather than one is deliberate. Each language owns what it does best.

**Rust handles serving.** The control plane needs concurrency, predictable latency, and strong resource isolation. Rust provides:

- Thread-per-connection with no GC pauses.
- Precise control over memory and blocking behavior.
- A type system that prevents data races across shared scheduler state.
- A mature HTTP/SSE ecosystem (even with raw TCP in this implementation).

**Python handles inference.** MLX is a Python-native framework. The model ecosystem, tokenizer integrations, and generation primitives are all in Python. Trying to port model execution to Rust would:

- Duplicate thousands of lines of model code.
- Create a permanent lag behind `mlx-lm` upstream changes.
- Lose access to the MLX community's optimized kernels.

The boundary between them is the IPC protocol — a contract, not a shared address space.

## Why Not Proxy to `mlx_lm.server`?

The naive approach would be an HTTP proxy:

```
Client → Rust → HTTP → mlx_lm.server → MLX
```

This is simple to implement but introduces:

- **Duplicate serving logic.** Both Rust and `mlx_lm.server` would parse requests, manage queues, and handle streaming.
- **Double HTTP overhead.** Every request passes through two HTTP stacks with separate serialization.
- **No worker supervision.** Rust cannot distinguish between `mlx_lm.server` being slow vs. crashed.
- **No end-to-end cancellation.** The proxy model makes clean cancellation difficult — the Rust layer can drop its connection, but the upstream `mlx_lm.server` may continue generating.

The project instead uses a custom Python worker that calls `mlx-lm` APIs directly:

```
Client → Rust (HTTP/SSE) → Unix Socket (JSON) → Python Worker → mlx-lm (direct calls)
```

Rust owns the client-facing experience. Python owns inference. The IPC contract between them is minimal.

## Control Plane vs. Worker Boundary

The boundary is defined by responsibility, not performance.

| Owned by Rust | Owned by Python |
|--------------|-----------------|
| HTTP parsing and response | Model loading and caching |
| SSE frame formatting | Tokenization |
| Queue admission and backpressure | Generation loop |
| Request cancellation | KV cache management |
| Client disconnect detection | Sampling |
| Worker process lifecycle | MLX tensor operations |
| Prometheus metrics | Batching |
| Structured logging | Prompt caching |

Rust never touches a tensor. Python never reads an HTTP header.

## IPC Design

The IPC uses a **Unix domain socket** with **newline-delimited JSON frames**.

### Why UDS + JSON?

- **UDS** provides lower latency and higher throughput than TCP loopback on macOS. No port allocation, no kernel TCP stack overhead.
- **UDS flow:** Rust binds a `UnixListener`, spawns the worker, the worker connects, bootstrap handshake completes, then all inference traffic flows over the same connection.
- **JSON** (not MessagePack, not Protobuf) is used in the current phase for debuggability. Every frame can be inspected with `nc -U` or a text editor. Encoding and decoding are trivial in both languages. The frame format is a single JSON object terminated by `\n` — no streaming parser required.
- **MessagePack** can replace JSON later if profiling shows serialization as a bottleneck. The protocol is designed so that only the encode/decode layer changes, not the message shapes.

### Bootstrap Sequence

```
1. Rust: bind UnixListener, set non-blocking
2. Rust: spawn worker (python -m mlx_worker.main)
3. Worker: connect to socket
4. Worker: send STATUS frames (loading_weights → initializing_runtime → warming_up)
5. Worker: warm up model with 1-token generation
6. Worker: send READY frame
7. Rust: create WorkerClient, mark service healthy
   (If step 3–6 fails or times out: Rust kills worker, returns 503)
```

### Message Flow

For non-streaming: Rust sends one `ChatCompletion` command, reads events until a `ChatCompletionResponse` or `Error` matches the `request_id`.

For streaming: Same structure, but intermediate `ChatCompletionDelta` events arrive before the final `ChatCompletionResponse`.

For cancellation: Rust sends a `CancelRequest` frame. The worker checks for cancellation at generation step boundaries by polling the socket (non-blocking) before each `mlx-lm` `stream_generate` step.

## Request Lifecycle

```
Client                  Rust Gateway                 Python Worker
  │                         │                             │
  │── POST /v1/chat/ ──────→│                             │
  │    completions          │                             │
  │                         │── validate request ─────────│
  │                         │── acquire admission permit  │
  │                         │── record RequestTracker     │
  │                         │                             │
  │                         │── IPC: ChatCompletion ─────→│
  │                         │       (JSON over UDS)       │── build_prompt_tokens()
  │                         │                             │── validate_token_limits()
  │                         │                             │── model.generate()
  │                         │                             │
  │     (streaming)         │◌─ IPC: Delta ───────────────│ (per token)
  │←── SSE: chunk ─────────│                             │
  │                         │                             │
  │                         │◌─ IPC: Final / Error ──────│
  │←── SSE: [DONE] ────────│                             │
  │                         │── finish RequestTracker     │
  │                         │── release admission permit  │
```

## Backpressure Model

Backpressure is implemented at the Rust admission layer, not at the worker.

```
                     ┌──────────────┐
  Request → Validate → Queue (FIFO) → Worker
                     │              │
                     │ max_pending  │ max_active
                     │ timeout      │
                     └──────────────┘
```

- If `max_active` slots are full, new requests wait in the queue.
- If the queue exceeds `max_pending`, the request gets **HTTP 429** immediately.
- If the queue wait exceeds `request_timeout_seconds`, the request gets **HTTP 429** with `QUEUE_TIMEOUT`.
- The worker itself is unaware of queue depth — it only receives one request at a time (sequential `request_lock` in the `WorkerClient`). This simplifies the Python side and avoids cascading worker-side queuing.

### Why Sequential Worker Dispatch?

The current implementation holds a mutex (`request_lock`) in `WorkerClient` that serializes requests to the Python worker. This is intentional for Phase 1:

1. It simplifies the Python worker — it never needs to handle concurrent IPC reads.
2. It makes backpressure predictable — the queue is fully visible in Rust.
3. The worker can still batch internally via `BatchGenerator` even though it receives requests one at a time.

Future phases can move to concurrent IPC dispatch when the worker supports it natively.

## Cancellation Semantics

When a streaming client disconnects:

1. The `DisconnectMonitor` thread detects EOF or socket error via `peek()`.
2. A `DisconnectCancellation` thread sends `CancelRequest` via IPC.
3. The Python worker, at the next `should_cancel()` call in the generation loop, stops generating and returns a `ChatCompletionResponse` with `finish_reason = "cancelled"`.
4. The Rust gateway discards the response, decrements the `RequestTracker`, and updates `mlx_requests_cancelled_total`.

Cancellation is **cooperative at generation step boundaries**. If the worker is inside an MLX forward pass, cancellation waits for that step to complete. This avoids crashing the MLX process and keeps the Metal device in a consistent state.

Non-streaming requests cannot be cancelled mid-flight (no client feedback channel), but the client disconnect is still detected and cleaned up.

## Worker Supervision

Rust owns the Python worker's lifecycle:

- **Startup:** Spawns worker, waits for `READY` frame, sets `mlx_worker_up = 1`.
- **Crash detection:** A watcher thread calls `child.wait()`. If the worker exits, Rust sets `mlx_worker_up = 0`, clears the client, and marks the model as failed.
- **Model loading failure:** If the worker sends `ERROR` or fails to send `READY` within the timeout, Rust kills the process and returns 503 for all requests.
- **Shutdown:** On `SIGTERM`/`SIGINT`, Rust kills the worker, waits for exit, and cleans up the socket file.

Worker restart is reserved for future phases.

## Telemetry Design

Metrics and logging are separate concerns.

**Metrics** are atomic counters, gauges, and histograms in Rust exposed as Prometheus text:

- Counters: request totals, error totals, token totals, IPC message totals.
- Gauges: active requests, queue depth, worker up, throughput.
- Histograms: TTFT, request latency.

The `MetricsRegistry` is shared across threads via `Arc`. Every request creates a `RequestTracker` that records timing, logs on completion, and updates metrics atomically.

**Logging** is structured JSON to stderr. Each request produces one log line on completion:

```json
{
  "request_id": "req-1",
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "prompt_tokens": 19,
  "max_tokens": 256,
  "stream": false,
  "queue_time_ms": 0,
  "ttft_ms": null,
  "latency_ms": 120,
  "completion_tokens": 8,
  "finish_reason": "stop",
  "cancelled": false,
  "error": null
}
```

The runtime does not bundle a log collector. Pipe stderr to your preferred collector (Vector, Fluentd, journald) for structured ingestion.

## Comparison with Alternatives

### vs. `mlx_lm.server`

| Aspect | `mlx_lm.server` | This Runtime |
|--------|----------------|--------------|
| Telemetry | Basic stdout | Prometheus + structured logs |
| Cancellation | Not supported | Step-boundary cancellation |
| Backpressure | None | Queue + admission control |
| Worker supervision | None | Process lifecycle management |
| Extensibility | Monolithic | Modular Rust/Python split |

### vs. Raw `mlx-lm` script

| Aspect | Raw `mlx-lm` | This Runtime |
|--------|-------------|--------------|
| HTTP client | None | OpenAI-compatible API |
| Streaming | stdout | SSE |
| Concurrency | None | Queue + worker dispatch |
| Observability | None | Full metrics |
| Reliability | None | Supervision + error handling |

## Future Design Considerations

The architecture is designed to evolve without rewrites:

- **Concurrent worker dispatch:** Replace `request_lock` with a concurrent IPC channel when the worker supports it.
- **Dedicated prefill/decode:** Split the generation step into prefill and decode phases for better scheduling.
- **Multiple workers:** Run one Python process per model for multi-model serving.
- **Rust tokenizer:** Optionally move tokenization to Rust for prompt length estimation before IPC.
- **Persistent KV cache:** Keep KV cache across requests for session-based applications.
