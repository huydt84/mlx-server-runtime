# HTTP API Reference

All endpoints are served on the configured host:port (default `127.0.0.1:8000`).

## Endpoints

### GET /live

Process health check. Always returns 200 while the gateway is running.

**Response 200:**

```json
{
  "status": "live",
  "uptime_seconds": 1234,
  "pid": 98765
}
```

---

### GET /ready

Model readiness probe. Returns 200 only when the model is loaded, warmed up, and servable.

**Response 200:**

```json
{
  "status": "ready",
  "ready": true,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "revision": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "loaded_at": 1718000000,
  "device": null,
  "dtype": null,
  "warmup_passed": true
}
```

**Response 503:**

```json
{
  "status": "not_ready",
  "ready": false,
  "reason": "model_loading",
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "state": "loading_weights",
  "last_error": null
}
```

---

### GET /startup

Startup phase probe. Reports the current bootstrap phase.

**Response 200 (starting):**

```json
{
  "status": "starting",
  "phase": "loading_weights",
  "elapsed_seconds": 5
}
```

**Response 200 (ready):**

```json
{
  "status": "started"
}
```

**Response 503 (failed):**

```json
{
  "status": "failed",
  "phase": "failed",
  "elapsed_seconds": 10,
  "error": { "code": "MODEL_LOAD_FAILED", "message": "..." }
}
```

---

### GET /health

Legacy health check. Plain text, backward-compatible.

**Response 200:** `healthy`

**Response 503:** `unhealthy`

---

### GET /version

Gateway package version used by clients that need to record the serving binary.

**Response 200:**

```json
{
  "gateway_version": "0.1.0"
}
```

---

### GET /models

List configured models and their status.

**Response 200:**

```json
{
  "models": [
    {
      "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
      "ready": true,
      "state": "ready"
    }
  ]
}
```

---

### GET /models/{model}/status

Detailed model lifecycle status. Model name is percent-encoded.

**Response 200:**

```json
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "state": "ready",
  "ready": true,
  "progress": null,
  "device": null,
  "dtype": null,
  "loaded_at": 1718000000,
  "started_loading_at": 1717999990,
  "last_transition_at": 1718000000,
  "last_error": null,
  "warmup_passed": true,
  "last_warmup_at": 1718000000,
  "last_warmup_latency_ms": 123
}
```

**Possible states:** `not_loaded`, `downloading`, `verifying`, `loading_weights`, `initializing_runtime`, `warming_up`, `ready`, `degraded`, `failed`, `unloading`

**Response 404:** Unknown model name.

---

### GET /models/{model}/ready

Model-specific readiness check.

**Response 200:**

```json
{ "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "ready": true, "state": "ready" }
```

**Response 503:**

```json
{ "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "ready": false, "state": "loading_weights", "reason": "model_loading" }
```

**Response 404:** Unknown model name.

---

### GET /metrics

Prometheus metrics exposition (configurable via `telemetry.metrics_path`). Disabled when `telemetry.enable_prometheus = false`.

**Response 200:** `text/plain; version=0.0.4; charset=utf-8` with Prometheus-formatted metrics.

**Response 404:** When Prometheus is disabled and path does not match.

---

### POST /v1/chat/completions

Generate a chat completion. Supports streaming and non-streaming modes.

**Request body:**

```json
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Must match configured `worker.model` |
| `messages` | array | yes | — | Chat messages, at least one non-empty |
| `max_tokens` | integer | no | config default | Max generation tokens |
| `temperature` | float | no | config default | Sampling temperature |
| `top_p` | float | no | config default | Nucleus sampling threshold |
| `stream` | boolean | no | false | Enable SSE streaming |

**Response 200 (non-streaming):**

```json
{
  "id": "chatcmpl-req-1",
  "object": "chat.completion",
  "created": 1718000000,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 8,
    "total_tokens": 27
  }
}
```

**Response 200 (streaming):** SSE stream. Each chunk:

```
data: {"id":"chatcmpl-req-2","object":"chat.completion.chunk","created":1718000000,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"token"},"finish_reason":null}]}
```

Final chunk has empty `delta` and a `finish_reason`. Stream ends with `data: [DONE]`.

## Error Responses

| Status | Code | Meaning |
|--------|------|---------|
| 400 | `INVALID_REQUEST` | Malformed JSON, empty model/message, token limits exceeded |
| 404 | — | Unknown endpoint or model |
| 429 | `QUEUE_FULL` | Max pending requests exceeded |
| 429 | `QUEUE_TIMEOUT` | Queue wait exceeded `request_timeout_seconds` |
| 503 | `MODEL_NOT_READY` | Model not yet loaded or warming up |
| 503 | `MODEL_LOAD_FAILED` | Worker failed to load the model |
| 500 | `INTERNAL_ERROR` | Unexpected server or protocol error |

Error body:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "model must not be empty"
  }
}
```
