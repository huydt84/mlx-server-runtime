# IPC Protocol Reference

The Rust gateway and Python worker communicate over a Unix domain socket using newline-delimited text frames.

## Bootstrap Phase

The bootstrap sequence establishes readiness:

```
Gateway                 Worker
   │                       │
   │── UnixListener ──────→│ connects
   │                       │── STATUS {model,state} ──→
   │                       │── STATUS {model,state} ──→
   │                       │── STATUS {model,state} ──→
   │                       │── STATUS {model,state} ──→
   │                       │── READY ─────────────────→
   │←── chat_completion ──→│
   │←── cancel_request ───→│
```

### Bootstrap Messages (Worker → Gateway)

**READY:**

```
READY\n
```

**ERROR:**

```
ERROR\t<message>\n
```

Structured startup failures may also use JSON payloads:

```json
ERROR\t{"message":"native-mlx startup failed","error":{"code":"UNSUPPORTED_ARCHITECTURE_CLASS","message":"...","at":123,"backend":"native-mlx","stage":"architecture_detection","category":"unsupported_class","detail":"LlamaForCausalLM"}}
```

**STATUS:**

```
STATUS\t<json>\n
```

Status JSON fields:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model identifier |
| `revision` | string or null | Model revision |
| `state` | string | Lifecycle state |
| `ready` | bool | Whether model is ready |
| `servable` | bool | Whether model can serve |
| `progress` | object or null | Loading progress details |
| `device` | string or null | Compute device |
| `dtype` | string or null | Weight dtype |
| `loaded_at` | int or null | Unix timestamp |
| `started_loading_at` | int or null | Unix timestamp |
| `last_transition_at` | int | Unix timestamp |
| `last_error` | object or null | Error details |
| `warmup_passed` | bool | Warmup result |
| `last_warmup_at` | int or null | Unix timestamp |
| `last_warmup_latency_ms` | int or null | Warmup duration |

When `last_error` is present, it may include:

| Field | Type | Description |
|-------|------|-------------|
| `code` | string | Stable error code |
| `message` | string | Human-readable error |
| `at` | int | Unix timestamp |
| `backend` | string or null | Backend that produced error |
| `stage` | string or null | Earliest failed startup/runtime stage |
| `category` | string or null | Triage class such as `unsupported_class` |
| `detail` | string or null | Stable mismatch detail |

## Inference Phase

After bootstrap, the gateway sends JSON commands and reads JSON events over the same connection.

### Gateway → Worker Commands

**ChatCompletion:**

```json
{
  "type": "chat_completion",
  "request": {
    "request_id": "req-1",
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 32,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_prompt_tokens": 32768,
    "max_completion_tokens": 4096,
    "max_total_tokens_per_request": 65536,
    "stream": false
  }
}
```

**CancelRequest:**

```json
{
  "type": "cancel_request",
  "request_id": "req-1"
}
```

### Worker → Gateway Events

**ChatCompletionResponse** (non-streaming response or final response):

```json
{
  "type": "chat_completion",
  "response": {
    "request_id": "req-1",
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "text": "Hello!",
    "finish_reason": "stop",
    "prompt_tokens": 14,
    "completion_tokens": 3
  }
}
```

**ChatCompletionDelta** (streaming token):

```json
{
  "type": "chat_completion_delta",
  "delta": {
    "request_id": "req-1",
    "delta": "Hello"
  }
}
```

**Error:**

```json
{
  "type": "error",
  "code": "INVALID_REQUEST",
  "request_id": "req-1",
  "message": "generation failed"
}
```

## Encoding

All frames are UTF-8 JSON objects terminated by `\n` (0x0A). Each frame is one complete JSON object — there are no streaming JSON parsers involved.

- Rust writes using `encode_gateway_command()` + `writeln!`
- Python writes using `encode_event()` / `encode_command()` + `sendall`
- Python reads with `socket.recv(4096)` + newline detection
- Rust reads with `BufReader::read_line()`
