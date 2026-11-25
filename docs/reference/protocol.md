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

Requests that offer tools also carry:

| Field | Type | Description |
|-------|------|-------------|
| `tools` | string | JSON text of the tool definitions array, as the client sent it. Text keeps the key order |
| `parallel_tool_calls` | bool | Whether one reply may hold several calls |

Messages carry these fields when they are set:

| Field | Role | Description |
|-------|------|-------------|
| `tool_calls` | `assistant` | `[{"id", "type": "function", "function": {"name", "arguments"}}]`; `arguments` is JSON text of an object |
| `reasoning_content` | `assistant` | Thinking of an earlier reply |
| `tool_call_id` | `tool` | Id of the call this message answers |
| `name` | `tool` | Optional function name |

**CancelRequest:**

```json
{
  "type": "cancel_request",
  "request_id": "req-1"
}
```

**Transcription:** an audio upload for the transcription model. The audio is
the uploaded file, base64 encoded, so a 25 MiB upload is a ~33 MiB frame.
`task` is `transcribe` or `translate`; optional fields are omitted when unset.

```json
{
  "type": "transcription",
  "request": {
    "request_id": "audio-1",
    "model": "mlx-community/whisper-large-v3-turbo-asr-fp16",
    "task": "transcribe",
    "audio_base64": "UklGRiQ...",
    "filename": "speech.wav",
    "language": "en",
    "prompt": "Kenobi",
    "temperature": 0.0,
    "word_timestamps": false
  }
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

A delta of a request with tools may also carry `reasoning` (string) and
`tool_calls`, an array of `{"index", "id", "type", "function"}` where each call
is complete. `delta` may then be empty.

The response of a request with tools adds these fields to `response`:

| Field | Type | Description |
|-------|------|-------------|
| `reasoning` | string | Thinking split from `text`; absent without any |
| `tool_calls` | array | Completed calls, in the shape above; absent without any |
| `tool_parser` | string | Parser that read the output |
| `tool_parse_failures` | int | Calls returned as text because they could not be parsed |

**Transcription** (result of a transcription command): `language` is the
lowercase Whisper language name, `duration` the decoded input length in
seconds, and `words` is empty unless `word_timestamps` was requested.

```json
{
  "type": "transcription",
  "response": {
    "request_id": "audio-1",
    "model": "mlx-community/whisper-large-v3-turbo-asr-fp16",
    "task": "transcribe",
    "text": " Hello there.",
    "language": "english",
    "duration": 1.62,
    "segments": [
      {"id": 0, "seek": 0, "start": 0.0, "end": 1.62, "text": " Hello there.",
       "tokens": [50365, 2425], "temperature": 0.0, "avg_logprob": -0.21,
       "compression_ratio": 0.75, "no_speech_prob": 0.01}
    ],
    "words": [],
    "audio_decode_latency_ms": 4,
    "inference_latency_ms": 310
  }
}
```

A transcription failure is an `error` event: `INVALID_REQUEST` for input the
worker cannot use (undecodable audio, a format that needs a missing `ffmpeg`,
an unknown language) and `WORKER_ERROR` for model failures.

**Error:**

```json
{
  "type": "error",
  "code": "INVALID_REQUEST",
  "request_id": "req-1",
  "message": "generation failed"
}
```

`code` is `INVALID_REQUEST`, `TOOLS_NOT_SUPPORTED` (the request offers tools and
the model has no tool-call parser), `WORKER_ERROR`, or another worker failure
code. The gateway answers the first two with HTTP 400.

## Encoding

All frames are UTF-8 JSON objects terminated by `\n` (0x0A). Each frame is one complete JSON object — there are no streaming JSON parsers involved.

- Rust writes using `encode_gateway_command()` + `writeln!`
- Python writes using `encode_event()` / `encode_command()` + `sendall`
- Python reads with `socket.recv(4096)` + newline detection
- Rust reads with `BufReader::read_line()`
