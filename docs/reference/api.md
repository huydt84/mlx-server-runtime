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

List configured models and their status. The transcription model, when
configured, is listed with its own state: it is `warming_up` while the worker
loads it after startup, and audio endpoints return 503 until it is `ready`.

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

### POST /v1/audio/transcriptions

Transcribe an audio file with the configured Whisper model
(`worker.transcription_model`, or `mlx-air serve --transcription-model`).
OpenAI-compatible `multipart/form-data` request.

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=mlx-community/whisper-large-v3-turbo-asr-fp16 \
  -F file=@speech.mp3 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"
```

**Form fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | — | Audio file, at most `limits.max_audio_bytes` (25 MiB by default). WAV, MP3, and FLAC decode in the worker; m4a, mp4, ogg, and webm need `ffmpeg` on the worker's `PATH` or in `/opt/homebrew/bin` or `/usr/local/bin` (detached instances run with launchd's minimal `PATH`) |
| `model` | string | yes | — | Must match the configured transcription model |
| `language` | string | no | detected | Input language as an ISO-639-1 code (`en`) or Whisper language name (`english`) |
| `prompt` | string | no | — | Text that conditions the first 30-second window, such as names or vocabulary |
| `response_format` | string | no | `json` | `json`, `text`, `srt`, `verbose_json`, or `vtt` |
| `temperature` | float | no | `0` | Between 0 and 1. `0` retries each window at 0.2, 0.4, … 1.0 when the decode looks repetitive or unlikely; any other value decodes at that temperature only |
| `timestamp_granularities[]` | string | no | `segment` | `segment` and/or `word`; requires `response_format=verbose_json`. `word` adds word-level timestamps |

`stream=true` is rejected with 400: Whisper transcription is not streamed.
Other OpenAI fields are ignored.

**Response 200 (`json`):**

```json
{ "text": "Hello there.", "usage": { "type": "duration", "seconds": 2 } }
```

**Response 200 (`verbose_json`):**

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 1.62,
  "text": "Hello there.",
  "usage": { "type": "duration", "seconds": 2 },
  "segments": [
    {
      "id": 0, "seek": 0, "start": 0.0, "end": 1.62, "text": " Hello there.",
      "tokens": [50365, 2425, 456, 13], "temperature": 0.0,
      "avg_logprob": -0.21, "compression_ratio": 0.75, "no_speech_prob": 0.01
    }
  ],
  "words": [
    { "word": " Hello", "start": 0.0, "end": 0.52 },
    { "word": " there.", "start": 0.52, "end": 1.62 }
  ]
}
```

`segments` is present unless only `word` granularity was requested; `words`
is present only when `word` granularity was requested.

`text` returns the transcript followed by a newline (`text/plain`). `srt`
(`text/plain`) and `vtt` (`text/vtt`) return one cue per segment.

The worker runs a transcription to completion before it advances chat
scheduling again, so concurrent chat requests wait for the transcription's
inference time. Audio requests share the gateway admission queue
(`max_active_requests`, `max_pending_requests`) with chat requests.

### POST /v1/audio/translations

Translate speech in any supported language into English text. Same form
fields and response formats as `/v1/audio/transcriptions`, except `language`
is rejected (the output is always English) and `verbose_json` reports
`"task": "translate"` with the detected source `language`.

`POST /v1/audio/speech` (text-to-speech) is not served: Whisper only converts
speech to text.

## Error Responses

| Status | Code | Meaning |
|--------|------|---------|
| 400 | `INVALID_REQUEST` | Malformed JSON, empty model/message, token limits exceeded |
| 400 | `TRANSCRIPTION_NOT_CONFIGURED` | Audio endpoint called without a configured transcription model |
| 404 | — | Unknown endpoint or model |
| 413 | `AUDIO_TOO_LARGE` | Audio upload exceeds `limits.max_audio_bytes` |
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
