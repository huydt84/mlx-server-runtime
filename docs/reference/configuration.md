# Configuration Reference

The runtime is configured via a TOML file. Default path is `config/runtime.toml`. Override with the `MLX_RUNTIME_CONFIG` environment variable.

If the config file is missing, all defaults apply.

Native-MLX cache, attention, scheduler, and profiling controls are environment
variables rather than TOML fields. See [Native MLX v2](native-mlx-v2.md),
including `MLX_RUNTIME_NATIVE_EXECUTION_BACKEND` and its cache/attention
compatibility contract.

## [server]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | HTTP server bind address |
| `port` | integer | `8000` | HTTP server port |

## [worker]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `python` | string | `"python/.venv/bin/python"` | Python interpreter path |
| `module` | string | `"mlx_worker.main"` | Python module entry point |
| `backend` | string | `"native-mlx"` | Backend selector: `native-mlx`, or `v1` (`mlx-lm`) for checkpoints native-mlx does not implement and for `vlm_model` |
| `model` | string | `"mlx-community/Qwen2.5-7B-Instruct-4bit"` | Hugging Face model ID |
| `vlm_model` | string or null | `null` | Optional VLM model ID |
| `transcription_model` | string or null | `null` | Optional Whisper checkpoint served by `/v1/audio/transcriptions` and `/v1/audio/translations` through `mlx-audio`, such as `mlx-community/whisper-large-v3-turbo-asr-fp16`. Works with both backends. The worker loads it right after startup; its weights are outside the native-mlx `[memory]` budget |
| `ipc_path` | string | `"/tmp/mlx-runtime.sock"` | Unix domain socket path |

## [generation]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | `0.7` | Default sampling temperature |
| `top_p` | float | `0.9` | Default nucleus sampling threshold |
| `max_tokens` | integer | `512` | Default max generation tokens |

## [limits]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_pending_requests` | integer | `64` | Max queued requests before 429 |
| `max_active_requests` | integer | `16` | Max concurrent active requests |
| `max_prompt_tokens` | integer | `32768` | Max prompt tokens per request |
| `max_completion_tokens` | integer | `4096` | Max completion tokens per request |
| `max_total_tokens_per_request` | integer | `65536` | Max prompt + completion per request. When unset, native-mlx lowers it to what its KV pool holds; when set and the pool cannot hold one such request, native-mlx refuses to start |
| `request_timeout_seconds` | integer | `300` | Queue wait timeout before 429 |
| `max_vlm_images` | integer | `5` | Max images in one VLM request |
| `max_audio_bytes` | integer or size string | `26214400` (25 MiB) | Max audio upload size; larger uploads get 413 before the body is read. Accepts sizes such as `"50MiB"` |

## [memory]

Sizes the native-mlx KV page pool, like vLLM's `--gpu-memory-utilization`.
After the weights load, the worker computes

```text
kv_pool = utilization x Metal recommended working set - weights - activation reserve
```

The activation reserve is the MLX buffer-cache limit
(`MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES`, 1 GiB by default), the hidden and
MLP activations of one full prefill step, last-position logits, and 256 MiB of
slack. The worker logs the result at startup, for example
`native-mlx memory kv_pool=3.41GiB source=memory_utilization utilization=0.5 ...`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `utilization` | float in (0, 1] | `0.5` | Share of the recommended working set for weights, activations, and the KV pool together. CLI: `mlx-air serve --memory-utilization` |
| `kv_cache_bytes` | integer or size string | unset | Absolute KV pool size such as `"4GiB"` or `"512MiB"`; overrides `utilization` for the pool. CLI: `mlx-air serve --kv-cache-memory` |

Startup fails with `KV_MEMORY_INSUFFICIENT` when the utilization cannot hold
the weights plus the reserve, or when an explicit
`limits.max_total_tokens_per_request` does not fit in the pool. The error
names the setting to change.

New requests wait in the queue while the pool cannot hold their prompt plus
`max_tokens` next to the requests already running, so a running request never
fails for lack of KV pages. `kv_admission_waits` in the worker metrics counts
those waits.

The legacy `MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES` environment variable still
pins the pool when `kv_cache_bytes` is unset. The v1 backend does not use the
KV pool; it keeps its own `mlx_lm` prompt cache.

## [telemetry]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_prometheus` | bool | `true` | Expose Prometheus metrics |
| `metrics_path` | string | `"/metrics"` | HTTP path for metrics |

## Example

```toml
[server]
host = "127.0.0.1"
port = 8000

[worker]
python = "python/.venv/bin/python"
module = "mlx_worker.main"
backend = "native-mlx"
model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ipc_path = "/tmp/mlx-runtime.sock"

[generation]
temperature = 0.7
top_p = 0.9
max_tokens = 512

[limits]
max_pending_requests = 64
max_active_requests = 16
max_prompt_tokens = 32768
max_completion_tokens = 4096
# max_total_tokens_per_request = 65536  # unset: derived from the KV pool
request_timeout_seconds = 300

[telemetry]
enable_prometheus = true
metrics_path = "/metrics"
```
