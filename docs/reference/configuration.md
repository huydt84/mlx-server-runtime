# Configuration Reference

The runtime is configured via a TOML file. Default path is `config/runtime.toml`. Override with the `MLX_RUNTIME_CONFIG` environment variable.

If the config file is missing, all defaults apply.

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
| `backend` | string | `"v1"` | Explicit backend selector: `v1` or experimental `native-mlx` |
| `model` | string | `"mlx-community/Qwen2.5-7B-Instruct-4bit"` | Hugging Face model ID |
| `vlm_model` | string or null | `null` | Optional VLM model ID |
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
| `max_total_tokens_per_request` | integer | `65536` | Max prompt + completion per request |
| `request_timeout_seconds` | integer | `300` | Queue wait timeout before 429 |

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
backend = "v1"
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
max_total_tokens_per_request = 65536
request_timeout_seconds = 300

[telemetry]
enable_prometheus = true
metrics_path = "/metrics"
```
