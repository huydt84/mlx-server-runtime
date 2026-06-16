# Getting Started

This tutorial walks you from a fresh clone to sending your first chat completion request.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4) — required for MLX
- Rust toolchain (1.75+ recommended)
- Python 3.10+
- `uv` package manager
- Model weights (e.g., a `mlx-community` quantized model)

## 1. Clone and Build

```bash
git clone <repo-url> mlx-server-runtime
cd mlx-server-runtime

# Build the Rust gateway and protocol crates
cargo build --workspace

# Set up the Python environment
cd python
uv sync
cd ..
```

## 2. Choose a Model

Set your model identifier. Small models (3B–8B parameters, 4-bit) are best for first runs:

```bash
export MLX_RUNTIME_MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"
```

The model will be downloaded from Hugging Face on first load if not cached locally.

## 3. Configure

Edit `config/runtime.toml` (or rely on defaults). The key fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `worker.model` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Hugging Face model ID |
| `worker.ipc_path` | `/tmp/mlx-runtime.sock` | Unix domain socket path |
| `server.port` | `8000` | HTTP server port |

## 4. Start the Gateway

```bash
cargo run --bin gateway
```

This command:

1. Starts the Rust HTTP server on `127.0.0.1:8000`.
2. Spawns the Python worker as a child process.
3. Waits for the worker to load the model and signal readiness.

Watch the logs. Model loading takes 5–60 seconds depending on model size.

## 5. Check Health

Probe readiness during startup:

```bash
curl http://127.0.0.1:8000/startup
```

Expected output while loading:

```json
{"status":"starting","phase":"loading_weights","elapsed_seconds":3}
```

When ready:

```bash
curl http://127.0.0.1:8000/ready
```

Expected output:

```json
{"status":"ready","ready":true,...}
```

The legacy health endpoint also works:

```bash
curl http://127.0.0.1:8000/health
# healthy
```

## 6. Send a Non-Streaming Completion

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 32
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-req-1",
  "object": "chat.completion",
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

## 7. Send a Streaming Completion

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Count to 5."}
    ],
    "max_tokens": 32,
    "stream": true
  }'
```

You will receive Server-Sent Events (SSE) with token deltas:

```
data: {"id":"chatcmpl-req-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-req-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":" 2"},"finish_reason":null}]}
...
data: {"id":"chatcmpl-req-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 8. View Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

This returns Prometheus-formatted text with counters, gauges, and histograms.

## 9. Check Model Status

```bash
curl http://127.0.0.1:8000/models
curl http://127.0.0.1:8000/models/mlx-community%2FQwen2.5-7B-Instruct-4bit/status
curl http://127.0.0.1:8000/models/mlx-community%2FQwen2.5-7B-Instruct-4bit/ready
```

## 10. Stop

Press `Ctrl+C` on the gateway process. Rust will terminate the Python worker and clean up the Unix domain socket.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `/ready` returns 503 | Model still loading | Wait. Check `/startup` for phase. |
| `/health` returns `unhealthy` | Worker not ready | Check logs for model load errors. |
| Connection refused | Gateway not running | Verify `cargo run` is executing. |
| Worker crashes after spawn | Missing Python deps | Run `cd python && uv sync`. |
| `model does not match` error | HTTP `model` field mismatch | Use the exact model name from config. |
