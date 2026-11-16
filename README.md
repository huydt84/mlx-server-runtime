# MLX Server Runtime

Rust control plane + Python MLX worker for local Apple Silicon inference.

> A serious local MLX inference service for Apple Silicon.
> This is **not** a proxy around `mlx_lm.server`.
> Rust owns serving. Python owns model execution.

## Documentation

Full documentation is in the [`docs/`](docs/) directory:

| Document | Description |
|----------|-------------|
| [Getting Started](docs/tutorial/getting-started.md) | Zero to first inference in 10 steps |
| [Architecture](docs/explanation/architecture.md) | Design rationale and system design |
| [HTTP API Reference](docs/reference/api.md) | All endpoints, schemas, error codes |
| [Configuration Reference](docs/reference/configuration.md) | All TOML fields and defaults |
| [IPC Protocol](docs/reference/protocol.md) | UDS handshake and frame format |
| [Profile Inference](docs/how-to/profiling.md) | Capture and interpret whole-pipeline timing |
| [Contributing](CONTRIBUTING.md) | Add model architectures, debug layer output, and profile model graphs |

## Quick Start

Install the latest Apple Silicon release:

```bash
curl -fsSL \
  https://github.com/huydt84/mlx-server-runtime/releases/latest/download/install.sh |
  sh

~/.local/bin/mlx-air version
~/.local/bin/mlx-air doctor
```

Run the same command again to update. To install or roll back to a specific
release, pass its version to the installer:

```bash
curl -fsSL \
  https://github.com/huydt84/mlx-server-runtime/releases/latest/download/install.sh |
  sh -s -- --version 0.1.0
```

Uninstall the distribution while preserving models, runtime state, logs, and
benchmark results:

```bash
curl -fsSL \
  https://github.com/huydt84/mlx-server-runtime/releases/latest/download/install.sh |
  sh -s -- --uninstall
```

To build from source instead:

```bash
# Build
cargo build --workspace
cd python && uv sync && cd ..

# Run
cargo run --bin gateway

# In another terminal
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen2.5-7B-Instruct-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Endpoints

- `GET /live` checks whether HTTP process is alive.
- `GET /ready` checks whether current model can serve requests.
- `GET /startup` reports startup phase.
- `GET /health` remains backwards-compatible with liveness-style checks.
- `GET /version` reports the gateway package version.
- `GET /models` lists configured model state.
- `GET /models/{model}/status` returns detailed model lifecycle state.
- `GET /models/{model}/ready` returns model-specific readiness.
- `GET /metrics` exposes Prometheus metrics.

## Telemetry

Default config enables Prometheus metrics:

```toml
[telemetry]
enable_prometheus = true
metrics_path = "/metrics"
```

Request logs are structured JSON lines written by the Rust gateway and include:

- `request_id`
- `model`
- `prompt_tokens`
- `max_tokens`
- `stream`
- `queue_time_ms`
- `ttft_ms`
- `latency_ms`
- `completion_tokens`
- `finish_reason`
- `cancelled`
- `error`

## Benchmarks

`mlx-air bench` is the benchmark entrypoint:

```bash
mlx-air bench run --suite smoke
mlx-air bench run --suite optimization --focus scheduler
```

Fair performance runs keep profiling disabled; diagnostics run as separate
processes and never become benchmark rows. Suites, focuses, profiling,
calibration, and the two-version comparison workflow are described in
[Run benchmarks](docs/how-to/run-benchmarks.md).

## Profiling

Whole-pipeline profiling helps explain latency across the Rust gateway, worker
transport, Python runtime, scheduler, executor, cache, model, MLX
synchronization, detokenization, and response streaming. It is disabled by
default and must remain off during fair benchmark runs.

Run the bounded profiling workflow on an Apple Silicon host:

```bash
bash scripts/profile.sh
```

The workflow writes request-correlated JSONL events, a Chrome Trace/Perfetto
timeline, and a Markdown summary. For configuration, optional Metal capture,
artifact interpretation, and failure handling, see
[Profile inference](docs/how-to/profiling.md).
