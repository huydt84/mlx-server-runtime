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

## Quick Start

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

Benchmark docs live in `docs/how-to/run-benchmarks.md`.

- Phase 6 text-only benchmark: `scripts/benchmark.sh`
- Phase 9 VLM benchmark: `scripts/benchmark-vlm.sh`

Default reports land in `benchmarks/results/`.

For the full argument list, smoke commands, and full-suite commands, read:

- [`docs/how-to/run-benchmarks.md`](docs/how-to/run-benchmarks.md)

Quick start:

```bash
bash scripts/benchmark.sh
```
