# MLX Server Runtime Documentation

A lightweight MLX inference runtime for Apple Silicon. Rust control plane + Python MLX worker.

## Documentation Map

| Quadrant | Document | Purpose |
|----------|----------|---------|
| **Tutorial** | [Getting Started](tutorial/getting-started.md) | Walk from zero to your first inference request |
| **How-to** | [Run Benchmarks](how-to/run-benchmarks.md) | Measure and compare inference performance |
| **How-to** | [Add an Endpoint](how-to/add-endpoint.md) | Extend the HTTP API surface |
| **How-to** | [Debug the Worker](how-to/debug-worker.md) | Inspect and troubleshoot the Python worker |
| **How-to** | [Deploy](how-to/deploy.md) | Run the runtime as a persistent service |
| **How-to** | [Interpret Metrics](how-to/interpret-metrics.md) | Understand Prometheus metric meaning |
| **How-to** | [Configure Backpressure](how-to/configure-backpressure.md) | Tune request admission and queue limits |
| **Reference** | [HTTP API](reference/api.md) | All endpoints, request/response schemas, status codes |
| **Reference** | [Configuration](reference/configuration.md) | All TOML fields with defaults |
| **Reference** | [IPC Protocol](reference/protocol.md) | UDS handshake and frame format |
| **Reference** | [Metrics](reference/metrics.md) | All Prometheus metrics, labels, types |
| **Reference** | [CLI](reference/cli.md) | Gateway and worker CLI flags |
| **Explanation** | [Architecture](explanation/architecture.md) | Rust/Python split, IPC, lifecycle, design rationale |

## Quick Links

- [PLAN.md](../PLAN.md) — Project plan and roadmap
- [protocol/schema.md](../protocol/schema.md) — Protocol schema
- [config/runtime.toml](../config/runtime.toml) — Default configuration
