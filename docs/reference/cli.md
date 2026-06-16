# CLI Reference

## Gateway (Rust Binary)

The gateway binary is built from `rust/crates/gateway/src/main.rs`.

```bash
cargo run --bin gateway
# or for production:
cargo build --release --bin gateway
./target/release/gateway
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_RUNTIME_CONFIG` | `config/runtime.toml` | Path to TOML configuration file |

The gateway has no CLI flags. All configuration is read from the TOML file. The binary:

1. Loads config from `MLX_RUNTIME_CONFIG` (or default path).
2. Creates a `RuntimeState` and starts the worker bootstrap thread.
3. Binds the HTTP server and begins accepting connections.

## Worker (Python Module)

The worker is spawned by the gateway as a child process. It can also be run directly for debugging.

```bash
cd python
uv run -m mlx_worker.main
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MLX_RUNTIME_SOCKET` | Unix domain socket path (must match `worker.ipc_path` in config) |
| `MLX_RUNTIME_MODEL` | Model identifier to load |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Normal shutdown |
| 1 | Model loading failed during bootstrap |
