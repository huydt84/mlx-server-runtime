# How to Debug the Python Worker

Inspect and troubleshoot the MLX worker process.

## Worker Output

The worker's stdout and stderr are inherited by the gateway and printed to the gateway's stderr. Start the gateway and look for Python tracebacks:

```bash
cargo run --bin gateway 2>&1 | grep -i error
```

## Run the Worker Directly

You can start the worker independently to test model loading and generation:

```bash
cd python
MLX_RUNTIME_SOCKET=/tmp/mlx-runtime.sock \
MLX_RUNTIME_MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit" \
uv run -m mlx_worker.main
```

This starts the worker, loads the model, and waits for IPC commands on the socket.

## Run with Test Harness

```bash
cd python
uv run pytest tests/ -v
```

Python unit tests exercise IPC encoding/decoding, token limit validation, and the batch completion backend with mocked models.

## Add Debug Logging

The worker uses `print()` (inherited stderr). Add temporary debug output:

```python
# In engine.py or batching.py
print("DEBUG: prompt tokens =", len(prompt_tokens), flush=True)
```

## Common Issues

| Issue | Check |
|-------|-------|
| `ModuleNotFoundError: mlx_lm` | Run `uv sync` in `python/` |
| Model loads slowly | First load downloads weights; subsequent loads use Hugging Face cache |
| Worker crashes silently | Check `MLX_RUNTIME_MODEL` is set or passed via config |
| IPC frame decode errors | Verify JSON line format matches protocol/schema.md |
