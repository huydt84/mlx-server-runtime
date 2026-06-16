# How to Deploy

Run the runtime as a persistent service.

## Manual Start

```bash
# Build Rust binary
cargo build --release --bin gateway

# Set up Python
cd python && uv sync && cd ..

# Run
./target/release/gateway
```

The gateway reads `config/runtime.toml` by default. Override with `MLX_RUNTIME_CONFIG`.

## launchd (macOS)

Create `~/Library/LaunchAgents/com.mlx-runtime.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx-runtime</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/mlx-server-runtime/target/release/gateway</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/mlx-server-runtime</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_RUNTIME_CONFIG</key>
        <string>/path/to/mlx-server-runtime/config/runtime.toml</string>
    </dict>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/mlx-runtime.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/mlx-runtime.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.mlx-runtime.plist
launchctl start com.mlx-runtime
```

## Health Checks

Configure your monitoring system to probe:

- `GET /live` — process alive check
- `GET /ready` — model ready to serve
- `GET /health` — legacy backward-compatible liveness

## Logging

The gateway emits structured JSON request logs to stderr:

```json
{"request_id":"req-1","model":"...","prompt_tokens":19,"stream":false,"queue_time_ms":0,"ttft_ms":null,"latency_ms":120,"completion_tokens":8,"finish_reason":"stop","cancelled":false,"error":null}
```

Pipe stderr to your log collector for structured ingestion.
