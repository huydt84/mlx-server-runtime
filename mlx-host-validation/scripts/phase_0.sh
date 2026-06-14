#!/usr/bin/env bash
#
# Phase 0 host-only MLX validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_0.sh
#
# What this verifies:
#   1. The host is arm64 Apple Silicon.
#   2. The Python environment can import `mlx.core` and execute a small MLX computation.
#   3. The Phase 0 Rust gateway starts successfully.
#   4. The Phase 0 Python worker connects to the gateway and the `/health` endpoint becomes healthy.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_compute_ok=` during the MLX smoke check.
#   - It prints `health_response=healthy` before exiting.
#   - If validation fails, the script exits non-zero and points to the captured gateway log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-gateway-phase-0.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-0-health.txt"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
        wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

echo "[1/4] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/4] Verify Apple Silicon and MLX compute"
uv run python - <<'PY'
import platform

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")

try:
    import mlx.core as mx
except Exception as exc:  # pragma: no cover - host-only path
    raise SystemExit(f"failed to import mlx.core: {exc}")

try:
    values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
except Exception as exc:  # pragma: no cover - host-only path
    raise SystemExit(f"mlx compute failed: {exc}")

print(f"mlx_compute_ok={values}")
PY

echo "[3/4] Start gateway"
cd "$ROOT"
rm -f "$GATEWAY_LOG" "$HEALTH_CAPTURE"
cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

echo "[4/4] Wait for /health to report healthy"
for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/health >"$HEALTH_CAPTURE"; then
        if grep -qx 'healthy' "$HEALTH_CAPTURE"; then
            echo "health_response=healthy"
            echo "gateway_log=$GATEWAY_LOG"
            exit 0
        fi
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

echo "timed out waiting for healthy response; inspect $GATEWAY_LOG" >&2
exit 1
