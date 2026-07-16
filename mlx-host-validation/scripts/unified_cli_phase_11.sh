#!/usr/bin/env bash
# Host-only archive installation and foreground lifecycle validation for Phase 11.
# Run on Apple Silicon with macOS 14+, Metal, uv, curl, and the test model available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-air-unified-cli-phase-11.XXXXXX")"
RELEASE_DIR="$WORK_DIR/release"
PREFIX="$WORK_DIR/prefix"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
DOCTOR_LOG="$WORK_DIR/doctor.log"
SERVER_LOG="$WORK_DIR/server.log"
VERSION="$(python3 "$ROOT/scripts/release_tool.py" version --repo-root "$ROOT")"
ARCHIVE="$RELEASE_DIR/mlx-air-$VERSION-darwin-arm64.tar.gz"
CHECKSUM="$ARCHIVE.sha256"
CLI="$PREFIX/bin/mlx-air"
MODEL="${MLX_AIR_PHASE11_MODEL:-mlx-community/gemma-3-270m-it-qat-8bit}"
PORT="${MLX_AIR_PHASE11_PORT:-18111}"
UV_CACHE="${MLX_AIR_PHASE11_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SOCKET_DIR="/tmp/mlx-air-$(id -u)"

cleanup() {
    local status=$?
    set +e
    for pid in "${DOCTOR_PID:-}" "${SERVER_PID:-}" "${WORKER_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM "$pid" >/dev/null 2>&1 || true
            sleep 1
            kill -KILL "$pid" >/dev/null 2>&1 || true
        fi
    done
    if [[ "$status" -ne 0 ]]; then
        echo "doctor output after validation failure:" >&2
        sed -n '1,260p' "$DOCTOR_LOG" >&2
        echo "server output after validation failure:" >&2
        sed -n '1,300p' "$SERVER_LOG" >&2
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}
trap cleanup EXIT

wait_bounded() {
    local pid="$1"
    local label="$2"
    local attempts="$3"
    for _ in $(seq 1 "$attempts"); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            wait "$pid"
            return $?
        fi
        sleep 0.1
    done
    echo "$label exceeded its time limit" >&2
    kill -TERM "$pid" >/dev/null 2>&1 || true
    return 1
}

echo "[1/7] Build the versioned archive with the release staging implementation"
"$ROOT/scripts/package-mlx-air.sh" --output-dir "$RELEASE_DIR" --version "$VERSION"
test -f "$ARCHIVE"
test -f "$CHECKSUM"

echo "[2/7] Verify checksum and install into a temporary prefix"
(
    cd "$RELEASE_DIR"
    shasum -a 256 -c "$(basename "$CHECKSUM")"
)
mkdir -p "$PREFIX" "$OUTSIDE_DIR" "$TEST_HOME"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$PREFIX"
test -x "$PREFIX/bin/mlx-air"
test -x "$PREFIX/bin/mlx_runtime_gateway"
test -f "$PREFIX/metadata/version.txt"
test -f "$PREFIX/metadata/layout.json"

echo "[3/7] Run installed help and version outside the repository"
cd "$OUTSIDE_DIR"
"$CLI" version | grep -F "mlx-air $VERSION" >/dev/null
"$CLI" help >/dev/null
"$CLI" bench --help >/dev/null

echo "[4/7] Run installed doctor with a fixed 30-minute bound"
HOME="$TEST_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$CLI" doctor >"$DOCTOR_LOG" 2>&1 &
DOCTOR_PID=$!
wait_bounded "$DOCTOR_PID" "installed doctor" 18000
unset DOCTOR_PID
grep -F "[PASS] Apple Silicon:" "$DOCTOR_LOG" >/dev/null
grep -F "[PASS] runtime environment:" "$DOCTOR_LOG" >/dev/null
grep -F "[PASS] Metal execution: metal_ok" "$DOCTOR_LOG" >/dev/null

echo "[5/7] Prove runtime and benchmark environments use different installed paths"
HOME="$TEST_HOME" UV_CACHE_DIR="$UV_CACHE" "$CLI" bench run --help >/dev/null
RUNTIME_SETUP="$(find "$TEST_HOME/Library/Application Support/mlx-air/environments/runtime" -name setup.json -print -quit)"
BENCHMARK_SETUP="$(find "$TEST_HOME/Library/Application Support/mlx-air/environments/benchmark" -name setup.json -print -quit)"
test -n "$RUNTIME_SETUP"
test -n "$BENCHMARK_SETUP"
test "$(dirname "$RUNTIME_SETUP")" != "$(dirname "$BENCHMARK_SETUP")"
python3 - "$RUNTIME_SETUP" "$BENCHMARK_SETUP" <<'PY'
import json
import pathlib
import sys

runtime = json.loads(pathlib.Path(sys.argv[1]).read_text())
benchmark = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert runtime["installed_dependency_groups"] == [], runtime
assert benchmark["installed_extras"] == ["bench"], benchmark
PY
echo "separate_environments_ok=1"

echo "[6/7] Start the installed foreground server and wait for readiness"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$CLI" serve --model "$MODEL" --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 18000); do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "installed server exited before readiness" >&2
        exit 1
    fi
    WORKER_PID="$(pgrep -P "$SERVER_PID" | head -n 1 || true)"
    if curl --silent --fail --max-time 1 "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/ready" >/dev/null
test -n "${WORKER_PID:-}"
echo "installed_server_ready=1"

echo "[7/7] Stop the foreground server and verify process/socket cleanup"
SERVER_PROCESS_ID="$SERVER_PID"
kill -TERM "$SERVER_PID"
wait_bounded "$SERVER_PID" "installed foreground server shutdown" 100
unset SERVER_PID
if kill -0 "$WORKER_PID" >/dev/null 2>&1; then
    echo "installed worker remains alive after shutdown: $WORKER_PID" >&2
    exit 1
fi
if find "$SOCKET_DIR" -maxdepth 1 -name "foreground-$SERVER_PROCESS_ID-*.sock" -print -quit 2>/dev/null | grep -q .; then
    echo "foreground socket remains after installed server shutdown" >&2
    exit 1
fi

echo "archive_checksum_ok=1"
echo "installed_cleanup_ok=1"
echo "unified_cli_phase_11_ok=1"
