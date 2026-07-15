#!/usr/bin/env bash
# Host-only process-isolation and diagnostic-artifact validation for MLX Air.
# Run on Apple Silicon with Metal, uv, and the fixture model available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-unified-cli-phase-10.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
COMMAND_LOG="$WORK_DIR/command.log"
FIXTURE="$ROOT/mlx-host-validation/fixtures/unified_cli_phase_10.toml"
UV_CACHE="${MLX_AIR_PHASE10_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cleanup() {
    local status=$?
    set +e
    if [[ -n "${COMMAND_PID:-}" ]] && kill -0 "$COMMAND_PID" >/dev/null 2>&1; then
        kill -TERM "$COMMAND_PID" >/dev/null 2>&1 || true
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "phase-10 command log after validation failure:" >&2
        sed -n '1,500p' "$COMMAND_LOG" >&2
        if [[ -n "${RESULT_DIR:-}" ]]; then
            find "$RESULT_DIR" -name '*.log' -type f -print -exec sed -n '1,240p' {} \; >&2 || true
        fi
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}

trap cleanup EXIT

wait_bounded() {
    local pid="$1"
    local label="$2"
    for _ in $(seq 1 36000); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            wait "$pid"
            return $?
        fi
        sleep 0.1
    done
    echo "$label exceeded the 3600 second bound" >&2
    kill -TERM "$pid" >/dev/null 2>&1 || true
    return 1
}

echo "[1/5] Stage the benchmark-capable distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/5] Run bounded measurements followed by one representative diagnostic process"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" bench run \
    --suite phase10 \
    --benchmark-config "$FIXTURE" \
    --profile representative \
    >"$COMMAND_LOG" 2>&1 &
COMMAND_PID=$!
wait_bounded "$COMMAND_PID" "phase-10 representative profile"
unset COMMAND_PID
RESULT_FILE="$(tail -n 1 "$COMMAND_LOG")"
RESULT_DIR="$(dirname "$RESULT_FILE")"

echo "[3/5] Validate measurement/diagnostic separation and linked pipeline artifacts"
python3 - "$RESULT_FILE" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
data = json.loads(result_path.read_text())
assert data["status"] == "succeeded", data
assert data["diagnostics"]["status"] == "succeeded", data["diagnostics"]
assert len(data["trials"]) == 2, data["trials"]
assert len(data["processes"]) == 2, data["processes"]
measured, diagnostic = data["processes"]
assert measured["purpose"] == "measurement", measured
assert diagnostic["purpose"] == "diagnostic:gateway", diagnostic
assert measured["pid"] != diagnostic["pid"], data["processes"]
assert measured["stopped_monotonic_ns"] <= diagnostic["started_monotonic_ns"], data["processes"]
for key in (
    "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE",
    "MLX_RUNTIME_NATIVE_GRAPH_PROFILE",
    "MLX_RUNTIME_NATIVE_METAL_CAPTURE",
):
    assert measured["profiling_environment"][key] == "0", measured
assert diagnostic["profiling_environment"]["MLX_RUNTIME_NATIVE_PIPELINE_PROFILE"] == "1", diagnostic

attempts = data["diagnostics"]["attempts"]
assert len(attempts) == 1, attempts
attempt = attempts[0]
assert attempt["family"] == "gateway", attempt
assert attempt["status"] == "succeeded", attempt
assert attempt["profilers"] == ["pipeline"], attempt
for artifact in attempt["artifacts"]:
    assert artifact["status"] == "succeeded", artifact
    assert (result_path.parent / artifact["path"]).is_file(), artifact

report = (result_path.parent / "report.md").read_text()
assert "## Diagnostic artifacts" in report, report
assert "pipeline-report.md" in report, report
PY
echo "representative_process_separation_ok=1"
echo "pipeline_artifact_links_ok=1"

echo "[4/5] Run only the explicitly requested decode diagnostic family"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" bench diagnose \
    --result "$RESULT_FILE" \
    --workload-family decode \
    >>"$COMMAND_LOG" 2>&1 &
COMMAND_PID=$!
wait_bounded "$COMMAND_PID" "phase-10 explicit decode diagnostic"
unset COMMAND_PID

python3 - "$RESULT_FILE" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
data = json.loads(result_path.read_text())
assert data["status"] == "succeeded", data
assert data["diagnostics"]["status"] == "succeeded", data["diagnostics"]
attempts = data["diagnostics"]["attempts"]
assert [attempt["family"] for attempt in attempts] == ["gateway", "decode"], attempts
decode = attempts[-1]
assert decode["profilers"] == ["graph"], decode
assert decode["status"] == "succeeded", decode
graph = next(artifact for artifact in decode["artifacts"] if artifact["kind"] == "graph-profile")
assert graph["status"] == "succeeded", graph
graph_path = result_path.parent / graph["path"]
payload = json.loads(graph_path.read_text())
assert payload["diagnostic_only"] is True, payload
assert payload["excluded_from_measurements"] is True, payload
assert payload["metrics"], payload
assert len(data["trials"]) == 2, data["trials"]
PY
echo "explicit_decode_family_only_ok=1"
echo "graph_profile_artifact_ok=1"

echo "[5/5] Verify every staged gateway and worker process was reaped"
if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
    echo "a staged phase-10 gateway remains alive" >&2
    exit 1
fi
echo "all_children_reaped=1"
echo "unified_cli_phase_10_ok=1"
