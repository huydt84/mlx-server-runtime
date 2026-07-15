#!/usr/bin/env bash
# Host-only statistical reporting and calibration validation for MLX Air.
# Run on Apple Silicon with Metal, uv, and the fixture model available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-unified-cli-phase-09.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
COMMAND_LOG="$WORK_DIR/command.log"
FIXTURE="$ROOT/mlx-host-validation/fixtures/unified_cli_phase_09.toml"
DELAY_PROXY="$ROOT/mlx-host-validation/scripts/python/unified_cli_phase9_delay_gateway.py"
UV_CACHE="${MLX_AIR_PHASE9_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cleanup() {
    local status=$?
    set +e
    if [[ -n "${COMMAND_PID:-}" ]] && kill -0 "$COMMAND_PID" >/dev/null 2>&1; then
        kill -TERM "$COMMAND_PID" >/dev/null 2>&1 || true
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "phase-9 command log after validation failure:" >&2
        sed -n '1,400p' "$COMMAND_LOG" >&2
        if [[ -n "${RESULT_DIR:-}" ]]; then
            find "$RESULT_DIR" -path '*/logs/*.log' -type f -print -exec sed -n '1,200p' {} \; >&2 || true
        fi
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}

trap cleanup EXIT

wait_bounded() {
    local pid="$1"
    for _ in $(seq 1 36000); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            wait "$pid"
            return $?
        fi
        sleep 0.1
    done
    echo "phase-9 calibration exceeded the 3600 second bound" >&2
    kill -TERM "$pid" >/dev/null 2>&1 || true
    return 1
}

echo "[1/4] Stage the benchmark-capable distribution and controlled-delay proxy"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mv "$STAGE_DIR/bin/mlx_runtime_gateway" "$STAGE_DIR/bin/mlx_runtime_gateway.real"
cp "$DELAY_PROXY" "$STAGE_DIR/bin/mlx_runtime_gateway"
chmod +x "$STAGE_DIR/bin/mlx_runtime_gateway"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/4] Run two bounded repetitions of the unchanged Phase 9 configuration"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
MLX_AIR_PHASE9_DELAY_MS=120 \
    "$STAGE_DIR/bin/mlx-air" bench calibrate \
    --suite phase9 \
    --benchmark-config "$FIXTURE" \
    --repetitions 2 \
    >"$COMMAND_LOG" 2>&1 &
COMMAND_PID=$!
wait_bounded "$COMMAND_PID"
unset COMMAND_PID
RESULT_FILE="$(tail -n 1 "$COMMAND_LOG")"
RESULT_DIR="$(dirname "$RESULT_FILE")"

echo "[3/4] Validate trial-level statistics, tails, calibration, and host observations"
python3 - "$RESULT_FILE" "$FIXTURE" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
data = json.loads(result_path.read_text())
assert data["status"] == "succeeded", data
assert data["command"] == "calibrate", data
assert data["configuration"]["benchmark_config"] == str(pathlib.Path(sys.argv[2]).resolve())
assert data["configuration"]["workloads"][0]["trials"] == 2
assert data["repetitions"] == {"requested": 2, "completed": 2}
assert len(data["runs"]) == 2, data["runs"]
assert len(data["host_observations"]) == 4, data["host_observations"]
for observation in data["host_observations"]:
    assert observation["phase"] in {"before", "after"}, observation
    assert set(("thermal_state", "power_state", "memory_pressure")) <= observation.keys()

measurements = data["repeated_measurements"]
assert len(measurements) == 1, measurements
measurement = measurements[0]
assert measurement["metric"] == "ttft", measurement
assert measurement["unit"] == "ms", measurement
assert measurement["better_direction"] == "lower", measurement
assert measurement["completed_repetition_count"] == 2, measurement
assert measurement["bootstrap_95_interval"]["resampling_unit"] == "run", measurement
assert len(measurement["repetition_values"]) == 2, measurement
assert measurement["run_to_run_range"]["unit"] == "ms", measurement

for run in data["runs"]:
    child = json.loads((result_path.parent / run["result_path"]).read_text())
    primary = child["analysis"]["primary_metrics"]
    assert len(primary) == 1, primary
    summary = primary[0]
    assert summary["independent_trial_count"] == 2, summary
    assert summary["request_count"] == 4, summary
    assert summary["bootstrap_95_interval"]["resampling_unit"] == "trial", summary
    assert summary["mean"] >= 100.0, summary
    assert len(child["analysis"]["tails"]) == 1, child["analysis"]
    tail = child["analysis"]["tails"][0]
    assert tail["request_ttft_p95"]["sample_count"] == 4, tail
    assert tail["trial_wall_time_p95"]["sample_count"] == 2, tail
PY
test -s "$RESULT_DIR/report.md"
echo "trial_statistics_ok=1"
echo "calibration_repeatability_ok=1"
echo "controlled_delay_visible_ok=1"

echo "[4/4] Verify every staged gateway and worker process was reaped"
if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
    echo "a staged phase-9 gateway remains alive" >&2
    exit 1
fi
echo "all_children_reaped=1"
echo "unified_cli_phase_09_ok=1"
