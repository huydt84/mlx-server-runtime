#!/usr/bin/env bash
# Host-only validation for the MLX Air distribution, environment, and doctor.
# Run on Apple Silicon with macOS 14+, Metal, and uv available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-air-unified-cli-phase-02.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
UNRELATED_DIR="$WORK_DIR/outside-repository"
DOCTOR_OUTPUT="$WORK_DIR/doctor.txt"

cleanup() {
    rm -rf "$WORK_DIR"
}

trap cleanup EXIT

echo "[1/4] Stage the relocatable distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"

test -x "$STAGE_DIR/bin/mlx-air"
test -x "$STAGE_DIR/bin/mlx_runtime_gateway"
test -f "$STAGE_DIR/python/pyproject.toml"
test -f "$STAGE_DIR/python/uv.lock"
test -d "$STAGE_DIR/python/mlx_worker"
test -f "$STAGE_DIR/config/runtime.toml"
test -f "$STAGE_DIR/licenses/LICENSE"

echo "[2/4] Run doctor outside the repository with an isolated home"
mkdir -p "$UNRELATED_DIR" "$WORK_DIR/home"
cd "$UNRELATED_DIR"
HOME="$WORK_DIR/home" \
UV_CACHE_DIR="$WORK_DIR/uv-cache" \
    "$STAGE_DIR/bin/mlx-air" doctor | tee "$DOCTOR_OUTPUT"

echo "[3/4] Verify every host-only diagnostic"
grep -F "[PASS] Apple Silicon:" "$DOCTOR_OUTPUT"
grep -F "[PASS] macOS version:" "$DOCTOR_OUTPUT"
grep -F "[PASS] uv:" "$DOCTOR_OUTPUT"
grep -F "[PASS] runtime directory:" "$DOCTOR_OUTPUT"
grep -F "[PASS] log directory:" "$DOCTOR_OUTPUT"
grep -F "[PASS] instance directory:" "$DOCTOR_OUTPUT"
grep -F "[PASS] socket directory:" "$DOCTOR_OUTPUT"
grep -F "[PASS] runtime environment:" "$DOCTOR_OUTPUT"
grep -F "[PASS] Python imports: imports_ok" "$DOCTOR_OUTPUT"
grep -F "[PASS] Metal execution: metal_ok" "$DOCTOR_OUTPUT"
grep -F "[PASS] port availability:" "$DOCTOR_OUTPUT"

echo "[4/4] Verify the managed environment record"
SETUP_RECORD="$(find "$WORK_DIR/home/Library/Application Support/mlx-air/environments/runtime" -name setup.json -print -quit)"
test -n "$SETUP_RECORD"
grep -F '"installed_dependency_groups": []' "$SETUP_RECORD"
grep -F '"python_executable"' "$SETUP_RECORD"
grep -F '"python_version"' "$SETUP_RECORD"

echo "unified_cli_phase_02_ok=1"
