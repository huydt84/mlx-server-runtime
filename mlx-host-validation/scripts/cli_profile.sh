#!/usr/bin/env bash
# Host-only validation of `mlx-air profile`: a bounded profiled capture against
# a real model through the staged distribution, the joined artifacts and HTML
# view, the `profiler` alias, `view`, `compare`, and gateway cleanup. With
# --metal it also captures a Metal `.gputrace` for the first request and checks
# that the captured request is kept out of the means (needs several GiB free;
# the trace is deleted after the check).
# Run on Apple Silicon with Metal, uv, and the checkpoint in the local cache.
#
# Usage:
#   bash mlx-host-validation/scripts/cli_profile.sh [options]
#   bash mlx-host-validation/scripts/cli_profile.sh --help
#
# Expected success: cli_profile_* lines and cli_profile_validation_ok=1.
# Expected failure: non-arm64 host, staging failure, readiness timeout, a failed
# request, a missing artifact, a leftover gateway, or a command over its bound.

# shellcheck source-path=SCRIPTDIR

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate mlx-air profile: profiled capture, joined artifacts, HTML view, view, compare, and cleanup."
cli_option CHECKPOINT --checkpoint MLX_CLI_PROFILE_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint profiled by every capture."
cli_option REQUESTS --requests MLX_CLI_PROFILE_REQUESTS int 4 "Profiled requests per capture."
cli_option CONCURRENCY --concurrency MLX_CLI_PROFILE_CONCURRENCY int 2 "Requests in flight per capture."
cli_option MAX_TOKENS --max-tokens MLX_CLI_PROFILE_MAX_TOKENS int 32 "Completion tokens per request."
cli_option COMMAND_TIMEOUT --command-timeout MLX_CLI_PROFILE_COMMAND_TIMEOUT int 1800 \
    "Seconds allowed for each profile capture."
cli_option UV_CACHE --uv-cache-dir MLX_AIR_HOST_UV_CACHE_DIR path "$HOME/Library/Caches/uv" \
    "uv cache shared with the isolated test home."
cli_option METAL --metal MLX_CLI_PROFILE_METAL bool 0 \
    "Also validate a bounded Metal .gputrace capture (needs several GiB of free disk)."
cli_option KEEP_WORK_DIR --keep-work-dir MLX_CLI_PROFILE_KEEP_WORK_DIR bool 0 \
    "Keep the temporary work directory (staged distribution and profiles) after exit."
cli_parse "$@"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "cli_profile_host_error=Apple Silicon arm64 is required" >&2
    exit 1
fi

TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-cli-profile.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
COMMAND_LOG="$WORK_DIR/command.log"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cleanup() {
    local status=$?
    set +e
    if [[ -n "${COMMAND_PID:-}" ]] && kill -0 "$COMMAND_PID" >/dev/null 2>&1; then
        kill -TERM "$COMMAND_PID" >/dev/null 2>&1 || true
    fi
    if [[ -d "$STAGE_DIR" ]]; then
        pkill -TERM -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1 || true
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "command log after validation failure:" >&2
        sed -n '1,300p' "$COMMAND_LOG" >&2
        find "$WORK_DIR" -name gateway.log -type f -print -exec tail -n 60 {} \; >&2
    fi
    if [[ "$KEEP_WORK_DIR" == "1" ]]; then
        echo "work_dir=$WORK_DIR" >&2
    else
        rm -rf "$WORK_DIR"
    fi
    exit "$status"
}
trap cleanup EXIT

# mlx_air COMMAND... runs the staged CLI with a bound and the isolated home.
mlx_air() {
    HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
        "$STAGE_DIR/bin/mlx-air" "$@" >"$WORK_DIR/stdout" 2>>"$COMMAND_LOG" &
    COMMAND_PID=$!
    local pid="$COMMAND_PID"
    for _ in $(seq 1 $((COMMAND_TIMEOUT * 10))); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            unset COMMAND_PID
            wait "$pid"
            return $?
        fi
        sleep 0.1
    done
    echo "mlx-air $1 exceeded the $COMMAND_TIMEOUT second bound" >&2
    return 1
}

assert_no_staged_gateway() {
    if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
        echo "a staged gateway remains alive after $1" >&2
        exit 1
    fi
}

capture() {
    local label="$1"
    shift
    mlx_air "$@" --model "$CHECKPOINT" --requests "$REQUESTS" \
        --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" \
        --output-dir "$WORK_DIR/$label"
    test "$(cat "$WORK_DIR/stdout")" = "$WORK_DIR/$label/profile.html"
    assert_no_staged_gateway "the $label capture"
}

STEPS=5
if [[ "$METAL" == "1" ]]; then
    STEPS=6
fi

echo "[1/$STEPS] Stage the distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/$STEPS] Profile a bounded non-streaming capture"
capture baseline profile run

echo "[3/$STEPS] Profile a bounded streaming capture through the profiler alias"
capture candidate profiler run --stream

echo "[4/$STEPS] Verify joined artifacts and the HTML view"
python3 - "$WORK_DIR" "$REQUESTS" <<'PY'
import json
import pathlib
import sys

work, requests = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for label in ("baseline", "candidate"):
    directory = work / label
    for name in ("pipeline-events.jsonl", "pipeline-trace.json", "pipeline-report.md",
                 "profile.html", "profile-summary.json", "runtime.toml", "gateway.log"):
        assert (directory / name).stat().st_size > 0, (label, name)
    events = [json.loads(line) for line in (directory / "pipeline-events.jsonl").read_text().splitlines()]
    public = {e["request_id"] for e in events if not e["request_id"].startswith("warmup")}
    gateway = {e["request_id"] for e in events if e["component"] == "gateway"}
    assert len(public) == requests and gateway == public, (label, public, gateway)
    for request_id in public:
        own = [e for e in events if e["request_id"] == request_id]
        terminal = next(e for e in own if (e["component"], e["stage"]) == ("runtime", "terminal"))
        assert terminal["state"] in {"length", "stop", "completed"}, terminal
        # IPC decode and tokenization happen before the request is enqueued.
        before_enqueue = {"terminal", "tokenization", "ipc_decode"}
        others = [e for e in own if e["stage"] not in before_enqueue and e["component"] != "gateway"]
        assert all(terminal["monotonic_ns"] <= e["monotonic_ns"] for e in others), request_id
    summary = json.loads((directory / "profile-summary.json").read_text())
    assert summary["totals"]["request_count"] == requests, summary["totals"]
    assert summary["totals"]["failed_requests"] == 0, summary["totals"]
    trace = json.loads((directory / "pipeline-trace.json").read_text())
    assert len(trace["traceEvents"]) == len(events), label
    page = (directory / "profile.html").read_text()
    assert "<script src" not in page and "forward_dispatch" in page, label
    print(f"cli_profile_{label}_mean_latency_ms={summary['totals']['mean_latency_ms']}")
    print(f"cli_profile_{label}_mean_ttft_ms={summary['totals']['mean_ttft_ms']}")
PY

echo "[5/$STEPS] Rebuild a view and compare the two captures"
mlx_air profile view "$WORK_DIR/baseline" --output "$WORK_DIR/rebuilt.html"
test "$(cat "$WORK_DIR/stdout")" = "$WORK_DIR/rebuilt.html"
test -s "$WORK_DIR/rebuilt.html"
mlx_air profile compare "$WORK_DIR/baseline" "$WORK_DIR/candidate" --output "$WORK_DIR/compare.md"
grep -q "Baseline: \`$WORK_DIR/baseline\`" "$WORK_DIR/compare.md"
grep -q "lower is better" "$WORK_DIR/compare.md"
cmp -s "$WORK_DIR/stdout" "$WORK_DIR/compare.md"

if [[ "$METAL" == "1" ]]; then
    echo "[6/$STEPS] Capture a Metal .gputrace for the first of two requests"
    MTL_CAPTURE_ENABLED=1 mlx_air profile run --metal --model "$CHECKPOINT" \
        --requests 2 --concurrency 1 --max-tokens 8 --output-dir "$WORK_DIR/metal"
    assert_no_staged_gateway "the Metal capture"
    test -d "$WORK_DIR/metal/pipeline.gputrace"
    test -n "$(ls -A "$WORK_DIR/metal/pipeline.gputrace")"
    python3 - "$WORK_DIR/metal" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
events = [json.loads(line) for line in (directory / "pipeline-events.jsonl").read_text().splitlines()]
markers = [(e["request_id"], e["stage"]) for e in events if e["component"] == "metal"]
public = sorted({e["request_id"] for e in events if e["component"] == "gateway"})
assert len(public) == 2, public
first = min(public, key=lambda rid: min(e["monotonic_ns"] for e in events if e["request_id"] == rid))
assert markers == [(first, "capture_start"), (first, "capture_stop")], markers
totals = json.loads((directory / "profile-summary.json").read_text())["totals"]
assert totals["metal_captured_requests"] == 1 and totals["metal_capture_excluded"], totals
assert totals["measured_request_count"] == 1, totals
print(f"cli_profile_metal_mean_latency_ms={totals['mean_latency_ms']}")
PY
    echo "cli_profile_metal_gputrace_kib=$(du -sk "$WORK_DIR/metal/pipeline.gputrace" | cut -f1)"
    rm -rf "$WORK_DIR/metal/pipeline.gputrace"
fi
echo "cli_profile_metal=$METAL"
echo "cli_profile_requests=$REQUESTS"
echo "cli_profile_concurrency=$CONCURRENCY"
echo "cli_profile_validation_ok=1"
