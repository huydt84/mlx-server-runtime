# shellcheck shell=bash
#
# Shared gateway readiness wait for host-validation scripts.
#
# This file is a library, not a validation script: source it after cli.sh.
#
#   source "$ROOT/mlx-host-validation/scripts/lib/gateway.sh"
#   gateway_wait_ready "$PORT" "$HEALTH_TIMEOUT" "$GATEWAY_PID" "$LOG_PATH"
#
# The wait fails fast instead of waiting out the timeout:
#   - when the gateway process exits before it is ready, and
#   - when `GET /startup` reports `"status":"failed"`, which the gateway keeps
#     reporting after a worker startup failure (bad configuration, missing
#     model, insufficient KV memory) while the process itself stays alive.
# On failure it prints the reason and the last lines of the log to stderr and
# returns 1, so a script running with `set -e` stops at that step.
#
# The library is compatible with macOS bash 3.2.

gateway_log_tail() {
    local log_path="${1:-}"
    if [[ -n "$log_path" && -f "$log_path" ]]; then
        echo "--- last 40 lines of $log_path ---" >&2
        tail -n 40 "$log_path" >&2
    fi
}

# gateway_wait_ready PORT TIMEOUT_SECONDS [PID] [LOG_PATH]
gateway_wait_ready() {
    local port="$1"
    local timeout="$2"
    local pid="${3:-}"
    local log_path="${4:-}"
    local deadline=$((SECONDS + timeout))
    local startup
    while ((SECONDS < deadline)); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
            echo "gateway (pid $pid) exited before it was ready" >&2
            gateway_log_tail "$log_path"
            return 1
        fi
        if curl --silent --fail --max-time 2 \
            "http://127.0.0.1:${port}/ready" >/dev/null 2>&1; then
            return 0
        fi
        startup="$(curl --silent --max-time 2 \
            "http://127.0.0.1:${port}/startup" 2>/dev/null || true)"
        if [[ "$startup" == *'"status":"failed"'* ]]; then
            echo "gateway startup failed: $startup" >&2
            gateway_log_tail "$log_path"
            return 1
        fi
        sleep 0.2
    done
    echo "gateway on port $port was not ready within ${timeout}s" >&2
    gateway_log_tail "$log_path"
    return 1
}
