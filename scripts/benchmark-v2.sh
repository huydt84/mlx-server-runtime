#!/usr/bin/env bash
# User-facing native-v2 benchmark and optimization regression gate.
#
# Run the comprehensive benchmark:
#   bash scripts/benchmark-v2.sh run
#
# Compare two independently captured source snapshots:
#   bash scripts/benchmark-v2.sh compare \
#     --baseline /path/to/before/results.json \
#     --candidate /path/to/after/results.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv --directory "$ROOT/python" run python "$ROOT/benchmarks/v2_benchmark.py" "$@"
