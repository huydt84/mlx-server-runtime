#!/usr/bin/env bash
# Run a bounded native-MLX whole-pipeline profile.
#
# Arguments are passed to native_pipeline_profile.sh; run with --help to list
# them (--checkpoint, --port, --metal, --artifacts-dir, ...). The matching
# MLX_PROFILE_* environment variables still apply when a flag is not given.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MLX_PROFILE_DIR="${MLX_PROFILE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-profile}"

exec bash "$ROOT/mlx-host-validation/scripts/native_pipeline_profile.sh" "$@"
