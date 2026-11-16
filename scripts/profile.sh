#!/usr/bin/env bash
# Run a bounded native-MLX whole-pipeline profile.
#
# Optional environment variables:
#   MLX_PROFILE_CHECKPOINT   Model checkpoint or local path.
#   MLX_PROFILE_DIR          Directory for profile artifacts.
#   MLX_PROFILE_PORT         Gateway port used during capture.
#   MLX_PROFILE_METAL=1      Capture a bounded Metal .gputrace.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MLX_PROFILE_DIR="${MLX_PROFILE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-profile}"

exec bash "$ROOT/mlx-host-validation/scripts/native_pipeline_profile.sh"
