#!/usr/bin/env bash
# Stage, validate, and archive the native MLX Air arm64 distribution.

set -euo pipefail

usage() {
    echo "Usage: $0 --output-dir PATH [--version VERSION]" >&2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR=""
VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            VERSION="$2"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    usage
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: release archives must be built on Apple Silicon macOS" >&2
    exit 1
fi

if [[ -z "$VERSION" ]]; then
    VERSION="$(python3 "$ROOT/scripts/release_tool.py" version --repo-root "$ROOT")"
else
    VERSION="$(python3 "$ROOT/scripts/release_tool.py" version --repo-root "$ROOT" --tag "v$VERSION")"
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "error: output path already exists: $OUTPUT_DIR" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-air-package.XXXXXX")"
cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

STAGE_DIR="$WORK_DIR/mlx-air-$VERSION"
EXTRACT_DIR="$WORK_DIR/extracted"
UNRELATED_DIR="$WORK_DIR/outside-repository"

"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR" --version "$VERSION"
file "$STAGE_DIR/bin/mlx-air" "$STAGE_DIR/bin/mlx_runtime_gateway" | grep -F "arm64" >/dev/null

python3 "$ROOT/scripts/release_tool.py" archive \
    --stage-dir "$STAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --version "$VERSION"

ARCHIVE="$OUTPUT_DIR/mlx-air-$VERSION-darwin-arm64.tar.gz"
CHECKSUM="$ARCHIVE.sha256"
(
    cd "$OUTPUT_DIR"
    shasum -a 256 -c "$(basename "$CHECKSUM")"
)

mkdir -p "$EXTRACT_DIR" "$UNRELATED_DIR"
tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"
CLI="$EXTRACT_DIR/mlx-air-$VERSION/bin/mlx-air"
(
    cd "$UNRELATED_DIR"
    "$CLI" version | grep -F "mlx-air $VERSION" >/dev/null
    "$CLI" help >/dev/null
    "$CLI" bench --help >/dev/null
)

echo "MLX Air release archive ready at $ARCHIVE"
