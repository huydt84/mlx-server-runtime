#!/usr/bin/env sh
# Install or update the versioned MLX Air distribution from GitHub Releases.

set -eu

if (set -o pipefail) >/dev/null 2>&1; then
    set -o pipefail
fi

OWNER="huydt84"
REPO="mlx-server-runtime"
APP="mlx-air"
MANAGED_MARKER="huydt84/mlx-server-runtime"

err() {
    printf '%s\n' "$*" >&2
}

die() {
    err "error: $*"
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"
}

usage() {
    cat <<'EOF'
Install or update MLX Air on Apple Silicon macOS.

Usage:
  install.sh [--version VERSION]
  install.sh --version VERSION --archive PATH --checksum PATH
  install.sh --uninstall

Options:
  --version VERSION       Install this release instead of the latest release
                          (or set $MLX_AIR_VERSION).
  --archive PATH          Install a local release archive (requires --checksum).
  --checksum PATH         Verify a local archive with this SHA-256 file.
  --install-root PATH     Versioned distributions (default: ~/.local/share/mlx-air,
                          or $MLX_AIR_INSTALL_ROOT).
  --bin-dir PATH          Public command directory (default: ~/.local/bin,
                          or $MLX_AIR_BIN_DIR).
  --no-modify-path        Do not add the command directory to a shell profile.
  --uninstall             Remove installer-managed distributions and launcher.
  -h, --help              Show this help.

Re-run without --version to update to the latest release. Previously installed
versions are retained, so installing an older version performs a rollback.
EOF
}

validate_version() {
    printf '%s\n' "$1" | grep -Eq \
        '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$' \
        || die "version is not semantic: $1"
}

profile_path() {
    shell_name=${SHELL##*/}
    case "$shell_name" in
        zsh) printf '%s' "${ZDOTDIR:-${HOME}}/.zshrc" ;;
        bash)
            if [ -f "${HOME}/.bashrc" ] || [ ! -f "${HOME}/.bash_profile" ]; then
                printf '%s' "${HOME}/.bashrc"
            else
                printf '%s' "${HOME}/.bash_profile"
            fi
            ;;
        *) printf '%s' "${HOME}/.profile" ;;
    esac
}

ensure_path() {
    install_dir=$1
    case ":${PATH}:" in
        *:"$install_dir":*) return 0 ;;
    esac

    profile=$(profile_path)
    path_line="export PATH=\"${install_dir}:\$PATH\""
    mkdir -p "$(dirname "$profile")"
    touch "$profile"
    if ! grep -Fqs "$path_line" "$profile"; then
        {
            printf '\n# Added by MLX Air installer\n'
            printf '%s\n' "$path_line"
        } >>"$profile"
        err "Added ${install_dir} to PATH in ${profile}"
    fi
    err "Open a new shell or run: . \"${profile}\""
}

resolve_latest_version() {
    need curl
    latest_url=$(curl -fsSL -o /dev/null -w '%{url_effective}' \
        "https://github.com/${OWNER}/${REPO}/releases/latest")
    tag=${latest_url##*/}
    case "$tag" in
        v*) printf '%s' "${tag#v}" ;;
        *) die "could not determine the latest release from ${latest_url}" ;;
    esac
}

verify_checksum() {
    archive=$1
    checksum=$2
    expected=$(awk 'NR == 1 { print $1 }' "$checksum")
    printf '%s\n' "$expected" | grep -Eq '^[0-9a-f]{64}$' \
        || die "checksum file does not start with a lowercase SHA-256 digest: $checksum"
    actual=$(shasum -a 256 "$archive" | awk '{ print $1 }')
    [ "$actual" = "$expected" ] \
        || die "checksum mismatch for $archive (expected $expected, got $actual)"
    err "Verified SHA-256: ${actual}"
}

validate_archive_entries() {
    archive=$1
    archive_root=$2
    list_file=$3
    verbose_file=$4

    tar -tzf "$archive" >"$list_file"
    [ -s "$list_file" ] || die "release archive is empty: $archive"
    while IFS= read -r entry; do
        normalized=${entry%/}
        case "$normalized" in
            "$archive_root"|"$archive_root"/*) ;;
            *) die "archive entry is outside ${archive_root}: ${entry}" ;;
        esac
        case "/$normalized/" in
            */../*|*/./*) die "archive entry contains an unsafe path: ${entry}" ;;
        esac
    done <"$list_file"

    tar -tvzf "$archive" >"$verbose_file"
    if awk 'substr($1, 1, 1) == "l" || substr($1, 1, 1) == "h" { exit 1 }' \
        "$verbose_file"; then
        :
    else
        die "release archive must not contain symbolic or hard links"
    fi
}

validate_distribution() {
    distribution=$1
    version=$2
    for relative in \
        bin/mlx-air \
        bin/mlx_runtime_gateway \
        python/pyproject.toml \
        python/uv.lock \
        config/runtime.toml \
        config/benchmark.toml \
        python/mlx_worker/__init__.py \
        python/mlx_benchmark/__init__.py \
        licenses/LICENSE \
        metadata/version.txt \
        metadata/layout.json
    do
        [ -f "$distribution/$relative" ] \
            || die "installed distribution is missing $relative"
    done
    [ -x "$distribution/bin/mlx-air" ] \
        || die "installed mlx-air is not executable"
    [ -x "$distribution/bin/mlx_runtime_gateway" ] \
        || die "installed mlx_runtime_gateway is not executable"
    installed_version=$(sed -n '1p' "$distribution/metadata/version.txt")
    [ "$installed_version" = "$version" ] \
        || die "distribution metadata version ${installed_version} does not match ${version}"
    "$distribution/bin/mlx-air" version | grep -Fx "mlx-air $version" >/dev/null \
        || die "installed CLI version check failed"
    "$distribution/bin/mlx-air" help >/dev/null \
        || die "installed CLI help check failed"
    "$distribution/bin/mlx-air" bench --help >/dev/null \
        || die "installed benchmark help check failed"
}

remove_installation() {
    marker="$INSTALL_ROOT/.installer-managed"
    [ -f "$marker" ] || die "refusing to remove unmanaged path: $INSTALL_ROOT"
    [ "$(sed -n '1p' "$marker")" = "$MANAGED_MARKER" ] \
        || die "refusing to remove path with an unknown management marker: $INSTALL_ROOT"

    launcher="$BIN_DIR/$APP"
    expected_target="$INSTALL_ROOT/current/bin/$APP"
    if [ -L "$launcher" ] && [ "$(readlink "$launcher")" = "$expected_target" ]; then
        rm "$launcher"
    elif [ -e "$launcher" ] || [ -L "$launcher" ]; then
        die "refusing to remove launcher not owned by this installation: $launcher"
    fi
    case "$INSTALL_ROOT" in
        ""|/|.|..) die "refusing to remove unsafe installation root: $INSTALL_ROOT" ;;
    esac
    rm -rf "$INSTALL_ROOT"
    printf 'Uninstalled %s; application data and model caches were preserved.\n' "$APP"
}

HOME=${HOME:?HOME must be set}
VERSION=${MLX_AIR_VERSION:-}
ARCHIVE_PATH=""
CHECKSUM_PATH=""
INSTALL_ROOT=${MLX_AIR_INSTALL_ROOT:-"$HOME/.local/share/mlx-air"}
BIN_DIR=${MLX_AIR_BIN_DIR:-"$HOME/.local/bin"}
MODIFY_PATH=1
UNINSTALL=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            [ "$#" -ge 2 ] || die "--version requires a value"
            VERSION=$2
            shift 2
            ;;
        --archive)
            [ "$#" -ge 2 ] || die "--archive requires a path"
            ARCHIVE_PATH=$2
            shift 2
            ;;
        --checksum)
            [ "$#" -ge 2 ] || die "--checksum requires a path"
            CHECKSUM_PATH=$2
            shift 2
            ;;
        --install-root)
            [ "$#" -ge 2 ] || die "--install-root requires a path"
            INSTALL_ROOT=$2
            shift 2
            ;;
        --bin-dir)
            [ "$#" -ge 2 ] || die "--bin-dir requires a path"
            BIN_DIR=$2
            shift 2
            ;;
        --no-modify-path)
            MODIFY_PATH=0
            shift
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown argument: $1"
            ;;
    esac
done

need grep
need sed
need awk
need readlink

if [ "$UNINSTALL" -eq 1 ]; then
    [ -z "$ARCHIVE_PATH$CHECKSUM_PATH$VERSION" ] \
        || die "--uninstall cannot be combined with release options"
    remove_installation
    exit 0
fi

[ "$(uname -s)" = "Darwin" ] || die "MLX Air supports macOS only"
[ "$(uname -m)" = "arm64" ] || die "MLX Air supports Apple Silicon only"
need tar
need shasum

if [ -n "$ARCHIVE_PATH" ] || [ -n "$CHECKSUM_PATH" ]; then
    [ -n "$ARCHIVE_PATH" ] && [ -n "$CHECKSUM_PATH" ] \
        || die "--archive and --checksum must be provided together"
    [ -n "$VERSION" ] || die "--version is required with a local archive"
    [ -f "$ARCHIVE_PATH" ] || die "archive does not exist: $ARCHIVE_PATH"
    [ -f "$CHECKSUM_PATH" ] || die "checksum does not exist: $CHECKSUM_PATH"
else
    if [ -z "$VERSION" ]; then
        VERSION=$(resolve_latest_version)
    fi
fi
validate_version "$VERSION"

current="$INSTALL_ROOT/current"
launcher="$BIN_DIR/$APP"
expected_launcher_target="$current/bin/$APP"
if [ -e "$current" ] && [ ! -L "$current" ]; then
    die "refusing to replace non-symlink current path: $current"
fi
if [ -e "$launcher" ] || [ -L "$launcher" ]; then
    if [ ! -L "$launcher" ] || [ "$(readlink "$launcher")" != "$expected_launcher_target" ]; then
        die "refusing to replace launcher not owned by this installation: $launcher"
    fi
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mlx-air-install.XXXXXX")
STAGING_DIR=""
cleanup() {
    if [ -n "$STAGING_DIR" ] && [ -d "$STAGING_DIR" ]; then
        rm -rf "$STAGING_DIR"
    fi
    rm -rf "$tmpdir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

archive_name="mlx-air-$VERSION-darwin-arm64.tar.gz"
if [ -z "$ARCHIVE_PATH" ]; then
    need curl
    base_url="https://github.com/${OWNER}/${REPO}/releases/download/v${VERSION}"
    ARCHIVE_PATH="$tmpdir/$archive_name"
    CHECKSUM_PATH="$tmpdir/$archive_name.sha256"
    err "Downloading ${base_url}/${archive_name}"
    curl -fsSL "${base_url}/${archive_name}" -o "$ARCHIVE_PATH"
    curl -fsSL "${base_url}/${archive_name}.sha256" -o "$CHECKSUM_PATH"
fi

verify_checksum "$ARCHIVE_PATH" "$CHECKSUM_PATH"
archive_root="mlx-air-$VERSION"
validate_archive_entries \
    "$ARCHIVE_PATH" "$archive_root" "$tmpdir/archive.list" "$tmpdir/archive.verbose"

if [ -e "$INSTALL_ROOT" ] && [ ! -f "$INSTALL_ROOT/.installer-managed" ]; then
    if [ -n "$(find "$INSTALL_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        die "refusing to install into non-empty unmanaged path: $INSTALL_ROOT"
    fi
fi
mkdir -p "$INSTALL_ROOT/versions" "$BIN_DIR"
if [ -f "$INSTALL_ROOT/.installer-managed" ]; then
    [ "$(sed -n '1p' "$INSTALL_ROOT/.installer-managed")" = "$MANAGED_MARKER" ] \
        || die "installation root has an unknown management marker: $INSTALL_ROOT"
else
    printf '%s\n' "$MANAGED_MARKER" >"$INSTALL_ROOT/.installer-managed"
fi

STAGING_DIR=$(mktemp -d "$INSTALL_ROOT/.staging.XXXXXX")
tar -xzf "$ARCHIVE_PATH" -C "$STAGING_DIR"
candidate="$STAGING_DIR/$archive_root"
validate_distribution "$candidate" "$VERSION"

destination="$INSTALL_ROOT/versions/$VERSION"
if [ -e "$destination" ]; then
    [ -d "$destination" ] || die "version destination is not a directory: $destination"
    validate_distribution "$destination" "$VERSION"
else
    mv "$candidate" "$destination"
fi

previous_target=""
if [ -L "$current" ]; then
    previous_target=$(readlink "$current")
fi
current_tmp="$INSTALL_ROOT/.current.$$"
launcher_tmp="$BIN_DIR/.mlx-air.$$"
rm -f "$current_tmp" "$launcher_tmp"
ln -s "versions/$VERSION" "$current_tmp"
ln -s "$expected_launcher_target" "$launcher_tmp"
mv -f "$launcher_tmp" "$launcher"
mv -fh "$current_tmp" "$current"

if ! "$launcher" version | grep -Fx "mlx-air $VERSION" >/dev/null; then
    rollback_tmp="$INSTALL_ROOT/.rollback.$$"
    if [ -n "$previous_target" ]; then
        ln -s "$previous_target" "$rollback_tmp"
        mv -fh "$rollback_tmp" "$current"
    else
        rm -f "$current"
    fi
    die "launcher verification failed; restored the previous active version"
fi

if [ "$MODIFY_PATH" -eq 1 ]; then
    ensure_path "$BIN_DIR"
fi

printf 'Installed %s %s to %s\n' "$APP" "$VERSION" "$destination"
printf 'Active command: %s\n' "$launcher"
