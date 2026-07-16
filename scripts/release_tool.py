#!/usr/bin/env python3
"""Build reproducible MLX Air release metadata, archives, and formula updates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import stat
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import tomllib


SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PLATFORM = "darwin-arm64"
LAYOUT_SCHEMA_VERSION = 1
LAYOUT_PATHS = {
    "cli": "bin/mlx-air",
    "gateway": "bin/mlx_runtime_gateway",
    "python_project": "python",
    "runtime_config": "config/runtime.toml",
    "benchmark_config": "config/benchmark.toml",
    "licenses": "licenses",
}
REQUIRED_FILES = (
    "bin/mlx-air",
    "bin/mlx_runtime_gateway",
    "python/pyproject.toml",
    "python/uv.lock",
    "python/mlx_worker/__init__.py",
    "python/mlx_benchmark/__init__.py",
    "config/runtime.toml",
    "config/benchmark.toml",
    "licenses/LICENSE",
    "metadata/version.txt",
    "metadata/layout.json",
)


class ReleaseError(ValueError):
    """Report invalid release inputs or distribution contents."""


def _load_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"failed to read {path}: {error}") from error


def _package_version(path: Path) -> str:
    data = _load_toml(path)
    package = data.get("package") or data.get("project")
    if not isinstance(package, dict) or not isinstance(package.get("version"), str):
        raise ReleaseError(f"package version is missing from {path}")
    return package["version"]


def resolve_version(repo_root: Path, tag: str | None) -> str:
    """Validate Rust and Python package versions and return the release version."""
    version_files = (
        repo_root / "rust/crates/gateway/Cargo.toml",
        repo_root / "rust/crates/protocol/Cargo.toml",
        repo_root / "python/pyproject.toml",
    )
    versions = {path: _package_version(path) for path in version_files}
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        details = ", ".join(
            f"{path.relative_to(repo_root)}={value}" for path, value in versions.items()
        )
        raise ReleaseError(f"package versions do not match: {details}")

    version = unique_versions.pop()
    if not SEMVER_PATTERN.fullmatch(version):
        raise ReleaseError(f"package version is not semantic: {version}")
    if tag:
        if not tag.startswith("v") or not SEMVER_PATTERN.fullmatch(tag[1:]):
            raise ReleaseError(
                f"release tag is not a semantic v-prefixed version: {tag}"
            )
        if tag[1:] != version:
            raise ReleaseError(
                f"release tag {tag} does not match package version {version}"
            )
    return version


def write_metadata(stage_dir: Path, version: str) -> None:
    """Write the version and machine-readable distribution layout metadata."""
    _validate_semver(version)
    metadata_dir = stage_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=False)
    (metadata_dir / "version.txt").write_text(f"{version}\n", encoding="utf-8")
    layout = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "distribution": "mlx-air",
        "version": version,
        "platform": PLATFORM,
        "paths": LAYOUT_PATHS,
    }
    (metadata_dir / "layout.json").write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_stage(stage_dir: Path, version: str) -> None:
    """Validate that a staged distribution contains the complete release layout."""
    _validate_semver(version)
    if not stage_dir.is_dir():
        raise ReleaseError(f"staged distribution does not exist: {stage_dir}")
    for relative_path in REQUIRED_FILES:
        path = stage_dir / relative_path
        if not path.is_file():
            raise ReleaseError(f"staged distribution is missing {relative_path}")
        if path.is_symlink():
            raise ReleaseError(
                f"staged distribution must not contain symlinked file {relative_path}"
            )

    for relative_path in ("bin/mlx-air", "bin/mlx_runtime_gateway"):
        mode = (stage_dir / relative_path).stat().st_mode
        if not mode & stat.S_IXUSR:
            raise ReleaseError(f"staged binary is not executable: {relative_path}")

    staged_version = (
        (stage_dir / "metadata/version.txt").read_text(encoding="utf-8").strip()
    )
    if staged_version != version:
        raise ReleaseError(
            f"staged version metadata {staged_version!r} does not match release version {version}"
        )
    try:
        layout = json.loads(
            (stage_dir / "metadata/layout.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"invalid staged layout metadata: {error}") from error
    expected_layout = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "distribution": "mlx-air",
        "version": version,
        "platform": PLATFORM,
        "paths": LAYOUT_PATHS,
    }
    if layout != expected_layout:
        raise ReleaseError(
            "staged layout metadata does not match the release layout contract"
        )


def create_archive(
    stage_dir: Path, output_dir: Path, version: str
) -> tuple[Path, Path]:
    """Create a deterministic tar.gz archive and its SHA-256 checksum file."""
    validate_stage(stage_dir, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"mlx-air-{version}-{PLATFORM}.tar.gz"
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    for path in (archive, checksum_file):
        if path.exists():
            raise ReleaseError(f"refusing to replace existing release output: {path}")

    archive_root = f"mlx-air-{version}"
    with archive.open("xb") as raw_archive:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_archive, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as tar:
                _add_tar_entry(tar, stage_dir, archive_root)
                for path in sorted(
                    stage_dir.rglob("*"), key=lambda item: item.as_posix()
                ):
                    relative_path = path.relative_to(stage_dir).as_posix()
                    _add_tar_entry(tar, path, f"{archive_root}/{relative_path}")

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_file.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, checksum_file


def _add_tar_entry(tar: tarfile.TarFile, path: Path, archive_name: str) -> None:
    info = tar.gettarinfo(str(path), arcname=archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "wheel"
    info.mtime = 0
    info.pax_headers = {}
    if path.is_dir():
        info.mode = 0o755
    elif archive_name.endswith(("/bin/mlx-air", "/bin/mlx_runtime_gateway")):
        info.mode = 0o755
    else:
        info.mode = 0o644
    if path.is_file():
        with path.open("rb") as file:
            tar.addfile(info, file)
    else:
        tar.addfile(info)


def update_homebrew_formula(
    tap_root: Path, version: str, url: str, sha256: str
) -> Path:
    """Write only Formula/mlx-air.rb in a checked-out Homebrew tap."""
    _validate_semver(version)
    if not SHA256_PATTERN.fullmatch(sha256):
        raise ReleaseError(
            "formula SHA-256 must contain exactly 64 lowercase hexadecimal characters"
        )
    parsed_url = urlparse(url)
    expected_asset = f"mlx-air-{version}-{PLATFORM}.tar.gz"
    if parsed_url.scheme != "https" or Path(parsed_url.path).name != expected_asset:
        raise ReleaseError(f"formula URL must be HTTPS and end with {expected_asset}")
    if not tap_root.is_dir():
        raise ReleaseError(f"tap checkout does not exist: {tap_root}")

    formula_dir = tap_root / "Formula"
    formula_dir.mkdir(exist_ok=True)
    formula_path = formula_dir / "mlx-air.rb"
    formula_path.write_text(_render_formula(version, url, sha256), encoding="utf-8")
    return formula_path


def _render_formula(version: str, url: str, sha256: str) -> str:
    return f'''require "timeout"

class MlxAir < Formula
  desc "Native MLX model serving and benchmarking"
  homepage "https://github.com/huydt84/mlx-server-runtime"
  url "{url}"
  version "{version}"
  sha256 "{sha256}"
  license "MIT"

  depends_on "uv"

  def install
    bin.install "bin/mlx-air", "bin/mlx_runtime_gateway"
    prefix.install "config", "licenses", "metadata", "python"
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/mlx-air version")
    shell_output("#{{bin}}/mlx-air help")
    shell_output("#{{bin}}/mlx-air bench --help")

    ENV["HOME"] = testpath/"home"
    ENV["UV_CACHE_DIR"] = testpath/"uv-cache"
    output = Timeout.timeout(900) {{ shell_output("#{{bin}}/mlx-air doctor") }}
    assert_match "[PASS] Apple Silicon:", output
    benchmark_root = testpath/"home/Library/Application Support/mlx-air/environments/benchmark"
    refute_predicate benchmark_root, :exist?
  end
end
'''


def _validate_semver(version: str) -> None:
    if not SEMVER_PATTERN.fullmatch(version):
        raise ReleaseError(f"version is not semantic: {version}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser(
        "version", help="validate and print package version"
    )
    version_parser.add_argument("--repo-root", type=Path, required=True)
    version_parser.add_argument("--tag")

    metadata_parser = subparsers.add_parser(
        "metadata", help="write staged layout metadata"
    )
    metadata_parser.add_argument("--stage-dir", type=Path, required=True)
    metadata_parser.add_argument("--version", required=True)

    archive_parser = subparsers.add_parser(
        "archive", help="validate and archive a staged layout"
    )
    archive_parser.add_argument("--stage-dir", type=Path, required=True)
    archive_parser.add_argument("--output-dir", type=Path, required=True)
    archive_parser.add_argument("--version", required=True)

    formula_parser = subparsers.add_parser("formula", help="update the tap formula")
    formula_parser.add_argument("--tap-root", type=Path, required=True)
    formula_parser.add_argument("--version", required=True)
    formula_parser.add_argument("--url", required=True)
    formula_parser.add_argument("--sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested release operation."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "version":
            print(resolve_version(args.repo_root.resolve(), args.tag))
        elif args.command == "metadata":
            write_metadata(args.stage_dir.resolve(), args.version)
        elif args.command == "archive":
            archive, checksum = create_archive(
                args.stage_dir.resolve(), args.output_dir.resolve(), args.version
            )
            print(archive)
            print(checksum)
        elif args.command == "formula":
            print(
                update_homebrew_formula(
                    args.tap_root.resolve(), args.version, args.url, args.sha256
                )
            )
        else:  # pragma: no cover - argparse enforces the command set.
            raise ReleaseError(f"unsupported command: {args.command}")
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
