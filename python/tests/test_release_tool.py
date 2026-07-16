"""Tests for deterministic distribution and Homebrew release tooling."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_TOOL = REPO_ROOT / "scripts/release_tool.py"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run the release tool with text output."""
    return subprocess.run(
        [sys.executable, str(RELEASE_TOOL), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _write_package(path: Path, table: str, name: str, version: str) -> None:
    """Write minimal Cargo or Python project metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'[{table}]\nname = "{name}"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def _fake_repo(root: Path, *, gateway: str, protocol: str, python: str) -> None:
    """Create the package metadata used by release version validation."""
    _write_package(
        root / "rust/crates/gateway/Cargo.toml", "package", "gateway", gateway
    )
    _write_package(
        root / "rust/crates/protocol/Cargo.toml", "package", "protocol", protocol
    )
    _write_package(root / "python/pyproject.toml", "project", "worker", python)


def _fake_stage(root: Path, version: str) -> None:
    """Create a minimal complete staged distribution."""
    files = {
        "bin/mlx-air": b"mlx-air-binary",
        "bin/mlx_runtime_gateway": b"gateway-binary",
        "python/pyproject.toml": b"[project]\nname='mlx-worker'\n",
        "python/uv.lock": b"version = 1\n",
        "python/mlx_worker/__init__.py": b"",
        "python/mlx_benchmark/__init__.py": b"",
        "config/runtime.toml": b"[server]\nport = 8000\n",
        "config/benchmark.toml": b"schema_version = 1\n",
        "licenses/LICENSE": b"MIT\n",
    }
    for relative_path, contents in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    for relative_path in ("bin/mlx-air", "bin/mlx_runtime_gateway"):
        os.chmod(root / relative_path, 0o755)
    _run("metadata", "--stage-dir", str(root), "--version", version)


def _file_snapshot(root: Path) -> dict[str, bytes]:
    """Return file contents keyed by path relative to a test checkout."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_version_accepts_matching_rust_python_and_tag(tmp_path: Path):
    """A semantic tag matching all package versions is accepted."""
    _fake_repo(tmp_path, gateway="1.2.3", protocol="1.2.3", python="1.2.3")

    result = _run("version", "--repo-root", str(tmp_path), "--tag", "v1.2.3")

    assert result.stdout == "1.2.3\n"


def test_version_rejects_mismatched_package_versions(tmp_path: Path):
    """A Python/Rust version mismatch stops release preparation."""
    _fake_repo(tmp_path, gateway="1.2.3", protocol="1.2.3", python="1.2.4")

    result = _run("version", "--repo-root", str(tmp_path), check=False)

    assert result.returncode == 1
    assert "package versions do not match" in result.stderr


def test_archive_is_deterministic_and_contains_versioned_layout(tmp_path: Path):
    """Repeated packaging of one stage produces identical archive bytes."""
    stage = tmp_path / "stage"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _fake_stage(stage, "1.2.3")

    _run(
        "archive",
        "--stage-dir",
        str(stage),
        "--output-dir",
        str(first_output),
        "--version",
        "1.2.3",
    )
    _run(
        "archive",
        "--stage-dir",
        str(stage),
        "--output-dir",
        str(second_output),
        "--version",
        "1.2.3",
    )

    archive_name = "mlx-air-1.2.3-darwin-arm64.tar.gz"
    first_archive = first_output / archive_name
    second_archive = second_output / archive_name
    assert first_archive.read_bytes() == second_archive.read_bytes()
    digest = hashlib.sha256(first_archive.read_bytes()).hexdigest()
    assert (first_output / f"{archive_name}.sha256").read_text() == (
        f"{digest}  {archive_name}\n"
    )
    with tarfile.open(first_archive, "r:gz") as archive:
        names = set(archive.getnames())
    assert "mlx-air-1.2.3/bin/mlx-air" in names
    assert "mlx-air-1.2.3/metadata/layout.json" in names


def test_formula_generator_changes_only_formula_file(tmp_path: Path):
    """A formula update leaves every other tap file untouched."""
    (tmp_path / "README.md").write_text("tap\n", encoding="utf-8")
    (tmp_path / "Formula").mkdir()
    (tmp_path / "Formula/other.rb").write_text("class Other; end\n", encoding="utf-8")
    (tmp_path / "Formula/mlx-air.rb").write_text("old formula\n", encoding="utf-8")
    before = _file_snapshot(tmp_path)
    digest = "a" * 64
    url = (
        "https://github.com/huydt84/mlx-server-runtime/releases/download/"
        "v1.2.3/mlx-air-1.2.3-darwin-arm64.tar.gz"
    )

    _run(
        "formula",
        "--tap-root",
        str(tmp_path),
        "--version",
        "1.2.3",
        "--url",
        url,
        "--sha256",
        digest,
    )

    after = _file_snapshot(tmp_path)
    changed = {path for path in before | after if before.get(path) != after.get(path)}
    assert changed == {"Formula/mlx-air.rb"}
    formula = (tmp_path / "Formula/mlx-air.rb").read_text(encoding="utf-8")
    assert f'url "{url}"' in formula
    assert f'sha256 "{digest}"' in formula
    assert 'depends_on "uv"' in formula
    assert 'prefix.install "config", "licenses", "metadata", "python"' in formula
    assert 'shell_output("#{bin}/mlx-air bench --help")' in formula
    assert "Timeout.timeout(900)" in formula


def test_formula_generator_rejects_invalid_checksum(tmp_path: Path):
    """Formula generation fails before writing an invalid checksum."""
    result = _run(
        "formula",
        "--tap-root",
        str(tmp_path),
        "--version",
        "1.2.3",
        "--url",
        "https://example.com/mlx-air-1.2.3-darwin-arm64.tar.gz",
        "--sha256",
        "bad",
        check=False,
    )

    assert result.returncode == 1
    assert "64 lowercase hexadecimal" in result.stderr
    assert not (tmp_path / "Formula/mlx-air.rb").exists()


def test_release_workflow_separates_dry_run_release_and_tap_permissions():
    """The manual path cannot publish and each publishing job has narrow permissions."""
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'tags:\n      - "v*.*.*"' in workflow
    assert "workflow_dispatch:" in workflow
    tag_only = "if: github.event_name == 'push' && github.ref_type == 'tag'"
    assert workflow.count(tag_only) == 3
    assert "contents: write\n    steps:" in workflow
    assert "id-token: write\n      attestations: write" in workflow
    assert "secrets.HOMEBREW_TAP_TOKEN" in workflow
    assert "--auto --squash" in workflow


def test_release_workflow_archives_the_staging_script_layout():
    """CI packaging delegates layout creation to the one staging implementation."""
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    package_script = (REPO_ROOT / "scripts/package-mlx-air.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/package-mlx-air.sh" in workflow
    assert '"$ROOT/scripts/stage-mlx-air.sh"' in package_script
