"""Tests for deterministic distribution and direct-install release tooling."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_TOOL = REPO_ROOT / "scripts/release_tool.py"
INSTALLER = REPO_ROOT / "install.sh"


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


def _fake_install_release(root: Path, version: str) -> tuple[Path, Path]:
    """Create an installable archive whose fake CLI exposes its version."""
    stage = root / f"stage-{version}"
    output = root / f"release-{version}"
    _fake_stage(stage, version)
    (stage / "bin/mlx-air").write_text(
        f"""#!/bin/sh
case "${{1:-}}" in
    version) printf 'mlx-air {version}\\n' ;;
    help) exit 0 ;;
    bench) test "${{2:-}}" = "--help" ;;
    *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    (stage / "bin/mlx_runtime_gateway").write_text(
        "#!/bin/sh\nexit 0\n", encoding="utf-8"
    )
    os.chmod(stage / "bin/mlx-air", 0o755)
    os.chmod(stage / "bin/mlx_runtime_gateway", 0o755)
    _run(
        "archive",
        "--stage-dir",
        str(stage),
        "--output-dir",
        str(output),
        "--version",
        version,
    )
    archive = output / f"mlx-air-{version}-darwin-arm64.tar.gz"
    return archive, Path(f"{archive}.sha256")


def _run_installer(
    home: Path,
    *args: str,
    check: bool = True,
    modify_path: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the installer against isolated installation and command roots."""
    install_root = home / ".local/share/mlx-air"
    bin_dir = home / ".local/bin"
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["SHELL"] = "/bin/sh"
    if extra_env:
        environment.update(extra_env)
    command = [
        str(INSTALLER),
        "--install-root",
        str(install_root),
        "--bin-dir",
        str(bin_dir),
    ]
    if not modify_path:
        command.append("--no-modify-path")
    command.extend(args)
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


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


def test_installer_supports_install_update_and_explicit_rollback(tmp_path: Path):
    """The active symlink switches while installed versions remain available."""
    home = tmp_path / "home"
    first_archive, first_checksum = _fake_install_release(tmp_path, "1.2.3")
    second_archive, second_checksum = _fake_install_release(tmp_path, "1.2.4")

    _run_installer(
        home,
        "--version",
        "1.2.3",
        "--archive",
        str(first_archive),
        "--checksum",
        str(first_checksum),
    )
    launcher = home / ".local/bin/mlx-air"
    install_root = home / ".local/share/mlx-air"
    assert (
        subprocess.check_output([launcher, "version"], text=True) == "mlx-air 1.2.3\n"
    )

    _run_installer(
        home,
        "--version",
        "1.2.4",
        "--archive",
        str(second_archive),
        "--checksum",
        str(second_checksum),
    )
    assert (
        subprocess.check_output([launcher, "version"], text=True) == "mlx-air 1.2.4\n"
    )
    assert os.readlink(install_root / "current") == "versions/1.2.4"
    assert (install_root / "versions/1.2.3").is_dir()
    assert (install_root / "versions/1.2.4").is_dir()

    _run_installer(
        home,
        "--version",
        "1.2.3",
        "--archive",
        str(first_archive),
        "--checksum",
        str(first_checksum),
    )
    assert (
        subprocess.check_output([launcher, "version"], text=True) == "mlx-air 1.2.3\n"
    )
    assert os.readlink(install_root / "current") == "versions/1.2.3"


def test_installer_resolves_latest_release_and_downloads_its_assets(tmp_path: Path):
    """An install without --version follows latest and verifies downloaded assets."""
    home = tmp_path / "home"
    archive, checksum = _fake_install_release(tmp_path, "1.2.3")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        """#!/bin/sh
output=''
url=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) output=$2; shift 2 ;;
        -w) shift 2 ;;
        http*) url=$1; shift ;;
        *) shift ;;
    esac
done
case "$url" in
    */releases/latest)
        printf '%s' 'https://github.com/huydt84/mlx-server-runtime/releases/tag/v1.2.3'
        ;;
    *.tar.gz.sha256) cp "$FAKE_CHECKSUM" "$output" ;;
    *.tar.gz) cp "$FAKE_ARCHIVE" "$output" ;;
    *) exit 1 ;;
esac
""",
        encoding="utf-8",
    )
    os.chmod(fake_curl, 0o755)
    original_path = os.environ["PATH"]

    result = _run_installer(
        home,
        extra_env={
            "PATH": f"{fake_bin}{os.pathsep}{original_path}",
            "FAKE_ARCHIVE": str(archive),
            "FAKE_CHECKSUM": str(checksum),
            # A generic VERSION in the caller's shell must not pin the install.
            "VERSION": "0.0.1",
        },
    )

    assert "Installed mlx-air 1.2.3" in result.stdout
    assert (
        subprocess.check_output([home / ".local/bin/mlx-air", "version"], text=True)
        == "mlx-air 1.2.3\n"
    )


def test_installer_checksum_failure_preserves_active_version(tmp_path: Path):
    """A failed update cannot change the active version or create its directory."""
    home = tmp_path / "home"
    first_archive, first_checksum = _fake_install_release(tmp_path, "1.2.3")
    second_archive, _ = _fake_install_release(tmp_path, "1.2.4")
    bad_checksum = tmp_path / "bad.sha256"
    bad_checksum.write_text(f"{'0' * 64}  {second_archive.name}\n", encoding="utf-8")
    _run_installer(
        home,
        "--version",
        "1.2.3",
        "--archive",
        str(first_archive),
        "--checksum",
        str(first_checksum),
    )

    result = _run_installer(
        home,
        "--version",
        "1.2.4",
        "--archive",
        str(second_archive),
        "--checksum",
        str(bad_checksum),
        check=False,
    )

    install_root = home / ".local/share/mlx-air"
    launcher = home / ".local/bin/mlx-air"
    assert result.returncode == 1
    assert "checksum mismatch" in result.stderr
    assert os.readlink(install_root / "current") == "versions/1.2.3"
    assert not (install_root / "versions/1.2.4").exists()
    assert (
        subprocess.check_output([launcher, "version"], text=True) == "mlx-air 1.2.3\n"
    )


def test_installer_rejects_unsafe_archive_paths_and_links(tmp_path: Path):
    """Checksummed archives still cannot escape their root or contain links."""
    for label, member in (
        ("traversal", tarfile.TarInfo("mlx-air-1.2.3/../../escape")),
        ("symlink", tarfile.TarInfo("mlx-air-1.2.3/link")),
    ):
        if label == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp"
        archive = tmp_path / f"{label}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.addfile(member)
        checksum = tmp_path / f"{label}.sha256"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

        result = _run_installer(
            tmp_path / f"home-{label}",
            "--version",
            "1.2.3",
            "--archive",
            str(archive),
            "--checksum",
            str(checksum),
            check=False,
        )

        assert result.returncode == 1
        if label == "traversal":
            assert "unsafe path" in result.stderr
        else:
            assert "must not contain symbolic or hard links" in result.stderr


def test_installer_refuses_nonempty_unmanaged_install_root(tmp_path: Path):
    """Existing user content cannot be adopted or overwritten by the installer."""
    home = tmp_path / "home"
    archive, checksum = _fake_install_release(tmp_path, "1.2.3")
    unmanaged_file = home / ".local/share/mlx-air/user-file"
    unmanaged_file.parent.mkdir(parents=True)
    unmanaged_file.write_text("keep\n", encoding="utf-8")

    result = _run_installer(
        home,
        "--version",
        "1.2.3",
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
        check=False,
    )

    assert result.returncode == 1
    assert "non-empty unmanaged path" in result.stderr
    assert unmanaged_file.read_text(encoding="utf-8") == "keep\n"
    assert not (home / ".local/bin/mlx-air").exists()


def test_installer_refuses_unmanaged_launcher_without_writing_install_root(
    tmp_path: Path,
):
    """An existing command cannot be replaced even when the install root is free."""
    home = tmp_path / "home"
    archive, checksum = _fake_install_release(tmp_path, "1.2.3")
    launcher = home / ".local/bin/mlx-air"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("user command\n", encoding="utf-8")

    result = _run_installer(
        home,
        "--version",
        "1.2.3",
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
        check=False,
    )

    assert result.returncode == 1
    assert "launcher not owned" in result.stderr
    assert launcher.read_text(encoding="utf-8") == "user command\n"
    assert not (home / ".local/share/mlx-air").exists()


def test_installer_adds_bin_directory_to_profile_once(tmp_path: Path):
    """Interactive installs configure PATH idempotently when it is missing."""
    home = tmp_path / "home"
    archive, checksum = _fake_install_release(tmp_path, "1.2.3")
    arguments = (
        "--version",
        "1.2.3",
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
    )

    _run_installer(home, *arguments, modify_path=True)
    _run_installer(home, *arguments, modify_path=True)

    profile = (home / ".profile").read_text(encoding="utf-8")
    path_line = f'export PATH="{home}/.local/bin:$PATH"'
    assert profile.count("# Added by MLX Air installer") == 1
    assert profile.count(path_line) == 1


def test_installer_uninstall_preserves_application_data(tmp_path: Path):
    """Uninstall removes managed distributions but not runtime state or caches."""
    home = tmp_path / "home"
    archive, checksum = _fake_install_release(tmp_path, "1.2.3")
    _run_installer(
        home,
        "--version",
        "1.2.3",
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
    )
    application_data = home / "Library/Application Support/mlx-air/instances/example"
    application_data.mkdir(parents=True)
    (application_data / "instance.json").write_text("{}\n", encoding="utf-8")

    _run_installer(home, "--uninstall")

    assert not (home / ".local/share/mlx-air").exists()
    assert not (home / ".local/bin/mlx-air").exists()
    assert (application_data / "instance.json").is_file()


def test_installer_has_valid_posix_shell_syntax():
    """The release installer parses with the system POSIX shell."""
    subprocess.run(["sh", "-n", str(INSTALLER)], check=True)


def test_release_workflow_separates_dry_run_and_release_permissions():
    """The manual path cannot publish and publishing permissions stay narrow."""
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'tags:\n      - "v*.*.*"' in workflow
    assert "workflow_dispatch:" in workflow
    tag_only = "if: github.event_name == 'push' && github.ref_type == 'tag'"
    assert workflow.count(tag_only) == 2
    assert "contents: write\n    steps:" in workflow
    assert "id-token: write\n      attestations: write" in workflow
    assert "HOMEBREW_TAP_TOKEN" not in workflow
    assert "update-homebrew" not in workflow
    assert '"release/install.sh"' in workflow


def test_release_workflow_archives_the_staging_script_layout():
    """CI packaging delegates layout creation to the one staging implementation."""
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    package_script = (REPO_ROOT / "scripts/package-mlx-air.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/package-mlx-air.sh" in workflow
    assert "release/install.sh" in workflow
    assert '"$ROOT/scripts/stage-mlx-air.sh"' in package_script
    assert 'cp "$ROOT/install.sh" "$OUTPUT_DIR/install.sh"' in package_script
