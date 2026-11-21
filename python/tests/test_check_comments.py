"""Tests for the comment ratchet lint in scripts/check_comments.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts/check_comments.py"

_spec = importlib.util.spec_from_file_location("check_comments", CHECKER)
check_comments = importlib.util.module_from_spec(_spec)
sys.modules["check_comments"] = check_comments
_spec.loader.exec_module(check_comments)


def _texts(path: str, source: str) -> list[str]:
    """Return the banned comment texts found in one source string."""
    return [comment.text for comment in check_comments.scan_file(path, source)]


def test_python_comments_exclude_docstrings_strings_and_directives() -> None:
    source = (
        "#!/usr/bin/env python3\n"
        '"""Module docstring # not a comment."""\n'
        "import os  # noqa: F401\n"
        "value = '# inside a string'\n"
        "# banned line\n"
        "other = 1  # banned trailing\n"
        "typed = []  # type: ignore[var-annotated]\n"
    )

    assert _texts("module.py", source) == ["# banned line", "# banned trailing"]


def test_rust_comments_exclude_doc_comments_strings_and_safety() -> None:
    source = (
        "//! Crate docs.\n"
        "/// Item docs.\n"
        "fn main() {\n"
        '    let url = "http://example.com"; // banned trailing\n'
        '    let raw = r#"// not a comment"#;\n'
        '    let lifetime: &\'static str = "x";\n'
        "    let quote = '\"';\n"
        "    // SAFETY: no preconditions.\n"
        "    /* banned block */\n"
        "    //// banned four slashes\n"
        "}\n"
    )

    assert _texts("lib.rs", source) == [
        "// banned trailing",
        "/* banned block */",
        "//// banned four slashes",
    ]


def test_metal_uses_rust_comment_rules() -> None:
    assert _texts("kernel.metal", "#pragma once\n// banned\nint x; /// doc\n") == [
        "// banned"
    ]


def test_shell_header_block_and_directives_are_allowed() -> None:
    source = (
        "#!/usr/bin/env bash\n"
        "#\n"
        "# Script header documentation.\n"
        "set -euo pipefail\n"
        "# shellcheck source=lib/cli.sh\n"
        'echo "# not a comment" ${#items[@]} $#\n'
        "# banned line\n"
        "run --flag  # banned trailing\n"
    )

    assert _texts("script.sh", source) == ["# banned line", "# banned trailing"]


def test_shell_scans_python_heredocs_but_not_other_heredocs() -> None:
    source = (
        "set -e\n"
        "uv run python - <<'PY' \"$ARG\"\n"
        "# banned python comment\n"
        "print('# fine')\n"
        "PY\n"
        "cat <<'EOF'\n"
        "# usage text\n"
        "EOF\n"
    )

    assert _texts("script.sh", source) == ["# banned python comment"]


def test_check_rejects_counts_above_baseline_and_unlisted_files() -> None:
    found = {
        "a.py": [check_comments.Comment("a.py", 1, "# one")],
        "b.py": [
            check_comments.Comment("b.py", 1, "# one"),
            check_comments.Comment("b.py", 2, "# two"),
        ],
    }

    errors = check_comments.check(found, {"b.py": 1})

    assert errors[0] == "a.py: 1 comment lines, baseline allows 0"
    assert "b.py: 2 comment lines, baseline allows 1" in errors
    assert check_comments.check(found, {"a.py": 1, "b.py": 2}) == []


def test_stale_entries_report_baselines_that_can_shrink() -> None:
    found = {"a.py": [check_comments.Comment("a.py", 1, "# one")]}

    assert check_comments.stale_entries(found, {"a.py": 3, "gone.py": 1}) == [
        "a.py",
        "gone.py",
    ]
    assert check_comments.stale_entries(found, {"a.py": 1}) == []


def _git_repo(root: Path, files: dict[str, str], baseline: dict[str, int]) -> None:
    """Create a git repository with tracked files and a comment baseline."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/comment-baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the checker against a repository root."""
    return subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_fails_on_new_comment_and_names_the_line(tmp_path: Path) -> None:
    _git_repo(tmp_path, {"pkg/mod.py": "x = 1\n# new comment\n"}, {})

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "pkg/mod.py:2: # new comment" in result.stderr


def test_cli_update_only_lowers_baseline(tmp_path: Path) -> None:
    _git_repo(tmp_path, {"mod.py": "# one\n# two\n"}, {"mod.py": 5, "gone.py": 2})

    assert _run(tmp_path).returncode == 1
    assert _run(tmp_path, "--update").returncode == 0

    baseline = json.loads(
        (tmp_path / "scripts/comment-baseline.json").read_text(encoding="utf-8")
    )
    assert baseline == {"mod.py": 2}
    assert _run(tmp_path).returncode == 0


def test_cli_update_never_raises_a_baseline(tmp_path: Path) -> None:
    _git_repo(tmp_path, {"mod.py": "# one\n# two\n"}, {"mod.py": 1})

    assert _run(tmp_path, "--update").returncode == 1


def test_repository_satisfies_its_comment_baseline() -> None:
    result = _run(REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
