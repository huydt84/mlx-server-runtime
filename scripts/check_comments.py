#!/usr/bin/env python3
"""Ban new code comments with a per-file ratchet baseline.

Documentation comments stay allowed: Rust ``///``/``//!`` doc comments, Python
docstrings, and the leading header block of a shell script. Tool directives
(shebangs, ``# noqa``, ``# shellcheck``, ``// SAFETY:`` and similar) are also
allowed. Every other comment line counts against the file's entry in the
baseline. A file may never exceed its baseline count, and a file without an
entry may not contain comments. Decision records belong in ``docs/adr``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

BASELINE_PATH = Path("scripts/comment-baseline.json")
RUST_SUFFIXES = {".rs", ".metal"}
PYTHON_SUFFIXES = {".py"}
SHELL_SUFFIXES = {".sh"}
SCANNED_SUFFIXES = RUST_SUFFIXES | PYTHON_SUFFIXES | SHELL_SUFFIXES
PYTHON_DIRECTIVE = re.compile(
    r"^#\s*(noqa\b|type:|pragma\b|fmt:|pyright:|ruff:|-\*-|shellcheck\b)"
)
SHELL_DIRECTIVE = re.compile(r"^#\s*shellcheck\b")
RUST_DIRECTIVE = re.compile(r"^//\s*(SAFETY:|clang-format\b|NOLINT)")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


@dataclass(frozen=True)
class Comment:
    """One banned comment line.

    Attributes:
        path: Repository-relative file path.
        line: One-based line number.
        text: Comment text including its marker.
    """

    path: str
    line: int
    text: str


def _python_comments(path: str, source: str) -> list[Comment]:
    comments: list[Comment] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return _hash_line_comments(path, source.splitlines(), 1)
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line = token.start[0]
        text = token.string
        if line == 1 and text.startswith("#!"):
            continue
        if PYTHON_DIRECTIVE.match(text):
            continue
        comments.append(Comment(path, line, text))
    return comments


def _hash_line_comments(path: str, lines: list[str], first: int) -> list[Comment]:
    comments: list[Comment] = []
    for offset, raw in enumerate(lines):
        text = raw.strip()
        if text.startswith("#") and not PYTHON_DIRECTIVE.match(text):
            comments.append(Comment(path, first + offset, text))
    return comments


def _rust_comments(path: str, source: str) -> list[Comment]:
    comments: list[Comment] = []
    index = 0
    line = 1
    length = len(source)
    while index < length:
        char = source[index]
        nxt = source[index + 1] if index + 1 < length else ""
        if char == "\n":
            line += 1
            index += 1
        elif char == "/" and nxt == "/":
            end = source.find("\n", index)
            end = length if end == -1 else end
            text = source[index:end].rstrip()
            is_doc = (text.startswith("///") and not text.startswith("////")) or (
                text.startswith("//!")
            )
            if not is_doc and not RUST_DIRECTIVE.match(text):
                comments.append(Comment(path, line, text))
            index = end
        elif char == "/" and nxt == "*":
            index, line = _rust_block_comment(path, source, index, line, comments)
        elif (
            char == "r"
            and re.match(r'r#*"', source[index : index + 64])
            and (
                index == 0
                or not (source[index - 1].isalnum() or source[index - 1] == "_")
            )
        ):
            hashes = len(re.match(r"r(#*)", source[index:]).group(1))
            terminator = '"' + "#" * hashes
            start = index + 2 + hashes
            end = source.find(terminator, start)
            end = length if end == -1 else end + len(terminator)
            line += source.count("\n", index, end)
            index = end
        elif char == '"':
            index += 1
            while index < length and source[index] != '"':
                if source[index] == "\\":
                    index += 1
                if index < length and source[index] == "\n":
                    line += 1
                index += 1
            index += 1
        elif char == "'":
            literal = re.match(r"'(\\.[^']*|[^'\\\n])'", source[index:])
            index += len(literal.group(0)) if literal else 1
        else:
            index += 1
    return comments


def _rust_block_comment(
    path: str, source: str, index: int, line: int, comments: list[Comment]
) -> tuple[int, int]:
    start = index
    start_line = line
    depth = 0
    length = len(source)
    while index < length:
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                break
        else:
            if source[index] == "\n":
                line += 1
            index += 1
    text = source[start:index]
    is_doc = (text.startswith("/**") and not text.startswith("/***")) or (
        text.startswith("/*!")
    )
    if not is_doc and text != "/**/":
        for offset, body in enumerate(text.splitlines()):
            comments.append(Comment(path, start_line + offset, body.strip()))
    return index, line


def _shell_comments(path: str, source: str) -> list[Comment]:
    comments: list[Comment] = []
    lines = source.splitlines()
    number = 0
    in_header = True
    heredoc: tuple[str, bool] | None = None
    heredoc_lines: list[str] = []
    heredoc_first = 0
    quote: str | None = None
    for number, raw in enumerate(lines, start=1):
        if heredoc is not None:
            delimiter, is_python = heredoc
            if raw.strip() == delimiter:
                if is_python:
                    comments.extend(
                        _hash_line_comments(path, heredoc_lines, heredoc_first)
                    )
                heredoc = None
            else:
                heredoc_lines.append(raw)
            continue
        stripped = raw.strip()
        if in_header:
            if number == 1 and stripped.startswith("#!"):
                continue
            if stripped.startswith("#"):
                continue
            in_header = False
        comment_at = None
        index = 0
        while index < len(raw):
            char = raw[index]
            if quote is not None:
                if char == "\\" and quote == '"':
                    index += 2
                    continue
                if char == quote:
                    quote = None
            elif char == "\\":
                index += 2
                continue
            elif char in ("'", '"'):
                quote = char
            elif char == "#" and (index == 0 or raw[index - 1] in " \t;|&("):
                comment_at = index
                break
            index += 1
        code = raw if comment_at is None else raw[:comment_at]
        if comment_at is not None:
            text = raw[comment_at:].rstrip()
            if not SHELL_DIRECTIVE.match(text):
                comments.append(Comment(path, number, text))
        if quote is None:
            match = HEREDOC.search(code)
            if match:
                delimiter = match.group(2)
                heredoc = (delimiter, delimiter.upper() in {"PY", "PYTHON"})
                heredoc_lines = []
                heredoc_first = number + 1
    return comments


def scan_file(path: str, source: str) -> list[Comment]:
    """Return the banned comment lines in one source file.

    Args:
        path: Repository-relative path, used for the suffix and in results.
        source: File contents.

    Returns:
        Banned comments in line order.
    """
    suffix = Path(path).suffix
    if suffix in PYTHON_SUFFIXES:
        return _python_comments(path, source)
    if suffix in RUST_SUFFIXES:
        return _rust_comments(path, source)
    if suffix in SHELL_SUFFIXES:
        return _shell_comments(path, source)
    return []


def tracked_files(root: Path) -> list[str]:
    """List git-tracked files that the comment ban applies to.

    Args:
        root: Repository root.

    Returns:
        Sorted repository-relative paths with a scanned suffix.
    """
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted(
        path
        for path in output.split("\0")
        if path and Path(path).suffix in SCANNED_SUFFIXES and (root / path).is_file()
    )


def scan_repository(root: Path) -> dict[str, list[Comment]]:
    """Scan every tracked source file.

    Args:
        root: Repository root.

    Returns:
        Banned comments keyed by path, omitting files without any.
    """
    found: dict[str, list[Comment]] = {}
    for path in tracked_files(root):
        source = (root / path).read_text(encoding="utf-8")
        comments = scan_file(path, source)
        if comments:
            found[path] = comments
    return found


def check(found: dict[str, list[Comment]], baseline: dict[str, int]) -> list[str]:
    """Compare scan results against the ratchet baseline.

    Args:
        found: Banned comments keyed by path.
        baseline: Allowed comment-line count per path.

    Returns:
        Human-readable violations; empty when the ratchet holds.
    """
    errors: list[str] = []
    for path, comments in sorted(found.items()):
        allowed = baseline.get(path, 0)
        if len(comments) > allowed:
            errors.append(
                f"{path}: {len(comments)} comment lines, baseline allows {allowed}"
            )
            for comment in comments:
                errors.append(f"  {comment.path}:{comment.line}: {comment.text}")
    return errors


def stale_entries(
    found: dict[str, list[Comment]], baseline: dict[str, int]
) -> list[str]:
    """List baseline entries that allow more comments than the file has.

    Args:
        found: Banned comments keyed by path.
        baseline: Allowed comment-line count per path.

    Returns:
        Paths whose baseline can be lowered.
    """
    return sorted(
        path for path, allowed in baseline.items() if len(found.get(path, [])) < allowed
    )


def _load_baseline(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_baseline(
    path: Path, found: dict[str, list[Comment]], old: dict[str, int]
) -> None:
    tightened = {
        name: min(len(comments), old.get(name, len(comments)))
        for name, comments in sorted(found.items())
    }
    path.write_text(json.dumps(tightened, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the comment ratchet check.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update",
        action="store_true",
        help="Lower baseline counts to the current scan; never raises a count.",
    )
    mode.add_argument(
        "--list",
        action="store_true",
        help="Print every banned comment line as path:line: text.",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    baseline_path = root / BASELINE_PATH
    found = scan_repository(root)
    baseline = _load_baseline(baseline_path)
    if args.list:
        for comments in found.values():
            for comment in comments:
                print(f"{comment.path}:{comment.line}: {comment.text}")
        return 0
    if args.update:
        _write_baseline(baseline_path, found, baseline)
        errors = check(found, _load_baseline(baseline_path))
        for error in errors:
            print(error, file=sys.stderr)
        return 1 if errors else 0
    errors = check(found, baseline)
    if errors:
        print(
            "New code comments are banned. Move decisions to docs/adr.", file=sys.stderr
        )
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    stale = stale_entries(found, baseline)
    total = sum(len(comments) for comments in found.values())
    print(f"comment ratchet ok: {total} baseline comment lines in {len(found)} files")
    if stale:
        print(
            "baseline can be lowered for: "
            + ", ".join(stale)
            + " (run scripts/check_comments.py --update)"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
