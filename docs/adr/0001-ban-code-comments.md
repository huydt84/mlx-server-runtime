# 0001. Ban code comments; record decisions in ADRs

- Status: Accepted
- Date: 2026-11-22

## Context

A 2026-11-22 audit of every code comment (Rust, Metal, Python, shell) found
374 comment lines. Most repeated what the code already said. Some referred to
code that no longer exists. A few excused a hack ("placeholder", "legacy pin",
"test fixtures") and let it spread. The comments worth keeping recorded
decisions, but they were scattered across call sites where nobody would look
for them as a set, and they went stale silently when the code moved.

## Decision

- Code comments are banned in tracked `.rs`, `.metal`, `.py`, and `.sh` files.
- Allowed, because they are API documentation or tool input rather than
  commentary:
  - Rust doc comments (`///`, `//!`, `/** */`, `/*! */`);
  - Python docstrings (string literals, not `#` comments);
  - the leading header block of a shell script, up to its first non-comment
    line, which documents usage and expected output signals;
  - tool directives: shebangs, `# noqa`, `# type:`, `# pragma`, `# fmt:`,
    `# pyright:`, `# ruff:`, `# shellcheck`, `// SAFETY:`, `// clang-format`,
    `// NOLINT`.
- An important, non-obvious decision goes in an ADR under `docs/adr/`.
- A comment that excuses a hack is not written. File an issue instead.
- `scripts/check_comments.py` enforces the ban as a ratchet. The per-file
  allowance in `scripts/comment-baseline.json` may only go down. A file not in
  the baseline may not contain comments. The check fails when a baseline entry
  is higher than the current count, so the allowance shrinks as soon as a
  comment is deleted. `--update` lowers counts and never raises them.
- Release CI runs the check before the Rust and Python checks.

## Consequences

- When this ADR was accepted, the audit deleted every banned comment, so the
  baseline is empty. The ratchet still applies if a later change has to
  restore a comment temporarily: the baseline entry records the exception, and
  it can only shrink.
- Python comments embedded in shell heredocs delimited by `PY` or `PYTHON` are
  scanned. Other heredoc bodies are treated as data.
- Reviewers have to look for decisions in ADRs, not at the call site. Every
  ADR's "Code" section names files and symbols to bridge that gap.
- Run the `/delete-comment` skill (`.claude/skills/delete-comment`,
  `.agents/skills/delete-comment`) to repeat the audit.

## Code

- `scripts/check_comments.py` — scanner and ratchet.
- `scripts/comment-baseline.json` — per-file allowance.
- `python/tests/test_check_comments.py` — scanner and ratchet behavior.
- `.github/workflows/release.yml` — CI step "Check comment ratchet".
