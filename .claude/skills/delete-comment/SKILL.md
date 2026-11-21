---
name: delete-comment
description: Audit code comments in this repository, sort each one into Tautology, Excuse, Stale, or Decision Record, delete it, file an issue for each Excuse, move each Decision Record into an ADR, and lower the comment ratchet baseline. Use when asked to delete, audit, or clean up code comments, when scripts/check_comments.py fails, or when the comment baseline needs to shrink.
---

# Delete Comments

Code comments are banned in this repository
([ADR-0001](../../../docs/adr/0001-ban-code-comments.md)).
`scripts/check_comments.py` enforces the ban as a ratchet against
`scripts/comment-baseline.json`. This skill repeats the audit that ADR-0001
started. Its goal is that every comment line it touches leaves the code.

## What Counts as a Comment

The scanner decides. Do not audit what it allows:

- Rust doc comments (`///`, `//!`, `/** */`, `/*! */`) and Python docstrings
  are API documentation.
- A shell script's leading header block (shebang plus the comment lines before
  the first command) documents usage and expected output signals.
- Tool directives: shebangs, `# noqa`, `# type:`, `# pragma`, `# fmt:`,
  `# pyright:`, `# ruff:`, `# shellcheck`, `// SAFETY:`, `// clang-format`,
  `// NOLINT`.
- TOML configuration comments are not scanned.

## Workflow

1. **Inventory.** Run the scanner and save its output:

   ```bash
   python3 scripts/check_comments.py --list > /tmp/comments.txt
   ```

   If the user named files or a directory, keep only those lines.

2. **Read each comment in context.** Treat consecutive comment lines as one
   entry. Read at least the surrounding function and any symbol the comment
   names. Never classify from the comment text alone.

3. **Classify each entry** as exactly one of these:

   | Category | Test | Action |
   |---|---|---|
   | Tautology | Repeats what the code, a name, a test name, or an assertion message already says, including section banners and arithmetic that the assertion shows | Delete |
   | Stale | Refers to code, defaults, phases, files, or behavior that no longer match the repository | Delete |
   | Excuse | Justifies a hack: "placeholder", "legacy", "workaround", "for tests", "simulate", "unreachable", "until X", a branch that exists only for fixtures, a signature mismatch papered over at the call site | Delete and file an issue |
   | Decision Record | States an important, non-obvious *why*: a platform constraint, a measured performance trade-off, a correctness invariant, a protocol rule | Move into an ADR, then delete |

   Tie-breaks: if a comment both records a decision and excuses a hack,
   record the decision in the ADR and also file the Excuse issue. If deleting
   it would lose a constant's meaning (magic numbers), prefer a named
   constant or a better name over keeping the comment, and classify it as a
   Tautology.

4. **Decision Records → ADRs.** Group them by topic. Extend an existing ADR in
   `docs/adr/` when the topic matches, for example 0004 for scheduler, 0005
   for KV cache, 0006 for decode kernel. Otherwise copy
   `docs/adr/template.md` to the next free number. Write the decision as a
   checkable statement, and name the code by file and symbol, never by line.
   Add new ADRs to the index in `docs/adr/README.md`.

5. **Excuses → issues.** Create one issue per hack:

   ```bash
   gh issue create --title "<type(scope)>: <fix>" --body-file <file>
   ```

   The body must quote the deleted comment and name the file and symbol. It
   must state the concrete problem and propose a fix. Link each issue from the
   ADR that mentions the behavior, if there is one. In zsh, arrays start at
   index 1, so check that each title matches its body after a batch create.
   Filing issues publishes to GitHub: confirm with the user first unless they
   already asked for issues.

6. **Delete.** Remove each comment line; for a trailing comment, remove only
   the comment and the whitespace before it. Then format:

   ```bash
   cargo fmt
   cd python && uv run ruff format . ../scripts && cd ..
   ```

   Run `bash -n` on every edited shell script.

7. **Ratchet.** Lower the baseline. The update never raises a count:

   ```bash
   python3 scripts/check_comments.py --update
   python3 scripts/check_comments.py
   ```

   If some comments must stay for now, the baseline keeps their count. Never
   raise a count by hand-editing `scripts/comment-baseline.json` without the
   user's explicit approval.

8. **Record.** Add a dated `docs/adr/comment-audit-YYYY-MM-DD.md` ledger
   listing each entry's location at the audited commit, category, abridged
   text, and action. Add an `Unreleased` entry to `CHANGELOG.md`.

9. **Validate** with the repository toolchain (see `AGENTS.md`):

   ```bash
   cargo fmt --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace --all-features
   ```

   ```bash
   cd python && uv run ruff format --check . && uv run ruff check . && uv run pytest
   ```

## Report

Give counts per category (entries and lines), the issues filed, the ADRs
created or extended, the baseline before and after, and the net lines removed
(`git diff --shortstat`).
