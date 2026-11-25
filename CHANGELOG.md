# Changelog

All notable changes to MLX Server Runtime and the `mlx-air` distribution are
recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-11-26

### Added

- `scripts/check_comments.py`: a lint that bans code comments in tracked
  Rust, Metal, Python, and shell files, enforced as a per-file ratchet against
  `scripts/comment-baseline.json`. Release CI runs it.
- Architecture Decision Records in `docs/adr/` (ADR-0001 to ADR-0010), plus
  the 2026-11-22 comment audit ledger.
- `/delete-comment` agent skill in `.claude/skills/` and `.agents/skills/` for
  repeating the comment audit.
- This changelog.
- OpenAI-compatible Whisper audio transcription endpoints.
- OpenAI-compatible tool calling on `/v1/chat/completions` for text models
  (`docs/explanation/tool-calling.md`).
- Native kernel warmup at startup. Metal kernels moved to `.metal` files.

### Changed

- Removed all 374 banned comment lines found by the audit: 186 Tautology,
  6 Stale, and 12 Excuse lines were deleted, and 170 Decision Record lines
  moved into ADRs 0002–0009. Runtime behavior is unchanged. The silent-WAV
  warmup header now uses named constants instead of field comments.
- Applied the same audit to the 13 comment lines added by the `mlx-air
  profile` command: 4 Tautology lines deleted, 9 Decision Record lines moved
  into ADR-0010.
- Filed issues for the hacks that the deleted Excuse comments justified:
  [#4](https://github.com/huydt84/mlx-server-runtime/issues/4)–[#10](https://github.com/huydt84/mlx-server-runtime/issues/10).
- `AGENTS.md` and `CONTRIBUTING.md` document the comment ban and the ratchet
  check.
- Packed Inkling-2-Mini steps of more than eight tokens multiply decoded weight
  tiles with up to 32 tokens as SIMD-group matrix products, and no longer
  expand dense weight banks in device memory. The soft-gated rotation computes
  each gated value once. Steps of at most eight tokens are unchanged.
- `packed_prefill` benchmark suite for long-prompt prefill on the 5.4 GB pack.

### Fixed

- Paged decode binds the KV dtype separately from the query dtype.
- The native pipeline profiler flushes incrementally.

## [0.3.0] - 2026-11-18

### Added

- Qwen3.5 architecture support.
- Command-line options for every host-validation script.

### Changed

- `native-mlx` is the default backend.
- Short-context decode closes the gap to `mlx_lm`.

### Fixed

- LFM2-MoE uses sigmoid expert routing.

## [0.2.0] - 2026-11-17

### Added

- `mlx-air` command interface: managed runtime, foreground serving, managed
  launchd instances, and isolated benchmark delegation.
- Declarative, bounded, cache-state benchmark execution with statistical
  reporting, calibration, and isolated profiling diagnostics.
- Automated distribution with a direct installer (replacing the Homebrew tap).

### Changed

- Overlap is the default native execution mode. Prefill overlaps decode, and
  consecutive decode steps are pipelined.
- Decode attention reads straight from KV pages, and logits are computed only
  for each row's sampled token.
- Python 3.12 or 3.13 is required.

### Fixed

- Gateway reads requests on a blocking socket.
- Paged KV is written into the page pool in place, and the MLX buffer cache is
  bounded.
- Decode rows are never padded to a prefill chunk.

[Unreleased]: https://github.com/huydt84/mlx-server-runtime/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/huydt84/mlx-server-runtime/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/huydt84/mlx-server-runtime/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/huydt84/mlx-server-runtime/releases/tag/v0.2.0
