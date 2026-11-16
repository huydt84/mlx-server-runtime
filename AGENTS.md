# AGENTS.md

## Default Workflow

Before editing:
1. Read the files directly relevant to the requested change.
2. Inspect existing tests for the area being changed.
3. Prefer the smallest change that satisfies the request.

While editing:
- Keep changes focused and reviewable.
- Match existing code style and local conventions.
- Do not rewrite large areas unless the task requires it.
- Do not add dependencies unless necessary for the current change.
- Update docs when behavior, configuration, protocol, commands, or validation changes.

After editing:
1. Run the narrowest relevant checks first.
2. Run broader validation when practical.
3. Report commands run, results, and anything skipped.

## Clarification Questions

When a user asks for questions about a plan, design, or conflicting documents:

- Ask about decisions that cannot be answered by reading the repository.
- Name the exact command, file, process, interface, or ownership boundary being
  decided.
- Explain the runtime or workflow consequence of each option in concrete terms.
- Define specialized terms before using them. Do not rely on shorthand such as
  "front door", "focus", "capture", or "hermetic" without explaining the
  exact command behavior meant by the term.
- Do not compress several technical decisions into one vague sentence. Use
  enough detail for the user to choose without guessing what an option changes.
- When asking several questions at once, keep each question limited to one
  decision and give a recommended option with the technical reason.

## Required Toolchain

Use the repository toolchain exactly. Do not silently replace tools.

Common required tools:
- `cargo`
- `uv`
- `ruff`
- `pytest`
- repository scripts
- `make` or `just` when the repo uses them

If a required tool is unavailable:
1. Try the repository setup/bootstrap path.
2. If that fails, stop.
3. Report the exact command and error.
4. Ask the user how to proceed.

Do not substitute another formatter, linter, test runner, or package manager without explicit user approval.

## Rust Checks

For Rust changes, use the workspace commands when practical:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Focused checks are fine during development, but final validation should use the broader workspace checks unless the environment prevents it.

Rust serving-path changes must not introduce blocking behavior into async execution.

## Python Checks

For Python changes, use `uv`, `ruff`, and `pytest`:

```bash
cd python
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Use Google-style docstrings for public Python modules, classes, and functions.

Do not replace this toolchain with `pip`, `black`, `unittest`, or other alternatives unless the user explicitly approves it.

## Testing Expectations

Add or update tests for behavior changes.

Prioritize tests for:
- request lifecycle
- scheduling behavior
- cancellation
- timeout handling
- backpressure
- protocol parsing
- worker crash handling
- streaming behavior
- configuration loading

Use mocks for fast unit tests when possible. Keep hardware/model integration checks separate from normal unit tests.

Do not rely only on manual testing for protocol, lifecycle, scheduler, or error-handling changes.

Do not add pytest tests for benchmark workloads or benchmark configuration
data. Validate them through command-level and host-only runs with fixed request
counts and explicit timeouts.

## Host-Only Validation

Some checks may require Apple Silicon, Metal, MLX, or local model files.

When validation is host-only:
- label it clearly as host-only
- do not claim it passed unless it actually ran on compatible hardware
- provide or update a runnable script under `mlx-host-validation/scripts` when the task requires repeatable host validation
- make the script produce deterministic success/failure output
- report exactly what was run and what still needs host execution
- Name each script after the capability it verifies (`native_<capability>.sh`
  for the native runtime, `cli_<command>.sh` for the `mlx-air` CLI), never
  after a plan phase. The same applies to its Python helpers, fixtures,
  environment variables (`MLX_<CAPABILITY>_*`), output keys, and report paths.
- Each script is a separate integration test. It must exercise only its own
  capability and must not invoke another host-validation script.
- When work for a capability lands, extend that capability's script instead of
  adding a new one. Delete a script whose checks another script already covers,
  and do not add completion gates that only re-run other scripts.
- A script may start a real server/model and assert public behavior,
  lifecycle, metrics, parity, and durable results. Performance conclusions use
  results produced by `mlx-air bench run`.
- Do not reference plan phases ("Phase N") in code, comments, docs, or test
  names that outlive the plan.

Do not replace host validation with prose-only instructions when a repeatable script is expected.

## Evaluation Methodology

Performance evaluation must be fair, comprehensive, and explicit:
- Select the suite, optional focus, and profiling mode from
  `benchmarks/config/default.toml` (described in
  `docs/how-to/run-benchmarks.md`) before measurement, and keep them and the
  configuration file unchanged for every result used in the same conclusion.
- A performance conclusion requires `results.json` produced by
  `mlx-air bench run`. Host-validation scripts provide integration evidence.
- When evaluating two software versions, follow "Compare Two Versions" in
  `docs/how-to/run-benchmarks.md`: prepare each distribution independently,
  run the same selection against both, and require matching workload inputs
  and output/token parity.
- Judge each workload by its `primary_metric`; the other metrics explain it.
  Run `mlx-air bench calibrate` first when run-to-run variation is unknown, and
  do not draw a conclusion from a row marked unstable.
- Keep profiling disabled during timed workloads. Run
  `mlx-air bench diagnose` afterward in separate processes when diagnostic
  evidence is needed.
- Use every model role required by the selected suite and focus. A single
  checkpoint is only a targeted smoke test.
- Report absolute latency first and state the direction in words: lower latency
  is better; higher latency is worse. Report absolute throughput and state that
  higher throughput is better.
- Keep correctness/parity, end-to-end latency, TTFT, TPOT, throughput, CPU
  overhead, graph profiles, and Metal idle-gap results separate.
- Do not present a signed percentage without naming the baseline, candidate,
  metric, unit, and whether the delta is better or worse.

## Dependency Policy

Before adding a dependency, check whether:
- the current task really needs it
- an existing dependency or standard library feature is enough
- it affects build time, runtime footprint, or deployment
- it is maintained and appropriate for this repo

Avoid dependencies for small convenience helpers.

## Files and Boundaries

Do not casually edit:
- generated files
- lockfiles, unless dependency changes require it
- secrets or environment files
- unrelated plans or docs
- broad architecture documents unrelated to the task

Never read, print, or modify secrets unless the user explicitly asks and the action is safe.

## Completion Report

Final responses for coding tasks should include:
- changed files
- important behavior changes
- tests/checks run
- checks skipped and why
- host-only validation still required, if any

Do not say the task is complete if required validation failed, was skipped, or was replaced by a non-equivalent tool.
