# Comment Audit: 2026-11-22

Every banned comment found by `scripts/check_comments.py` at commit `b7aa1e2`,
classified under [ADR-0001](0001-ban-code-comments.md). Consecutive comment
lines are one entry. Line numbers refer to `b7aa1e2`.

| Category | Entries | Lines | Action |
|---|---|---|---|
| Tautology | 137 | 186 | Deleted |
| Stale | 2 | 6 | Deleted |
| Excuse | 7 | 12 | Deleted and an issue filed |
| Decision Record | 83 | 170 | Moved into ADRs 0002–0009 and deleted |
| **Total** | 229 | 374 | |

Out of scope, and allowed by ADR-0001: Rust doc comments, Python docstrings,
shell script header blocks, tool directives (`// SAFETY:`, `# shellcheck`,
`# noqa`, `# type:`), and TOML configuration comments.

## Entries

| Location | Lines | Category | Comment (abridged) | Action |
|---|---|---|---|---|
| `mlx-host-validation/scripts/attention_backend_registry.sh:123` | 2 | Decision Record | Decode steps with a head dimension that is a multiple of 32 run the page-table decode k... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `mlx-host-validation/scripts/cli_bench.sh:135` | 1 | Tautology | Start a regular gateway from the cache-state run's runtime config. | Deleted |
| `mlx-host-validation/scripts/cli_bench.sh:320` | 2 | Decision Record | The foreground gateway exits when its worker fails to start; a gateway that stays up re... | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `mlx-host-validation/scripts/cli_bench.sh:325` | 1 | Tautology | The readiness timeout is 1800 s; failing fast means well under it. | Deleted |
| `mlx-host-validation/scripts/cli_bench.sh:574` | 3 | Decision Record | Calibration and the failed-trial check run last because they replace the staged gateway... | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `mlx-host-validation/scripts/cli_bench.sh:645` | 1 | Tautology | Two workloads are configured; the run stops after the first one's trial. | Deleted |
| `mlx-host-validation/scripts/cli_serve_detached.sh:149` | 2 | Decision Record | 1% of the working set cannot hold the weights, so the native worker refuses to start wi... | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `mlx-host-validation/scripts/cli_serve_detached.sh:175` | 1 | Decision Record | The failure names the setting to change. | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `mlx-host-validation/scripts/cli_serve_detached.sh:178` | 1 | Decision Record | launchd finishes 'bootout' asynchronously; allow it a few seconds. | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `mlx-host-validation/scripts/lib/cli.sh:247` | 3 | Tautology | Fail fast when a port the script will bind is already taken.  Arguments: PORT FLAG. FLA... | Deleted |
| `mlx-host-validation/scripts/lib/cli.sh:260` | 1 | Tautology | Split a list option into the named array: cli_split_list NAME VALUE. | Deleted |
| `mlx-host-validation/scripts/lib/gateway.sh:28` | 1 | Tautology | gateway_wait_ready PORT TIMEOUT_SECONDS [PID] [LOG_PATH] | Deleted |
| `mlx-host-validation/scripts/native_audio_transcription.sh:80` | 1 | Decision Record | Unix socket paths are limited to 104 bytes, so keep it out of ARTIFACTS_DIR. | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `mlx-host-validation/scripts/native_audio_transcription.sh:94` | 1 | Tautology | post_audio NAME ENDPOINT EXPECTED_STATUS [curl form args...] | Deleted |
| `mlx-host-validation/scripts/native_continuous_batching.sh:253` | 2 | Tautology | Publish immediately: the main thread starts the next stream as soon as this one has pro... | Deleted |
| `mlx-host-validation/scripts/native_continuous_batching.sh:320` | 4 | Decision Record | The scheduler runs decode and prefill in separate steps, so batch growth shows up on th... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `mlx-host-validation/scripts/native_overlap.sh:80` | 1 | Decision Record | Metal refuses to overwrite an existing capture, so drop the last run's. | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `mlx-host-validation/scripts/native_paged_attention.sh:349` | 2 | Tautology | A non-streaming request that decodes many tokens, so every decode step runs the paged d... | Deleted |
| `mlx-host-validation/scripts/native_paged_attention.sh:491` | 2 | Decision Record | Paged decode kernel against a float32 reference for each query/KV dtype pairing, over s... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `mlx-host-validation/scripts/native_paged_attention.sh:607` | 1 | Tautology | Record the query/KV dtypes the real checkpoint hands the paged decode kernel. | Deleted |
| `mlx-host-validation/scripts/native_paged_attention.sh:659` | 1 | Tautology | A decode-only step for both rows takes the paged decode kernel path. | Deleted |
| `mlx-host-validation/scripts/native_streaming.sh:227` | 3 | Stale | Keep enough paged KV capacity for the larger probe families.  The worker's 8 MiB defaul... | Deleted: describes the 8 MiB worker default as the native pool size; native-mlx now sizes its pool from memory utilization unless pinned |
| `mlx-host-validation/scripts/python/overlap_benchmark.py:398` | 4 | Decision Record | A step's events are recorded together when it resolves, so each synchronize_eval belong... | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `mlx-host-validation/scripts/python/scheduling_policies.py:156` | 3 | Stale | Rust-waiting is exercised by a short timeout/queue-full config in the phase script. Pyt... | Deleted: refers to a plan-phase script that no longer exists |
| `python/mlx_benchmark/loadgen.py:42` | 3 | Decision Record | A transport failure after write/drain is ambiguous: the server may already have execute... | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `python/mlx_benchmark/loadgen.py:326` | 1 | Tautology | Configuration validation makes this unreachable. | Deleted |
| `python/mlx_benchmark/mail_assistant.py:69` | 1 | Decision Record | Header, wrapper, language line, and chat-template overhead around the body. | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `python/mlx_benchmark/mail_assistant.py:71` | 1 | Decision Record | Rough tokens per generated body word; bodies are sized from it. | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `python/mlx_benchmark/runner.py:1271` | 1 | Decision Record | A failed worker startup does not recover; stop waiting now. | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `python/mlx_benchmark/statistics.py:10` | 1 | Tautology | Metric name -> (unit, better direction). | Deleted |
| `python/mlx_worker/batching.py:1008` | 4 | Excuse | ''all_tokens'' was added after the pinned legacy mlx-lm release.  It is an optional cac... | Deleted; [#7](https://github.com/huydt84/mlx-server-runtime/issues/7) |
| `python/mlx_worker/config.py:153` | 2 | Excuse | Legacy pin for the native pool. Zero only disables the v1 prompt cache, so native-mlx t... | Deleted; [#8](https://github.com/huydt84/mlx-server-runtime/issues/8) |
| `python/mlx_worker/main.py:164` | 2 | Decision Record | Default VLM engine factory when env var is set (production path). Engine is NOT constru... | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `python/mlx_worker/main.py:336` | 1 | Decision Record | Model-first dispatch: route by model name, not image presence. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `python/mlx_worker/main.py:342` | 1 | Decision Record | Lazy initialize VLM engine on first request. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `python/mlx_worker/main.py:362` | 1 | Tautology | Text-only engine cannot process image content. | Deleted |
| `python/mlx_worker/main.py:369` | 6 | Decision Record | VLM engine initialization is deferred to ''_generate_vlm'' / ''_stream_vlm'' where the ... | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `python/mlx_worker/main.py:383` | 2 | Excuse | Non-stream: only VLM engine accepts should_cancel. Text engine's complete_chat does not... | Deleted; [#5](https://github.com/huydt84/mlx-server-runtime/issues/5) |
| `python/mlx_worker/native_mlx/attention.py:270` | 2 | Decision Record | Decode reads keys straight from the pages instead of gathering every row's cache into d... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/attention.py:289` | 2 | Decision Record | A single row with no cached prefix attends only to its own new K/V, which are already d... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/attention.py:516` | 3 | Decision Record | Keep mask construction lazy and device-side.  The previous Python loop called ''tolist'... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/bootstrap.py:216` | 2 | Decision Record | A model may refine its config from the loaded weights, for example the K/V activation d... | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/mlx_worker/native_mlx/bootstrap.py:433` | 2 | Decision Record | Some valid MLX exports (including LFM2) store special-token metadata in tokenizer.json/... | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/mlx_worker/native_mlx/bootstrap.py:467` | 2 | Decision Record | Hadamard packs list every end-of-generation id in generation_config.json; the tokenizer... | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/mlx_worker/native_mlx/cache.py:643` | 3 | Decision Record | No row is padded (every decode step, for one): the slot order is the row-major (row, po... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/cache.py:675` | 2 | Decision Record | Consecutive slots (one decode row, for one) take a plain slice copy, which is cheaper o... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/cache.py:685` | 2 | Decision Record | Drop the pool's reference to the previous buffer so the scatter above can donate it rat... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/cache.py:920` | 2 | Decision Record | State-only layers keep an empty placeholder so per-layer indexing stays uniform; stagin... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/cache.py:1200` | 2 | Decision Record | Only pages this step stopped referencing can have become unused; pages whose last pin d... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/cache.py:1211` | 2 | Decision Record | A finished reservation must not keep pool buffers alive, or the next step's in-place pa... | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/executor.py:535` | 3 | Decision Record | Keep model, cache, and async evaluation on the owner thread's default GPU stream.  A se... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/kernels/online_softmax.metal:1` | 6 | Decision Record | Online (streaming) softmax helpers shared by attention kernels.  A kernel that sees sco... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/online_softmax.metal:24` | 2 | Decision Record | Adds 'score' to the running state. Returns the factor that rescales values accumulated ... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/online_softmax.metal:35` | 2 | Decision Record | Factor that rescales a partial result with maximum 'partial_max' to 'global_max'. An em... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/online_softmax.metal:41` | 1 | Tautology | namespace mlx_air | Deleted |
| `python/mlx_worker/native_mlx/kernels/paged_decode.metal:1` | 18 | Decision Record | Decode attention read directly from the paged KV pool.  Launch geometry (see paged_deco... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/paged_decode.metal:72` | 2 | Decision Record | Each simdgroup accumulates an online softmax over a strided subset of this split's keys. | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/paged_decode.metal:116` | 1 | Decision Record | Merge the simdgroups through threadgroup memory. | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/paged_decode.metal:147` | 1 | Decision Record | Unnormalized partial result, scaled to this split's maximum. | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/paged_decode.metal:158` | 2 | Decision Record | Combines the SPLITS partial results of one head. One threadgroup per head, one thread p... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/kernels/paged_decode.metal:183` | 1 | Tautology | namespace mlx_air | Deleted |
| `python/mlx_worker/native_mlx/metal_source.py:60` | 1 | Decision Record | Pragma once is meaningless in a single string and warns there. | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/paged_decode.py:35` | 1 | Decision Record | Split a row's keys until a step launches about this many threadgroups... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/paged_decode.py:37` | 1 | Decision Record | ...but give every split at least this many keys. | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/paged_decode.py:41` | 2 | Decision Record | The kernels live in kernels/paged_decode.metal; these bodies only declare threadgroup s... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/paged_decode.py:148` | 2 | Decision Record | Split count ''s'' covers [32 * s, 64 * s) keys; aim for the middle so a slightly longer... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/mlx_worker/native_mlx/pipeline_profile.py:314` | 3 | Decision Record | 'json.dumps({"traceEvents": [...]}, indent=2) + "\n"' split into a fixed head, one bloc... | Moved to [ADR-0009](0009-benchmark-measurement-integrity.md) |
| `python/mlx_worker/native_mlx/prefix_cache.py:392` | 1 | Decision Record | Every entry holds exactly one page of its own. | Moved to [ADR-0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) |
| `python/mlx_worker/native_mlx/registry.py:205` | 2 | Decision Record | Checkpoints that declare no ''architectures'' but a model type whose weight format a re... | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/mlx_worker/native_mlx/runtime.py:26` | 2 | Decision Record | Startup warmup requests (see NativeRuntime.warmup). Each decodes a few tokens so the ov... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/sampling.py:69` | 2 | Decision Record | Drop a token once the more likely tokens already reach top_p; the most likely token alw... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/sampling.py:86` | 1 | Decision Record | SplitMix64-style mixing so neighbouring positions get unrelated keys. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:103` | 2 | Decision Record | OrderedDict insertion order is the arrival order used by FCFS.  Do not allocate a sorte... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:363` | 1 | Decision Record | Resolving the step reports a launch failure for its rows. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:499` | 1 | Decision Record | Resolving ''step'' reports the commit failure for its rows. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:568` | 1 | Excuse | Placeholder: the executor reads the device-resident token. | Deleted; [#6](https://github.com/huydt84/mlx-server-runtime/issues/6) |
| `python/mlx_worker/native_mlx/scheduler.py:701` | 2 | Decision Record | Every prefill candidate is waiting for KV pages; keep decoding so running requests fini... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:760` | 1 | Decision Record | Eviction may have dropped the matched entry. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:767` | 2 | Decision Record | FCFS: later arrivals wait behind this one so a long request is not starved by shorter o... | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:851` | 1 | Decision Record | One extra page covers a copy-on-write split of a shared last page. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/scheduler.py:933` | 2 | Decision Record | A chained decode step can run one token past a terminal token; that token is discarded. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/worker.py:47` | 1 | Decision Record | Largest single socket read; audio uploads arrive as multi-megabyte lines. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/native_mlx/worker.py:158` | 1 | Decision Record | Runs to completion here; text scheduling resumes after. | Moved to [ADR-0004](0004-native-scheduler-and-sampling.md) |
| `python/mlx_worker/vlm_engine.py:46` | 3 | Tautology | --------------------------------------------------------------------------- Image sourc... | Deleted |
| `python/mlx_worker/vlm_engine.py:433` | 1 | Tautology | Deferred model state — populated by initialize(). | Deleted |
| `python/mlx_worker/vlm_engine.py:439` | 1 | Tautology | Telemetry state. | Deleted |
| `python/mlx_worker/vlm_engine.py:446` | 3 | Tautology | ------------------------------------------------------------------ Initialization -----... | Deleted |
| `python/mlx_worker/vlm_engine.py:469` | 1 | Tautology | caller handles cancelled response | Deleted |
| `python/mlx_worker/vlm_engine.py:473` | 3 | Decision Record | Load completed but request was cancelled. Keep loaded model/processor so next request r... | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `python/mlx_worker/vlm_engine.py:485` | 3 | Tautology | ------------------------------------------------------------------ Config loader ------... | Deleted |
| `python/mlx_worker/vlm_engine.py:670` | 1 | Tautology | Use the full mlx-vlm prompt utility stack. | Deleted |
| `python/mlx_worker/vlm_engine.py:685` | 2 | Tautology | Some model types return a dict or list; finalise through the template. | Deleted |
| `python/mlx_worker/vlm_engine.py:692` | 1 | Excuse | No loaded config (test fixtures): use processor directly. | Deleted; [#4](https://github.com/huydt84/mlx-server-runtime/issues/4) |
| `python/mlx_worker/vlm_engine.py:901` | 3 | Tautology | ------------------------------------------------------------------ Telemetry accessors ... | Deleted |
| `python/tests/test_main.py:854` | 1 | Tautology | Text engine should NOT have been called. | Deleted |
| `python/tests/test_main.py:906` | 1 | Tautology | Text-only request targeting VLM model. | Deleted |
| `python/tests/test_main.py:1150` | 1 | Tautology | Request targets the text model — VLM should stay uninitialized. | Deleted |
| `python/tests/test_main.py:1186` | 1 | Tautology | Must patch before main() triggers the local import. | Deleted |
| `python/tests/test_main.py:1260` | 1 | Tautology | VLM engine was lazily constructed once on first VLM request. | Deleted |
| `python/tests/test_main.py:1265` | 1 | Tautology | Verify the VLM request actually routed to the VLM engine. | Deleted |
| `python/tests/test_main.py:1290` | 1 | Tautology | Simulate the real engine: check cancel before initialize(). | Deleted |
| `python/tests/test_main.py:1327` | 1 | Tautology | VLM request followed immediately by cancel for the same request. | Deleted |
| `python/tests/test_main.py:1350` | 1 | Tautology | Force select to return readable so should_cancel picks up the cancel. | Deleted |
| `python/tests/test_main.py:1378` | 1 | Tautology | skip deferred init for this test | Deleted |
| `python/tests/test_main.py:1391` | 1 | Tautology | Simulate cancellation check inside the engine. | Deleted |
| `python/tests/test_main.py:1427` | 1 | Tautology | VLM non-stream request followed by cancel. | Deleted |
| `python/tests/test_main.py:1461` | 1 | Tautology | The VLM engine saw cancel and returned cancelled. | Deleted |
| `python/tests/test_native_mlx.py:741` | 1 | Tautology | The child's append pins the shared page and copies it on write. | Deleted |
| `python/tests/test_native_mlx.py:748` | 2 | Tautology | A commit moves the child off the shared page, which no request now references; it must ... | Deleted |
| `python/tests/test_native_mlx.py:1432` | 1 | Tautology | Each request needs ceil((4 + 8) / 4) + 1 = 4 pages through its last token. | Deleted |
| `python/tests/test_native_mlx.py:1643` | 1 | Tautology | Prefill is dispatched asynchronously and resolves on the next tick. | Deleted |
| `python/tests/test_native_mlx.py:1706` | 1 | Tautology | prefill dispatched | Deleted |
| `python/tests/test_native_mlx.py:1709` | 2 | Tautology | Resolving the prefill already built the first decode, which is enqueued on the next tic... | Deleted |
| `python/tests/test_native_mlx.py:1714` | 1 | Tautology | ...and the decode after that decode is enqueued before it resolves. | Deleted |
| `python/tests/test_native_mlx.py:1724` | 1 | Tautology | max_tokens is known ahead, so no step runs past the last token. | Deleted |
| `python/tests/test_native_mlx.py:1743` | 2 | Tautology | 1 -> 2 -> 3 -> 0 (terminal). The step after 0 was already running when 0 resolved; its ... | Deleted |
| `python/tests/test_native_mlx.py:1766` | 2 | Tautology | "short" leaves the chain at max_tokens, so later chained steps feed "long" only its own... | Deleted |
| `python/tests/test_native_mlx.py:1784` | 2 | Tautology | prefill dispatched first decode dispatched | Deleted |
| `python/tests/test_native_mlx.py:1790` | 2 | Tautology | The pending prefill was built before the first decode resolved, and is enqueued only on... | Deleted |
| `python/tests/test_native_mlx.py:1798` | 2 | Tautology | The decode after it mixes the prefill's device-resident token with the resolved token o... | Deleted |
| `python/tests/test_native_mlx_metal_source.py:33` | 1 | Tautology | System includes are left for the Metal compiler. | Deleted |
| `python/tests/test_native_mlx_model_families.py:274` | 2 | Tautology | Sigmoid scores are ~[0.881, 0.5, 0.269]; the bias makes expert 2 win the selection over... | Deleted |
| `python/tests/test_native_mlx_paged_attention.py:92` | 2 | Tautology | head_dim 8 takes the gather path for every step; 32 sends decode steps through the page... | Deleted |
| `python/tests/test_native_mlx_paged_attention.py:158` | 2 | Decision Record | Quantized checkpoints can run float16 activations over a bfloat16 cache (and the revers... | Moved to [ADR-0006](0006-paged-decode-attention-kernel.md) |
| `python/tests/test_native_mlx_paged_attention.py:215` | 1 | Tautology | The reference attends in float32 over the exact stored values. | Deleted |
| `python/tests/test_native_mlx_paged_attention.py:238` | 1 | Tautology | Every split keeps at least 32 keys. | Deleted |
| `python/tests/test_native_mlx_paged_attention.py:252` | 1 | Tautology | One row's slots are consecutive, even across a page boundary. | Deleted |
| `python/tests/test_native_mlx_paged_attention.py:257` | 1 | Tautology | Two rows land on different pages, so their slots need a scatter. | Deleted |
| `python/tests/test_native_mlx_prefix_reclaim.py:26` | 1 | Tautology | One layer, one head, head_dim 2, float16: 64 bytes per 8-token page. | Deleted |
| `python/tests/test_native_mlx_prefix_reclaim.py:73` | 1 | Tautology | The old leaf's second page is shared with nothing else. | Deleted |
| `python/tests/test_native_mlx_prefix_reclaim.py:141` | 1 | Tautology | Each request needs ceil((4 + 8) / 4) + 1 = 4 pages through its last token. | Deleted |
| `python/tests/test_native_mlx_qwen3_5.py:347` | 1 | Tautology | row i = rotate(e_i) | Deleted |
| `python/tests/test_native_mlx_qwen3_5.py:349` | 1 | Decision Record | Linear: stored W' = W R^T, so W' rotate(x) = W x. | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/tests/test_native_mlx_qwen3_5.py:357` | 1 | Decision Record | Embedding: stored rows are rotate(e), returned rows are e. | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/tests/test_native_mlx_qwen3_5.py:458` | 1 | Decision Record | float32 norms keep the reference residual stream, and so K/V, in float32. | Moved to [ADR-0008](0008-checkpoint-loading-compatibility.md) |
| `python/tests/test_native_mlx_sampling.py:86` | 1 | Tautology | 0.4 + 0.3 reaches 0.5 only with the second token; the rest are dropped. | Deleted |
| `python/tests/test_native_mlx_sampling.py:143` | 2 | Tautology | The recording model's logits are [0, 1, 2, 3]; at temperature 1.5 more than the greedy ... | Deleted |
| `python/tests/test_native_mlx_warmup.py:23` | 2 | Tautology | Each length keeps its split count when the prompt runs a little long or short, or the w... | Deleted |
| `python/tests/test_native_mlx_warmup.py:136` | 1 | Tautology | Single requests run alone; the pair shares steps. | Deleted |
| `python/tests/test_native_mlx_warmup.py:139` | 1 | Tautology | Each request decodes more than one step after its first token. | Deleted |
| `python/tests/test_release_tool.py:287` | 1 | Tautology | A generic VERSION in the caller's shell must not pin the install. | Deleted |
| `python/tests/test_vlm_engine.py:848` | 2 | Tautology | Cancel on the 9th call = inside the 2nd loop iteration, after the first token delta has... | Deleted |
| `python/tests/test_vlm_engine.py:860` | 2 | Tautology | no full text accumulator on loop cancel only first token emitted before cancel | Deleted |
| `python/tests/test_vlm_engine.py:887` | 1 | Tautology | no deltas emitted at all | Deleted |
| `python/tests/test_vlm_engine.py:914` | 1 | Tautology | Cancel on the 6th call = after preprocessing, before prompt build. | Deleted |
| `python/tests/test_vlm_engine.py:930` | 1 | Tautology | Telemetry reflects preprocessing that already ran. | Deleted |
| `python/tests/test_vlm_engine.py:1042` | 1 | Tautology | one image in _vlm_request | Deleted |
| `python/tests/test_vlm_engine.py:1117` | 3 | Tautology | ------------------------------------------------------------------ Cancel via public co... | Deleted |
| `python/tests/test_vlm_engine.py:1168` | 3 | Tautology | ------------------------------------------------------------------ Image source validat... | Deleted |
| `python/tests/test_vlm_engine.py:1226` | 3 | Tautology | ------------------------------------------------------------------ Image count validati... | Deleted |
| `python/tests/test_vlm_engine.py:1319` | 3 | Tautology | ------------------------------------------------------------------ Lazy initialization ... | Deleted |
| `python/tests/test_vlm_engine.py:1383` | 1 | Tautology | second call should be safe | Deleted |
| `python/tests/test_vlm_engine.py:1408` | 1 | Tautology | Cancel BEFORE the load call. | Deleted |
| `python/tests/test_vlm_engine.py:1429` | 1 | Tautology | Return False on first call (before load), True on second (after). | Deleted |
| `python/tests/test_vlm_engine.py:1442` | 2 | Tautology | Model was loaded (first should_cancel returned False). Even though second should_cancel... | Deleted |
| `python/tests/test_vlm_engine.py:1488` | 3 | Tautology | Call 1 = generate_vlm check before init, Call 2 = initialize check before load, Call 3 ... | Deleted |
| `python/tests/test_vlm_engine.py:1504` | 1 | Tautology | First request — cancelled after load. | Deleted |
| `python/tests/test_vlm_engine.py:1512` | 1 | Tautology | Second request — should succeed without reloading. | Deleted |
| `rust/crates/gateway/src/audio.rs:437` | 2 | Tautology | PCM mono | Deleted; replaced by named constants |
| `rust/crates/gateway/src/audio.rs:440` | 3 | Tautology | byte rate block align bits per sample | Deleted; replaced by named constants |
| `rust/crates/gateway/src/environment.rs:100` | 2 | Decision Record | The worker is a local, non-editable package: rebuild it from this distribution instead ... | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `rust/crates/gateway/src/http.rs:377` | 2 | Decision Record | On macOS an accepted socket inherits the listener's non-blocking flag, so a request who... | Moved to [ADR-0003](0003-gateway-socket-and-sse-writes.md) |
| `rust/crates/gateway/src/http.rs:380` | 3 | Decision Record | Disable Nagle's algorithm: small SSE frames are time-sensitive and should not be delaye... | Moved to [ADR-0003](0003-gateway-socket-and-sse-writes.md) |
| `rust/crates/gateway/src/http.rs:931` | 1 | Decision Record | Model-first routing: route based on requested model name, not image content. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:953` | 1 | Tautology | Validate: images require VLM-capable model | Deleted |
| `rust/crates/gateway/src/http.rs:963` | 1 | Decision Record | Enforce VLM image-count cap before queue admission. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:1075` | 1 | Decision Record | VLM request lifecycle: first VLM request failure marks it failed. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:1250` | 1 | Decision Record | Model-first routing: route based on requested model name, not image content. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:1287` | 1 | Tautology | Validate: images require VLM-capable model | Deleted |
| `rust/crates/gateway/src/http.rs:1303` | 1 | Decision Record | Enforce VLM image-count cap before queue admission. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:1369` | 2 | Decision Record | Save the requested model name before worker_model is moved into RequestTracker; used fo... | Moved to [ADR-0003](0003-gateway-socket-and-sse-writes.md) |
| `rust/crates/gateway/src/http.rs:1449` | 2 | Decision Record | First delta: combine SSE headers + data into a single write to save one write(2) syscal... | Moved to [ADR-0003](0003-gateway-socket-and-sse-writes.md) |
| `rust/crates/gateway/src/http.rs:1480` | 1 | Decision Record | VLM request lifecycle: first VLM request failure marks it failed. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:1858` | 1 | Decision Record | Only non-InvalidRequest errors indicate load failure (not bad user input). | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/http.rs:2422` | 2 | Tautology | Validate combined-write optimization: headers immediately followed by first data event ... | Deleted |
| `rust/crates/gateway/src/http.rs:2435` | 1 | Tautology | All SSE data events must use the same model field. | Deleted |
| `rust/crates/gateway/src/http.rs:2601` | 1 | Tautology | The stream comes from accept() without TCP_NODELAY by default. | Deleted |
| `rust/crates/gateway/src/http.rs:2603` | 1 | Excuse | Now simulate what handle_connection does. | Deleted; [#10](https://github.com/huydt84/mlx-server-runtime/issues/10) |
| `rust/crates/gateway/src/http.rs:2608` | 1 | Tautology | Ensure stream options are not inherited from client side. | Deleted |
| `rust/crates/gateway/src/http.rs:2896` | 1 | Tautology | ── VLM routing tests ───────────────────────────────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:2969` | 1 | Tautology | Only one model in the list | Deleted |
| `rust/crates/gateway/src/http.rs:3053` | 2 | Tautology | VLM model must NOT be marked ready from generic worker READY; it requires explicit warm... | Deleted |
| `rust/crates/gateway/src/http.rs:3091` | 1 | Tautology | ── VLM HTTP request handling tests ─────────────────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:3095` | 1 | Tautology | POST with image content, VLM configured → 200 OK | Deleted |
| `rust/crates/gateway/src/http.rs:3162` | 1 | Tautology | POST with image content, VLM NOT configured → 400 VLM_NOT_CONFIGURED | Deleted |
| `rust/crates/gateway/src/http.rs:3191` | 1 | Tautology | POST stream=true with VLM content → SSE events | Deleted |
| `rust/crates/gateway/src/http.rs:3248` | 1 | Tautology | ── Model-first routing tests ──────────────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:3252` | 1 | Tautology | Text-only request targeting VLM model must reach VLM-capable path. | Deleted |
| `rust/crates/gateway/src/http.rs:3292` | 1 | Tautology | Text-only request explicitly targeting the VLM model name | Deleted |
| `rust/crates/gateway/src/http.rs:3309` | 1 | Tautology | ── Image URL validation tests ────────────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:3345` | 1 | Tautology | data: URIs are unsupported image sources. | Deleted |
| `rust/crates/gateway/src/http.rs:3543` | 1 | Tautology | HTTPS image URLs must be accepted through the full HTTP path. | Deleted |
| `rust/crates/gateway/src/http.rs:3633` | 1 | Tautology | ── VLM lifecycle recovery tests ──────────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:3642` | 1 | Tautology | VLM starts as NotLoaded — deliberately not marking ready yet. | Deleted |
| `rust/crates/gateway/src/http.rs:3644` | 1 | Tautology | Verify initial VLM state is NotLoaded | Deleted |
| `rust/crates/gateway/src/http.rs:3649` | 2 | Tautology | First request fails with a transient (non-InvalidRequest) error → VLM should transition... | Deleted |
| `rust/crates/gateway/src/http.rs:3659` | 1 | Tautology | Explicit warmup completion → VLM should recover to Ready. | Deleted |
| `rust/crates/gateway/src/http.rs:3669` | 2 | Tautology | GatewayError::InvalidRequest (bad user input) must NOT mark VLM as failed; only backend... | Deleted |
| `rust/crates/gateway/src/http.rs:3677` | 1 | Tautology | An InvalidRequest error should NOT trigger VLM failure transition | Deleted |
| `rust/crates/gateway/src/http.rs:3692` | 2 | Tautology | Repeating explicit warmup completion when VLM is already Ready must be a safe no-op (no... | Deleted |
| `rust/crates/gateway/src/http.rs:3701` | 1 | Tautology | Call warmup completion again — should stay Ready, not error. | Deleted |
| `rust/crates/gateway/src/http.rs:3711` | 2 | Tautology | Calling track_vlm_lifecycle_failure when VLM is already Failed must be a safe no-op. | Deleted |
| `rust/crates/gateway/src/http.rs:3720` | 1 | Tautology | Call failure again — should stay Failed, not error | Deleted |
| `rust/crates/gateway/src/http.rs:3728` | 1 | Tautology | The original error message should be preserved, not overwritten | Deleted |
| `rust/crates/gateway/src/http.rs:3737` | 2 | Tautology | Full HTTP-path test: user request failures do not change explicit VLM warmup readiness. | Deleted |
| `rust/crates/gateway/src/http.rs:3746` | 1 | Tautology | Backend that fails on first call, succeeds on second. | Deleted |
| `rust/crates/gateway/src/http.rs:3791` | 1 | Tautology | First request: fails, but explicit warmup state remains Ready. | Deleted |
| `rust/crates/gateway/src/http.rs:3802` | 1 | Tautology | Second request: succeeds, and readiness still stays Ready. | Deleted |
| `rust/crates/gateway/src/http.rs:3817` | 1 | Tautology | Image-bearing request targeting text model must fail when VLM configured. | Deleted |
| `rust/crates/gateway/src/http.rs:3869` | 1 | Tautology | ── VLM image-count cap tests ────────────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:4107` | 1 | Tautology | Text-only request targeting VLM model must not hit the image cap. | Deleted |
| `rust/crates/gateway/src/http.rs:4160` | 1 | Tautology | ── VLM lifecycle cancelled/no-load guard tests ──────────────────────── | Deleted |
| `rust/crates/gateway/src/http.rs:4188` | 1 | Tautology | Emit a delta to confirm streaming started before cancelled finish. | Deleted |
| `rust/crates/gateway/src/http.rs:4204` | 1 | Tautology | Non-streaming cancelled VLM request must not alter warmup readiness. | Deleted |
| `rust/crates/gateway/src/http.rs:4257` | 1 | Tautology | Streaming cancelled VLM request must not alter warmup readiness. | Deleted |
| `rust/crates/gateway/src/instances.rs:405` | 2 | Decision Record | launchd restarts a gateway that exits; a new pid before readiness means startup failed ... | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `rust/crates/gateway/src/instances.rs:429` | 2 | Decision Record | A failed worker startup does not recover by waiting; report it now instead of at the st... | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `rust/crates/gateway/src/instances.rs:766` | 2 | Decision Record | Rendered only when set explicitly, so a detached worker still derives the request limit... | Moved to [ADR-0007](0007-worker-startup-failure-handling.md) |
| `rust/crates/gateway/src/openai.rs:444` | 1 | Tautology | No VLM model configured, so "vlm-model" should be rejected | Deleted |
| `rust/crates/gateway/src/supervisor.rs:176` | 1 | Decision Record | VLM not yet initialized: return NotLoaded fallback. | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/supervisor.rs:558` | 3 | Decision Record | Only update the text model lifecycle from this STATUS. VLM loads lazily on first VLM re... | Moved to [ADR-0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) |
| `rust/crates/gateway/src/telemetry.rs:85` | 1 | Tautology | VLM-specific counters | Deleted |
| `rust/crates/gateway/src/telemetry.rs:255` | 1 | Tautology | ── VLM metric methods ────────────────────────────────────── | Deleted |
| `rust/crates/gateway/src/telemetry.rs:505` | 1 | Tautology | VLM-specific metrics | Deleted |
| `rust/crates/gateway/src/telemetry.rs:1062` | 1 | Tautology | VLM metrics | Deleted |
| `rust/crates/gateway/tests/bench_delegation.rs:1014` | 1 | Tautology | The run stops after the first trial (two sequential requests) fails. | Deleted |
| `rust/crates/gateway/tests/bench_delegation.rs:1046` | 3 | Tautology | 6 warmup requests (one per concurrency slot) plus the first trial's 2 measured requests... | Deleted |
| `rust/crates/protocol/src/events.rs:610` | 1 | Tautology | Verify the JSON structure has the expected content parts | Deleted |
| `rust/crates/protocol/src/events.rs:673` | 1 | Excuse | Remove the escaped quotes we inserted for readability | Deleted; [#9](https://github.com/huydt84/mlx-server-runtime/issues/9) |
