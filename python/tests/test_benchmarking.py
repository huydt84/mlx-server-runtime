from __future__ import annotations

import io
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.compare import (
    DEFAULT_PROMPT,
    _ProgressTracker,
    PromptCase,
    StreamResult,
    _build_prompt_cases,
    _default_prompt_suite,
    _extract_port,
    _is_port_free,
    _percentile,
    _prompt_summary,
    _reduce_measurements,
    _result_summary,
    _progress_detail,
    _prepare_project_config,
    _replace_config_value,
    _request_completion,
    make_long_prompts,
)
from mlx_worker.benchmarking import (
    BenchmarkResult,
    BenchmarkRun,
    _format_number,
    _format_overhead,
    _overhead_summary,
    now_utc_iso,
    summarize_report,
    summarize_results,
)


# ===========================================================================
# _is_port_free
# ===========================================================================


class TestIsPortFree:
    """Verify port-liveness check used for stale-port rejection."""

    def test_free_when_connection_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "socket.create_connection",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionRefusedError("refused")),
        )
        assert _is_port_free("127.0.0.1", 9999) is True

    def test_free_when_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "socket.create_connection",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("oops")),
        )
        assert _is_port_free("127.0.0.1", 9999) is True

    def test_free_when_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "socket.create_connection",
            lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("timed out")),
        )
        assert _is_port_free("127.0.0.1", 9999) is True

    def test_occupied_when_connection_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_sock = MagicMock()
        monkeypatch.setattr("socket.create_connection", lambda *a, **kw: mock_sock)
        assert _is_port_free("127.0.0.1", 9999) is False


# ===========================================================================
# _extract_port
# ===========================================================================


class TestExtractPort:
    def test_simple_http(self) -> None:
        assert _extract_port("http://127.0.0.1:8000") == 8000

    def test_with_path_not_supported(self) -> None:
        """_extract_port uses naive rsplit — only works for bare host:port URLs.
        This documents current limitation, not a desired contract.
        """
        assert _extract_port("http://127.0.0.1:8080") == 8080

    def test_https(self) -> None:
        assert _extract_port("https://example.com:443") == 443

    def test_high_port(self) -> None:
        assert _extract_port("http://localhost:65535") == 65535


# ===========================================================================
# _replace_config_value
# ===========================================================================


class TestReplaceConfigValue:
    def test_replace_numeric(self) -> None:
        text = 'port = 8000\nhost = "127.0.0.1"\n'
        result = _replace_config_value(text, "port", "9000")
        assert result == 'port = 9000\nhost = "127.0.0.1"\n'

    def test_replace_string(self) -> None:
        text = 'port = 8000\nipc_path = "/tmp/old.sock"\n'
        result = _replace_config_value(text, "ipc_path", "/tmp/new.sock")
        assert result == 'port = 8000\nipc_path = "/tmp/new.sock"\n'

    def test_no_match_leaves_unchanged(self) -> None:
        text = 'port = 8000\nhost = "127.0.0.1"\n'
        result = _replace_config_value(text, "unknown_key", "value")
        assert result == text

    def test_preserves_indentation(self) -> None:
        text = "  port = 8000\n"
        result = _replace_config_value(text, "port", "9000")
        assert result == "  port = 9000\n"

    def test_numeric_detection_dotted_value(self) -> None:
        text = "timeout = 30\n"
        # "30" is all digits → no quotes
        result = _replace_config_value(text, "timeout", "30")
        assert result == "timeout = 30\n"

    def test_mixed_quoted_value(self) -> None:
        text = 'name = "old"\n'
        result = _replace_config_value(text, "name", "new-name")
        # "new-name" contains non-digit → gets quoted
        assert result == 'name = "new-name"\n'

    def test_replaces_all_matching_keys(self) -> None:
        """Current implementation replaces EVERY matching line."""
        text = "port = 8000\nport = 8001\n"
        result = _replace_config_value(text, "port", "9000")
        assert result == "port = 9000\nport = 9000\n"


class TestPrepareProjectConfig:
    def test_overrides_model_port_and_ipc_path(self, tmp_path: Path) -> None:
        path = _prepare_project_config(
            "mlx-community/Qwen3-8B-4bit", 8123, config_dir=tmp_path
        )
        content = path.read_text(encoding="utf-8")
        assert 'model = "mlx-community/Qwen3-8B-4bit"' in content
        assert "port = 8123" in content
        assert ".sock" in content


# ===========================================================================
# _format_number
# ===========================================================================


class TestFormatNumber:
    def test_positive(self) -> None:
        assert _format_number(12.345) == "12.3"

    def test_zero(self) -> None:
        assert _format_number(0.0) == "0.0"

    def test_integer_float(self) -> None:
        assert _format_number(5.0) == "5.0"

    def test_negative(self) -> None:
        assert _format_number(-1.5) == "-1.5"

    def test_large_value(self) -> None:
        assert _format_number(1234.56) == "1234.6"


# ===========================================================================
# _format_overhead
# ===========================================================================


class TestFormatOverhead:
    def test_positive_delta(self) -> None:
        assert _format_overhead(20.0, 10.0) == "+10.0"

    def test_negative_delta(self) -> None:
        assert _format_overhead(5.0, 10.0) == "-5.0"

    def test_zero_delta(self) -> None:
        assert _format_overhead(10.0, 10.0) == "+0.0"

    def test_no_baseline(self) -> None:
        assert _format_overhead(20.0, None) == "-"

    def test_large_delta(self) -> None:
        assert _format_overhead(150.0, 42.0) == "+108.0"


# ===========================================================================
# _reduce_measurements
# ===========================================================================


def _stream_result(
    ttft_ms: float = 10.0,
    latency_ms: float = 50.0,
    prompt_tokens: int = 8,
    completion_tokens: int = 4,
    text: str = "hello",
    notes: tuple[str, ...] = (),
) -> StreamResult:
    return StreamResult(
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        text=text,
        notes=notes,
    )


def _prompt_case(prompt_tokens: int = 8) -> PromptCase:
    return PromptCase(
        name="prompt-1",
        messages=[{"role": "user", "content": "hi"}],
        prompt_input="User: hi\nAssistant:",
        prompt_tokens=prompt_tokens,
    )


def _benchmark_result(
    backend: str,
    *,
    samples: int = 1,
    errors: int = 0,
    error_rate: float = 0.0,
    ttft_mean_ms: float | None = 10.0,
    ttft_p50_ms: float | None = 10.0,
    ttft_p95_ms: float | None = None,
    ttft_p99_ms: float | None = None,
    latency_mean_ms: float | None = 50.0,
    latency_p50_ms: float | None = 50.0,
    latency_p95_ms: float | None = None,
    latency_p99_ms: float | None = None,
    prompt_tokens_mean: float | None = 8.0,
    completion_tokens_mean: float | None = 4.0,
    total_tokens_mean: float | None = 12.0,
    decode_time_mean_ms: float | None = 40.0,
    decode_tokens_per_second_mean: float | None = 100.0,
    decode_tokens_per_second_p50: float | None = 100.0,
    end_to_end_tokens_per_second_mean: float | None = 80.0,
    end_to_end_tokens_per_second_p50: float | None = 80.0,
    notes: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend,
        samples=samples,
        errors=errors,
        error_rate=error_rate,
        ttft_mean_ms=ttft_mean_ms,
        ttft_p50_ms=ttft_p50_ms,
        ttft_p95_ms=ttft_p95_ms,
        ttft_p99_ms=ttft_p99_ms,
        latency_mean_ms=latency_mean_ms,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        latency_p99_ms=latency_p99_ms,
        prompt_tokens_mean=prompt_tokens_mean,
        completion_tokens_mean=completion_tokens_mean,
        total_tokens_mean=total_tokens_mean,
        decode_time_mean_ms=decode_time_mean_ms,
        decode_tokens_per_second_mean=decode_tokens_per_second_mean,
        decode_tokens_per_second_p50=decode_tokens_per_second_p50,
        end_to_end_tokens_per_second_mean=end_to_end_tokens_per_second_mean,
        end_to_end_tokens_per_second_p50=end_to_end_tokens_per_second_p50,
        notes=notes,
        warnings=warnings,
    )


class _SimpleTokenizer:
    def __init__(self, *, has_chat_template: bool = False) -> None:
        self.has_chat_template = has_chat_template

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        token_count = sum(len(message["content"].split()) for message in messages)
        return list(range(token_count + 1))


class TestReduceMeasurements:
    def test_single_measurement(self) -> None:
        result = _reduce_measurements("test-backend", [_stream_result()])
        assert result.backend == "test-backend"
        assert result.ttft_ms == 10.0
        assert result.latency_ms == 50.0
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4
        assert result.notes == ()

    def test_averages_multiple_measurements(self) -> None:
        results = _reduce_measurements(
            "avg-backend",
            [
                _stream_result(ttft_ms=10.0, latency_ms=40.0),
                _stream_result(ttft_ms=20.0, latency_ms=60.0),
            ],
        )
        assert results.ttft_ms == 15.0  # (10 + 20) / 2
        assert results.latency_ms == 50.0  # (40 + 60) / 2

    def test_uses_last_prompt_completion_tokens(self) -> None:
        results = _reduce_measurements(
            "tokens-backend",
            [
                _stream_result(prompt_tokens=5, completion_tokens=3),
                _stream_result(prompt_tokens=8, completion_tokens=6),
            ],
        )
        # prompt/completion tokens are suite means across successful samples
        assert results.prompt_tokens == 6.5
        assert results.completion_tokens == 4.5

    def test_empty_measurements_raises(self) -> None:
        with pytest.raises(ValueError, match="produced no benchmark measurements"):
            _reduce_measurements("empty-backend", [])

    def test_deduplicates_notes_across_samples(self) -> None:
        results = _reduce_measurements(
            "notes-backend",
            [
                _stream_result(notes=("a", "b")),
                _stream_result(notes=("b", "c")),
            ],
        )
        # deduplicated preserving order of first occurrence
        assert results.notes == ("a", "b", "c")

    def test_empty_notes_when_all_none(self) -> None:
        results = _reduce_measurements(
            "no-notes-backend",
            [
                _stream_result(notes=()),
                _stream_result(notes=()),
            ],
        )
        assert results.notes == ()

    def test_distribution_summary_adds_percentile_notes(self) -> None:
        results = _reduce_measurements(
            "summary-backend",
            [
                _stream_result(ttft_ms=10.0, latency_ms=40.0, notes=("source",)),
                _stream_result(ttft_ms=20.0, latency_ms=60.0),
                _stream_result(ttft_ms=30.0, latency_ms=80.0),
            ],
            summarize_distribution=True,
        )

        assert results.notes == ("source",)


class TestPercentile:
    def test_interpolates_small_samples(self) -> None:
        assert _percentile([60.0, 40.0], 50) == 50.0
        assert _percentile([40.0, 60.0], 95) == 59.0

    def test_empty_values_return_zero(self) -> None:
        assert _percentile([], 50) == 0.0


class TestPromptSuites:
    def test_build_prompt_cases_falls_back_to_default_prompt(self) -> None:
        cases = _build_prompt_cases(_SimpleTokenizer(), [])

        assert len(cases) == 1
        assert cases[0].messages == [{"role": "user", "content": DEFAULT_PROMPT}]
        assert cases[0].prompt_tokens > 0

    def test_build_prompt_cases_uses_chat_template_tokens(self) -> None:
        cases = _build_prompt_cases(
            _SimpleTokenizer(has_chat_template=True), ["hello world"]
        )

        assert cases[0].prompt_input == [0, 1, 2]
        assert cases[0].prompt_tokens == 3

    def test_default_prompt_suite_limit(self) -> None:
        prompts = _default_prompt_suite(
            _SimpleTokenizer(),
            include_long=False,
            prompt_limit=3,
            prefill_step_size=4,
            long_prompt_multiplier=2,
        )

        assert len(prompts) == 3

    def test_default_prompt_suite_can_include_long_prompts(self) -> None:
        short_prompts = _default_prompt_suite(
            _SimpleTokenizer(),
            include_long=False,
            prompt_limit=0,
            prefill_step_size=4,
            long_prompt_multiplier=2,
        )
        long_prompts = _default_prompt_suite(
            _SimpleTokenizer(),
            include_long=True,
            prompt_limit=0,
            prefill_step_size=4,
            long_prompt_multiplier=2,
        )

        assert len(long_prompts) > len(short_prompts)

    def test_make_long_prompts_meet_target_token_count(self) -> None:
        tokenizer = _SimpleTokenizer()

        prompts = make_long_prompts(tokenizer, prefill_step_size=4, multiplier=2)

        assert prompts
        assert all(len(tokenizer.encode(prompt)) >= 40 for prompt in prompts)

    def test_prompt_summary_reports_suite_shape(self) -> None:
        summary = _prompt_summary([_prompt_case(3), _prompt_case(5)])

        assert summary == "prompt suite: 2 cases, 8 prompt tokens total"


class TestProgressHelpers:
    def test_progress_tracker_text_mode_logs_updates(self) -> None:
        stream = io.StringIO()
        tracker = _ProgressTracker("raw warmup", 2, stream=stream, use_tqdm=False)

        tracker.advance("trial 1/2, prompt-1, prompt_tokens=8")
        tracker.advance("trial 2/2, prompt-2, prompt_tokens=13")
        tracker.close()

        output = stream.getvalue()
        assert "raw warmup: 0/2" in output
        assert "raw warmup: 1/2 (50%) - trial 1/2, prompt-1, prompt_tokens=8" in output
        assert (
            "raw warmup: 2/2 (100%) - trial 2/2, prompt-2, prompt_tokens=13" in output
        )
        assert "raw warmup: completed" in output

    def test_progress_detail_and_result_summary(self) -> None:
        detail = _progress_detail(
            _prompt_case(prompt_tokens=21), phase_index=2, phase_total=3
        )
        summary = _result_summary(
            "mlx-community/test-model",
            _benchmark_result(
                "raw mlx-lm",
                ttft_mean_ms=40.0,
                latency_mean_ms=200.0,
                prompt_tokens_mean=21.0,
                completion_tokens_mean=10.0,
                total_tokens_mean=31.0,
                decode_time_mean_ms=160.0,
                decode_tokens_per_second_mean=62.5,
                decode_tokens_per_second_p50=62.5,
                end_to_end_tokens_per_second_mean=50.0,
                end_to_end_tokens_per_second_p50=50.0,
            ),
        )

        assert detail == "trial 2/3, prompt-1, prompt_tokens=21"
        assert "[model mlx-community/test-model] raw mlx-lm done:" in summary
        assert "latency_mean=200.0 ms" in summary
        assert "ttft_mean=40.0 ms" in summary
        assert "output_tps_mean=50.0" in summary


# ===========================================================================
# _overhead_summary
# ===========================================================================


class TestOverheadSummary:
    def test_no_raw_baseline(self) -> None:
        msg = _overhead_summary(None, [])
        assert msg == "Raw mlx-lm baseline was not recorded."

    def test_no_slower_backends(self) -> None:
        raw = _benchmark_result("raw mlx-lm")
        faster = _benchmark_result(
            "fast",
            ttft_mean_ms=5.0,
            ttft_p50_ms=5.0,
            latency_mean_ms=25.0,
            latency_p50_ms=25.0,
            decode_time_mean_ms=20.0,
            decode_tokens_per_second_mean=200.0,
            decode_tokens_per_second_p50=200.0,
            end_to_end_tokens_per_second_mean=160.0,
            end_to_end_tokens_per_second_p50=160.0,
        )
        msg = _overhead_summary(raw, [raw, faster])
        assert (
            msg
            == "No backend exceeded the raw mlx-lm baseline in measured mean latency."
        )

    def test_with_slower_backend(self) -> None:
        raw = _benchmark_result("raw mlx-lm")
        slow = _benchmark_result(
            "slow-backend",
            ttft_mean_ms=20.0,
            ttft_p50_ms=20.0,
            latency_mean_ms=100.0,
            latency_p50_ms=100.0,
            decode_time_mean_ms=80.0,
            decode_tokens_per_second_mean=50.0,
            decode_tokens_per_second_p50=50.0,
            end_to_end_tokens_per_second_mean=40.0,
            end_to_end_tokens_per_second_p50=40.0,
        )
        msg = _overhead_summary(raw, [raw, slow])
        assert "slow-backend was 50.0 ms slower than raw mlx-lm" in msg
        assert "10.0 ms slower on mean TTFT" in msg

    def test_worst_backend_selected(self) -> None:
        raw = _benchmark_result("raw mlx-lm")
        mid = _benchmark_result("mid", ttft_mean_ms=15.0, latency_mean_ms=70.0)
        worst = _benchmark_result("worst", ttft_mean_ms=30.0, latency_mean_ms=120.0)
        msg = _overhead_summary(raw, [raw, mid, worst])
        assert "worst" in msg
        assert "70.0 ms slower" in msg  # 120 - 50 = 70

    def test_raw_itself_not_counted_as_slower(self) -> None:
        raw = _benchmark_result("raw mlx-lm")
        msg = _overhead_summary(raw, [raw])
        assert "No backend exceeded" in msg


# ===========================================================================
# summarize_results — edge cases
# ===========================================================================


class TestSummarizeResultsEdgeCases:
    def test_single_backend_no_raw(self) -> None:
        """Single backend (not raw) — overhead columns show '-'."""
        run = BenchmarkRun(
            model="m",
            prompt="p",
            max_tokens=16,
            generated_at="2026-06-14T00:00:00+00:00",
            results=(
                _benchmark_result(
                    "only-backend",
                    ttft_mean_ms=15.0,
                    ttft_p50_ms=15.0,
                    latency_mean_ms=45.0,
                    latency_p50_ms=45.0,
                ),
            ),
        )
        report = summarize_results(run)
        assert (
            "| only-backend | 1 | 0 | 0.0% | 15.0 | 15.0 | - | - | 45.0 | 45.0 | - | - | 8.0 | 4.0 | 12.0 | 40.0 |"
            in report
        )
        assert "Raw mlx-lm baseline was not recorded." in report

    def test_no_raw_baseline_among_results(self) -> None:
        """No result has backend='raw mlx-lm' — overhead is '-'."""
        run = BenchmarkRun(
            model="m",
            prompt="p",
            max_tokens=16,
            generated_at="2026-06-14T00:00:00+00:00",
            results=(
                _benchmark_result(
                    "backend-a",
                    ttft_mean_ms=10.0,
                    ttft_p50_ms=10.0,
                    latency_mean_ms=30.0,
                    latency_p50_ms=30.0,
                    decode_time_mean_ms=20.0,
                    decode_tokens_per_second_mean=200.0,
                    decode_tokens_per_second_p50=200.0,
                    end_to_end_tokens_per_second_mean=133.3333333333,
                    end_to_end_tokens_per_second_p50=133.3333333333,
                ),
                _benchmark_result(
                    "backend-b",
                    ttft_mean_ms=20.0,
                    ttft_p50_ms=20.0,
                    latency_mean_ms=60.0,
                    latency_p50_ms=60.0,
                    decode_time_mean_ms=40.0,
                    decode_tokens_per_second_mean=100.0,
                    decode_tokens_per_second_p50=100.0,
                    end_to_end_tokens_per_second_mean=66.6666666667,
                    end_to_end_tokens_per_second_p50=66.6666666667,
                ),
            ),
        )
        report = summarize_results(run)
        assert "| backend-a | - | - | - | - | - | - | - | - |" in report
        assert "| backend-b | - | - | - | - | - | - | - | - |" in report

    def test_notes_dedup_across_results(self) -> None:
        """Each result's notes are joined with '; ' — duplicate notes per result are kept as-is."""
        run = BenchmarkRun(
            model="m",
            prompt="p",
            max_tokens=16,
            generated_at="2026-06-14T00:00:00+00:00",
            results=(
                _benchmark_result(
                    "raw mlx-lm", latency_mean_ms=30.0, latency_p50_ms=30.0
                ),
                _benchmark_result(
                    "backend",
                    ttft_mean_ms=15.0,
                    ttft_p50_ms=15.0,
                    latency_mean_ms=45.0,
                    latency_p50_ms=45.0,
                    notes=("note-a", "note-b", "note-a"),
                ),
            ),
        )
        report = summarize_results(run)
        assert "- backend: note-a" in report
        assert "- backend: note-b" in report

    def test_empty_results_raises(self) -> None:
        run = BenchmarkRun(
            model="m",
            prompt="p",
            max_tokens=16,
            generated_at="2026-06-14T00:00:00+00:00",
            results=(),
        )
        with pytest.raises(ValueError, match="at least one result"):
            summarize_results(run)


class TestSummarizeReport:
    def test_multiple_runs_render_separate_sections(self) -> None:
        raw = _benchmark_result(
            "raw mlx-lm",
            latency_mean_ms=30.0,
            latency_p50_ms=30.0,
            decode_time_mean_ms=20.0,
            decode_tokens_per_second_mean=200.0,
            decode_tokens_per_second_p50=200.0,
            end_to_end_tokens_per_second_mean=133.3333333333,
            end_to_end_tokens_per_second_p50=133.3333333333,
        )
        project = _benchmark_result(
            "this project",
            ttft_mean_ms=12.0,
            ttft_p50_ms=12.0,
            latency_mean_ms=35.0,
            latency_p50_ms=35.0,
            decode_time_mean_ms=23.0,
            decode_tokens_per_second_mean=173.9130434783,
            decode_tokens_per_second_p50=173.9130434783,
            end_to_end_tokens_per_second_mean=114.2857142857,
            end_to_end_tokens_per_second_p50=114.2857142857,
        )
        report = summarize_report(
            [
                BenchmarkRun(
                    model="model-a",
                    prompt="p",
                    max_tokens=16,
                    generated_at="2026-06-14T00:00:00+00:00",
                    results=(raw, project),
                ),
                BenchmarkRun(
                    model="model-b",
                    prompt="p",
                    max_tokens=16,
                    generated_at="2026-06-14T00:00:01+00:00",
                    results=(raw, project),
                ),
            ]
        )
        assert report.count("# Phase 6 Benchmark Report") == 1
        assert "## Model: model-a" in report
        assert "## Model: model-b" in report
        assert (
            report.count(
                "| backend | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_p99_ms | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | decode_time_mean_ms |"
            )
            == 2
        )

    def test_no_runs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one run"):
            summarize_report([])


# ===========================================================================
# now_utc_iso
# ===========================================================================


class TestNowUtcIso:
    def test_returns_iso_format_string(self) -> None:
        ts = now_utc_iso()
        assert isinstance(ts, str)
        # Should look like "2026-06-14T12:00:00+00:00"
        assert "+00:00" in ts or "Z" in ts  # UTC timezone marker
        assert "T" in ts  # ISO separator
        parts = ts.split("T")
        assert len(parts) == 2
        date_part = parts[0]
        # Date should be YYYY-MM-DD
        assert len(date_part.split("-")) == 3

    def test_returns_current_year(self) -> None:
        ts = now_utc_iso()
        current_year = str(time.gmtime().tm_year)
        assert ts.startswith(current_year)


# ===========================================================================
# Dataclass construction
# ===========================================================================


class TestStreamResultConstruction:
    def test_minimal(self) -> None:
        sr = StreamResult(
            ttft_ms=5.0,
            latency_ms=20.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="hi",
        )
        assert sr.ttft_ms == 5.0
        assert sr.latency_ms == 20.0
        assert sr.prompt_tokens == 8
        assert sr.completion_tokens == 4
        assert sr.text == "hi"
        assert sr.notes == ()

    def test_with_notes(self) -> None:
        sr = StreamResult(
            ttft_ms=5.0,
            latency_ms=20.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="hi",
            notes=("a", "b"),
        )
        assert sr.notes == ("a", "b")


class TestBenchmarkResultConstruction:
    def test_minimal(self) -> None:
        br = _benchmark_result(
            "b",
            ttft_mean_ms=5.0,
            ttft_p50_ms=5.0,
            latency_mean_ms=20.0,
            latency_p50_ms=20.0,
            decode_time_mean_ms=15.0,
            decode_tokens_per_second_mean=266.6666666667,
            decode_tokens_per_second_p50=266.6666666667,
            end_to_end_tokens_per_second_mean=200.0,
            end_to_end_tokens_per_second_p50=200.0,
        )
        assert br.backend == "b"
        assert br.notes == ()

    def test_with_notes(self) -> None:
        br = _benchmark_result("b", notes=("x",))
        assert br.notes == ("x",)


class TestBenchmarkRunConstruction:
    def test_minimal(self) -> None:
        br = _benchmark_result(
            "b",
            ttft_mean_ms=1.0,
            ttft_p50_ms=1.0,
            latency_mean_ms=2.0,
            latency_p50_ms=2.0,
            prompt_tokens_mean=3.0,
            completion_tokens_mean=4.0,
            total_tokens_mean=7.0,
            decode_time_mean_ms=1.0,
            decode_tokens_per_second_mean=4000.0,
            decode_tokens_per_second_p50=4000.0,
            end_to_end_tokens_per_second_mean=2000.0,
            end_to_end_tokens_per_second_p50=2000.0,
        )
        run = BenchmarkRun(
            model="m", prompt="p", max_tokens=16, generated_at="now", results=(br,)
        )
        assert run.model == "m"
        assert run.prompt == "p"
        assert run.max_tokens == 16
        assert run.generated_at == "now"
        assert run.results == (br,)


# ===========================================================================
# _request_completion — SSE parsing
# ===========================================================================


class _MockResponse:
    """Simulate an HTTPResponse yielding SSE lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self) -> _MockResponse:
        return self

    def __next__(self) -> bytes:
        if not self._lines:
            raise StopIteration()
        return self._lines.pop(0)


class TestRequestCompletionSseParsing:
    """Verify SSE stream is correctly parsed into a StreamResult."""

    def test_basic_sse_stream(self) -> None:
        """Parses data: lines, handles [DONE], extracts usage and content."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}],"usage":{"completion_tokens":2,"prompt_tokens":5}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=None,
                prompt_tokens=5,
            )

        assert result.text == "Hello world"
        assert result.prompt_tokens == 5  # from parameter
        assert result.completion_tokens == 2  # from usage in SSE
        assert result.ttft_ms >= 0
        assert result.latency_ms >= result.ttft_ms

    def test_falls_back_when_usage_missing(self) -> None:
        """When no usage in SSE and no tokenizer, completion_tokens stays 0."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=None,
                prompt_tokens=8,
            )

        assert result.text == "Hello"
        assert result.completion_tokens == 0  # no usage data
        assert result.prompt_tokens == 8

    def test_extracts_prompt_tokens_from_usage_when_none_given(self) -> None:
        """When prompt_tokens is None, it's read from usage dict."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"A"}}],"usage":{"completion_tokens":1,"prompt_tokens":10}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=None,
                prompt_tokens=None,
            )

        assert result.prompt_tokens == 10  # from SSE usage

    def test_content_delta_via_message_field(self) -> None:
        """When delta is empty, falls back to choice['message']['content']."""
        sse_lines = [
            b'data: {"choices":[{"message":{"content":"Fallback content"}}],"usage":{"completion_tokens":1,"prompt_tokens":3}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=None,
                prompt_tokens=3,
            )

        assert result.text == "Fallback content"

    def test_content_delta_via_reasoning_field(self) -> None:
        """Counts mlx_lm.server reasoning deltas as generated output."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"reasoning":"hidden text"}}],"usage":{"completion_tokens":0,"prompt_tokens":3}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=_SimpleTokenizer(),
                prompt_tokens=3,
            )

        assert result.text == "hidden text"
        assert result.completion_tokens == 2

    def test_skips_non_data_lines(self) -> None:
        """Lines not starting with 'data:' are ignored."""
        sse_lines = [
            b":comment\n",
            b"\n",
            b'data: {"choices":[{"delta":{"content":"valid"}}],"usage":{"completion_tokens":1,"prompt_tokens":2}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=None,
                prompt_tokens=2,
            )

        assert result.text == "valid"

    def test_no_content_first_delta_fallback(self) -> None:
        """When no content deltas, first_delta_at falls back to end time."""
        sse_lines = [
            b'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":0,"prompt_tokens":2}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            result = _request_completion(
                base_url="http://127.0.0.1:8000",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                tokenizer=None,
                prompt_tokens=2,
            )

        assert result.text == ""
        assert result.ttft_ms >= 0


# ===========================================================================
# Subprocess cleanup in _benchmark_http_service
# ===========================================================================


# ===========================================================================
# _wait_for_service_ready
# ===========================================================================


class TestWaitForServiceReady:
    """Verify readiness-check behavior with and without readiness_url.

    This is the function whose branch changed: previously _benchmark_project
    passed readiness_url="/health" (fast health-check path). Now it passes
    None, falling through to the streaming-completion readiness path used
    by _benchmark_mlx_lm_server -- making the readiness check fair between
    both backends.
    """

    def _make_response(self, status: int) -> MagicMock:
        """Build a context-manager mock that simulates urlopen response."""
        resp = MagicMock()
        resp.status = status
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    def test_with_readiness_url_success(self) -> None:
        """When readiness_url is set and endpoint returns < 500, returns."""
        from benchmarks.compare import _wait_for_service_ready

        mock_resp = self._make_response(200)

        with patch("benchmarks.compare.urlopen", return_value=mock_resp):
            _wait_for_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url="/health",
                model="m",
                messages=[],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )
        # No exception means success

    def test_with_readiness_url_server_error_retries(self) -> None:
        """When readiness_url returns 503 first, then 200 -- retries."""
        from benchmarks.compare import _wait_for_service_ready

        mock_503 = self._make_response(503)
        mock_200 = self._make_response(200)

        with (
            patch(
                "benchmarks.compare.urlopen", side_effect=[mock_503, mock_200]
            ) as mock_urlopen,
            patch("benchmarks.compare.time.sleep", return_value=None),
        ):
            _wait_for_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url="/health",
                model="m",
                messages=[],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )
        assert mock_urlopen.call_count == 2

    def test_with_readiness_url_timeout(self) -> None:
        """When readiness_url always returns 503, raises RuntimeError."""
        from benchmarks.compare import _wait_for_service_ready

        mock_resp = self._make_response(503)

        with (
            patch("benchmarks.compare.urlopen", return_value=mock_resp),
            patch("benchmarks.compare.time.sleep", return_value=None),
            patch(
                "benchmarks.compare.time.monotonic",
                side_effect=[100.0, 100.1, 100.2, 100.3, 120.0],
            ),
        ):
            with pytest.raises(RuntimeError, match="service did not become ready"):
                _wait_for_service_ready(
                    base_url="http://127.0.0.1:8000",
                    readiness_url="/health",
                    model="m",
                    messages=[],
                    max_tokens=16,
                    timeout_s=10,
                    label="svc",
                )

    def test_without_readiness_url_success(self) -> None:
        """When readiness_url is None, calls _request_completion and returns."""
        from benchmarks.compare import _wait_for_service_ready

        mock_result = StreamResult(
            ttft_ms=5.0,
            latency_ms=15.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="ready",
        )

        with patch(
            "benchmarks.compare._request_completion", return_value=mock_result
        ) as mock_req:
            _wait_for_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )
        mock_req.assert_called_once_with(
            "http://127.0.0.1:8000",
            "m",
            [{"role": "user", "content": "hi"}],
            16,
            tokenizer=None,
            prompt_tokens=None,
        )

    def test_without_readiness_url_retries_then_succeeds(self) -> None:
        """When _request_completion fails first, then succeeds -- retries."""
        from benchmarks.compare import _wait_for_service_ready

        mock_result = StreamResult(
            ttft_ms=5.0,
            latency_ms=15.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="ready",
        )

        with (
            patch(
                "benchmarks.compare._request_completion",
                side_effect=[RuntimeError("not ready"), mock_result],
            ) as mock_req,
            patch("benchmarks.compare.time.sleep", return_value=None),
        ):
            _wait_for_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )
        assert mock_req.call_count == 2

    def test_without_readiness_url_timeout(self) -> None:
        """When _request_completion always fails, raises RuntimeError."""
        from benchmarks.compare import _wait_for_service_ready

        with (
            patch(
                "benchmarks.compare._request_completion",
                side_effect=RuntimeError("model not loaded"),
            ),
            patch("benchmarks.compare.time.sleep", return_value=None),
            patch(
                "benchmarks.compare.time.monotonic",
                side_effect=[100.0, 100.1, 100.2, 100.3, 120.0],
            ),
        ):
            with pytest.raises(
                RuntimeError, match="service did not become ready: model not loaded"
            ):
                _wait_for_service_ready(
                    base_url="http://127.0.0.1:8000",
                    readiness_url=None,
                    model="m",
                    messages=[],
                    max_tokens=16,
                    timeout_s=10,
                    label="svc",
                )

    def test_without_readiness_url_http_error(self) -> None:
        """When readiness_url is None and _request_completion raises HTTPError."""
        from benchmarks.compare import _wait_for_service_ready

        from urllib.error import HTTPError

        with (
            patch(
                "benchmarks.compare._request_completion",
                side_effect=HTTPError(
                    "http://127.0.0.1:8000/v1/chat/completions",
                    503,
                    "Service Unavailable",
                    {},
                    None,
                ),
            ),
            patch("benchmarks.compare.time.sleep", return_value=None),
            patch(
                "benchmarks.compare.time.monotonic",
                side_effect=[100.0, 100.1, 100.2, 100.3, 120.0],
            ),
        ):
            with pytest.raises(
                RuntimeError,
                match="service did not become ready: HTTP Error 503: Service Unavailable",
            ):
                _wait_for_service_ready(
                    base_url="http://127.0.0.1:8000",
                    readiness_url=None,
                    model="m",
                    messages=[],
                    max_tokens=16,
                    timeout_s=10,
                    label="svc",
                )


class TestBenchmarkHttpServiceCleanup:
    """Verify process group kill and file cleanup on success path.

    Mock subprocess.Popen and all networking so the function runs
    synchronously without real subprocesses or ports.
    """

    def test_cleanup_called_on_success(self) -> None:
        from benchmarks.compare import _benchmark_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_001
        mock_process.wait.return_value = 0
        killpg = MagicMock()
        getpgid = MagicMock(return_value=42_001)

        # Build a valid StreamResult for _request_completion to return
        mock_stream = StreamResult(
            ttft_ms=5.0,
            latency_ms=15.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="mock",
            notes=(),
        )

        with (
            patch(
                "benchmarks.compare.subprocess.Popen", return_value=mock_process
            ) as mock_popen,
            patch("benchmarks.compare._is_port_free", return_value=True),
            patch("benchmarks.compare._wait_for_process_port", return_value=True),
            patch("benchmarks.compare._wait_for_service_ready", return_value=None),
            patch("benchmarks.compare._request_completion", return_value=mock_stream),
            patch(
                "benchmarks.compare._reduce_measurements",
                side_effect=lambda name, ms, **_: _benchmark_result(
                    name,
                    ttft_mean_ms=5.0,
                    ttft_p50_ms=5.0,
                    latency_mean_ms=15.0,
                    latency_p50_ms=15.0,
                    decode_time_mean_ms=10.0,
                    decode_tokens_per_second_mean=400.0,
                    decode_tokens_per_second_p50=400.0,
                    end_to_end_tokens_per_second_mean=266.6666666667,
                    end_to_end_tokens_per_second_p50=266.6666666667,
                ),
            ),
            patch("benchmarks.compare.os.killpg", killpg),
            patch("benchmarks.compare.os.getpgid", getpgid),
        ):
            result = _benchmark_http_service(
                backend_name="test-svc",
                command_variants=[["/fake/binary", "--arg"]],
                base_url="http://127.0.0.1:9999",
                model="m",
                prompt_cases=[_prompt_case()],
                tokenizer=None,
                max_tokens=16,
                warmup_trials=0,
                trials=1,
            )

        # Popen called with expected command
        assert mock_popen.call_count >= 1
        popen_call = mock_popen.call_args_list[0]
        assert popen_call[0][0] == ["/fake/binary", "--arg"]

        # Process group killed
        killpg.assert_called_once_with(42_001, signal.SIGTERM)

        # Process waited on
        assert mock_process.wait.call_count >= 1

        # Result returned successfully
        assert isinstance(result, BenchmarkResult)
        assert result.backend == "test-svc"
        assert result.ttft_ms == 5.0

    def test_cleanup_after_wait_for_process_port_failure(self) -> None:
        """When _wait_for_process_port returns False, cleanup still runs."""
        from benchmarks.compare import _benchmark_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_002
        mock_process.wait.return_value = 0
        killpg = MagicMock()
        getpgid = MagicMock(return_value=42_002)

        with (
            patch("benchmarks.compare.subprocess.Popen", return_value=mock_process),
            patch("benchmarks.compare._is_port_free", return_value=True),
            patch("benchmarks.compare._wait_for_process_port", return_value=False),
            patch("benchmarks.compare._wait_for_service_ready"),
            patch("benchmarks.compare._request_completion"),
            patch("benchmarks.compare._reduce_measurements"),
            patch("benchmarks.compare.os.killpg", killpg),
            patch("benchmarks.compare.os.getpgid", getpgid),
        ):
            with pytest.raises(RuntimeError, match="test-svc benchmark failed"):
                _benchmark_http_service(
                    backend_name="test-svc",
                    command_variants=[["/fake/binary"]],
                    base_url="http://127.0.0.1:9998",
                    model="m",
                    prompt_cases=[_prompt_case()],
                    tokenizer=None,
                    max_tokens=16,
                    warmup_trials=0,
                    trials=1,
                )

        # Cleanup still ran despite _wait_for_process_port failing
        killpg.assert_called_once_with(42_002, signal.SIGTERM)
        mock_process.wait.assert_called_once_with(timeout=30)

    def test_skips_occupied_port(self) -> None:
        """When port is in use, logs error and tries next variant."""
        from benchmarks.compare import _benchmark_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_003
        mock_process.wait.return_value = 0
        killpg = MagicMock()
        getpgid = MagicMock(return_value=42_003)

        stream = StreamResult(
            ttft_ms=1.0, latency_ms=2.0, prompt_tokens=3, completion_tokens=4, text="x"
        )

        with (
            patch(
                "benchmarks.compare.subprocess.Popen", return_value=mock_process
            ) as mock_popen,
            # First port check fails, second succeeds
            patch(
                "benchmarks.compare._is_port_free",
                side_effect=[False, True],
            ),
            patch("benchmarks.compare._wait_for_process_port", return_value=True),
            patch("benchmarks.compare._wait_for_service_ready"),
            patch("benchmarks.compare._request_completion", return_value=stream),
            patch(
                "benchmarks.compare._reduce_measurements",
                side_effect=lambda name, ms, **_: _benchmark_result(
                    name,
                    ttft_mean_ms=1.0,
                    ttft_p50_ms=1.0,
                    latency_mean_ms=2.0,
                    latency_p50_ms=2.0,
                    prompt_tokens_mean=3.0,
                    completion_tokens_mean=4.0,
                    total_tokens_mean=7.0,
                    decode_time_mean_ms=1.0,
                    decode_tokens_per_second_mean=4000.0,
                    decode_tokens_per_second_p50=4000.0,
                    end_to_end_tokens_per_second_mean=2000.0,
                    end_to_end_tokens_per_second_p50=2000.0,
                ),
            ),
            patch("benchmarks.compare.os.killpg", killpg),
            patch("benchmarks.compare.os.getpgid", getpgid),
        ):
            result = _benchmark_http_service(
                backend_name="test-svc",
                command_variants=[["first"], ["second"]],
                base_url="http://127.0.0.1:9997",
                model="m",
                prompt_cases=[_prompt_case(prompt_tokens=3)],
                tokenizer=None,
                max_tokens=16,
                warmup_trials=0,
                trials=1,
            )

        # Second variant was used
        assert isinstance(result, BenchmarkResult)
        assert result.backend == "test-svc"

        # Popen was called only for the second variant (first was skipped)
        assert mock_popen.call_count == 1
        assert mock_popen.call_args[0][0] == ["second"]

    def test_timeout_path_sends_killpg_sigkill(self) -> None:
        """When SIGTERM wait times out, sends SIGKILL to process group."""
        from benchmarks.compare import _benchmark_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_004
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 30),
            0,
        ]
        killpg_sigterm = MagicMock()
        killpg_sigkill = MagicMock()
        getpgid = MagicMock(return_value=42_004)

        def _killpg_impl(pgid: int, sig: int) -> None:
            if sig == signal.SIGTERM:
                killpg_sigterm(pgid, sig)
            elif sig == signal.SIGKILL:
                killpg_sigkill(pgid, sig)

        with (
            patch("benchmarks.compare.subprocess.Popen", return_value=mock_process),
            patch("benchmarks.compare._is_port_free", return_value=True),
            patch("benchmarks.compare._wait_for_process_port", return_value=False),
            patch("benchmarks.compare._wait_for_service_ready"),
            patch("benchmarks.compare._request_completion"),
            patch("benchmarks.compare._reduce_measurements"),
            patch("benchmarks.compare.os.killpg", side_effect=_killpg_impl),
            patch("benchmarks.compare.os.getpgid", getpgid),
        ):
            with pytest.raises(RuntimeError, match="test-svc benchmark failed"):
                _benchmark_http_service(
                    backend_name="test-svc",
                    command_variants=[["/fake/binary"]],
                    base_url="http://127.0.0.1:9996",
                    model="m",
                    prompt_cases=[_prompt_case(prompt_tokens=3)],
                    tokenizer=None,
                    max_tokens=16,
                    warmup_trials=0,
                    trials=1,
                )

        killpg_sigterm.assert_called_once_with(42_004, signal.SIGTERM)
        killpg_sigkill.assert_called_once_with(42_004, signal.SIGKILL)
        assert mock_process.wait.call_count == 2
