"""Tests for Phase 9 VLM benchmark fixtures and runner."""

from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.vlm_fixtures import (
    VlmFixture,
    prepare_fixtures,
    _gradient_ppm,
    _chart_pattern_ppm,
    _text_pattern_ppm,
    _CHECKED_IN_IMAGES_DIR,
    _copy_checked_in_images,
)
from benchmarks.compare_vlm import (
    VlmPromptCase,
    VlmStreamResult,
    NopSampler,
    _build_vlm_cases,
    _build_vlm_prompt,
    _extract_port,
    _failed_vlm_stream_result,
    _is_port_free,
    _prepare_project_config,
    _reduce_vlm_measurements,
    _replace_config_value,
    _request_vlm_completion,
    _result_summary,
    _set_vlm_config,
    _wait_for_vlm_service_ready,
)
from mlx_worker.benchmarking import (
    BenchmarkResult,
    BenchmarkRun,
    write_vlm_report,
)


# ===========================================================================
# VLM fixture generation
# ===========================================================================


class TestCopyCheckedInImages:
    """Verify checked-in image copying."""

    def test_copies_existing_images(self, tmp_path: Path) -> None:
        if not _CHECKED_IN_IMAGES_DIR.is_dir():
            pytest.skip("benchmarks/images/ directory not found")
        # Create a dummy checked-in image
        checked_dir = tmp_path / "checked"
        checked_dir.mkdir()
        (checked_dir / "test.png").write_text("png content")
        # Override the constant to point to our temp dir
        import benchmarks.vlm_fixtures as vf

        original = vf._CHECKED_IN_IMAGES_DIR
        vf._CHECKED_IN_IMAGES_DIR = checked_dir
        try:
            result = _copy_checked_in_images(tmp_path / "dest")
            assert "test" in result
            assert (tmp_path / "dest" / "test.png").exists()
        finally:
            vf._CHECKED_IN_IMAGES_DIR = original

    def test_returns_empty_when_no_images(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        import benchmarks.vlm_fixtures as vf

        original = vf._CHECKED_IN_IMAGES_DIR
        vf._CHECKED_IN_IMAGES_DIR = empty_dir
        try:
            result = _copy_checked_in_images(tmp_path / "dest2")
            assert result == {}
        finally:
            vf._CHECKED_IN_IMAGES_DIR = original

    def test_skips_non_image_files(self, tmp_path: Path) -> None:
        checked_dir = tmp_path / "mixed"
        checked_dir.mkdir()
        (checked_dir / "readme.txt").write_text("not an image")
        (checked_dir / "photo.jpg").write_text("jpg data")
        import benchmarks.vlm_fixtures as vf

        original = vf._CHECKED_IN_IMAGES_DIR
        vf._CHECKED_IN_IMAGES_DIR = checked_dir
        try:
            result = _copy_checked_in_images(tmp_path / "dest3")
            assert "photo" in result
            assert "readme" not in result
        finally:
            vf._CHECKED_IN_IMAGES_DIR = original


class TestVlmFixtures:
    """Verify synthetic image generators produce valid PPM files."""

    def test_gradient_ppm_creates_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "gradient.ppm"
        result = _gradient_ppm(dest)
        assert result == dest
        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert content.startswith("P3\n64 64\n255\n")

    def test_chart_pattern_ppm_creates_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "chart.ppm"
        result = _chart_pattern_ppm(dest)
        assert result == dest
        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert content.startswith("P3\n64 64\n255\n")

    def test_text_pattern_ppm_creates_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "text.ppm"
        result = _text_pattern_ppm(dest)
        assert result == dest
        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert content.startswith("P3\n64 64\n255\n")

    def test_prepare_fixtures_returns_three_cases(self, tmp_path: Path) -> None:
        fixtures = prepare_fixtures(tmp_path, use_checked_in=False)
        assert len(fixtures) == 3
        names = [f.name for f in fixtures]
        assert "natural" in names
        assert "chart" in names
        assert "ocr" in names

    def test_prepare_fixtures_all_have_image_paths(self, tmp_path: Path) -> None:
        fixtures = prepare_fixtures(tmp_path)
        # When checked-in images exist, fixtures use those instead of synthetic.
        assert len(fixtures) >= 1
        for fixture in fixtures:
            assert fixture.image_path is not None
            assert fixture.image_path.exists()
            assert fixture.prompt_text
            assert isinstance(fixture.tags, tuple)

    def test_prepare_fixtures_images_differ(self, tmp_path: Path) -> None:
        fixtures = prepare_fixtures(tmp_path)
        contents = {
            f.name: f.image_path.read_bytes() if f.image_path else b"" for f in fixtures
        }
        # All images should have different content
        if len(contents) > 1:
            assert len(set(contents.values())) == len(contents)

    def test_prepare_fixtures_idempotent(self, tmp_path: Path) -> None:
        """Calling prepare_fixtures twice on same dir overwrites without error."""
        _ = prepare_fixtures(tmp_path)
        fixtures2 = prepare_fixtures(tmp_path)
        assert len(fixtures2) >= 1


class TestVlmFixtureDataclass:
    """Verify VlmFixture construction and field access."""

    def test_minimal_construction(self) -> None:
        fixture = VlmFixture(name="test", prompt_text="describe")
        assert fixture.name == "test"
        assert fixture.prompt_text == "describe"

    def test_image_path_optional(self) -> None:
        fixture = VlmFixture(name="x", prompt_text="x")
        assert fixture.image_path is None


# ===========================================================================
# _build_vlm_cases
# ===========================================================================


class TestBuildVlmCases:
    def test_converts_fixtures_to_prompt_cases(self, tmp_path: Path) -> None:
        fixtures = [
            VlmFixture(
                name="natural",
                prompt_text="describe the scene",
                image_path=tmp_path / "img.ppm",
                tags=("vlm",),
            )
        ]
        # Create the image file
        (tmp_path / "img.ppm").write_text("P3\n1 1\n255\n0 0 0\n")

        cases = _build_vlm_cases(fixtures)
        assert len(cases) == 2
        case = cases[0]
        assert case.name == "single-1-natural"
        assert case.prompt_tokens_estimate > 0
        assert len(case.image_paths) == 1
        assert str(tmp_path / "img.ppm") in case.image_paths[0]

    def test_messages_contain_text_and_image_parts(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.ppm"
        img_path.write_text("P3\n1 1\n255\n255 0 0\n")
        fixtures = [
            VlmFixture(
                name="ocr",
                prompt_text="read the text",
                image_path=img_path,
            )
        ]
        cases = _build_vlm_cases(fixtures)
        messages = cases[0].messages
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert "image" in content[0]["text"].lower()
        assert content[1]["type"] == "image_url"
        assert str(img_path) in content[1]["image_url"]["url"]

    def test_builds_multi_image_mix_when_three_fixtures_present(
        self, tmp_path: Path
    ) -> None:
        fixture_names = ["HappyFish", "fruits", "lake"]
        fixtures: list[VlmFixture] = []
        for name in fixture_names:
            image_path = tmp_path / f"{name}.ppm"
            image_path.write_text("P3\n1 1\n255\n0 0 0\n")
            fixtures.append(
                VlmFixture(
                    name=name,
                    prompt_text=f"describe {name}",
                    image_path=image_path,
                )
            )

        cases = _build_vlm_cases(fixtures)

        assert any(case.name == "multi-image-summary" for case in cases)
        long_case = next(
            case for case in cases if case.name == "long-multi-image-analysis"
        )
        assert len(long_case.image_paths) == 3
        assert long_case.prompt_tokens_estimate > 100


# ===========================================================================
# VlmStreamResult
# ===========================================================================


class TestVlmStreamResult:
    def test_minimal_construction(self) -> None:
        result = VlmStreamResult(
            ttft_ms=10.0,
            latency_ms=50.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="output",
            image_count=1,
            image_preprocess_ms=5.0,
        )
        assert result.ttft_ms == 10.0
        assert result.latency_ms == 50.0
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4
        assert result.text == "output"
        assert result.image_count == 1
        assert result.image_preprocess_ms == 5.0
        assert result.succeeded is True

    def test_error_marks_as_failed(self) -> None:
        result = VlmStreamResult(
            ttft_ms=None,
            latency_ms=None,
            prompt_tokens=0,
            completion_tokens=None,
            text="",
            error="RuntimeError: failure",
        )
        assert result.succeeded is False
        assert result.error is not None

    def test_default_image_count_zero(self) -> None:
        result = VlmStreamResult(
            ttft_ms=0.0,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            text="",
        )
        assert result.image_count == 0
        assert result.image_preprocess_ms is None


# ===========================================================================
# _reduce_vlm_measurements
# ===========================================================================


class TestReduceVlmMeasurements:
    def test_single_measurement(self) -> None:
        m = [
            VlmStreamResult(
                ttft_ms=10.0,
                latency_ms=50.0,
                prompt_tokens=8,
                completion_tokens=4,
                text="hi",
                image_count=1,
                image_preprocess_ms=5.0,
            )
        ]
        result = _reduce_vlm_measurements("vlm-backend", m)
        assert result.backend == "vlm-backend"
        assert result.samples == 1
        assert result.errors == 0
        assert result.ttft_mean_ms == 10.0
        assert result.latency_mean_ms == 50.0
        assert result.prompt_tokens_mean == 8.0
        assert result.completion_tokens_mean == 4.0
        assert result.image_preprocess_latency_ms_mean == 5.0
        assert result.image_count_mean == 1.0

    def test_averages_multiple_measurements(self) -> None:
        m = [
            VlmStreamResult(
                ttft_ms=10.0,
                latency_ms=40.0,
                prompt_tokens=8,
                completion_tokens=4,
                text="a",
                image_count=1,
            ),
            VlmStreamResult(
                ttft_ms=20.0,
                latency_ms=60.0,
                prompt_tokens=12,
                completion_tokens=6,
                text="b",
                image_count=2,
            ),
        ]
        result = _reduce_vlm_measurements("avg", m)
        assert result.ttft_mean_ms == 15.0
        assert result.latency_mean_ms == 50.0
        assert result.prompt_tokens_mean == 10.0
        assert result.completion_tokens_mean == 5.0
        assert result.image_count_mean == 1.5

    def test_errors_excluded(self) -> None:
        m = [
            VlmStreamResult(
                ttft_ms=10.0,
                latency_ms=40.0,
                prompt_tokens=8,
                completion_tokens=4,
                text="ok",
                image_count=1,
            ),
            VlmStreamResult(
                ttft_ms=None,
                latency_ms=None,
                prompt_tokens=8,
                completion_tokens=None,
                text="",
                image_count=0,
                error="fail",
            ),
        ]
        result = _reduce_vlm_measurements("err", m)
        assert result.samples == 1
        assert result.errors == 1
        assert result.ttft_mean_ms == 10.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="produced no VLM benchmark"):
            _reduce_vlm_measurements("empty", [])

    def test_vlm_load_time_carried_through(self) -> None:
        m = [
            VlmStreamResult(
                ttft_ms=10.0,
                latency_ms=50.0,
                prompt_tokens=8,
                completion_tokens=4,
                text="hi",
                image_count=1,
            )
        ]
        result = _reduce_vlm_measurements("loaded", m, load_time_ms=1234.5)
        assert result.vlm_load_time_ms == 1234.5

    def test_warning_on_ttft_greater_than_latency(self) -> None:
        m = [
            VlmStreamResult(
                ttft_ms=60.0,
                latency_ms=50.0,
                prompt_tokens=8,
                completion_tokens=4,
                text="weird",
                image_count=0,
            )
        ]
        result = _reduce_vlm_measurements("warn", m)
        assert any("TTFT greater" in w for w in result.warnings)


# ===========================================================================
# _failed_vlm_stream_result
# ===========================================================================


class TestFailedVlmStreamResult:
    def test_creates_failed_result(self) -> None:
        pc = _make_prompt_case()
        exc = RuntimeError("model exploded")
        result = _failed_vlm_stream_result(pc, exc)
        assert result.succeeded is False
        assert "RuntimeError: model exploded" in result.error
        assert result.prompt_tokens == 256

    def test_with_image_paths(self) -> None:
        pc = VlmPromptCase(
            name="img",
            messages=[],
            image_paths=("/tmp/a.ppm", "/tmp/b.ppm"),
            prompt_tokens_estimate=128,
        )
        exc = ValueError("bad image")
        result = _failed_vlm_stream_result(pc, exc)
        assert result.image_count == 2


# ===========================================================================
# _extract_port
# ===========================================================================


class TestExtractPort:
    def test_simple_http(self) -> None:
        assert _extract_port("http://127.0.0.1:8000") == 8000

    def test_https(self) -> None:
        assert _extract_port("https://example.com:443") == 443

    def test_high_port(self) -> None:
        assert _extract_port("http://localhost:65535") == 65535


# ===========================================================================
# _is_port_free
# ===========================================================================


class TestIsPortFree:
    def test_free_when_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "socket.create_connection",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionRefusedError()),
        )
        assert _is_port_free("127.0.0.1", 9999) is True

    def test_free_when_oserror(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "socket.create_connection",
            lambda *a, **kw: (_ for _ in ()).throw(OSError()),
        )
        assert _is_port_free("127.0.0.1", 9999) is True

    def test_occupied(self, monkeypatch) -> None:
        mock_sock = MagicMock()
        monkeypatch.setattr("socket.create_connection", lambda *a, **kw: mock_sock)
        assert _is_port_free("127.0.0.1", 9999) is False


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

    def test_no_match_unchanged(self) -> None:
        text = "a = 1\n"
        result = _replace_config_value(text, "b", "2")
        assert result == text

    def test_numeric_detection(self) -> None:
        text = "timeout = 30\n"
        result = _replace_config_value(text, "timeout", "45")
        assert result == "timeout = 45\n"


# ===========================================================================
# _set_vlm_config
# ===========================================================================


class TestSetVlmConfig:
    """Verify _set_vlm_config uncomments and sets vlm_model correctly."""

    def test_uncomments_commented_line(self) -> None:
        text = '# vlm_model = "old-model"\n'
        result = _set_vlm_config(text, "new-vlm")
        assert 'vlm_model = "new-vlm"' in result
        assert "#" not in result.split("vlm_model")[0].strip()

    def test_replaces_existing_uncommented_line(self) -> None:
        text = 'vlm_model = "old-model"\n'
        result = _set_vlm_config(text, "new-vlm")
        assert 'vlm_model = "new-vlm"' in result

    def test_preserves_indent(self) -> None:
        text = '  # vlm_model = "old"\n'
        result = _set_vlm_config(text, "new")
        for line in result.splitlines():
            if "vlm_model" in line:
                assert line.startswith("  ")
                break

    def test_appends_when_missing(self) -> None:
        text = "port = 8000\n"
        result = _set_vlm_config(text, "added-vlm")
        assert 'vlm_model = "added-vlm"' in result

    def test_appends_without_breaking_existing(self) -> None:
        text = 'model = "test"\nport = 8000\n'
        result = _set_vlm_config(text, "vlm-model")
        assert 'model = "test"' in result
        assert 'vlm_model = "vlm-model"' in result


# ===========================================================================
# _prepare_project_config
# ===========================================================================


class TestPrepareProjectConfig:
    def test_overrides_model_and_port(self, tmp_path: Path) -> None:
        path = _prepare_project_config(
            "mlx-community/test-vlm-4bit", 8222, config_dir=tmp_path
        )
        content = path.read_text(encoding="utf-8")
        assert 'model = "mlx-community/test-vlm-4bit"' in content
        assert "port = 8222" in content
        assert ".sock" in content

    def test_sets_vlm_model_when_provided(self, tmp_path: Path) -> None:
        path = _prepare_project_config(
            "mlx-community/test-vlm-4bit",
            8222,
            config_dir=tmp_path,
            vlm_model="mlx-community/test-vlm-4bit",
        )
        content = path.read_text(encoding="utf-8")
        assert 'vlm_model = "mlx-community/test-vlm-4bit"' in content

    def test_does_not_set_vlm_model_when_not_provided(self, tmp_path: Path) -> None:
        path = _prepare_project_config(
            "mlx-community/test-vlm-4bit", 8222, config_dir=tmp_path
        )
        content = path.read_text(encoding="utf-8")
        # Default config has vlm_model commented out; without vlm_model arg,
        # it stays commented out.
        assert "# vlm_model" in content

    def test_uncomments_vlm_model_in_config(self, tmp_path: Path) -> None:
        """Setting vlm_model uncomments the line and avoids leading '#'."""
        path = _prepare_project_config(
            "mlx-community/test-vlm-4bit",
            8222,
            config_dir=tmp_path,
            vlm_model="mlx-community/test-vlm-4bit",
        )
        content = path.read_text(encoding="utf-8")
        # The line must NOT start with '#' when vlm_model is set.
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("vlm_model ="):
                assert not stripped.startswith("#"), (
                    f"vlm_model line must not be commented: {line!r}"
                )
                break
        else:
            pytest.fail("no vlm_model line found in generated config")

    def test_preserves_text_model_when_vlm_set(self, tmp_path: Path) -> None:
        """When vlm_model is set, the original text model is preserved."""
        path = _prepare_project_config(
            "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            8222,
            config_dir=tmp_path,
            vlm_model="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        )
        content = path.read_text(encoding="utf-8")
        # Text model should remain as the default config value, NOT
        # overwritten by the VLM model name.
        assert 'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"' in content, (
            "text model must be preserved when vlm_model is set"
        )
        assert 'vlm_model = "mlx-community/Qwen2-VL-2B-Instruct-4bit"' in content
        assert "port = 8222" in content

    def test_uses_short_ipc_socket_path(self, tmp_path: Path) -> None:
        path = _prepare_project_config(
            "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            8222,
            config_dir=tmp_path,
            vlm_model="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        )
        content = path.read_text(encoding="utf-8")
        assert f'ipc_path = "{tmp_path / "m.sock"}"' in content


# ===========================================================================
# _result_summary
# ===========================================================================


class TestResultSummary:
    def test_formats_result_line(self) -> None:
        result = BenchmarkResult(
            backend="raw mlx-vlm",
            samples=3,
            errors=0,
            error_rate=0.0,
            ttft_mean_ms=20.0,
            ttft_p50_ms=20.0,
            ttft_p95_ms=None,
            ttft_p99_ms=None,
            latency_mean_ms=100.0,
            latency_p50_ms=100.0,
            latency_p95_ms=None,
            latency_p99_ms=None,
            prompt_tokens_mean=256.0,
            completion_tokens_mean=10.0,
            completion_tokens_p50=10.0,
            total_tokens_mean=266.0,
            decode_time_mean_ms=80.0,
            latency_per_completion_token_ms=10.0,
            decode_time_per_completion_token_ms=8.0,
            latency_p50_per_completion_token_ms=10.0,
            decode_tokens_per_second_mean=125.0,
            decode_tokens_per_second_p50=125.0,
            end_to_end_tokens_per_second_mean=100.0,
            end_to_end_tokens_per_second_p50=100.0,
        )
        summary = _result_summary("test-model", result)
        assert "[VLM model test-model] raw mlx-vlm done:" in summary
        assert "latency_mean=100.0 ms" in summary
        assert "ttft_mean=20.0 ms" in summary
        assert "samples=3" in summary


# ===========================================================================
# VLM report rendering
# ===========================================================================


class TestVlmReportWriting:
    def test_write_vlm_report_creates_file(self, tmp_path: Path) -> None:
        br = BenchmarkResult(
            backend="raw mlx-vlm",
            samples=1,
            errors=0,
            error_rate=0.0,
            ttft_mean_ms=20.0,
            ttft_p50_ms=20.0,
            ttft_p95_ms=None,
            ttft_p99_ms=None,
            latency_mean_ms=100.0,
            latency_p50_ms=100.0,
            latency_p95_ms=None,
            latency_p99_ms=None,
            prompt_tokens_mean=256.0,
            completion_tokens_mean=10.0,
            completion_tokens_p50=10.0,
            total_tokens_mean=266.0,
            decode_time_mean_ms=80.0,
            latency_per_completion_token_ms=10.0,
            decode_time_per_completion_token_ms=8.0,
            latency_p50_per_completion_token_ms=10.0,
            decode_tokens_per_second_mean=125.0,
            decode_tokens_per_second_p50=125.0,
            end_to_end_tokens_per_second_mean=100.0,
            end_to_end_tokens_per_second_p50=100.0,
            image_preprocess_latency_ms_mean=5.0,
            image_count_mean=1.0,
            vlm_load_time_ms=2000.0,
        )
        from mlx_worker.benchmarking import BenchmarkRun

        run = BenchmarkRun(
            model="test-vlm",
            prompt="VLM fixtures",
            max_tokens=128,
            generated_at="2026-06-20T00:00:00+00:00",
            results=(br,),
        )
        path = tmp_path / "phase_9_vlm_report.md"
        write_vlm_report(path, [run])
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Phase 9" in content
        assert "test-vlm" in content
        assert "raw mlx-vlm" in content
        assert "Fairness Notes" in content
        assert "image_preprocess_ms_mean" in content
        assert "5.0" in content


# ===========================================================================
# Request completion (HTTP SSE parsing for VLM)
# ===========================================================================


class _MockResponse:
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


class TestRequestVlmCompletionSseParsing:
    """Verify SSE stream parsing for VLM HTTP requests."""

    def test_basic_sse_stream(self) -> None:
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}],'
            b'"usage":{"completion_tokens":2,"prompt_tokens":10}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare_vlm.urlopen", return_value=mock_resp):
            result = _request_vlm_completion(
                base_url="http://127.0.0.1:8000",
                model="test-vlm",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "desc"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "/tmp/img.ppm"},
                            },
                        ],
                    }
                ],
                max_tokens=16,
            )

        assert result.text == "Hello world"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 2
        assert result.image_count == 1
        assert result.ttft_ms >= 0
        assert result.latency_ms >= result.ttft_ms
        assert result.succeeded

    def test_skips_non_data_lines(self) -> None:
        sse_lines = [
            b":comment\n",
            b"\n",
            b'data: {"choices":[{"delta":{"content":"valid"}}],'
            b'"usage":{"completion_tokens":1,"prompt_tokens":3}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare_vlm.urlopen", return_value=mock_resp):
            result = _request_vlm_completion(
                base_url="http://127.0.0.1:8000",
                model="test-vlm",
                messages=[],
                max_tokens=16,
            )
        assert result.text == "valid"

    def test_empty_content_fallback(self) -> None:
        """When no content deltas, first_delta_at falls back to end time."""
        sse_lines = [
            b'data: {"choices":[{"delta":{}}],'
            b'"usage":{"completion_tokens":0,"prompt_tokens":5}}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare_vlm.urlopen", return_value=mock_resp):
            result = _request_vlm_completion(
                base_url="http://127.0.0.1:8000",
                model="test-vlm",
                messages=[],
                max_tokens=16,
            )
        assert result.text == ""
        assert result.ttft_ms >= 0

    def test_usage_absent_fallback(self) -> None:
        """When no usage in SSE, tokens estimated from text length."""
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = _MockResponse(sse_lines)

        with patch("benchmarks.compare_vlm.urlopen", return_value=mock_resp):
            result = _request_vlm_completion(
                base_url="http://127.0.0.1:8000",
                model="test-vlm",
                messages=[],
                max_tokens=16,
            )
        assert result.text == "Hi"
        assert result.prompt_tokens == 256  # default fallback
        assert result.completion_tokens == 1  # len("Hi") // 4 = 0, max(1,0) = 1


# ===========================================================================
# NopSampler
# ===========================================================================


class TestNopSampler:
    def test_instantiable(self) -> None:
        s = NopSampler()
        assert isinstance(s, NopSampler)


# ===========================================================================
# _build_vlm_prompt — prompt string construction
# ===========================================================================


class TestBuildVlmPrompt:
    """Verify _build_vlm_prompt constructs prompt strings correctly."""

    def test_uses_apply_chat_template_when_available(self) -> None:
        pc = _make_prompt_case()
        model = SimpleNamespace()
        processor = MagicMock()
        processor.apply_chat_template.return_value = "processed prompt"

        result = _build_vlm_prompt(model, processor, pc)
        assert result == "processed prompt"
        processor.apply_chat_template.assert_called_once()
        args, kwargs = processor.apply_chat_template.call_args
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True

    def test_fallback_when_processor_has_no_chat_template(self) -> None:
        pc = _make_prompt_case()
        model = SimpleNamespace()
        processor = object()  # no apply_chat_template

        result = _build_vlm_prompt(model, processor, pc)
        # Falls back to naive string join
        assert "user:" in result.lower() or "User:" in result
        assert "describe" in result

    def test_passes_messages_and_max_tokens_to_template(self) -> None:
        """apply_chat_template receives the prompt messages."""
        pc = _make_prompt_case()
        model = SimpleNamespace()
        processor = MagicMock()
        processor.apply_chat_template.return_value = "processed"

        _build_vlm_prompt(model, processor, pc)
        args, kwargs = processor.apply_chat_template.call_args
        assert args[0] == pc.messages
        assert kwargs.get("tokenize") is False
        assert kwargs.get("add_generation_prompt") is True


# ===========================================================================
# _wait_for_vlm_service_ready — VLM readiness check
# ===========================================================================


class TestWaitForVlmServiceReady:
    """Verify VLM readiness-check behavior with and without readiness_url."""

    def _make_response(self, status: int) -> MagicMock:
        resp = MagicMock()
        resp.status = status
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    def test_with_readiness_url_success(self) -> None:
        mock_resp = self._make_response(200)
        with patch("benchmarks.compare_vlm.urlopen", return_value=mock_resp):
            _wait_for_vlm_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url="/health",
                model="m",
                messages=[],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )

    def test_with_readiness_url_server_error_retries(self) -> None:
        mock_503 = self._make_response(503)
        mock_200 = self._make_response(200)
        with (
            patch(
                "benchmarks.compare_vlm.urlopen",
                side_effect=[mock_503, mock_200],
            ) as mock_urlopen,
            patch("benchmarks.compare_vlm.time.sleep", return_value=None),
        ):
            _wait_for_vlm_service_ready(
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
        mock_resp = self._make_response(503)
        with (
            patch("benchmarks.compare_vlm.urlopen", return_value=mock_resp),
            patch("benchmarks.compare_vlm.time.sleep", return_value=None),
            patch(
                "benchmarks.compare_vlm.time.monotonic",
                side_effect=[100.0, 100.1, 100.2, 100.3, 120.0],
            ),
        ):
            with pytest.raises(RuntimeError, match="service did not become ready"):
                _wait_for_vlm_service_ready(
                    base_url="http://127.0.0.1:8000",
                    readiness_url="/health",
                    model="m",
                    messages=[],
                    max_tokens=16,
                    timeout_s=10,
                    label="svc",
                )

    def test_without_readiness_url_success(self) -> None:
        mock_result = VlmStreamResult(
            ttft_ms=5.0,
            latency_ms=15.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="ready",
            image_count=1,
        )
        with patch(
            "benchmarks.compare_vlm._request_vlm_completion",
            return_value=mock_result,
        ) as mock_req:
            _wait_for_vlm_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url=None,
                model="m",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )
        mock_req.assert_called_once()

    def test_without_readiness_url_retries_then_succeeds(self) -> None:
        mock_result = VlmStreamResult(
            ttft_ms=5.0,
            latency_ms=15.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="ready",
            image_count=0,
        )
        with (
            patch(
                "benchmarks.compare_vlm._request_vlm_completion",
                side_effect=[RuntimeError("not ready"), mock_result],
            ) as mock_req,
            patch("benchmarks.compare_vlm.time.sleep", return_value=None),
        ):
            _wait_for_vlm_service_ready(
                base_url="http://127.0.0.1:8000",
                readiness_url=None,
                model="m",
                messages=[],
                max_tokens=16,
                timeout_s=10,
                label="svc",
            )
        assert mock_req.call_count == 2

    def test_without_readiness_url_timeout(self) -> None:
        with (
            patch(
                "benchmarks.compare_vlm._request_vlm_completion",
                side_effect=RuntimeError("model not loaded"),
            ),
            patch("benchmarks.compare_vlm.time.sleep", return_value=None),
            patch(
                "benchmarks.compare_vlm.time.monotonic",
                side_effect=[100.0, 100.1, 100.2, 100.3, 120.0],
            ),
        ):
            with pytest.raises(
                RuntimeError,
                match="service did not become ready: model not loaded",
            ):
                _wait_for_vlm_service_ready(
                    base_url="http://127.0.0.1:8000",
                    readiness_url=None,
                    model="m",
                    messages=[],
                    max_tokens=16,
                    timeout_s=10,
                    label="svc",
                )


# ===========================================================================
# VLM report fairness notes
# ===========================================================================


class TestVlmReportFairnessNotes:
    """Verify Phase 9 VLM report contains fairness caveat text."""

    def _make_result(self, backend: str) -> BenchmarkResult:
        return BenchmarkResult(
            backend=backend,
            samples=1,
            errors=0,
            error_rate=0.0,
            ttft_mean_ms=20.0,
            ttft_p50_ms=20.0,
            ttft_p95_ms=None,
            ttft_p99_ms=None,
            latency_mean_ms=100.0,
            latency_p50_ms=100.0,
            latency_p95_ms=None,
            latency_p99_ms=None,
            prompt_tokens_mean=256.0,
            completion_tokens_mean=10.0,
            completion_tokens_p50=10.0,
            total_tokens_mean=266.0,
            decode_time_mean_ms=80.0,
            latency_per_completion_token_ms=10.0,
            decode_time_per_completion_token_ms=8.0,
            latency_p50_per_completion_token_ms=10.0,
            decode_tokens_per_second_mean=125.0,
            decode_tokens_per_second_p50=125.0,
            end_to_end_tokens_per_second_mean=100.0,
            end_to_end_tokens_per_second_p50=100.0,
            image_preprocess_latency_ms_mean=5.0,
            image_count_mean=1.0,
            vlm_load_time_ms=2000.0,
        )

    def test_fairness_notes_present(self, tmp_path: Path) -> None:
        run = BenchmarkRun(
            model="test-vlm",
            prompt="VLM fixtures",
            max_tokens=128,
            generated_at="2026-06-20T00:00:00+00:00",
            results=(self._make_result("raw mlx-vlm"),),
        )
        path = tmp_path / "fairness_report.md"
        write_vlm_report(path, [run])
        content = path.read_text(encoding="utf-8")
        assert "Fairness Notes" in content
        assert "Image sizes" in content
        assert "prompt templates" in content
        assert "do not compare raw latency" in content
        assert "raw mlx-vlm" in content
        assert "max_tokens` comes from the benchmark command line" in content

    def test_fairness_backend_differences(self, tmp_path: Path) -> None:
        run = BenchmarkRun(
            model="test-vlm",
            prompt="VLM fixtures (natural, chart, ocr)",
            max_tokens=128,
            generated_at="2026-06-20T00:00:00+00:00",
            results=(
                self._make_result("raw mlx-vlm"),
                self._make_result("mlx_vlm.server"),
                self._make_result("this project"),
            ),
        )
        path = tmp_path / "backends_report.md"
        write_vlm_report(path, [run])
        content = path.read_text(encoding="utf-8")
        assert "raw mlx-vlm" in content
        assert "mlx_vlm.server" in content
        assert "this project" in content

    def test_report_contains_metric_columns(self, tmp_path: Path) -> None:
        run = BenchmarkRun(
            model="test-vlm",
            prompt="VLM fixtures",
            max_tokens=128,
            generated_at="2026-06-20T00:00:00+00:00",
            results=(self._make_result("raw mlx-vlm"),),
        )
        path = tmp_path / "metrics_report.md"
        write_vlm_report(path, [run])
        content = path.read_text(encoding="utf-8")
        assert "ttft_mean_ms" in content
        assert "latency_mean_ms" in content
        assert "completion_tokens_mean" in content
        assert "image_preprocess_ms_mean" in content
        assert "decode_tps_mean" in content
        assert "e2e_tps_mean" in content
        assert "image_preprocess_ms_mean" in content
        assert "error_rate" in content

    def test_report_has_overhead_section(self, tmp_path: Path) -> None:
        results = (
            self._make_result("raw mlx-vlm"),
            self._make_result("mlx_vlm.server"),
        )
        run = BenchmarkRun(
            model="test-vlm",
            prompt="VLM fixtures",
            max_tokens=128,
            generated_at="2026-06-20T00:00:00+00:00",
            results=results,
        )
        path = tmp_path / "overhead_report.md"
        write_vlm_report(path, [run])
        content = path.read_text(encoding="utf-8")
        assert "Overhead" in content
        assert "ttft_mean_delta" in content


# ===========================================================================
# VLM benchmark HTTP service process cleanup
# ===========================================================================


class TestBenchmarkVlmHttpServiceCleanup:
    """Verify process group kill and file cleanup in _benchmark_vlm_http_service."""

    def test_cleanup_called_on_success(self) -> None:
        from benchmarks.compare_vlm import _benchmark_vlm_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_101
        mock_process.wait.return_value = 0
        killpg = MagicMock()
        getpgid = MagicMock(return_value=42_101)

        mock_stream = VlmStreamResult(
            ttft_ms=5.0,
            latency_ms=15.0,
            prompt_tokens=8,
            completion_tokens=4,
            text="mock",
            image_count=1,
        )

        with (
            patch(
                "benchmarks.compare_vlm.subprocess.Popen",
                return_value=mock_process,
            ) as mock_popen,
            patch("benchmarks.compare_vlm._is_port_free", return_value=True),
            patch(
                "benchmarks.compare_vlm._wait_for_process_port",
                return_value=True,
            ),
            patch(
                "benchmarks.compare_vlm._wait_for_vlm_service_ready",
                return_value=None,
            ),
            patch(
                "benchmarks.compare_vlm._request_vlm_completion",
                return_value=mock_stream,
            ),
            patch(
                "benchmarks.compare_vlm._reduce_vlm_measurements",
                side_effect=lambda name, ms, **kw: BenchmarkResult(
                    backend=name,
                    samples=1,
                    errors=0,
                    error_rate=0.0,
                    ttft_mean_ms=5.0,
                    ttft_p50_ms=5.0,
                    ttft_p95_ms=None,
                    ttft_p99_ms=None,
                    latency_mean_ms=15.0,
                    latency_p50_ms=15.0,
                    latency_p95_ms=None,
                    latency_p99_ms=None,
                    prompt_tokens_mean=8.0,
                    completion_tokens_mean=4.0,
                    completion_tokens_p50=4.0,
                    total_tokens_mean=12.0,
                    decode_time_mean_ms=10.0,
                    latency_per_completion_token_ms=3.75,
                    decode_time_per_completion_token_ms=2.5,
                    latency_p50_per_completion_token_ms=3.75,
                    decode_tokens_per_second_mean=400.0,
                    decode_tokens_per_second_p50=400.0,
                    end_to_end_tokens_per_second_mean=266.6666666667,
                    end_to_end_tokens_per_second_p50=266.6666666667,
                ),
            ),
            patch("benchmarks.compare_vlm.os.killpg", killpg),
            patch("benchmarks.compare_vlm.os.getpgid", getpgid),
        ):
            result = _benchmark_vlm_http_service(
                backend_name="test-vlm-svc",
                command_variants=[["/fake/binary", "--arg"]],
                base_url="http://127.0.0.1:9999",
                model="m",
                prompt_cases=[_make_prompt_case()],
                max_tokens=16,
                warmup_trials=0,
                trials=1,
            )

        assert mock_popen.call_count >= 1
        popen_call = mock_popen.call_args_list[0]
        assert popen_call[0][0] == ["/fake/binary", "--arg"]
        killpg.assert_called_once_with(42_101, signal.SIGTERM)
        assert mock_process.wait.call_count >= 1
        assert isinstance(result, BenchmarkResult)
        assert result.backend == "test-vlm-svc"

    def test_cleanup_after_wait_for_process_port_failure(self) -> None:
        from benchmarks.compare_vlm import _benchmark_vlm_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_102
        mock_process.wait.return_value = 0
        killpg = MagicMock()
        getpgid = MagicMock(return_value=42_102)

        with (
            patch(
                "benchmarks.compare_vlm.subprocess.Popen",
                return_value=mock_process,
            ),
            patch("benchmarks.compare_vlm._is_port_free", return_value=True),
            patch(
                "benchmarks.compare_vlm._wait_for_process_port",
                return_value=False,
            ),
            patch("benchmarks.compare_vlm._wait_for_vlm_service_ready"),
            patch("benchmarks.compare_vlm._request_vlm_completion"),
            patch(
                "benchmarks.compare_vlm._reduce_vlm_measurements",
                side_effect=lambda name, ms, **kw: BenchmarkResult(
                    backend=name,
                    samples=0,
                    errors=1,
                    error_rate=1.0,
                    ttft_mean_ms=None,
                    ttft_p50_ms=None,
                    ttft_p95_ms=None,
                    ttft_p99_ms=None,
                    latency_mean_ms=None,
                    latency_p50_ms=None,
                    latency_p95_ms=None,
                    latency_p99_ms=None,
                    prompt_tokens_mean=None,
                    completion_tokens_mean=None,
                    completion_tokens_p50=None,
                    total_tokens_mean=None,
                    decode_time_mean_ms=None,
                    latency_per_completion_token_ms=None,
                    decode_time_per_completion_token_ms=None,
                    latency_p50_per_completion_token_ms=None,
                    decode_tokens_per_second_mean=None,
                    decode_tokens_per_second_p50=None,
                    end_to_end_tokens_per_second_mean=None,
                    end_to_end_tokens_per_second_p50=None,
                    notes=(),
                    warnings=(),
                ),
            ),
            patch("benchmarks.compare_vlm.os.killpg", killpg),
            patch("benchmarks.compare_vlm.os.getpgid", getpgid),
        ):
            result = _benchmark_vlm_http_service(
                backend_name="test-vlm-svc",
                command_variants=[["/fake/binary"]],
                base_url="http://127.0.0.1:9998",
                model="m",
                prompt_cases=[_make_prompt_case()],
                max_tokens=16,
                warmup_trials=0,
                trials=1,
            )
            # Function returns a BenchmarkResult with errors, not raising.
            assert isinstance(result, BenchmarkResult)
            assert result.errors == 1
            assert result.backend == "test-vlm-svc"

        killpg.assert_called_once_with(42_102, signal.SIGTERM)
        mock_process.wait.assert_called_once_with(timeout=30)

    def test_skips_occupied_port(self) -> None:
        from benchmarks.compare_vlm import _benchmark_vlm_http_service

        mock_process = MagicMock()
        mock_process.pid = 42_103
        mock_process.wait.return_value = 0
        killpg = MagicMock()
        getpgid = MagicMock(return_value=42_103)

        mock_stream = VlmStreamResult(
            ttft_ms=1.0,
            latency_ms=2.0,
            prompt_tokens=3,
            completion_tokens=4,
            text="x",
            image_count=0,
        )

        with (
            patch(
                "benchmarks.compare_vlm.subprocess.Popen",
                return_value=mock_process,
            ) as mock_popen,
            patch(
                "benchmarks.compare_vlm._is_port_free",
                side_effect=[False, True],
            ),
            patch(
                "benchmarks.compare_vlm._wait_for_process_port",
                return_value=True,
            ),
            patch(
                "benchmarks.compare_vlm._wait_for_vlm_service_ready",
            ),
            patch(
                "benchmarks.compare_vlm._request_vlm_completion",
                return_value=mock_stream,
            ),
            patch(
                "benchmarks.compare_vlm._reduce_vlm_measurements",
                side_effect=lambda name, ms, **kw: BenchmarkResult(
                    backend=name,
                    samples=1,
                    errors=0,
                    error_rate=0.0,
                    ttft_mean_ms=1.0,
                    ttft_p50_ms=1.0,
                    ttft_p95_ms=None,
                    ttft_p99_ms=None,
                    latency_mean_ms=2.0,
                    latency_p50_ms=2.0,
                    latency_p95_ms=None,
                    latency_p99_ms=None,
                    prompt_tokens_mean=3.0,
                    completion_tokens_mean=4.0,
                    completion_tokens_p50=4.0,
                    total_tokens_mean=7.0,
                    decode_time_mean_ms=1.0,
                    latency_per_completion_token_ms=0.5,
                    decode_time_per_completion_token_ms=0.25,
                    latency_p50_per_completion_token_ms=0.5,
                    decode_tokens_per_second_mean=4000.0,
                    decode_tokens_per_second_p50=4000.0,
                    end_to_end_tokens_per_second_mean=2000.0,
                    end_to_end_tokens_per_second_p50=2000.0,
                ),
            ),
            patch("benchmarks.compare_vlm.os.killpg", killpg),
            patch("benchmarks.compare_vlm.os.getpgid", getpgid),
        ):
            result = _benchmark_vlm_http_service(
                backend_name="test-vlm-svc",
                command_variants=[["first"], ["second"]],
                base_url="http://127.0.0.1:9997",
                model="m",
                prompt_cases=[_make_prompt_case()],
                max_tokens=16,
                warmup_trials=0,
                trials=1,
            )

        assert isinstance(result, BenchmarkResult)
        assert result.backend == "test-vlm-svc"
        assert mock_popen.call_count == 1
        assert mock_popen.call_args[0][0] == ["second"]


# ===========================================================================
# Helper to construct VlmPromptCase for tests
# ===========================================================================


def _make_prompt_case(
    name: str = "test-case",
    image_paths: tuple[str, ...] = ("/tmp/img.ppm",),
) -> VlmPromptCase:
    return VlmPromptCase(
        name=name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "/tmp/img.ppm"},
                    },
                ],
            }
        ],
        image_paths=image_paths,
        prompt_tokens_estimate=256,
    )
