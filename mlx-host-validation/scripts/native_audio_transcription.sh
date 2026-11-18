#!/usr/bin/env bash
#
# Whisper speech-to-text host-only validation through the public gateway.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_audio_transcription.sh [options]
#   bash mlx-host-validation/scripts/native_audio_transcription.sh --help
#
# What it checks, against one native-mlx gateway that serves a chat model and
# a Whisper transcription model in the same worker:
#   - the transcription model warms up and reports ready on /models/{model}/ready
#   - /v1/audio/transcriptions returns the spoken sentence for json, text,
#     verbose_json (segments + words), srt, and vtt
#   - /v1/audio/translations runs the translate task
#   - an m4a upload succeeds when ffmpeg is on PATH, and returns 400 naming
#     ffmpeg when it is not
#   - a missing file returns 400 and /v1/audio/speech returns 404
#   - a chat completion still succeeds after the audio requests
#   - /metrics exposes the audio duration counter and audio latency gauges
#
# Speech comes from the macOS `say` command, so the expected words are fixed.
#
# Host requirements:
#   - Apple Silicon (`arm64`) with Metal
#   - `uv` environment for `python/`, `cargo`, and macOS `say`
#   - network access or a warm Hugging Face cache for both checkpoints
#
# Expected success signals:
#   - `transcription_ready_ok=1`
#   - `transcription_json_ok=1`
#   - `transcription_formats_ok=1`
#   - `transcription_verbose_ok=1`
#   - `translation_ok=1`
#   - `ffmpeg_format_ok=1`
#   - `audio_errors_ok=1`
#   - `chat_after_audio_ok=1`
#   - `audio_metrics_ok=1`
#   - `audio_transcription_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway or transcription model never becomes ready
#   - a transcript missing the spoken words, or an unexpected HTTP status

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
# shellcheck source=lib/gateway.sh
source "$ROOT/mlx-host-validation/scripts/lib/gateway.sh"
cli_describe "Validate Whisper transcription and translation endpoints next to a native-mlx chat model through the public gateway."
cli_option CHECKPOINT --checkpoint MLX_AUDIO_TRANSCRIPTION_CHECKPOINT string \
    "mlx-community/Qwen3-4B-Instruct-2507-4bit" "Chat checkpoint served by native-mlx next to Whisper."
cli_option TRANSCRIPTION_MODEL --transcription-model MLX_AUDIO_TRANSCRIPTION_MODEL string \
    "mlx-community/whisper-large-v3-turbo-asr-fp16" "Whisper checkpoint served on the audio endpoints."
cli_option PORT --port MLX_AUDIO_TRANSCRIPTION_PORT port 18200 "Gateway port."
cli_option HEALTH_TIMEOUT --health-timeout MLX_AUDIO_TRANSCRIPTION_HEALTH_TIMEOUT int 600 \
    "Seconds to wait for the chat model and then the transcription model to become ready."
cli_option REQUEST_TIMEOUT --request-timeout MLX_AUDIO_TRANSCRIPTION_REQUEST_TIMEOUT int 300 \
    "Seconds allowed for each HTTP request."
cli_option ARTIFACTS_DIR --artifacts-dir MLX_AUDIO_TRANSCRIPTION_ARTIFACTS_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-audio-transcription" "Directory for logs, configs, audio, and captures."
cli_option SKIP_BUILD --skip-build MLX_AUDIO_TRANSCRIPTION_SKIP_BUILD bool 0 \
    "Skip uv sync and cargo build; reuse the existing environment and gateway binary."
cli_example "bash mlx-host-validation/scripts/native_audio_transcription.sh --port 18210"
cli_example "bash mlx-host-validation/scripts/native_audio_transcription.sh --transcription-model mlx-community/whisper-tiny-asr-fp16"
cli_parse "$@"
cli_require_free_port "$PORT" --port

BASE_URL="http://127.0.0.1:$PORT"
GATEWAY_LOG="$ARTIFACTS_DIR/gateway.log"
CONFIG_PATH="$ARTIFACTS_DIR/runtime.toml"
CAPTURE_DIR="$ARTIFACTS_DIR/captures"
SPEECH_WAV="$ARTIFACTS_DIR/speech.wav"
SPEECH_M4A="$ARTIFACTS_DIR/speech.m4a"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
# Unix socket paths are limited to 104 bytes, so keep it out of ARTIFACTS_DIR.
SOCKET_PATH="/tmp/mlx-audio-transcription-$PORT.sock"
SENTENCE="The quick brown fox jumps over the lazy dog."
mkdir -p "$CAPTURE_DIR"

GATEWAY_PID=""
cleanup() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# post_audio NAME ENDPOINT EXPECTED_STATUS [curl form args...]
post_audio() {
    local name="$1" endpoint="$2" expected="$3"
    shift 3
    local status
    status=$(curl -sS --max-time "$REQUEST_TIMEOUT" -o "$CAPTURE_DIR/$name" -w '%{http_code}' \
        -X POST "$BASE_URL/v1/audio/$endpoint" "$@")
    if [[ "$status" != "$expected" ]]; then
        echo "$name: expected HTTP $expected, got $status; inspect $CAPTURE_DIR/$name and $GATEWAY_LOG" >&2
        exit 1
    fi
}

echo "[1/6] Sync Python environment and build the gateway"
if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "build_skipped=1"
else
    uv --directory "$PYTHON_DIR" sync --group dev
    cargo build --manifest-path "$ROOT/Cargo.toml" -p mlx_runtime_gateway
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "expected Apple Silicon arm64 host" >&2
    exit 1
fi

echo "[2/6] Synthesize speech and write the runtime config"
say -o "$SPEECH_WAV" --data-format=LEI16@16000 "$SENTENCE"
say -o "$SPEECH_M4A" --file-format=m4af --data-format=aac "$SENTENCE"
uv --directory "$PYTHON_DIR" run python - <<'PY' \
    "$ROOT/config/runtime.toml" "$CONFIG_PATH" "$CHECKPOINT" "$TRANSCRIPTION_MODEL" "$PORT" "$SOCKET_PATH"
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
for needle in ("port = 8000", 'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"', 'ipc_path = "/tmp/mlx-runtime.sock"'):
    if needle not in source:
        raise SystemExit(f"config/runtime.toml no longer contains {needle!r}")
source = source.replace("port = 8000", f"port = {sys.argv[5]}", 1)
source = source.replace('backend = "v1"', 'backend = "native-mlx"')
source = source.replace(
    'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
    f'model = "{sys.argv[3]}"\ntranscription_model = "{sys.argv[4]}"',
    1,
)
source = source.replace(
    'ipc_path = "/tmp/mlx-runtime.sock"',
    f'ipc_path = "{sys.argv[6]}"',
    1,
)
pathlib.Path(sys.argv[2]).write_text(source)
PY

echo "[3/6] Start the gateway and wait for both models"
rm -f "$GATEWAY_LOG" "$SOCKET_PATH"
(
    cd "$ROOT"
    exec env MLX_RUNTIME_CONFIG="$CONFIG_PATH" "$GATEWAY_BIN"
) >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!
gateway_wait_ready "$PORT" "$HEALTH_TIMEOUT" "$GATEWAY_PID" "$GATEWAY_LOG"
ENCODED_MODEL="${TRANSCRIPTION_MODEL//\//%2F}"
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -sf --max-time 2 "$BASE_URL/models/$ENCODED_MODEL/ready" >/dev/null; do
    state="$(curl -s --max-time 2 "$BASE_URL/models/$ENCODED_MODEL/status" || true)"
    if [[ "$state" == *'"state":"failed"'* ]]; then
        echo "transcription model failed to load: $state" >&2
        gateway_log_tail "$GATEWAY_LOG"
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "transcription model not ready within ${HEALTH_TIMEOUT}s: $state" >&2
        gateway_log_tail "$GATEWAY_LOG"
        exit 1
    fi
    sleep 1
done
echo "transcription_ready_ok=1"

echo "[4/6] Transcribe and translate"
MODEL_FIELD=(-F "model=$TRANSCRIPTION_MODEL")
post_audio json.json transcriptions 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_WAV" -F language=en
post_audio text.txt transcriptions 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_WAV" -F response_format=text
post_audio subtitles.srt transcriptions 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_WAV" -F response_format=srt
post_audio subtitles.vtt transcriptions 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_WAV" -F response_format=vtt
post_audio verbose.json transcriptions 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_WAV" \
    -F response_format=verbose_json -F "timestamp_granularities[]=word" -F "timestamp_granularities[]=segment"
post_audio translation.json translations 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_WAV" -F response_format=verbose_json
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_PRESENT=1
    post_audio m4a.json transcriptions 200 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_M4A"
else
    FFMPEG_PRESENT=0
    post_audio m4a.json transcriptions 400 "${MODEL_FIELD[@]}" -F "file=@$SPEECH_M4A"
fi

echo "[5/6] Error paths and chat after audio"
post_audio missing-file.json transcriptions 400 "${MODEL_FIELD[@]}"
SPEECH_STATUS=$(curl -sS --max-time "$REQUEST_TIMEOUT" -o "$CAPTURE_DIR/speech.txt" -w '%{http_code}' \
    -X POST "$BASE_URL/v1/audio/speech" -H 'Content-Type: application/json' \
    --data '{"model":"tts","input":"hello","voice":"alloy"}')
if [[ "$SPEECH_STATUS" != "404" ]]; then
    echo "/v1/audio/speech: expected HTTP 404, got $SPEECH_STATUS" >&2
    exit 1
fi
CHAT_STATUS=$(curl -sS --max-time "$REQUEST_TIMEOUT" -o "$CAPTURE_DIR/chat.json" -w '%{http_code}' \
    -X POST "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
    --data "{\"model\":\"$CHECKPOINT\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":8,\"temperature\":0}")
if [[ "$CHAT_STATUS" != "200" ]]; then
    echo "chat after audio: expected HTTP 200, got $CHAT_STATUS; inspect $CAPTURE_DIR/chat.json" >&2
    exit 1
fi
curl -fsS --max-time 10 "$BASE_URL/metrics" >"$CAPTURE_DIR/metrics.txt"

echo "[6/6] Verify captured responses"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$CAPTURE_DIR" "$FFMPEG_PRESENT"
import json
import pathlib
import re
import sys

captures = pathlib.Path(sys.argv[1])
ffmpeg_present = sys.argv[2] == "1"
expected = {"quick", "brown", "fox", "lazy", "dog"}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def require_sentence(name: str, text: str) -> None:
    missing = expected - words(text)
    if missing:
        raise SystemExit(f"{name}: transcript {text!r} is missing {sorted(missing)}")


payload = json.loads((captures / "json.json").read_text())
require_sentence("json", payload["text"])
if payload["usage"]["type"] != "duration" or payload["usage"]["seconds"] < 1:
    raise SystemExit(f"json: unexpected usage {payload['usage']}")
print(f"transcription_json_text={payload['text']!r}")
print("transcription_json_ok=1")

text = (captures / "text.txt").read_text()
require_sentence("text", text)
srt = (captures / "subtitles.srt").read_text()
if not re.match(r"1\n\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}\n", srt):
    raise SystemExit(f"srt: unexpected body {srt!r}")
require_sentence("srt", srt)
vtt = (captures / "subtitles.vtt").read_text()
if not re.match(r"WEBVTT\n\n\d\d:\d\d:\d\d\.\d{3} --> ", vtt):
    raise SystemExit(f"vtt: unexpected body {vtt!r}")
print("transcription_formats_ok=1")

verbose = json.loads((captures / "verbose.json").read_text())
if verbose["task"] != "transcribe" or verbose["language"] != "english":
    raise SystemExit(f"verbose_json: task/language {verbose['task']}/{verbose['language']}")
if not verbose["segments"] or not verbose["words"]:
    raise SystemExit("verbose_json: segments or words are empty")
if not 1.0 < verbose["duration"] < 10.0:
    raise SystemExit(f"verbose_json: implausible duration {verbose['duration']}")
ends = [word["end"] for word in verbose["words"]]
if ends != sorted(ends) or ends[-1] > verbose["duration"] + 0.5:
    raise SystemExit(f"verbose_json: word timestamps out of order or range: {ends}")
require_sentence("verbose_json", " ".join(word["word"] for word in verbose["words"]))
print(f"transcription_word_count={len(verbose['words'])}")
print("transcription_verbose_ok=1")

translation = json.loads((captures / "translation.json").read_text())
if translation["task"] != "translate":
    raise SystemExit(f"translation: task {translation['task']}")
require_sentence("translation", translation["text"])
print("translation_ok=1")

m4a = json.loads((captures / "m4a.json").read_text())
if ffmpeg_present:
    require_sentence("m4a", m4a["text"])
    print("ffmpeg_format_branch=decoded")
else:
    if "ffmpeg" not in m4a["error"]["message"]:
        raise SystemExit(f"m4a without ffmpeg: error does not name ffmpeg: {m4a}")
    print("ffmpeg_format_branch=rejected_without_ffmpeg")
print("ffmpeg_format_ok=1")

missing = json.loads((captures / "missing-file.json").read_text())
if "'file' is required" not in missing["error"]["message"]:
    raise SystemExit(f"missing file: unexpected error {missing}")
print("audio_errors_ok=1")

chat = json.loads((captures / "chat.json").read_text())
if not chat["choices"][0]["message"]["content"].strip():
    raise SystemExit("chat after audio returned empty content")
print("chat_after_audio_ok=1")

metrics = (captures / "metrics.txt").read_text()
for needle in (
    'mlx_audio_input_ms_by_backend_total{backend="mlx-audio",task="transcribe"}',
    'mlx_audio_input_ms_by_backend_total{backend="mlx-audio",task="translate"}',
    'kind="audio_inference"',
    'modality="audio",kind="gateway_queue"',
):
    if needle not in metrics:
        raise SystemExit(f"metrics missing {needle}")
print("audio_metrics_ok=1")
PY

echo "gateway_log=$GATEWAY_LOG"
echo "audio_transcription_validation_ok=1"
