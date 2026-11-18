//! OpenAI-compatible audio endpoints: multipart parsing, request validation,
//! and response rendering for `/v1/audio/transcriptions` and
//! `/v1/audio/translations`.

use mlx_runtime_protocol::{AudioTask, TranscriptionRequest, TranscriptionResponse};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};

static AUDIO_REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Bytes allowed above `limits.max_audio_bytes` for multipart framing and the
/// non-file form fields.
pub(crate) const MULTIPART_OVERHEAD_BYTES: usize = 64 * 1024;

/// Maps a request line to the audio task it serves, if any.
pub(crate) fn audio_route(request_line: &str) -> Option<AudioTask> {
    if request_line.starts_with("POST /v1/audio/transcriptions ") {
        Some(AudioTask::Transcribe)
    } else if request_line.starts_with("POST /v1/audio/translations ") {
        Some(AudioTask::Translate)
    } else {
        None
    }
}

/// A client-facing audio request failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AudioRequestError {
    /// HTTP status line text, such as `400 Bad Request`.
    pub status: &'static str,
    /// Stable machine-readable error code.
    pub code: &'static str,
    /// Human-readable reason.
    pub message: String,
}

impl AudioRequestError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: "400 Bad Request",
            code: "INVALID_REQUEST",
            message: message.into(),
        }
    }

    pub(crate) fn too_large(max_audio_bytes: usize) -> Self {
        Self {
            status: "413 Payload Too Large",
            code: "AUDIO_TOO_LARGE",
            message: format!("audio file exceeds the maximum size of {max_audio_bytes} bytes"),
        }
    }
}

/// OpenAI `response_format` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

impl ResponseFormat {
    fn parse(raw: &str) -> Result<Self, AudioRequestError> {
        match raw {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "srt" => Ok(Self::Srt),
            "verbose_json" => Ok(Self::VerboseJson),
            "vtt" => Ok(Self::Vtt),
            other => Err(AudioRequestError::invalid(format!(
                "unsupported response_format '{other}': must be one of json, text, srt, verbose_json, vtt"
            ))),
        }
    }
}

/// A validated audio request ready for the worker.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AudioRequest {
    /// Worker request.
    pub request: TranscriptionRequest,
    /// How the HTTP response is rendered.
    pub response_format: ResponseFormat,
    /// Whether verbose JSON includes segments.
    pub include_segments: bool,
    /// Whether verbose JSON includes words.
    pub include_words: bool,
}

/// One part of a `multipart/form-data` body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MultipartPart<'a> {
    /// Form field name.
    pub name: String,
    /// Uploaded file name, when the part is a file.
    pub filename: Option<String>,
    /// Raw part bytes.
    pub data: &'a [u8],
}

/// Extracts the boundary from a `multipart/form-data` content type.
pub(crate) fn multipart_boundary(content_type: Option<&str>) -> Result<String, AudioRequestError> {
    let content_type = content_type
        .ok_or_else(|| AudioRequestError::invalid("Content-Type must be multipart/form-data"))?;
    let mut params = content_type.split(';');
    let mime = params.next().unwrap_or_default().trim();
    if !mime.eq_ignore_ascii_case("multipart/form-data") {
        return Err(AudioRequestError::invalid(format!(
            "Content-Type must be multipart/form-data, got '{mime}'"
        )));
    }
    params
        .filter_map(|param| param.split_once('='))
        .find(|(key, _)| key.trim().eq_ignore_ascii_case("boundary"))
        .map(|(_, value)| value.trim().trim_matches('"').to_string())
        .filter(|boundary| !boundary.is_empty() && boundary.len() <= 70)
        .ok_or_else(|| AudioRequestError::invalid("multipart boundary is missing or invalid"))
}

/// Parses a `multipart/form-data` body into its parts.
pub(crate) fn parse_multipart<'a>(
    body: &'a [u8],
    boundary: &str,
) -> Result<Vec<MultipartPart<'a>>, AudioRequestError> {
    let delimiter = format!("--{boundary}").into_bytes();
    let separator = format!("\r\n--{boundary}").into_bytes();
    let malformed = || AudioRequestError::invalid("malformed multipart body");

    let start = find(body, &delimiter, 0).ok_or_else(malformed)?;
    let mut cursor = start + delimiter.len();
    let mut parts = Vec::new();
    loop {
        let rest = &body[cursor..];
        if rest.starts_with(b"--") {
            return Ok(parts);
        }
        if !rest.starts_with(b"\r\n") {
            return Err(malformed());
        }
        cursor += 2;
        let header_end = find(body, b"\r\n\r\n", cursor).ok_or_else(malformed)?;
        let headers = std::str::from_utf8(&body[cursor..header_end])
            .map_err(|_| AudioRequestError::invalid("multipart headers must be UTF-8"))?;
        let data_start = header_end + 4;
        let data_end = find(body, &separator, data_start).ok_or_else(malformed)?;
        let (name, filename) = parse_part_headers(headers)?;
        parts.push(MultipartPart {
            name,
            filename,
            data: &body[data_start..data_end],
        });
        cursor = data_end + separator.len();
    }
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > haystack.len() || needle.is_empty() {
        return None;
    }
    let first = needle[0];
    let mut index = from;
    while index + needle.len() <= haystack.len() {
        let offset = haystack[index..=haystack.len() - needle.len()]
            .iter()
            .position(|byte| *byte == first)?;
        index += offset;
        if haystack[index..].starts_with(needle) {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn parse_part_headers(headers: &str) -> Result<(String, Option<String>), AudioRequestError> {
    let disposition = headers
        .split("\r\n")
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-disposition"))
        .map(|(_, value)| value)
        .ok_or_else(|| AudioRequestError::invalid("multipart part has no Content-Disposition"))?;
    let mut name = None;
    let mut filename = None;
    for param in disposition.split(';').skip(1) {
        let Some((key, value)) = param.split_once('=') else {
            continue;
        };
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|inner| inner.strip_suffix('"'))
            .unwrap_or(value)
            .to_string();
        match key.trim().to_ascii_lowercase().as_str() {
            "name" => name = Some(value),
            "filename" => filename = Some(value),
            _ => {}
        }
    }
    let name = name.ok_or_else(|| AudioRequestError::invalid("multipart part has no name"))?;
    Ok((name, filename))
}

/// Validates the multipart form of an audio request and builds the worker request.
pub(crate) fn parse_audio_request(
    task: AudioTask,
    content_type: Option<&str>,
    body: &[u8],
    max_audio_bytes: usize,
) -> Result<AudioRequest, AudioRequestError> {
    let boundary = multipart_boundary(content_type)?;
    let parts = parse_multipart(body, &boundary)?;

    let mut file: Option<&MultipartPart<'_>> = None;
    let mut model = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = ResponseFormat::Json;
    let mut temperature = 0.0_f32;
    let mut granularities: Vec<String> = Vec::new();
    for part in &parts {
        if part.name == "file" {
            if file.is_some() {
                return Err(AudioRequestError::invalid("exactly one file is allowed"));
            }
            file = Some(part);
            continue;
        }
        let value = std::str::from_utf8(part.data)
            .map_err(|_| {
                AudioRequestError::invalid(format!("field '{}' must be UTF-8", part.name))
            })?
            .trim();
        match part.name.as_str() {
            "model" => model = Some(value.to_string()),
            "language" if !value.is_empty() => language = Some(value.to_ascii_lowercase()),
            "prompt" if !value.is_empty() => prompt = Some(value.to_string()),
            "response_format" => response_format = ResponseFormat::parse(value)?,
            "temperature" => {
                temperature = value
                    .parse::<f32>()
                    .ok()
                    .filter(|value| (0.0..=1.0).contains(value))
                    .ok_or_else(|| {
                        AudioRequestError::invalid("temperature must be a number between 0 and 1")
                    })?;
            }
            "timestamp_granularities[]" | "timestamp_granularities" => {
                for granularity in value.split(',').map(str::trim).filter(|v| !v.is_empty()) {
                    if granularity != "word" && granularity != "segment" {
                        return Err(AudioRequestError::invalid(format!(
                            "unsupported timestamp granularity '{granularity}': must be word or segment"
                        )));
                    }
                    granularities.push(granularity.to_string());
                }
            }
            "stream" if value.eq_ignore_ascii_case("true") => {
                return Err(AudioRequestError::invalid(
                    "streaming is not supported for Whisper transcription",
                ));
            }
            _ => {}
        }
    }

    let file = file.ok_or_else(|| AudioRequestError::invalid("form field 'file' is required"))?;
    if file.data.is_empty() {
        return Err(AudioRequestError::invalid("uploaded audio file is empty"));
    }
    if file.data.len() > max_audio_bytes {
        return Err(AudioRequestError::too_large(max_audio_bytes));
    }
    let model = model
        .filter(|model| !model.is_empty())
        .ok_or_else(|| AudioRequestError::invalid("form field 'model' is required"))?;
    if task == AudioTask::Translate && language.is_some() {
        return Err(AudioRequestError::invalid(
            "language is not supported for translations; the output is always English",
        ));
    }
    if !granularities.is_empty() && response_format != ResponseFormat::VerboseJson {
        return Err(AudioRequestError::invalid(
            "timestamp_granularities requires response_format=verbose_json",
        ));
    }
    let include_words = granularities.iter().any(|value| value == "word");
    let include_segments = granularities.is_empty() || granularities.iter().any(|v| v == "segment");

    Ok(AudioRequest {
        request: TranscriptionRequest {
            request_id: format!(
                "audio-{}",
                AUDIO_REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
            ),
            model,
            task,
            audio_base64: base64_encode(file.data),
            filename: file.filename.clone().filter(|name| !name.is_empty()),
            language,
            prompt,
            temperature,
            word_timestamps: include_words,
        },
        response_format,
        include_segments,
        include_words,
    })
}

/// Rendered HTTP response body and content type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderedAudioResponse {
    pub content_type: &'static str,
    pub body: String,
}

/// Renders a worker transcription in the requested OpenAI response format.
pub(crate) fn render_response(
    response: &TranscriptionResponse,
    request: &AudioRequest,
) -> RenderedAudioResponse {
    let text = response.text.trim();
    let usage = json!({
        "type": "duration",
        "seconds": response.duration.max(0.0).ceil() as u64,
    });
    match request.response_format {
        ResponseFormat::Json => RenderedAudioResponse {
            content_type: "application/json",
            body: json!({ "text": text, "usage": usage }).to_string(),
        },
        ResponseFormat::VerboseJson => {
            let mut payload = json!({
                "task": response.task.as_str(),
                "language": response.language,
                "duration": response.duration,
                "text": text,
                "usage": usage,
            });
            if request.include_segments {
                payload["segments"] = Value::Array(
                    response
                        .segments
                        .iter()
                        .map(|segment| json!(segment))
                        .collect(),
                );
            }
            if request.include_words {
                payload["words"] = json!(response.words);
            }
            RenderedAudioResponse {
                content_type: "application/json",
                body: payload.to_string(),
            }
        }
        ResponseFormat::Text => RenderedAudioResponse {
            content_type: "text/plain; charset=utf-8",
            body: format!("{text}\n"),
        },
        ResponseFormat::Srt => RenderedAudioResponse {
            content_type: "text/plain; charset=utf-8",
            body: render_subtitles(response, false),
        },
        ResponseFormat::Vtt => RenderedAudioResponse {
            content_type: "text/vtt; charset=utf-8",
            body: render_subtitles(response, true),
        },
    }
}

fn render_subtitles(response: &TranscriptionResponse, vtt: bool) -> String {
    let mut output = String::new();
    if vtt {
        output.push_str("WEBVTT\n\n");
    }
    for (index, segment) in response.segments.iter().enumerate() {
        if !vtt {
            output.push_str(&format!("{}\n", index + 1));
        }
        output.push_str(&format!(
            "{} --> {}\n{}\n\n",
            subtitle_timestamp(segment.start, vtt),
            subtitle_timestamp(segment.end, vtt),
            segment.text.trim()
        ));
    }
    output
}

fn subtitle_timestamp(seconds: f64, vtt: bool) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms / 60_000) % 60;
    let secs = (total_ms / 1000) % 60;
    let millis = total_ms % 1000;
    let separator = if vtt { '.' } else { ',' };
    format!("{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}")
}

/// Standard base64 with padding (RFC 4648 section 4).
pub(crate) fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let bytes = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let triple = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
        for position in 0..4 {
            if position <= chunk.len() {
                output.push(ALPHABET[((triple >> (18 - 6 * position)) & 0x3f) as usize] as char);
            } else {
                output.push('=');
            }
        }
    }
    output
}

/// One second of 16 kHz mono 16-bit PCM silence as a WAV file, used to load
/// and warm the transcription model after worker startup.
pub(crate) fn silent_wav() -> Vec<u8> {
    const SAMPLE_RATE: u32 = 16_000;
    let data_len: u32 = SAMPLE_RATE * 2;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1_u16.to_le_bytes()); // mono
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2_u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(44 + data_len as usize, 0);
    wav
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use mlx_runtime_protocol::{TranscriptionSegment, TranscriptionWord};

    pub(crate) const BOUNDARY: &str = "XyZ123";

    pub(crate) fn multipart_body(fields: &[(&str, &str)], file: Option<(&str, &[u8])>) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        if let Some((filename, data)) = file {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        body
    }

    fn content_type() -> String {
        format!("multipart/form-data; boundary={BOUNDARY}")
    }

    fn response() -> TranscriptionResponse {
        TranscriptionResponse {
            request_id: "audio-1".to_string(),
            model: "whisper".to_string(),
            task: AudioTask::Transcribe,
            text: " Hello there. General Kenobi.".to_string(),
            language: "english".to_string(),
            duration: 3661.2,
            segments: vec![
                TranscriptionSegment {
                    id: 0,
                    seek: 0,
                    start: 0.0,
                    end: 1.5,
                    text: " Hello there.".to_string(),
                    tokens: vec![1],
                    temperature: 0.0,
                    avg_logprob: -0.1,
                    compression_ratio: 1.0,
                    no_speech_prob: 0.0,
                },
                TranscriptionSegment {
                    id: 1,
                    seek: 0,
                    start: 3660.25,
                    end: 3661.2,
                    text: " General Kenobi.".to_string(),
                    tokens: vec![2],
                    temperature: 0.0,
                    avg_logprob: -0.1,
                    compression_ratio: 1.0,
                    no_speech_prob: 0.0,
                },
            ],
            words: vec![TranscriptionWord {
                word: " Hello".to_string(),
                start: 0.0,
                end: 0.4,
            }],
            audio_decode_latency_ms: None,
            inference_latency_ms: None,
        }
    }

    fn parse(
        fields: &[(&str, &str)],
        file: Option<(&str, &[u8])>,
    ) -> Result<AudioRequest, AudioRequestError> {
        let body = multipart_body(fields, file);
        parse_audio_request(AudioTask::Transcribe, Some(&content_type()), &body, 1024)
    }

    #[test]
    fn routes_audio_request_lines() {
        assert_eq!(
            audio_route("POST /v1/audio/transcriptions HTTP/1.1\r\n"),
            Some(AudioTask::Transcribe)
        );
        assert_eq!(
            audio_route("POST /v1/audio/translations HTTP/1.1\r\n"),
            Some(AudioTask::Translate)
        );
        assert_eq!(audio_route("POST /v1/audio/speech HTTP/1.1\r\n"), None);
        assert_eq!(
            audio_route("GET /v1/audio/transcriptions HTTP/1.1\r\n"),
            None
        );
    }

    #[test]
    fn boundary_requires_multipart_form_data() {
        assert_eq!(
            multipart_boundary(Some("multipart/form-data; boundary=\"abc\"")).unwrap(),
            "abc"
        );
        assert!(multipart_boundary(Some("application/json")).is_err());
        assert!(multipart_boundary(Some("multipart/form-data")).is_err());
        assert!(multipart_boundary(None).is_err());
    }

    #[test]
    fn parses_fields_and_binary_file_with_embedded_crlf() {
        let audio = b"RIFF\r\n--not-a-boundary\r\n\x00\xff";
        let body = multipart_body(&[("model", "whisper")], Some(("clip.wav", audio)));
        let parts = parse_multipart(&body, BOUNDARY).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "model");
        assert_eq!(parts[0].data, b"whisper");
        assert_eq!(parts[1].filename.as_deref(), Some("clip.wav"));
        assert_eq!(parts[1].data, audio);
    }

    #[test]
    fn rejects_truncated_multipart_body() {
        let mut body = multipart_body(&[("model", "whisper")], Some(("a.wav", b"abc")));
        body.truncate(body.len() - 12);
        assert!(parse_multipart(&body, BOUNDARY).is_err());
    }

    #[test]
    fn builds_worker_request_with_openai_defaults() {
        let request = parse(&[("model", "whisper")], Some(("clip.wav", b"abc"))).unwrap();
        assert_eq!(request.response_format, ResponseFormat::Json);
        assert_eq!(request.request.model, "whisper");
        assert_eq!(request.request.task, AudioTask::Transcribe);
        assert_eq!(request.request.audio_base64, "YWJj");
        assert_eq!(request.request.filename.as_deref(), Some("clip.wav"));
        assert_eq!(request.request.temperature, 0.0);
        assert!(!request.request.word_timestamps);
        assert!(request.include_segments);
        assert!(request.request.request_id.starts_with("audio-"));
    }

    #[test]
    fn word_granularity_enables_word_timestamps() {
        let request = parse(
            &[
                ("model", "whisper"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("language", "EN"),
                ("prompt", "Kenobi"),
                ("temperature", "0.2"),
            ],
            Some(("clip.wav", b"abc")),
        )
        .unwrap();
        assert!(request.request.word_timestamps);
        assert!(request.include_words);
        assert!(!request.include_segments);
        assert_eq!(request.request.language.as_deref(), Some("en"));
        assert_eq!(request.request.prompt.as_deref(), Some("Kenobi"));
        assert_eq!(request.request.temperature, 0.2);
    }

    #[test]
    fn rejects_invalid_forms() {
        let file = Some(("clip.wav", b"abc".as_slice()));
        type Case<'a> = (
            Vec<(&'a str, &'a str)>,
            Option<(&'a str, &'a [u8])>,
            &'a str,
        );
        let cases: Vec<Case<'_>> = vec![
            (vec![("model", "whisper")], None, "'file' is required"),
            (vec![], file, "'model' is required"),
            (vec![("model", "whisper")], Some(("a.wav", b"")), "empty"),
            (
                vec![("model", "whisper"), ("response_format", "xml")],
                file,
                "response_format",
            ),
            (
                vec![("model", "whisper"), ("temperature", "2")],
                file,
                "temperature",
            ),
            (
                vec![("model", "whisper"), ("stream", "true")],
                file,
                "streaming",
            ),
            (
                vec![("model", "whisper"), ("timestamp_granularities[]", "word")],
                file,
                "verbose_json",
            ),
            (
                vec![
                    ("model", "whisper"),
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "char"),
                ],
                file,
                "granularity",
            ),
        ];
        for (fields, file, expected) in cases {
            let error = parse(&fields, file).unwrap_err();
            assert_eq!(error.status, "400 Bad Request");
            assert!(
                error.message.contains(expected),
                "{expected}: {}",
                error.message
            );
        }
    }

    #[test]
    fn rejects_oversized_file_with_413() {
        let audio = vec![1_u8; 1025];
        let error = parse(&[("model", "whisper")], Some(("a.wav", &audio))).unwrap_err();
        assert_eq!(error.status, "413 Payload Too Large");
        assert_eq!(error.code, "AUDIO_TOO_LARGE");
    }

    #[test]
    fn translations_reject_language() {
        let body = multipart_body(
            &[("model", "w"), ("language", "fr")],
            Some(("a.wav", b"abc")),
        );
        let error = parse_audio_request(AudioTask::Translate, Some(&content_type()), &body, 1024)
            .unwrap_err();
        assert!(error.message.contains("translations"));
    }

    #[test]
    fn renders_json_text_and_verbose_json() {
        let mut request = parse(&[("model", "whisper")], Some(("a.wav", b"abc"))).unwrap();
        let json_body: Value =
            serde_json::from_str(&render_response(&response(), &request).body).unwrap();
        assert_eq!(json_body["text"], "Hello there. General Kenobi.");
        assert_eq!(json_body["usage"]["seconds"], 3662);

        request.response_format = ResponseFormat::Text;
        let rendered = render_response(&response(), &request);
        assert_eq!(rendered.body, "Hello there. General Kenobi.\n");
        assert!(rendered.content_type.starts_with("text/plain"));

        request.response_format = ResponseFormat::VerboseJson;
        request.include_words = true;
        let verbose: Value =
            serde_json::from_str(&render_response(&response(), &request).body).unwrap();
        assert_eq!(verbose["task"], "transcribe");
        assert_eq!(verbose["language"], "english");
        assert_eq!(verbose["segments"].as_array().unwrap().len(), 2);
        assert_eq!(verbose["segments"][0]["text"], " Hello there.");
        assert_eq!(verbose["words"][0]["word"], " Hello");
    }

    #[test]
    fn renders_srt_and_vtt_timestamps() {
        let mut request = parse(&[("model", "whisper")], Some(("a.wav", b"abc"))).unwrap();
        request.response_format = ResponseFormat::Srt;
        assert_eq!(
            render_response(&response(), &request).body,
            "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n2\n01:01:00,250 --> 01:01:01,200\nGeneral Kenobi.\n\n"
        );
        request.response_format = ResponseFormat::Vtt;
        let rendered = render_response(&response(), &request);
        assert!(rendered
            .body
            .starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.\n\n"));
        assert_eq!(rendered.content_type, "text/vtt; charset=utf-8");
    }

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), expected);
        }
    }

    #[test]
    fn silent_wav_has_valid_riff_header() {
        let wav = silent_wav();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 32_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 32_000);
    }
}
