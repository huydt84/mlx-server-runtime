use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

/// Runtime server settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    /// Bind host for the HTTP server.
    pub host: String,
    /// Bind port for the HTTP server.
    pub port: u16,
}

/// Runtime worker settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    /// Python executable path.
    pub python: String,
    /// Python module entrypoint.
    pub module: String,
    /// Worker backend. `native-mlx` is the default; `v1` serves through
    /// `mlx-lm` for checkpoints and VLMs native-mlx does not implement.
    pub backend: BackendKind,
    /// Text model identifier.
    pub model: String,
    /// Optional VLM model identifier.
    pub vlm_model: Option<String>,
    /// Unix socket path used for bootstrap IPC.
    pub ipc_path: String,
}

/// Supported worker backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Existing `mlx_lm` runtime path.
    V1,
    /// Native MLX runtime path, the default.
    NativeMlx,
}

impl BackendKind {
    /// Returns config/env string for this backend.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::NativeMlx => "native-mlx",
        }
    }

    fn parse(raw: &str) -> io::Result<Self> {
        match raw {
            "v1" => Ok(Self::V1),
            "native-mlx" => Ok(Self::NativeMlx),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported worker.backend: {other}"),
            )),
        }
    }
}

/// Default generation settings used by the gateway for omitted request fields.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    /// Default temperature.
    pub temperature: f32,
    /// Default top-p.
    pub top_p: f32,
    /// Default maximum completion length.
    pub max_tokens: u32,
}

/// Backpressure limits for inbound requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestLimits {
    /// Maximum queued requests waiting for a worker slot.
    pub max_pending_requests: usize,
    /// Maximum concurrent active requests.
    pub max_active_requests: usize,
    /// Maximum prompt tokens allowed per request.
    pub max_prompt_tokens: usize,
    /// Maximum completion tokens allowed per request.
    pub max_completion_tokens: usize,
    /// Maximum prompt + completion tokens allowed per request.
    pub max_total_tokens_per_request: usize,
    /// Seconds to wait for a slot before returning 429.
    pub request_timeout_seconds: u64,
    /// Maximum images allowed in a single VLM request (enforced before queue).
    pub max_vlm_images: usize,
}

/// Default share of the Metal recommended working set the worker may use.
pub const DEFAULT_MEMORY_UTILIZATION: f32 = 0.5;

/// Worker memory sizing.
///
/// The native-mlx worker sizes its KV page pool from these values after the
/// weights load: `utilization x recommended working set - weights - activation
/// reserve`, unless `kv_cache_bytes` pins an absolute pool size.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryConfig {
    /// Fraction in (0, 1] of the Metal recommended working set for weights,
    /// activations, and the KV pool together.
    pub utilization: f32,
    /// Absolute KV pool size in bytes; overrides `utilization` when set.
    pub kv_cache_bytes: Option<u64>,
    /// Whether `limits.max_total_tokens_per_request` was set explicitly. When
    /// false, the worker lowers it to the KV pool capacity; when true and the
    /// pool cannot hold one such request, the worker refuses to start.
    pub max_total_tokens_explicit: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            utilization: DEFAULT_MEMORY_UTILIZATION,
            kv_cache_bytes: None,
            max_total_tokens_explicit: false,
        }
    }
}

impl MemoryConfig {
    /// Validates a memory utilization fraction.
    pub fn validate_utilization(value: f64) -> Result<f32, String> {
        if value.is_finite() && value > 0.0 && value <= 1.0 {
            Ok(value as f32)
        } else {
            Err(format!(
                "memory utilization must be greater than 0 and at most 1, got {value}"
            ))
        }
    }
}

/// Parses a byte size such as `4GiB`, `512MiB`, `2GB`, or `1073741824`.
pub fn parse_byte_size(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    let split = trimmed
        .find(|character: char| !(character.is_ascii_digit() || character == '.'))
        .unwrap_or(trimmed.len());
    let (number, unit) = trimmed.split_at(split);
    let multiplier: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" => 1e3,
        "m" | "mb" => 1e6,
        "g" | "gb" => 1e9,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unsupported byte-size unit {other:?} in {raw:?}")),
    };
    let value: f64 = number
        .parse()
        .map_err(|_| format!("invalid byte size {raw:?}"))?;
    let bytes = (value * multiplier).round();
    if !bytes.is_finite() || bytes < 1.0 || bytes > u64::MAX as f64 {
        return Err(format!("byte size must be positive, got {raw:?}"));
    }
    Ok(bytes as u64)
}

/// Telemetry settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryConfig {
    /// Whether Prometheus metrics are enabled.
    pub enable_prometheus: bool,
    /// HTTP path for metrics exposition.
    pub metrics_path: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: true,
            metrics_path: "/metrics".to_string(),
        }
    }
}

/// Runtime configuration loaded from TOML.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    /// HTTP server configuration.
    pub server: ServerConfig,
    /// Worker process configuration.
    pub worker: WorkerConfig,
    /// Generation defaults.
    pub generation: GenerationConfig,
    /// Request backpressure limits.
    pub limits: RequestLimits,
    /// Worker memory sizing.
    pub memory: MemoryConfig,
    /// Telemetry settings.
    pub telemetry: TelemetryConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8000,
            },
            worker: WorkerConfig {
                python: "python/.venv/bin/python".to_string(),
                module: "mlx_worker.main".to_string(),
                backend: BackendKind::NativeMlx,
                model: "mlx-community/Qwen2.5-7B-Instruct-4bit".to_string(),
                vlm_model: None,
                ipc_path: "/tmp/mlx-runtime.sock".to_string(),
            },
            generation: GenerationConfig {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 512,
            },
            limits: RequestLimits {
                max_pending_requests: 64,
                max_active_requests: 16,
                max_prompt_tokens: 32_768,
                max_completion_tokens: 4_096,
                max_total_tokens_per_request: 65_536,
                request_timeout_seconds: 300,
                max_vlm_images: 5,
            },
            memory: MemoryConfig::default(),
            telemetry: TelemetryConfig {
                enable_prometheus: true,
                metrics_path: "/metrics".to_string(),
            },
        }
    }
}

impl RuntimeConfig {
    /// Loads a runtime config from a tiny TOML subset.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(err) => return Err(err),
        };

        Self::apply_contents(Self::default(), &contents)
    }

    /// Loads a required runtime config, returning an error when it is missing.
    pub fn load_required(path: impl AsRef<Path>) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        Self::apply_contents(Self::default(), &contents)
    }

    /// Applies values from a required config file over this configuration.
    pub fn overlay(path: impl AsRef<Path>, base: Self) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        Self::apply_contents(base, &contents)
    }

    /// Parses config text over the defaults; test helper for renderers.
    #[cfg(test)]
    pub(crate) fn overlay_contents_for_test(contents: &str) -> Self {
        Self::apply_contents(Self::default(), contents).expect("rendered config parses")
    }

    fn apply_contents(mut config: Self, contents: &str) -> io::Result<Self> {
        let mut section = String::new();

        for raw_line in contents.lines() {
            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(section_name) = parse_section(line) {
                section = section_name;
                continue;
            }

            let (key, value) = match line.split_once('=') {
                Some(pair) => pair,
                None => continue,
            };

            let key = key.trim();
            let value = parse_value(value.trim())?;
            match (section.as_str(), key, value) {
                ("server", "host", Value::String(host)) => config.server.host = host,
                ("server", "port", Value::Integer(port)) => {
                    config.server.port = u16::try_from(port).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "server.port out of range")
                    })?
                }
                ("worker", "python", Value::String(python)) => config.worker.python = python,
                ("worker", "module", Value::String(module)) => config.worker.module = module,
                ("worker", "backend", Value::String(backend)) => {
                    config.worker.backend = BackendKind::parse(&backend)?
                }
                ("worker", "model", Value::String(model)) => config.worker.model = model,
                ("worker", "vlm_model", Value::String(vlm_model)) => {
                    config.worker.vlm_model = Some(vlm_model)
                }
                ("worker", "ipc_path", Value::String(ipc_path)) => {
                    config.worker.ipc_path = ipc_path
                }
                ("generation", "temperature", Value::Float(temperature)) => {
                    config.generation.temperature = temperature
                }
                ("generation", "top_p", Value::Float(top_p)) => config.generation.top_p = top_p,
                ("generation", "max_tokens", Value::Integer(max_tokens)) => {
                    config.generation.max_tokens = u32::try_from(max_tokens).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "generation.max_tokens out of range",
                        )
                    })?
                }
                ("limits", "max_pending_requests", Value::Integer(value)) => {
                    config.limits.max_pending_requests = usize::try_from(value).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "limits.max_pending_requests out of range",
                        )
                    })?
                }
                ("limits", "max_active_requests", Value::Integer(value)) => {
                    config.limits.max_active_requests = usize::try_from(value).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "limits.max_active_requests out of range",
                        )
                    })?
                }
                ("limits", "max_prompt_tokens", Value::Integer(value)) => {
                    config.limits.max_prompt_tokens = usize::try_from(value).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "limits.max_prompt_tokens out of range",
                        )
                    })?
                }
                ("limits", "max_completion_tokens", Value::Integer(value)) => {
                    config.limits.max_completion_tokens = usize::try_from(value).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "limits.max_completion_tokens out of range",
                        )
                    })?
                }
                ("limits", "max_total_tokens_per_request", Value::Integer(value)) => {
                    config.limits.max_total_tokens_per_request =
                        usize::try_from(value).map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "limits.max_total_tokens_per_request out of range",
                            )
                        })?;
                    config.memory.max_total_tokens_explicit = true;
                }
                ("memory", "utilization", Value::Float(value)) => {
                    config.memory.utilization = MemoryConfig::validate_utilization(value.into())
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?
                }
                ("memory", "utilization", Value::Integer(value)) => {
                    config.memory.utilization = MemoryConfig::validate_utilization(value as f64)
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?
                }
                ("memory", "kv_cache_bytes", Value::Integer(value)) => {
                    let bytes = u64::try_from(value)
                        .ok()
                        .filter(|bytes| *bytes > 0)
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "memory.kv_cache_bytes must be positive",
                            )
                        })?;
                    config.memory.kv_cache_bytes = Some(bytes);
                }
                ("memory", "kv_cache_bytes", Value::String(value)) => {
                    config.memory.kv_cache_bytes = Some(
                        parse_byte_size(&value)
                            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?,
                    );
                }
                ("limits", "request_timeout_seconds", Value::Integer(value)) => {
                    config.limits.request_timeout_seconds = u64::try_from(value).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "limits.request_timeout_seconds out of range",
                        )
                    })?
                }
                ("limits", "max_vlm_images", Value::Integer(value)) => {
                    config.limits.max_vlm_images = usize::try_from(value).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "limits.max_vlm_images out of range",
                        )
                    })?
                }
                ("telemetry", "enable_prometheus", Value::Bool(value)) => {
                    config.telemetry.enable_prometheus = value
                }
                ("telemetry", "metrics_path", Value::String(metrics_path)) => {
                    config.telemetry.metrics_path = metrics_path
                }
                _ => {}
            }
        }

        Ok(config)
    }

    /// Renders startup configuration for logs.
    pub fn startup_log(&self, config_path: &str) -> String {
        let mut rendered = String::new();
        let _ = writeln!(rendered, "runtime_config path={config_path}");
        let _ = writeln!(
            rendered,
            "server host={} port={}",
            self.server.host, self.server.port
        );
        let _ = writeln!(
            rendered,
            "worker python={} module={} backend={} model={} vlm_model={} ipc_path={}",
            self.worker.python,
            self.worker.module,
            self.worker.backend.as_str(),
            self.worker.model,
            self.worker.vlm_model.as_deref().unwrap_or("<none>"),
            self.worker.ipc_path
        );
        let _ = writeln!(
            rendered,
            "generation temperature={} top_p={} max_tokens={}",
            self.generation.temperature, self.generation.top_p, self.generation.max_tokens
        );
        let _ = writeln!(
            rendered,
            "limits max_pending_requests={} max_active_requests={} max_prompt_tokens={} max_completion_tokens={} max_total_tokens_per_request={} request_timeout_seconds={} max_vlm_images={}",
            self.limits.max_pending_requests,
            self.limits.max_active_requests,
            self.limits.max_prompt_tokens,
            self.limits.max_completion_tokens,
            self.limits.max_total_tokens_per_request,
            self.limits.request_timeout_seconds,
            self.limits.max_vlm_images
        );
        let _ = writeln!(
            rendered,
            "memory utilization={} kv_cache_bytes={} max_total_tokens_explicit={}",
            self.memory.utilization,
            self.memory
                .kv_cache_bytes
                .map_or_else(|| "<derived>".to_string(), |bytes| bytes.to_string()),
            self.memory.max_total_tokens_explicit
        );
        let _ = write!(
            rendered,
            "telemetry enable_prometheus={} metrics_path={}",
            self.telemetry.enable_prometheus, self.telemetry.metrics_path
        );
        rendered
    }
}

fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..index],
            _ => {}
        }
    }
    line
}

fn parse_section(line: &str) -> Option<String> {
    let section = line.strip_prefix('[')?.strip_suffix(']')?;
    Some(section.trim().to_string())
}

enum Value {
    String(String),
    Integer(i64),
    Float(f32),
    Bool(bool),
}

fn parse_value(raw: &str) -> io::Result<Value> {
    if let Some(value) = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Ok(Value::String(parse_basic_string(value)?));
    }

    if let Ok(value) = raw.parse::<i64>() {
        return Ok(Value::Integer(value));
    }

    if let Ok(value) = raw.parse::<f32>() {
        return Ok(Value::Float(value));
    }

    match raw {
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported value: {other}"),
        )),
    }
}

fn parse_basic_string(raw: &str) -> io::Result<String> {
    let mut parsed = String::with_capacity(raw.len());
    let mut characters = raw.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            parsed.push(character);
            continue;
        }
        let escaped = characters.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unterminated string escape")
        })?;
        parsed.push(match escaped {
            '\\' => '\\',
            '"' => '"',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported string escape: \\{other}"),
                ));
            }
        });
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn load_text(contents: &str) -> io::Result<RuntimeConfig> {
        RuntimeConfig::apply_contents(RuntimeConfig::default(), contents)
    }

    #[test]
    fn memory_defaults_derive_the_pool_and_request_limit() {
        let config = RuntimeConfig::default();
        assert_eq!(config.memory.utilization, DEFAULT_MEMORY_UTILIZATION);
        assert_eq!(config.memory.kv_cache_bytes, None);
        assert!(!config.memory.max_total_tokens_explicit);
    }

    #[test]
    fn memory_section_sets_utilization_and_absolute_pool() {
        let config = load_text("[memory]\nutilization = 0.7\nkv_cache_bytes = \"4GiB\"\n").unwrap();
        assert!((config.memory.utilization - 0.7).abs() < 1e-6);
        assert_eq!(config.memory.kv_cache_bytes, Some(4 * 1024 * 1024 * 1024));

        let config = load_text("[memory]\nutilization = 1\nkv_cache_bytes = 1048576\n").unwrap();
        assert_eq!(config.memory.utilization, 1.0);
        assert_eq!(config.memory.kv_cache_bytes, Some(1_048_576));
    }

    #[test]
    fn memory_section_rejects_out_of_range_values() {
        for contents in [
            "[memory]\nutilization = 0\n",
            "[memory]\nutilization = 1.5\n",
            "[memory]\nkv_cache_bytes = 0\n",
            "[memory]\nkv_cache_bytes = \"4 TiB\"\n",
        ] {
            let error = load_text(contents).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{contents}");
        }
    }

    #[test]
    fn explicit_max_total_tokens_is_tracked() {
        let config = load_text("[limits]\nmax_total_tokens_per_request = 8192\n").unwrap();
        assert_eq!(config.limits.max_total_tokens_per_request, 8192);
        assert!(config.memory.max_total_tokens_explicit);
    }

    #[test]
    fn byte_sizes_accept_binary_and_decimal_units() {
        assert_eq!(parse_byte_size("1073741824").unwrap(), 1 << 30);
        assert_eq!(parse_byte_size("512MiB").unwrap(), 512 << 20);
        assert_eq!(parse_byte_size("1.5GiB").unwrap(), 3 << 29);
        assert_eq!(parse_byte_size("2GB").unwrap(), 2_000_000_000);
        assert_eq!(parse_byte_size(" 8 gib ").unwrap(), 8 << 30);
        assert!(parse_byte_size("0").is_err());
        assert!(parse_byte_size("GiB").is_err());
        assert!(parse_byte_size("3 parsecs").is_err());
    }

    #[test]
    fn load_returns_defaults_when_file_is_missing() {
        let config = RuntimeConfig::load("/path/that/does/not/exist.toml").unwrap();
        assert_eq!(config, RuntimeConfig::default());
    }

    #[test]
    fn load_overrides_known_values() {
        let temp_path = std::env::temp_dir().join(format!(
            "mlx-runtime-config-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        fs::write(
            &temp_path,
            r#"
            [server]
            host = "0.0.0.0"
            port = 9000

            [worker]
            python = "/usr/bin/python3"
            module = "mlx_worker.main"
            backend = "native-mlx"
            model = "test-model"
            vlm_model = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
            ipc_path = "/tmp/test.sock"

            [generation]
            temperature = 0.1
            top_p = 0.8
            max_tokens = 42

            [limits]
            max_pending_requests = 8
            max_active_requests = 4
            max_prompt_tokens = 1024
            max_completion_tokens = 256
            max_total_tokens_per_request = 2048
            request_timeout_seconds = 10
            max_vlm_images = 5

            [telemetry]
            enable_prometheus = false
            metrics_path = "/custom-metrics"
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load(&temp_path).unwrap();
        let _ = fs::remove_file(&temp_path);

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.worker.python, "/usr/bin/python3");
        assert_eq!(config.worker.backend, BackendKind::NativeMlx);
        assert_eq!(config.worker.model, "test-model");
        assert_eq!(
            config.worker.vlm_model,
            Some("mlx-community/Qwen3-VL-2B-Instruct-4bit".to_string())
        );
        assert_eq!(config.worker.ipc_path, "/tmp/test.sock");
        assert_eq!(config.generation.temperature, 0.1);
        assert_eq!(config.generation.top_p, 0.8);
        assert_eq!(config.generation.max_tokens, 42);
        assert_eq!(config.limits.max_pending_requests, 8);
        assert_eq!(config.limits.max_active_requests, 4);
        assert_eq!(config.limits.max_prompt_tokens, 1024);
        assert_eq!(config.limits.max_completion_tokens, 256);
        assert_eq!(config.limits.max_total_tokens_per_request, 2048);
        assert_eq!(config.limits.request_timeout_seconds, 10);
        assert_eq!(config.limits.max_vlm_images, 5);
        assert!(!config.telemetry.enable_prometheus);
        assert_eq!(config.telemetry.metrics_path, "/custom-metrics");
    }

    #[test]
    fn startup_log_includes_loaded_backend_and_paths() {
        let mut config = RuntimeConfig::default();
        config.worker.backend = BackendKind::NativeMlx;
        config.worker.model = "native-model".to_string();
        config.worker.vlm_model = Some("vlm-model".to_string());
        config.worker.ipc_path = "/tmp/native.sock".to_string();

        let rendered = config.startup_log("/tmp/runtime.toml");

        assert!(rendered.contains("runtime_config path=/tmp/runtime.toml"));
        assert!(rendered.contains("backend=native-mlx"));
        assert!(rendered.contains("model=native-model"));
        assert!(rendered.contains("vlm_model=vlm-model"));
        assert!(rendered.contains("ipc_path=/tmp/native.sock"));
    }

    #[test]
    fn load_preserves_hashes_and_escaped_characters_inside_strings() {
        let temp_path = std::env::temp_dir().join(format!(
            "mlx-runtime-config-escaped-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(
            &temp_path,
            "[worker]\nmodel = \"model#one\\\"quoted\" # comment\nipc_path = \"/tmp/back\\\\slash.sock\"\n",
        )
        .unwrap();

        let config = RuntimeConfig::load(&temp_path).unwrap();
        fs::remove_file(temp_path).unwrap();

        assert_eq!(config.worker.model, "model#one\"quoted");
        assert_eq!(config.worker.ipc_path, "/tmp/back\\slash.sock");
    }
}
