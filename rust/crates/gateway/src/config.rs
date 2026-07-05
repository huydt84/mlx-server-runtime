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
    /// Explicit worker backend. `v1` remains default.
    pub backend: BackendKind,
    /// Text model identifier.
    pub model: String,
    /// Optional VLM model identifier (Phase 8).
    pub vlm_model: Option<String>,
    /// Unix socket path used for bootstrap IPC.
    pub ipc_path: String,
}

/// Supported worker backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Existing `mlx_lm` runtime path.
    V1,
    /// Experimental native MLX runtime path.
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

/// Runtime configuration for the Phase 0 skeleton.
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
                backend: BackendKind::V1,
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

        let mut config = Self::default();
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
                        })?
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
        let _ = write!(
            rendered,
            "telemetry enable_prometheus={} metrics_path={}",
            self.telemetry.enable_prometheus, self.telemetry.metrics_path
        );
        rendered
    }
}

fn strip_comment(line: &str) -> &str {
    match line.split_once('#') {
        Some((left, _)) => left,
        None => line,
    }
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
        return Ok(Value::String(value.to_string()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
}
