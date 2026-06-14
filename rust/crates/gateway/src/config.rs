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
    /// Model identifier.
    pub model: String,
    /// Unix socket path used for bootstrap IPC.
    pub ipc_path: String,
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

/// Runtime configuration for the Phase 0 skeleton.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    /// HTTP server configuration.
    pub server: ServerConfig,
    /// Worker process configuration.
    pub worker: WorkerConfig,
    /// Generation defaults.
    pub generation: GenerationConfig,
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
                model: "mlx-community/Qwen2.5-7B-Instruct-4bit".to_string(),
                ipc_path: "/tmp/mlx-runtime.sock".to_string(),
            },
            generation: GenerationConfig {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 512,
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
                ("worker", "model", Value::String(model)) => config.worker.model = model,
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
                _ => {}
            }
        }

        Ok(config)
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
    Bool(()),
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
        "true" | "false" => Ok(Value::Bool(())),
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
            model = "test-model"
            ipc_path = "/tmp/test.sock"

            [generation]
            temperature = 0.1
            top_p = 0.8
            max_tokens = 42
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load(&temp_path).unwrap();
        let _ = fs::remove_file(&temp_path);

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.worker.python, "/usr/bin/python3");
        assert_eq!(config.worker.model, "test-model");
        assert_eq!(config.worker.ipc_path, "/tmp/test.sock");
        assert_eq!(config.generation.temperature, 0.1);
        assert_eq!(config.generation.top_p, 0.8);
        assert_eq!(config.generation.max_tokens, 42);
    }
}
