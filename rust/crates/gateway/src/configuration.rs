use crate::{BackendKind, RuntimeConfig};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CliConfigOverrides {
    pub(crate) model: Option<String>,
    pub(crate) backend: Option<BackendKind>,
    pub(crate) port: Option<u16>,
}

#[derive(Debug)]
pub(crate) struct ConfigError(String);

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

pub(crate) fn resolve_runtime_config(
    default_config: &Path,
    environment_config: Option<&Path>,
    explicit_config: Option<&Path>,
    overrides: &CliConfigOverrides,
    managed_python: &Path,
) -> Result<RuntimeConfig, ConfigError> {
    let mut config = load_required(default_config, "bundled default configuration")?;
    if let Some(path) = environment_config {
        config = overlay_required(path, config, "MLX_RUNTIME_CONFIG")?;
    }
    if let Some(path) = explicit_config {
        config = overlay_required(path, config, "--config")?;
    }

    if let Some(model) = &overrides.model {
        config.worker.model.clone_from(model);
    }
    if let Some(backend) = overrides.backend {
        config.worker.backend = backend;
    }
    if let Some(port) = overrides.port {
        config.server.port = port;
    }
    if config.worker.model.trim().is_empty() {
        return Err(ConfigError::new(
            "a model is required through --model or resolved configuration",
        ));
    }
    config.worker.python = managed_python.to_string_lossy().into_owned();
    Ok(config)
}

fn load_required(path: &Path, source: &str) -> Result<RuntimeConfig, ConfigError> {
    RuntimeConfig::load_required(path).map_err(|err| {
        ConfigError::new(format!(
            "failed to load {source} at {}: {err}",
            path.display()
        ))
    })
}

fn overlay_required(
    path: &Path,
    base: RuntimeConfig,
    source: &str,
) -> Result<RuntimeConfig, ConfigError> {
    RuntimeConfig::overlay(path, base).map_err(|err| {
        ConfigError::new(format!(
            "failed to load {source} file at {}: {err}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::tests::temp_path;
    use std::fs;

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn resolution_applies_default_environment_file_and_cli_precedence() {
        let root = temp_path("config-precedence");
        let default_path = root.join("default.toml");
        let environment_path = root.join("environment.toml");
        let explicit_path = root.join("explicit.toml");
        write(
            &default_path,
            "[server]\nport = 8000\n[worker]\nmodel = \"default-model\"\nbackend = \"v1\"\n[limits]\nmax_pending_requests = 7\n",
        );
        write(
            &environment_path,
            "[server]\nport = 8001\n[worker]\nmodel = \"environment-model\"\n[generation]\nmax_tokens = 100\n[telemetry]\nmetrics_path = \"/environment-metrics\"\n",
        );
        write(
            &explicit_path,
            "[server]\nport = 8002\n[worker]\nmodel = \"explicit-model\"\n[generation]\nmax_tokens = 200\n",
        );
        let overrides = CliConfigOverrides {
            model: Some("cli-model".to_string()),
            backend: Some(BackendKind::NativeMlx),
            port: Some(8003),
        };

        let config = resolve_runtime_config(
            &default_path,
            Some(&environment_path),
            Some(&explicit_path),
            &overrides,
            Path::new("/managed/bin/python"),
        )
        .unwrap();

        assert_eq!(config.worker.model, "cli-model");
        assert_eq!(config.worker.backend, BackendKind::NativeMlx);
        assert_eq!(config.server.port, 8003);
        assert_eq!(config.worker.python, "/managed/bin/python");
        assert_eq!(config.limits.max_pending_requests, 7);
        assert_eq!(config.generation.max_tokens, 200);
        assert_eq!(config.telemetry.metrics_path, "/environment-metrics");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn omitted_user_config_uses_bundled_defaults() {
        let root = temp_path("config-default");
        let default_path = root.join("default.toml");
        write(&default_path, "[worker]\nmodel = \"default-model\"\n");

        let config = resolve_runtime_config(
            &default_path,
            None,
            None,
            &CliConfigOverrides::default(),
            Path::new("/managed/bin/python"),
        )
        .unwrap();

        assert_eq!(config.worker.model, "default-model");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicitly_named_missing_config_is_an_error() {
        let root = temp_path("config-missing");
        let default_path = root.join("default.toml");
        let missing_path = root.join("missing.toml");
        write(&default_path, "[worker]\nmodel = \"default-model\"\n");

        let error = resolve_runtime_config(
            &default_path,
            None,
            Some(&missing_path),
            &CliConfigOverrides::default(),
            Path::new("/managed/bin/python"),
        )
        .unwrap_err();

        assert!(error.to_string().contains("--config"));
        assert!(error.to_string().contains("missing.toml"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_resolved_model_is_an_error() {
        let root = temp_path("config-empty-model");
        let default_path = root.join("default.toml");
        write(&default_path, "[worker]\nmodel = \"\"\n");

        let error = resolve_runtime_config(
            &default_path,
            None,
            None,
            &CliConfigOverrides::default(),
            Path::new("/managed/bin/python"),
        )
        .unwrap_err();

        assert!(error.to_string().contains("a model is required"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn environment_config_missing_path_is_an_error() {
        let root = temp_path("config-env-missing");
        let default_path = root.join("default.toml");
        let missing_path = root.join("environment-missing.toml");
        write(&default_path, "[worker]\nmodel = \"default-model\"\n");

        let error = resolve_runtime_config(
            &default_path,
            Some(&missing_path),
            None,
            &CliConfigOverrides::default(),
            Path::new("/managed/bin/python"),
        )
        .unwrap_err();

        assert!(error.to_string().contains("MLX_RUNTIME_CONFIG"));
        fs::remove_dir_all(root).unwrap();
    }
}
