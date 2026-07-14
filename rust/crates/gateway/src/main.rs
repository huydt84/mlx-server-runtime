use mlx_runtime_gateway::{GatewayError, RuntimeConfig};

fn main() -> Result<(), GatewayError> {
    let config_path =
        std::env::var("MLX_RUNTIME_CONFIG").unwrap_or_else(|_| "config/runtime.toml".to_string());
    let config = RuntimeConfig::load(&config_path)?;
    eprintln!("{}", config.startup_log(&config_path));
    mlx_runtime_gateway::run(config)
}
