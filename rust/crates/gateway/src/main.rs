mod config;
mod errors;
mod http;
mod ipc;
mod openai;
mod supervisor;
mod telemetry;

use crate::config::RuntimeConfig;
use crate::errors::GatewayError;

fn main() -> Result<(), GatewayError> {
    let config_path =
        std::env::var("MLX_RUNTIME_CONFIG").unwrap_or_else(|_| "config/runtime.toml".to_string());
    let config = RuntimeConfig::load(config_path)?;
    let runtime = supervisor::Supervisor::start(config.clone())?;
    http::serve(config, runtime)?;

    Ok(())
}
