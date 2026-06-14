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
    let config = RuntimeConfig::load("config/runtime.toml")?;
    let runtime = supervisor::Supervisor::start(config.clone())?;
    http::serve(config, runtime)?;

    Ok(())
}
