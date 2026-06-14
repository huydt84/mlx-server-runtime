mod config;
mod errors;
mod http;
mod ipc;
mod supervisor;
mod telemetry;

use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use std::sync::{atomic::AtomicBool, Arc};

fn main() -> Result<(), GatewayError> {
    let config = RuntimeConfig::load("config/runtime.toml")?;
    let healthy = Arc::new(AtomicBool::new(false));

    let _supervisor = supervisor::Supervisor::start(config.clone(), healthy.clone())?;
    http::serve(&config.server, healthy)?;

    Ok(())
}
