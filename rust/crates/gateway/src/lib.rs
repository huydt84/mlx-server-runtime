//! Reusable gateway and public `mlx-air` command-line interfaces.

pub mod cli;
mod config;
mod errors;
mod http;
mod ipc;
mod openai;
mod supervisor;
mod telemetry;

pub use config::{
    BackendKind, GenerationConfig, RequestLimits, RuntimeConfig, ServerConfig, TelemetryConfig,
    WorkerConfig,
};
pub use errors::GatewayError;

/// Starts the worker supervisor and serves HTTP requests until termination.
///
/// # Errors
///
/// Returns a [`GatewayError`] when worker startup, IPC, or HTTP serving fails.
pub fn run(config: RuntimeConfig) -> Result<(), GatewayError> {
    let runtime = supervisor::Supervisor::start(config.clone())?;
    http::serve(config, runtime)
}
