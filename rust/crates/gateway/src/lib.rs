//! Reusable gateway and public `mlx-air` command-line interfaces.

pub mod cli;
mod command_runner;
mod config;
mod configuration;
mod distribution;
mod doctor;
mod environment;
mod errors;
mod foreground;
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
    foreground::run(config)
}
