use std::error::Error;
use std::fmt;
use std::io;

/// Errors produced by the gateway bootstrap slice.
#[derive(Debug)]
pub enum GatewayError {
    /// I/O failure.
    Io(io::Error),
    /// Worker startup failure.
    WorkerStartup(String),
    /// Request/response protocol failure.
    Protocol(String),
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GatewayError::Io(err) => write!(f, "io error: {err}"),
            GatewayError::WorkerStartup(message) => write!(f, "worker startup error: {message}"),
            GatewayError::Protocol(message) => write!(f, "protocol error: {message}"),
        }
    }
}

impl Error for GatewayError {}

impl From<io::Error> for GatewayError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
