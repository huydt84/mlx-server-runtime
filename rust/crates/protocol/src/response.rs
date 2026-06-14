/// Bootstrap success acknowledgment from the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerReady;

/// Bootstrap failure details from the worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerError {
    /// Human-readable failure reason.
    pub message: String,
}
