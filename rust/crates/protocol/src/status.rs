use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Lifecycle states for a model instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    /// No model has been loaded yet.
    NotLoaded,
    /// The model artifacts are being downloaded.
    Downloading,
    /// The model artifacts are being verified.
    Verifying,
    /// Model weights are being loaded.
    LoadingWeights,
    /// The runtime is being initialized.
    InitializingRuntime,
    /// A warmup inference is running.
    WarmingUp,
    /// The model is ready to serve traffic.
    Ready,
    /// The model is serving but not fully healthy.
    Degraded,
    /// The model failed to load or serve.
    Failed,
    /// The model is shutting down or reloading.
    Unloading,
}

/// Error metadata for a model lifecycle transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelError {
    /// Stable error code.
    pub code: String,
    /// Human-readable error description.
    pub message: String,
    /// Unix timestamp for the error event.
    pub at: u64,
}

/// Optional progress information during model loading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLoadProgress {
    /// Downloaded bytes so far.
    pub downloaded_bytes: Option<u64>,
    /// Total bytes expected.
    pub total_bytes: Option<u64>,
    /// Loaded tensors so far.
    pub loaded_tensors: Option<u32>,
    /// Total tensors expected.
    pub total_tensors: Option<u32>,
    /// Current loading phase, if known.
    pub current_phase: Option<String>,
}

/// Detailed model status used by the gateway and bootstrap protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    /// Model identifier.
    pub model: String,
    /// Revision or source identifier for the model.
    pub revision: Option<String>,
    /// Current lifecycle state.
    pub state: ModelState,
    /// Whether the model can currently serve inference.
    pub ready: bool,
    /// Whether the model is usable for traffic.
    pub servable: bool,
    /// Loading progress, if available.
    pub progress: Option<ModelLoadProgress>,
    /// Device used for inference, if known.
    pub device: Option<String>,
    /// Numeric dtype used for inference, if known.
    pub dtype: Option<String>,
    /// Unix timestamp when the model became ready.
    pub loaded_at: Option<u64>,
    /// Unix timestamp when loading first began.
    pub started_loading_at: Option<u64>,
    /// Unix timestamp of the latest state transition.
    pub last_transition_at: u64,
    /// Last recorded error, if any.
    pub last_error: Option<ModelError>,
    /// Whether warmup has passed.
    pub warmup_passed: bool,
    /// Unix timestamp of the last warmup success.
    pub last_warmup_at: Option<u64>,
    /// Warmup latency in milliseconds.
    pub last_warmup_latency_ms: Option<u64>,
}

/// Summary data for `/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model identifier.
    pub name: String,
    /// Current lifecycle state.
    pub state: ModelState,
    /// Whether the model is ready.
    pub ready: bool,
    /// Revision or source identifier.
    pub revision: Option<String>,
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl ModelStatus {
    /// Creates a not-loaded status for the configured model.
    pub fn new(model: impl Into<String>) -> Self {
        let now = now_unix_seconds();
        let model = model.into();
        Self {
            model: model.clone(),
            revision: Some(model),
            state: ModelState::NotLoaded,
            ready: false,
            servable: false,
            progress: None,
            device: None,
            dtype: None,
            loaded_at: None,
            started_loading_at: None,
            last_transition_at: now,
            last_error: None,
            warmup_passed: false,
            last_warmup_at: None,
            last_warmup_latency_ms: None,
        }
    }

    /// Returns the current status as a summary item.
    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            name: self.model.clone(),
            state: self.state,
            ready: self.ready,
            revision: self.revision.clone(),
        }
    }

    /// Updates the lifecycle state and derived readiness flags.
    pub fn set_state(&mut self, state: ModelState) {
        let now = now_unix_seconds();
        if self.started_loading_at.is_none() && state != ModelState::NotLoaded {
            self.started_loading_at = Some(now);
        }
        self.state = state;
        self.last_transition_at = now;
        self.ready = matches!(state, ModelState::Ready);
        self.servable = matches!(state, ModelState::Ready | ModelState::Degraded);
        if matches!(state, ModelState::Ready) {
            self.loaded_at = Some(now);
            self.warmup_passed = true;
            self.last_warmup_at = Some(now);
            self.last_error = None;
            self.progress = None;
        }
    }

    /// Records the current loading progress, if any.
    pub fn set_progress(&mut self, progress: Option<ModelLoadProgress>) {
        self.progress = progress;
        self.last_transition_at = now_unix_seconds();
    }

    /// Records the model device and dtype metadata.
    pub fn set_runtime_metadata(&mut self, device: Option<String>, dtype: Option<String>) {
        self.device = device;
        self.dtype = dtype;
        self.last_transition_at = now_unix_seconds();
    }

    /// Marks the model as failed with a stable error code.
    pub fn mark_failed(&mut self, code: impl Into<String>, message: impl Into<String>) {
        let now = now_unix_seconds();
        self.state = ModelState::Failed;
        self.ready = false;
        self.servable = false;
        self.last_transition_at = now;
        self.last_error = Some(ModelError {
            code: code.into(),
            message: message.into(),
            at: now,
        });
    }

    /// Marks a warmup success and transitions to ready.
    pub fn mark_ready(
        &mut self,
        device: Option<String>,
        dtype: Option<String>,
        warmup_latency_ms: u64,
    ) {
        let now = now_unix_seconds();
        self.set_state(ModelState::Ready);
        self.device = device;
        self.dtype = dtype;
        self.warmup_passed = true;
        self.last_warmup_at = Some(now);
        self.last_warmup_latency_ms = Some(warmup_latency_ms);
        self.last_transition_at = now;
        self.last_error = None;
        self.progress = None;
    }
}
