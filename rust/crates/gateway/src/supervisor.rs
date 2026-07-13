use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::ipc::WorkerClient;
use crate::telemetry::MetricsRegistry;
use mlx_runtime_protocol::{
    decode_worker_message, ChatCompletionRequest, ChatMessage, MessageContent, MessageRole,
    ModelError, ModelState, ModelStatus, WorkerError, WorkerMessage, WorkerReady,
};
use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Shared runtime state managed by the supervisor.
#[derive(Clone)]
pub struct RuntimeState {
    /// When the gateway process started.
    pub started_at: Instant,
    /// Process id for the gateway.
    pub pid: u32,
    /// Current text model lifecycle snapshot.
    pub model_status: Arc<RwLock<ModelStatus>>,
    /// The active worker client, if the bootstrap handshake completed.
    pub worker_client: Arc<Mutex<Option<Arc<WorkerClient>>>>,
    /// Shared telemetry registry.
    pub metrics: Arc<MetricsRegistry>,
    /// VLM model name if configured (Phase 8).
    pub vlm_model_name: Arc<RwLock<Option<String>>>,
    /// VLM model lifecycle status (Phase 8).
    pub vlm_status: Arc<RwLock<Option<ModelStatus>>>,
}

impl RuntimeState {
    /// Creates a new runtime snapshot for the configured model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            started_at: Instant::now(),
            pid: std::process::id(),
            model_status: Arc::new(RwLock::new(ModelStatus::new(model))),
            worker_client: Arc::new(Mutex::new(None)),
            metrics: {
                let metrics = Arc::new(MetricsRegistry::new());
                metrics.set_worker_memory_bytes(0);
                metrics.set_kv_cache_bytes(0);
                metrics
            },
            vlm_model_name: Arc::new(RwLock::new(None)),
            vlm_status: Arc::new(RwLock::new(None)),
        }
    }

    /// Creates a runtime state with a shared metrics registry (for tests).
    #[cfg(test)]
    pub fn new_with_metrics(model: impl Into<String>, metrics: Arc<MetricsRegistry>) -> Self {
        Self {
            started_at: Instant::now(),
            pid: std::process::id(),
            model_status: Arc::new(RwLock::new(ModelStatus::new(model))),
            worker_client: Arc::new(Mutex::new(None)),
            metrics,
            vlm_model_name: Arc::new(RwLock::new(None)),
            vlm_status: Arc::new(RwLock::new(None)),
        }
    }

    /// Returns the latest model status snapshot.
    pub fn snapshot(&self) -> Result<ModelStatus, GatewayError> {
        self.model_status
            .read()
            .map(|guard| guard.clone())
            .map_err(|_| GatewayError::Protocol("model status lock poisoned".to_string()))
    }

    /// Replaces the current model status snapshot.
    pub fn set_status(&self, status: ModelStatus) -> Result<(), GatewayError> {
        let mut guard = self
            .model_status
            .write()
            .map_err(|_| GatewayError::Protocol("model status lock poisoned".to_string()))?;
        let previous_state = guard.state;
        let new_state = status.state;
        if previous_state != new_state {
            eprintln!(
                "model_state_transition model={} from={:?} to={:?}",
                status.model, previous_state, new_state
            );
        }
        self.metrics.set_worker_up(status.ready);
        *guard = status;
        Ok(())
    }

    /// Updates only the lifecycle state.
    pub fn set_state(&self, state: ModelState) -> Result<(), GatewayError> {
        let mut guard = self
            .model_status
            .write()
            .map_err(|_| GatewayError::Protocol("model status lock poisoned".to_string()))?;
        let previous_state = guard.state;
        guard.set_state(state);
        if previous_state != state {
            eprintln!(
                "model_state_transition model={} from={:?} to={:?}",
                guard.model, previous_state, state
            );
        }
        self.metrics.set_worker_up(guard.ready);
        Ok(())
    }

    /// Initializes VLM model lifecycle tracking.
    pub fn set_vlm_model(&self, vlm_model: String) {
        if let Ok(mut name_guard) = self.vlm_model_name.write() {
            *name_guard = Some(vlm_model.clone());
        }
        let status = ModelStatus::new(vlm_model);
        if let Ok(mut guard) = self.vlm_status.write() {
            *guard = Some(status);
        }
    }

    /// Updates only VLM lifecycle state.
    pub fn set_vlm_state(&self, state: ModelState) -> Result<(), GatewayError> {
        let mut guard = self
            .vlm_status
            .write()
            .map_err(|_| GatewayError::Protocol("vlm status lock poisoned".to_string()))?;
        if let Some(ref mut status) = *guard {
            let previous_state = status.state;
            status.set_state(state);
            if previous_state != state {
                eprintln!(
                    "vlm_model_state_transition model={} from={:?} to={:?}",
                    status.model, previous_state, state
                );
            }
        }
        Ok(())
    }

    /// Returns the VLM model lifecycle snapshot.
    pub fn snapshot_vlm(&self) -> Result<ModelStatus, GatewayError> {
        let guard = self
            .vlm_status
            .read()
            .map_err(|_| GatewayError::Protocol("vlm status lock poisoned".to_string()))?;
        Ok(guard.clone().unwrap_or_else(|| {
            // VLM not yet initialized: return NotLoaded fallback.
            let name = self
                .vlm_model_name
                .read()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_else(|| "vlm".to_string());
            ModelStatus::new(name)
        }))
    }

    /// Marks the VLM model as ready after warmup.
    pub fn mark_vlm_ready(&self, warmup_latency_ms: u64) -> Result<(), GatewayError> {
        let mut guard = self
            .vlm_status
            .write()
            .map_err(|_| GatewayError::Protocol("vlm status lock poisoned".to_string()))?;
        if let Some(ref mut status) = *guard {
            status.mark_ready(None, None, warmup_latency_ms);
        }
        Ok(())
    }

    /// Marks the VLM model as failed with a stable error code.
    pub fn mark_vlm_failed(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), GatewayError> {
        let mut guard = self
            .vlm_status
            .write()
            .map_err(|_| GatewayError::Protocol("vlm status lock poisoned".to_string()))?;
        if let Some(ref mut status) = *guard {
            let previous_state = status.state;
            status.mark_failed(code, message);
            if previous_state != status.state {
                eprintln!(
                    "vlm_model_state_transition model={} from={:?} to={:?}",
                    status.model, previous_state, status.state
                );
            }
            self.metrics.increment_vlm_load_errors_total();
        }
        Ok(())
    }

    /// Marks the model as failed and updates the bootstrap snapshot.
    pub fn mark_failed(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), GatewayError> {
        let mut guard = self
            .model_status
            .write()
            .map_err(|_| GatewayError::Protocol("model status lock poisoned".to_string()))?;
        let previous_state = guard.state;
        guard.mark_failed(code, message);
        if previous_state != guard.state {
            eprintln!(
                "model_state_transition model={} from={:?} to={:?}",
                guard.model, previous_state, guard.state
            );
        }
        self.metrics.set_worker_up(false);
        Ok(())
    }

    /// Marks the model as failed with a structured worker error.
    pub fn mark_failed_with_error(&self, error: ModelError) -> Result<(), GatewayError> {
        let mut guard = self
            .model_status
            .write()
            .map_err(|_| GatewayError::Protocol("model status lock poisoned".to_string()))?;
        let previous_state = guard.state;
        guard.mark_failed_with_error(error);
        if previous_state != guard.state {
            eprintln!(
                "model_state_transition model={} from={:?} to={:?}",
                guard.model, previous_state, guard.state
            );
        }
        self.metrics.set_worker_up(false);
        Ok(())
    }
}

/// Background worker supervision for the current runtime slice.
pub struct Supervisor;

impl Supervisor {
    /// Starts the worker bootstrap flow on a background thread.
    pub fn start(config: RuntimeConfig) -> Result<RuntimeState, GatewayError> {
        let runtime = RuntimeState::new(config.worker.model.clone());

        // Initialize VLM model lifecycle if configured.
        if let Some(ref vlm) = config.worker.vlm_model {
            runtime.set_vlm_model(vlm.clone());
        }

        let bootstrap_runtime = runtime.clone();

        thread::spawn(move || {
            if let Err(err) = bootstrap_worker(&config, bootstrap_runtime.clone()) {
                eprintln!("worker bootstrap failed: {err}");
                if let Ok(mut guard) = bootstrap_runtime.worker_client.lock() {
                    *guard = None;
                }
                let _ = bootstrap_runtime.mark_failed("WORKER_BOOTSTRAP_FAILED", err.to_string());
            }
        });

        Ok(runtime)
    }
}

fn bootstrap_worker(config: &RuntimeConfig, runtime: RuntimeState) -> Result<(), GatewayError> {
    let socket_path = &config.worker.ipc_path;
    let _ = fs::remove_file(socket_path);

    let listener = UnixListener::bind(socket_path)?;
    listener.set_nonblocking(true)?;
    let mut child = spawn_worker(config)?;
    let deadline = Instant::now() + Duration::from_secs(1800);
    let _ = runtime.set_state(ModelState::LoadingWeights);

    let connection = loop {
        match listener.accept() {
            Ok((stream, _addr)) => break stream,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return fail_child_message(
                            &mut child,
                            format!("worker exited before ready: {status}"),
                            &runtime,
                        );
                    }
                    Ok(None) => {}
                    Err(err) => return fail_child_message(&mut child, err.to_string(), &runtime),
                }
                if Instant::now() >= deadline {
                    return fail_child_message(
                        &mut child,
                        "worker did not become ready in time".to_string(),
                        &runtime,
                    );
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return fail_child_message(&mut child, err.to_string(), &runtime),
        }
    };

    if let Err(err) = connection.set_nonblocking(false) {
        return fail_child_message(&mut child, err.to_string(), &runtime);
    }

    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or_else(|| Duration::from_secs(0));

    let reader_stream = connection.try_clone()?;
    if let Err(err) = reader_stream.set_read_timeout(Some(remaining)) {
        return fail_child_message(&mut child, err.to_string(), &runtime);
    }
    let mut reader = BufReader::new(reader_stream);
    let mut line = String::new();
    loop {
        let bytes = match reader.read_line(&mut line) {
            Ok(bytes) => bytes,
            Err(err)
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
                return fail_child_message(
                    &mut child,
                    "worker did not become ready in time".to_string(),
                    &runtime,
                );
            }
            Err(err) => return fail_child_message(&mut child, err.to_string(), &runtime),
        };
        if bytes == 0 {
            return fail_child_message(
                &mut child,
                "worker closed the bootstrap socket".to_string(),
                &runtime,
            );
        }

        match decode_worker_message(&line) {
            Some(WorkerMessage::Status(status)) => {
                // Only update the text model lifecycle from this STATUS.
                // VLM loads lazily on first VLM request (Phase 8) and must NOT
                // be marked ready from the generic text bootstrap STATUS.
                let _ = runtime.set_status(*status);
            }
            Some(WorkerMessage::Ready(WorkerReady)) => {
                let client = Arc::new(WorkerClient::new(
                    connection.try_clone()?,
                    runtime.metrics.clone(),
                )?);
                let mut guard = runtime.worker_client.lock().map_err(|_| {
                    GatewayError::Protocol("worker client lock poisoned".to_string())
                })?;
                *guard = Some(client.clone());
                if let Some(vlm_model) = config.worker.vlm_model.clone() {
                    let warmup_runtime = runtime.clone();
                    thread::spawn(move || warmup_vlm_model(warmup_runtime, client, vlm_model));
                }
                let mut status = runtime.snapshot()?;
                if !status.ready {
                    status.mark_ready(None, None, 1);
                    let _ = runtime.set_status(status);
                }
                break;
            }
            Some(WorkerMessage::Error(error)) => {
                return fail_child(&mut child, error, &runtime);
            }
            None => {
                return fail_child_message(
                    &mut child,
                    format!("unrecognized bootstrap message: {}", line.trim()),
                    &runtime,
                );
            }
        }
        line.clear();
    }

    thread::spawn(move || {
        let wait_result = child.wait();
        if let Ok(mut guard) = runtime.worker_client.lock() {
            *guard = None;
        }
        let _ = runtime.mark_failed("WORKER_EXITED", "worker process exited");
        let _ = runtime.mark_vlm_failed("WORKER_EXITED", "worker process exited");
        if let Ok(status) = wait_result {
            eprintln!("worker exited: {status}");
        }
    });

    Ok(())
}

fn warmup_vlm_model(runtime: RuntimeState, client: Arc<WorkerClient>, vlm_model: String) {
    let started = Instant::now();
    let _ = runtime.set_vlm_state(ModelState::WarmingUp);
    let request = ChatCompletionRequest {
        request_id: "vlm-warmup".to_string(),
        model: vlm_model,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("warmup".to_string()),
        }],
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0,
        max_prompt_tokens: 64,
        max_completion_tokens: 64,
        max_total_tokens_per_request: 128,
        stop: vec![],
        stream: false,
    };

    match client.complete_chat(request) {
        Ok(_) => {
            let latency_ms = started.elapsed().as_millis() as u64;
            let _ = runtime.mark_vlm_ready(latency_ms.max(1));
        }
        Err(err) => {
            let _ = runtime.mark_vlm_failed("VLM_WARMUP_FAILED", err.to_string());
        }
    }
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn fail_child(
    child: &mut Child,
    error: WorkerError,
    runtime: &RuntimeState,
) -> Result<(), GatewayError> {
    let message = error.message.clone();
    terminate_child(child);
    if let Some(model_error) = error.error {
        let _ = runtime.mark_failed_with_error(model_error);
    } else {
        let _ = runtime.mark_failed("WORKER_BOOTSTRAP_FAILED", message.clone());
    }
    let _ = runtime.mark_vlm_failed("WORKER_BOOTSTRAP_FAILED", message.clone());
    Err(GatewayError::WorkerStartup(message))
}

fn fail_child_message(
    child: &mut Child,
    message: String,
    runtime: &RuntimeState,
) -> Result<(), GatewayError> {
    fail_child(
        child,
        WorkerError {
            message,
            error: None,
        },
        runtime,
    )
}

fn worker_env(config: &RuntimeConfig) -> Vec<(&'static str, String)> {
    let mut values = vec![
        ("MLX_RUNTIME_SOCKET", config.worker.ipc_path.clone()),
        (
            "MLX_RUNTIME_BACKEND",
            config.worker.backend.as_str().to_string(),
        ),
        ("MLX_RUNTIME_MODEL", config.worker.model.clone()),
        (
            "MLX_RUNTIME_VLM_MODEL",
            config.worker.vlm_model.clone().unwrap_or_default(),
        ),
        (
            "MLX_RUNTIME_MAX_VLM_IMAGES",
            config.limits.max_vlm_images.to_string(),
        ),
        ("PYTHONPATH", "python".to_string()),
    ];
    for key in [
        "MLX_RUNTIME_NATIVE_EXECUTION_MODE",
        "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE",
        "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR",
        "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID",
        "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD",
        "MLX_RUNTIME_NATIVE_METAL_CAPTURE",
        "MTL_CAPTURE_ENABLED",
    ] {
        if let Ok(value) = std::env::var(key) {
            values.push((key, value));
        }
    }
    values
}

fn spawn_worker(config: &RuntimeConfig) -> Result<Child, GatewayError> {
    let mut command = Command::new(&config.worker.python);
    command.arg("-m").arg(&config.worker.module);
    for (key, value) in worker_env(config) {
        command.env(key, value);
    }
    command
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let child = command.spawn()?;
    Ok(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackendKind;

    #[test]
    fn worker_env_defaults_to_v1_backend() {
        let config = RuntimeConfig::default();
        let env = worker_env(&config);

        assert!(
            env.iter()
                .any(|(key, value)| *key == "MLX_RUNTIME_BACKEND"
                    && value == BackendKind::V1.as_str())
        );
    }

    #[test]
    fn worker_env_uses_explicit_native_backend() {
        let mut config = RuntimeConfig::default();
        config.worker.backend = BackendKind::NativeMlx;

        let env = worker_env(&config);

        assert!(env.iter().any(|(key, value)| {
            *key == "MLX_RUNTIME_BACKEND" && value == BackendKind::NativeMlx.as_str()
        }));
    }

    #[test]
    fn worker_env_forwards_pipeline_profile_configuration() {
        let config = RuntimeConfig::default();
        unsafe { std::env::set_var("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", "1") };

        let env = worker_env(&config);

        unsafe { std::env::remove_var("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE") };
        assert!(env
            .iter()
            .any(|(key, value)| { *key == "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE" && value == "1" }));
    }

    #[test]
    fn worker_env_forwards_native_execution_mode() {
        let config = RuntimeConfig::default();
        unsafe { std::env::set_var("MLX_RUNTIME_NATIVE_EXECUTION_MODE", "overlap") };

        let env = worker_env(&config);

        unsafe { std::env::remove_var("MLX_RUNTIME_NATIVE_EXECUTION_MODE") };
        assert!(env.iter().any(|(key, value)| {
            *key == "MLX_RUNTIME_NATIVE_EXECUTION_MODE" && value == "overlap"
        }));
    }
}
