use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::ipc::WorkerClient;
use crate::telemetry::MetricsRegistry;
use mlx_runtime_protocol::{
    decode_worker_message, ChatCompletionRequest, ChatMessage, MessageContent, MessageRole,
    ModelError, ModelState, ModelStatus, WorkerMessage, WorkerReady,
};
use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
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
        if let Some(progress) = model_progress_log(&status) {
            eprintln!("{progress}");
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

fn model_progress_log(status: &ModelStatus) -> Option<String> {
    let progress = status.progress.as_ref()?;
    Some(format!(
        "model_load_progress model={} phase={} downloaded_bytes={} total_bytes={} loaded_tensors={} total_tensors={}",
        status.model,
        progress.current_phase.as_deref().unwrap_or("unknown"),
        progress
            .downloaded_bytes
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
        progress
            .total_bytes
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
        progress
            .loaded_tensors
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
        progress
            .total_tensors
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
    ))
}

pub(crate) trait WorkerLauncher: Send + Sync {
    fn spawn(&self, config: &RuntimeConfig) -> Result<Child, GatewayError>;
}

struct ProcessWorkerLauncher;

impl WorkerLauncher for ProcessWorkerLauncher {
    fn spawn(&self, config: &RuntimeConfig) -> Result<Child, GatewayError> {
        spawn_worker(config)
    }
}

/// Owns worker bootstrap, monitoring, termination, and socket cleanup.
pub struct Supervisor {
    runtime: RuntimeState,
    shutdown: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    worker_thread: Option<thread::JoinHandle<()>>,
    socket_path: PathBuf,
}

impl Supervisor {
    /// Starts the worker bootstrap flow while retaining lifecycle ownership.
    pub fn start(config: RuntimeConfig, shutdown: Arc<AtomicBool>) -> Result<Self, GatewayError> {
        Self::start_with(
            config,
            shutdown,
            Duration::from_secs(1_800),
            Arc::new(ProcessWorkerLauncher),
        )
    }

    pub(crate) fn start_with(
        config: RuntimeConfig,
        shutdown: Arc<AtomicBool>,
        startup_timeout: Duration,
        launcher: Arc<dyn WorkerLauncher>,
    ) -> Result<Self, GatewayError> {
        let runtime = RuntimeState::new(config.worker.model.clone());
        if let Some(ref vlm) = config.worker.vlm_model {
            runtime.set_vlm_model(vlm.clone());
        }

        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let listener = UnixListener::bind(&socket_path)?;
        listener.set_nonblocking(true)?;
        let child = match launcher.spawn(&config) {
            Ok(child) => Arc::new(Mutex::new(Some(child))),
            Err(err) => {
                let _ = fs::remove_file(&socket_path);
                return Err(err);
            }
        };
        let worker_child = child.clone();
        let worker_runtime = runtime.clone();
        let worker_shutdown = shutdown.clone();
        let worker_socket = socket_path.clone();
        let worker_thread = thread::spawn(move || {
            if let Err(err) = bootstrap_worker(
                listener,
                &config,
                worker_runtime.clone(),
                &worker_child,
                &worker_shutdown,
                startup_timeout,
            ) {
                let interrupted = worker_shutdown.load(Ordering::Relaxed);
                if !interrupted {
                    eprintln!("worker bootstrap failed: {err}");
                }
                if let Ok(mut guard) = worker_runtime.worker_client.lock() {
                    *guard = None;
                }
                let already_failed = worker_runtime
                    .snapshot()
                    .is_ok_and(|status| status.state == ModelState::Failed);
                if !interrupted && !already_failed {
                    let _ = worker_runtime.mark_failed("WORKER_BOOTSTRAP_FAILED", err.to_string());
                }
                let _ = terminate_worker(&worker_child, Duration::from_secs(1));
                let _ = fs::remove_file(&worker_socket);
                return;
            }

            monitor_worker(worker_child, worker_runtime, worker_shutdown);
        });

        Ok(Self {
            runtime,
            shutdown,
            child,
            worker_thread: Some(worker_thread),
            socket_path,
        })
    }

    /// Returns a shared runtime snapshot for the HTTP layer.
    pub fn runtime(&self) -> RuntimeState {
        self.runtime.clone()
    }

    /// Waits until the worker handshake succeeds, fails, or is interrupted.
    pub fn wait_until_ready(&self) -> Result<bool, GatewayError> {
        loop {
            let status = self.runtime.snapshot()?;
            if status.ready {
                return Ok(true);
            }
            if status.state == ModelState::Failed {
                let message = status.last_error.map_or_else(
                    || "worker startup failed".to_string(),
                    |error| error.message,
                );
                return Err(GatewayError::WorkerStartup(message));
            }
            if self.shutdown.load(Ordering::Relaxed) {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// Gracefully terminates and reaps the worker, then removes its socket.
    pub fn shutdown(&mut self, timeout: Duration) -> Result<(), GatewayError> {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Ok(mut guard) = self.runtime.worker_client.lock() {
            *guard = None;
        }
        terminate_worker(&self.child, timeout)?;
        if let Some(handle) = self.worker_thread.take() {
            handle
                .join()
                .map_err(|_| GatewayError::WorkerStartup("worker monitor panicked".to_string()))?;
        }
        match fs::remove_file(&self.socket_path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(GatewayError::Io(err)),
        }
    }

    #[cfg(test)]
    fn child_id(&self) -> Option<u32> {
        self.child
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(Child::id))
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        let _ = self.shutdown(Duration::from_secs(1));
    }
}

fn bootstrap_worker(
    listener: UnixListener,
    config: &RuntimeConfig,
    runtime: RuntimeState,
    child: &Arc<Mutex<Option<Child>>>,
    shutdown: &Arc<AtomicBool>,
    startup_timeout: Duration,
) -> Result<(), GatewayError> {
    let deadline = Instant::now() + startup_timeout;
    let _ = runtime.set_state(ModelState::LoadingWeights);

    let connection = loop {
        if shutdown.load(Ordering::Relaxed) {
            return Err(GatewayError::WorkerStartup(
                "worker startup interrupted".to_string(),
            ));
        }
        match listener.accept() {
            Ok((stream, _addr)) => break stream,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                ensure_worker_running(child)?;
                if Instant::now() >= deadline {
                    return Err(GatewayError::WorkerStartup(
                        "worker did not become ready in time".to_string(),
                    ));
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return Err(GatewayError::Io(err)),
        }
    };

    connection.set_nonblocking(false)?;
    connection.set_read_timeout(Some(Duration::from_millis(50)))?;
    let mut reader = BufReader::new(connection.try_clone()?);
    let mut line = String::new();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return Err(GatewayError::WorkerStartup(
                "worker startup interrupted".to_string(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(GatewayError::WorkerStartup(
                "worker did not become ready in time".to_string(),
            ));
        }
        ensure_worker_running(child)?;
        line.clear();
        let bytes = match reader.read_line(&mut line) {
            Ok(bytes) => bytes,
            Err(err)
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(err) => return Err(GatewayError::Io(err)),
        };
        if bytes == 0 {
            return Err(GatewayError::WorkerStartup(
                "worker closed the bootstrap socket".to_string(),
            ));
        }

        match decode_worker_message(&line) {
            Some(WorkerMessage::Status(status)) => {
                // Only update the text model lifecycle from this STATUS.
                // VLM loads lazily on first VLM request (Phase 8) and must NOT
                // be marked ready from the generic text bootstrap STATUS.
                let _ = runtime.set_status(*status);
            }
            Some(WorkerMessage::Ready(WorkerReady)) => {
                connection.set_read_timeout(None)?;
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
                let message = error.message.clone();
                if let Some(model_error) = error.error {
                    let _ = runtime.mark_failed_with_error(model_error);
                }
                return Err(GatewayError::WorkerStartup(message));
            }
            None => {
                return Err(GatewayError::WorkerStartup(format!(
                    "unrecognized bootstrap message: {}",
                    line.trim()
                )));
            }
        }
    }

    Ok(())
}

fn monitor_worker(
    child: Arc<Mutex<Option<Child>>>,
    runtime: RuntimeState,
    shutdown: Arc<AtomicBool>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        let exited = match child.lock() {
            Ok(mut guard) => match guard.as_mut().map(Child::try_wait) {
                Some(Ok(Some(status))) => {
                    eprintln!("worker exited: {status}");
                    *guard = None;
                    true
                }
                Some(Ok(None)) | None => false,
                Some(Err(err)) => {
                    eprintln!("failed to monitor worker: {err}");
                    true
                }
            },
            Err(_) => true,
        };
        if exited {
            if let Ok(mut guard) = runtime.worker_client.lock() {
                *guard = None;
            }
            let _ = runtime.mark_failed("WORKER_EXITED", "worker process exited");
            let _ = runtime.mark_vlm_failed("WORKER_EXITED", "worker process exited");
            shutdown.store(true, Ordering::Relaxed);
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn ensure_worker_running(child: &Arc<Mutex<Option<Child>>>) -> Result<(), GatewayError> {
    let mut guard = child
        .lock()
        .map_err(|_| GatewayError::WorkerStartup("worker process lock poisoned".to_string()))?;
    let Some(child) = guard.as_mut() else {
        return Err(GatewayError::WorkerStartup(
            "worker process is not running".to_string(),
        ));
    };
    match child.try_wait()? {
        Some(status) => {
            *guard = None;
            Err(GatewayError::WorkerStartup(format!(
                "worker exited before ready: {status}"
            )))
        }
        None => Ok(()),
    }
}

fn terminate_worker(
    child: &Arc<Mutex<Option<Child>>>,
    timeout: Duration,
) -> Result<(), GatewayError> {
    let pid = child
        .lock()
        .map_err(|_| GatewayError::WorkerStartup("worker process lock poisoned".to_string()))?
        .as_ref()
        .map(Child::id);
    let Some(pid) = pid else {
        return Ok(());
    };

    let pid = i32::try_from(pid)
        .map_err(|_| GatewayError::WorkerStartup("worker pid is out of range".to_string()))?;
    // SAFETY: `pid` belongs to the child retained above and SIGTERM is a valid signal.
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    let deadline = Instant::now() + timeout;
    loop {
        let mut guard = child
            .lock()
            .map_err(|_| GatewayError::WorkerStartup("worker process lock poisoned".to_string()))?;
        let Some(process) = guard.as_mut() else {
            return Ok(());
        };
        if process.try_wait()?.is_some() {
            *guard = None;
            return Ok(());
        }
        if Instant::now() >= deadline {
            process.kill()?;
            process.wait()?;
            *guard = None;
            return Ok(());
        }
        drop(guard);
        thread::sleep(Duration::from_millis(20));
    }
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
    use mlx_runtime_protocol::ModelLoadProgress;
    use std::io::Write as _;
    use std::os::unix::net::UnixStream;

    struct SleepLauncher;

    impl WorkerLauncher for SleepLauncher {
        fn spawn(&self, _config: &RuntimeConfig) -> Result<Child, GatewayError> {
            Command::new("/bin/sleep")
                .arg("60")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(GatewayError::Io)
        }
    }

    fn supervisor_config(_label: &str) -> (PathBuf, RuntimeConfig) {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(format!("/tmp/ma-sup-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut config = RuntimeConfig::default();
        config.worker.ipc_path = root.join("worker.sock").to_string_lossy().into_owned();
        (root, config)
    }

    fn send_bootstrap_message(
        socket_path: PathBuf,
        message: &'static str,
        shutdown: Arc<AtomicBool>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut stream = loop {
                match UnixStream::connect(&socket_path) {
                    Ok(stream) => break stream,
                    Err(_) => thread::sleep(Duration::from_millis(10)),
                }
            };
            writeln!(stream, "{message}").unwrap();
            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(10));
            }
        })
    }

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
    fn worker_env_does_not_force_repository_pythonpath() {
        let env = worker_env(&RuntimeConfig::default());

        assert!(!env.iter().any(|(key, _)| *key == "PYTHONPATH"));
    }

    #[test]
    fn model_progress_log_reports_available_download_progress() {
        let mut status = ModelStatus::new("test-model");
        status.set_progress(Some(ModelLoadProgress {
            downloaded_bytes: Some(64),
            total_bytes: Some(128),
            loaded_tensors: None,
            total_tensors: None,
            current_phase: Some("downloading".to_string()),
        }));

        let rendered = model_progress_log(&status).unwrap();

        assert_eq!(
            rendered,
            "model_load_progress model=test-model phase=downloading downloaded_bytes=64 total_bytes=128 loaded_tensors=unknown total_tensors=unknown"
        );
    }

    #[test]
    fn supervisor_reaches_readiness_and_reaps_worker_on_shutdown() {
        let (root, config) = supervisor_config("supervisor-ready");
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let connector = send_bootstrap_message(socket_path.clone(), "READY", shutdown.clone());
        let mut supervisor = Supervisor::start_with(
            config,
            shutdown.clone(),
            Duration::from_secs(1),
            Arc::new(SleepLauncher),
        )
        .unwrap();

        assert!(supervisor.wait_until_ready().unwrap());
        thread::sleep(Duration::from_millis(100));
        let client = supervisor
            .runtime()
            .worker_client
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        assert!(!client.is_closed());
        assert!(supervisor.child_id().is_some());
        supervisor.shutdown(Duration::from_secs(1)).unwrap();
        connector.join().unwrap();

        assert!(supervisor.child_id().is_none());
        assert!(!socket_path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervisor_reports_worker_bootstrap_failure_and_cleans_up() {
        let (root, config) = supervisor_config("supervisor-failure");
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let connector =
            send_bootstrap_message(socket_path.clone(), "ERROR\tfailed", shutdown.clone());
        let mut supervisor = Supervisor::start_with(
            config,
            shutdown.clone(),
            Duration::from_secs(1),
            Arc::new(SleepLauncher),
        )
        .unwrap();

        let error = supervisor.wait_until_ready().unwrap_err();
        supervisor.shutdown(Duration::from_secs(1)).unwrap();
        connector.join().unwrap();

        assert!(error.to_string().contains("failed"));
        assert!(!socket_path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervisor_times_out_and_reaps_worker() {
        let (root, config) = supervisor_config("supervisor-timeout");
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut supervisor = Supervisor::start_with(
            config,
            shutdown,
            Duration::from_millis(100),
            Arc::new(SleepLauncher),
        )
        .unwrap();

        let error = supervisor.wait_until_ready().unwrap_err();
        supervisor.shutdown(Duration::from_secs(1)).unwrap();

        assert!(error.to_string().contains("did not become ready"));
        assert!(supervisor.child_id().is_none());
        assert!(!socket_path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn active_socket_cannot_be_reused_by_second_supervisor() {
        let (root, config) = supervisor_config("supervisor-socket-conflict");
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut first = Supervisor::start_with(
            config.clone(),
            shutdown,
            Duration::from_secs(5),
            Arc::new(SleepLauncher),
        )
        .unwrap();

        let error = match Supervisor::start_with(
            config,
            Arc::new(AtomicBool::new(false)),
            Duration::from_secs(1),
            Arc::new(SleepLauncher),
        ) {
            Ok(mut supervisor) => {
                supervisor.shutdown(Duration::from_secs(1)).unwrap();
                panic!("second supervisor unexpectedly reused active socket")
            }
            Err(error) => error,
        };

        assert_eq!(
            match error {
                GatewayError::Io(error) => Some(error.kind()),
                _ => None,
            },
            Some(std::io::ErrorKind::AddrInUse)
        );
        first.shutdown(Duration::from_secs(1)).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shutdown_during_startup_is_normal_and_cleans_up() {
        let (root, config) = supervisor_config("supervisor-interrupt");
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut supervisor = Supervisor::start_with(
            config,
            shutdown.clone(),
            Duration::from_secs(5),
            Arc::new(SleepLauncher),
        )
        .unwrap();
        shutdown.store(true, Ordering::Relaxed);

        assert!(!supervisor.wait_until_ready().unwrap());
        supervisor.shutdown(Duration::from_secs(1)).unwrap();

        assert!(supervisor.child_id().is_none());
        assert!(!socket_path.exists());
        fs::remove_dir_all(root).unwrap();
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
