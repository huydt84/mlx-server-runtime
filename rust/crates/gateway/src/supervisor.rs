use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::ipc::WorkerClient;
use crate::telemetry::MetricsRegistry;
use mlx_runtime_protocol::{
    decode_worker_message, ModelState, ModelStatus, WorkerError, WorkerMessage, WorkerReady,
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
    /// Current model lifecycle snapshot.
    pub model_status: Arc<RwLock<ModelStatus>>,
    /// The active worker client, if the bootstrap handshake completed.
    pub worker_client: Arc<Mutex<Option<Arc<WorkerClient>>>>,
    /// Shared telemetry registry.
    pub metrics: Arc<MetricsRegistry>,
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
}

/// Background worker supervision for the current runtime slice.
pub struct Supervisor;

impl Supervisor {
    /// Starts the worker bootstrap flow on a background thread.
    pub fn start(config: RuntimeConfig) -> Result<RuntimeState, GatewayError> {
        let runtime = RuntimeState::new(config.worker.model.clone());
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
                        return fail_child(
                            &mut child,
                            format!("worker exited before ready: {status}"),
                            &runtime,
                        );
                    }
                    Ok(None) => {}
                    Err(err) => return fail_child(&mut child, err.to_string(), &runtime),
                }
                if Instant::now() >= deadline {
                    return fail_child(
                        &mut child,
                        "worker did not become ready in time".to_string(),
                        &runtime,
                    );
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return fail_child(&mut child, err.to_string(), &runtime),
        }
    };

    if let Err(err) = connection.set_nonblocking(false) {
        return fail_child(&mut child, err.to_string(), &runtime);
    }

    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or_else(|| Duration::from_secs(0));

    let reader_stream = connection.try_clone()?;
    if let Err(err) = reader_stream.set_read_timeout(Some(remaining)) {
        return fail_child(&mut child, err.to_string(), &runtime);
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
                return fail_child(
                    &mut child,
                    "worker did not become ready in time".to_string(),
                    &runtime,
                );
            }
            Err(err) => return fail_child(&mut child, err.to_string(), &runtime),
        };
        if bytes == 0 {
            return fail_child(
                &mut child,
                "worker closed the bootstrap socket".to_string(),
                &runtime,
            );
        }

        match decode_worker_message(&line) {
            Some(WorkerMessage::Status(status)) => {
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
                *guard = Some(client);
                let mut status = runtime.snapshot()?;
                if !status.ready {
                    status.mark_ready(None, None, 0);
                    let _ = runtime.set_status(status);
                }
                break;
            }
            Some(WorkerMessage::Error(WorkerError { message })) => {
                return fail_child(&mut child, message, &runtime);
            }
            None => {
                return fail_child(
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
        if let Ok(status) = wait_result {
            eprintln!("worker exited: {status}");
        }
    });

    Ok(())
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn fail_child(
    child: &mut Child,
    message: String,
    runtime: &RuntimeState,
) -> Result<(), GatewayError> {
    terminate_child(child);
    let _ = runtime.mark_failed("WORKER_BOOTSTRAP_FAILED", message.clone());
    Err(GatewayError::WorkerStartup(message))
}

fn spawn_worker(config: &RuntimeConfig) -> Result<Child, GatewayError> {
    let mut command = Command::new(&config.worker.python);
    command
        .arg("-m")
        .arg(&config.worker.module)
        .env("MLX_RUNTIME_SOCKET", &config.worker.ipc_path)
        .env("MLX_RUNTIME_MODEL", &config.worker.model)
        .env("PYTHONPATH", "python")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let child = command.spawn()?;
    Ok(child)
}
