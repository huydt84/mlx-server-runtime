use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::ipc::WorkerClient;
use mlx_runtime_protocol::{decode_worker_message, WorkerError, WorkerMessage, WorkerReady};
use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

/// Shared runtime state managed by the supervisor.
pub struct RuntimeState {
    /// Whether the worker is ready for requests.
    pub healthy: Arc<AtomicBool>,
    /// The active worker client, if the bootstrap handshake completed.
    pub worker_client: Arc<Mutex<Option<Arc<WorkerClient>>>>,
}

/// Background worker supervision for the current runtime slice.
pub struct Supervisor;

impl Supervisor {
    /// Starts the worker bootstrap flow on a background thread.
    pub fn start(config: RuntimeConfig) -> Result<RuntimeState, GatewayError> {
        let healthy = Arc::new(AtomicBool::new(false));
        let worker_client = Arc::new(Mutex::new(None));
        let runtime = RuntimeState {
            healthy: healthy.clone(),
            worker_client: worker_client.clone(),
        };

        thread::spawn(move || {
            if let Err(err) = bootstrap_worker(&config, healthy.clone(), worker_client.clone()) {
                eprintln!("worker bootstrap failed: {err}");
                healthy.store(false, Ordering::SeqCst);
                if let Ok(mut guard) = worker_client.lock() {
                    *guard = None;
                }
            }
        });

        Ok(runtime)
    }
}

fn bootstrap_worker(
    config: &RuntimeConfig,
    healthy: Arc<AtomicBool>,
    worker_client: Arc<Mutex<Option<Arc<WorkerClient>>>>,
) -> Result<(), GatewayError> {
    let socket_path = &config.worker.ipc_path;
    let _ = fs::remove_file(socket_path);

    let listener = UnixListener::bind(socket_path)?;
    listener.set_nonblocking(true)?;
    let mut child = spawn_worker(config)?;
    let deadline = Instant::now() + Duration::from_secs(1800);

    let connection = loop {
        match listener.accept() {
            Ok((stream, _addr)) => break stream,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return fail_child(
                            &mut child,
                            format!("worker exited before ready: {status}"),
                        );
                    }
                    Ok(None) => {}
                    Err(err) => return fail_child(&mut child, err.to_string()),
                }
                if Instant::now() >= deadline {
                    return fail_child(
                        &mut child,
                        "worker did not become ready in time".to_string(),
                    );
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return fail_child(&mut child, err.to_string()),
        }
    };

    if let Err(err) = connection.set_nonblocking(false) {
        return fail_child(&mut child, err.to_string());
    }

    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or_else(|| Duration::from_secs(0));

    let reader_stream = connection.try_clone()?;
    if let Err(err) = reader_stream.set_read_timeout(Some(remaining)) {
        return fail_child(&mut child, err.to_string());
    }
    let mut reader = BufReader::new(reader_stream);
    let mut line = String::new();
    let bytes = match reader.read_line(&mut line) {
        Ok(bytes) => bytes,
        Err(err)
            if err.kind() == std::io::ErrorKind::WouldBlock
                || err.kind() == std::io::ErrorKind::TimedOut =>
        {
            return fail_child(
                &mut child,
                "worker did not become ready in time".to_string(),
            );
        }
        Err(err) => return fail_child(&mut child, err.to_string()),
    };
    if bytes == 0 {
        return fail_child(&mut child, "worker closed the bootstrap socket".to_string());
    }

    match decode_worker_message(&line) {
        Some(WorkerMessage::Ready(WorkerReady)) => {
            healthy.store(true, Ordering::SeqCst);
            let client = Arc::new(WorkerClient::new(connection.try_clone()?)?);
            let mut guard = worker_client
                .lock()
                .map_err(|_| GatewayError::Protocol("worker client lock poisoned".to_string()))?;
            *guard = Some(client);
        }
        Some(WorkerMessage::Error(WorkerError { message })) => {
            return fail_child(&mut child, message);
        }
        None => {
            return fail_child(
                &mut child,
                format!("unrecognized bootstrap message: {}", line.trim()),
            );
        }
    }

    thread::spawn(move || {
        let wait_result = child.wait();
        healthy.store(false, Ordering::SeqCst);
        if let Ok(mut guard) = worker_client.lock() {
            *guard = None;
        }
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

fn fail_child(child: &mut Child, message: String) -> Result<(), GatewayError> {
    terminate_child(child);
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
