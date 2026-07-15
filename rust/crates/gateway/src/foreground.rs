use crate::http::HttpServer;
use crate::supervisor::{Supervisor, WorkerLauncher};
use crate::{GatewayError, RuntimeConfig};
use signal_hook::consts::{SIGINT, SIGTERM};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECTION_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) fn run(config: RuntimeConfig) -> Result<(), GatewayError> {
    let shutdown = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(SIGINT, shutdown.clone())?;
    signal_hook::flag::register(SIGTERM, shutdown.clone())?;
    run_with(
        config,
        shutdown,
        Duration::from_secs(1_800),
        CONNECTION_DRAIN_TIMEOUT,
        WORKER_SHUTDOWN_TIMEOUT,
        None,
    )
}

fn run_with(
    config: RuntimeConfig,
    shutdown: Arc<AtomicBool>,
    startup_timeout: Duration,
    drain_timeout: Duration,
    worker_shutdown_timeout: Duration,
    launcher: Option<Arc<dyn WorkerLauncher>>,
) -> Result<(), GatewayError> {
    if config.server.host != "127.0.0.1" {
        return Err(GatewayError::WorkerStartup(format!(
            "mlx-air serves only on 127.0.0.1; configured host is {}",
            config.server.host
        )));
    }

    let mut supervisor = match launcher {
        Some(launcher) => {
            Supervisor::start_with(config.clone(), shutdown.clone(), startup_timeout, launcher)?
        }
        None => Supervisor::start(config.clone(), shutdown.clone())?,
    };
    let server = match HttpServer::bind(config.clone(), supervisor.runtime()) {
        Ok(server) => server,
        Err(err) => {
            let _ = supervisor.shutdown(worker_shutdown_timeout);
            return Err(err);
        }
    };

    match supervisor.wait_until_ready() {
        Ok(true) => {
            eprintln!(
                "MLX Air is ready at http://{}:{}",
                config.server.host, config.server.port
            );
        }
        Ok(false) => {
            supervisor.shutdown(worker_shutdown_timeout)?;
            return Ok(());
        }
        Err(err) => {
            let _ = supervisor.shutdown(worker_shutdown_timeout);
            return Err(err);
        }
    }

    let serve_result = server.serve_until(&shutdown, drain_timeout);
    let final_status = supervisor.runtime().snapshot();
    let shutdown_result = supervisor.shutdown(worker_shutdown_timeout);
    serve_result?;
    shutdown_result?;

    let final_status = final_status?;
    if final_status.state == mlx_runtime_protocol::ModelState::Failed {
        let message = final_status.last_error.map_or_else(
            || "worker exited while serving requests".to_string(),
            |error| error.message,
        );
        return Err(GatewayError::WorkerStartup(message));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn run_with_launcher(
    config: RuntimeConfig,
    shutdown: Arc<AtomicBool>,
    startup_timeout: Duration,
    drain_timeout: Duration,
    worker_shutdown_timeout: Duration,
    launcher: Arc<dyn WorkerLauncher>,
) -> Result<(), GatewayError> {
    run_with(
        config,
        shutdown,
        startup_timeout,
        drain_timeout,
        worker_shutdown_timeout,
        Some(launcher),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::WorkerLauncher;
    use std::fs;
    use std::io::Write as _;
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::thread;

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

    struct ShortLivedLauncher;

    impl WorkerLauncher for ShortLivedLauncher {
        fn spawn(&self, _config: &RuntimeConfig) -> Result<Child, GatewayError> {
            Command::new("/bin/sleep")
                .arg("0.2")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(GatewayError::Io)
        }
    }

    fn test_config() -> (PathBuf, RuntimeConfig) {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(format!("/tmp/ma-fg-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut config = RuntimeConfig::default();
        config.server.port = 0;
        config.worker.ipc_path = root.join("worker.sock").to_string_lossy().into_owned();
        (root, config)
    }

    fn ready_worker(socket_path: PathBuf, shutdown: Arc<AtomicBool>) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut stream = loop {
                match UnixStream::connect(&socket_path) {
                    Ok(stream) => break stream,
                    Err(_) => thread::sleep(Duration::from_millis(10)),
                }
            };
            writeln!(stream, "READY").unwrap();
            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(10));
            }
        })
    }

    #[test]
    fn shutdown_during_startup_returns_normally_and_removes_socket() {
        let (root, config) = test_config();
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let trigger = shutdown.clone();
        let signal_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            trigger.store(true, Ordering::Relaxed);
        });

        run_with_launcher(
            config,
            shutdown,
            Duration::from_secs(5),
            Duration::from_millis(100),
            Duration::from_secs(1),
            Arc::new(SleepLauncher),
        )
        .unwrap();
        signal_thread.join().unwrap();

        assert!(!socket_path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shutdown_while_idle_returns_normally_and_removes_socket() {
        let (root, config) = test_config();
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let connector = ready_worker(socket_path.clone(), shutdown.clone());
        let trigger = shutdown.clone();
        let signal_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            trigger.store(true, Ordering::Relaxed);
        });

        run_with_launcher(
            config,
            shutdown,
            Duration::from_secs(1),
            Duration::from_millis(100),
            Duration::from_secs(1),
            Arc::new(SleepLauncher),
        )
        .unwrap();
        signal_thread.join().unwrap();
        connector.join().unwrap();

        assert!(!socket_path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn public_bind_address_is_rejected_before_worker_start() {
        let (root, mut config) = test_config();
        config.server.host = "0.0.0.0".to_string();

        let error = run_with_launcher(
            config,
            Arc::new(AtomicBool::new(false)),
            Duration::from_secs(1),
            Duration::from_millis(100),
            Duration::from_secs(1),
            Arc::new(SleepLauncher),
        )
        .unwrap_err();

        assert!(error.to_string().contains("only on 127.0.0.1"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn worker_exit_after_readiness_returns_an_error() {
        let (root, config) = test_config();
        let socket_path = PathBuf::from(&config.worker.ipc_path);
        let shutdown = Arc::new(AtomicBool::new(false));
        let connector = ready_worker(socket_path, shutdown.clone());

        let error = run_with_launcher(
            config,
            shutdown,
            Duration::from_secs(1),
            Duration::from_millis(100),
            Duration::from_secs(1),
            Arc::new(ShortLivedLauncher),
        )
        .unwrap_err();
        connector.join().unwrap();

        assert!(error.to_string().contains("worker process exited"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sigint_and_sigterm_set_the_shutdown_flag() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let sigint = signal_hook::flag::register(SIGINT, shutdown.clone()).unwrap();
        let sigterm = signal_hook::flag::register(SIGTERM, shutdown.clone()).unwrap();

        signal_hook::low_level::raise(SIGINT).unwrap();
        for _ in 0..100 {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(shutdown.load(Ordering::Relaxed));

        shutdown.store(false, Ordering::Relaxed);
        signal_hook::low_level::raise(SIGTERM).unwrap();
        for _ in 0..100 {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(shutdown.load(Ordering::Relaxed));
        signal_hook::low_level::unregister(sigint);
        signal_hook::low_level::unregister(sigterm);
    }
}
