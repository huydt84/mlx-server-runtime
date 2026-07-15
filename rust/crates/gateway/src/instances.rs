//! Per-user launchd instance registration and lifecycle management.

use crate::command_runner::{path_arg, CommandRunner, CommandSpec, OutputMode};
use crate::distribution::{effective_user_id, ApplicationPaths};
use crate::RuntimeConfig;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fmt::{self, Write as FmtWrite};
use std::fs;
use std::io::{Read as IoRead, Write as IoWrite};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{Duration, Instant};

const LABEL_PREFIX: &str = "com.mlx-air.instance";
const INSTANCE_SCHEMA_VERSION: u32 = 1;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const ATTACH_TIMEOUT: Duration = Duration::from_secs(7 * 24 * 60 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstanceErrorKind {
    Conflict,
    Launchd,
    Serving,
}

#[derive(Debug)]
pub(crate) struct InstanceError {
    kind: InstanceErrorKind,
    message: String,
}

impl InstanceError {
    pub(crate) fn new(kind: InstanceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn kind(&self) -> InstanceErrorKind {
        self.kind
    }
}

impl fmt::Display for InstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for InstanceError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InstanceMetadata {
    schema_version: u32,
    name: String,
    label: String,
    model: String,
    port: u16,
    url: String,
    pid: Option<u32>,
    gateway_executable: PathBuf,
    config_path: PathBuf,
    plist_path: PathBuf,
    socket_path: PathBuf,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StartResult {
    pub(crate) name: String,
    pub(crate) url: String,
    pub(crate) pid: u32,
}

impl StartResult {
    pub(crate) fn render(&self) -> String {
        format!("Started {} at {} (PID {})\n", self.name, self.url, self.pid)
    }
}

pub(crate) trait ReadinessProbe {
    fn is_ready(&self, port: u16) -> bool;
}

pub(crate) struct HttpReadinessProbe;

impl ReadinessProbe for HttpReadinessProbe {
    fn is_ready(&self, port: u16) -> bool {
        http_ready(port)
    }
}

pub(crate) struct InstanceManager<'a> {
    application: &'a ApplicationPaths,
    runner: &'a dyn CommandRunner,
    readiness: &'a dyn ReadinessProbe,
    startup_timeout: Duration,
    poll_interval: Duration,
    stop_timeout: Duration,
}

impl<'a> InstanceManager<'a> {
    pub(crate) fn new(
        application: &'a ApplicationPaths,
        runner: &'a dyn CommandRunner,
        readiness: &'a dyn ReadinessProbe,
    ) -> Self {
        Self {
            application,
            runner,
            readiness,
            startup_timeout: Duration::from_secs(1_800),
            poll_interval: Duration::from_millis(100),
            stop_timeout: Duration::from_secs(10),
        }
    }

    #[cfg(test)]
    fn with_timeouts(mut self, startup_timeout: Duration, stop_timeout: Duration) -> Self {
        self.startup_timeout = startup_timeout;
        self.poll_interval = Duration::ZERO;
        self.stop_timeout = stop_timeout;
        self
    }

    pub(crate) fn start(
        &self,
        name: &str,
        config: &RuntimeConfig,
        gateway_executable: &Path,
    ) -> Result<StartResult, InstanceError> {
        validate_instance_name(name)?;
        let paths = InstancePaths::new(self.application, name);
        if paths.root.exists() {
            return Err(InstanceError::new(
                InstanceErrorKind::Conflict,
                format!("managed instance '{name}' is already registered"),
            ));
        }
        self.reject_registered_port(config.server.port)?;
        ensure_port_available(config.server.port)?;

        prepare_instance_directories(self.application, &paths)?;
        create_log_file(&paths.stdout_log)?;
        create_log_file(&paths.stderr_log)?;

        let mut persisted_config = config.clone();
        persisted_config.worker.ipc_path = paths.socket.to_string_lossy().into_owned();
        write_private(
            &paths.config,
            render_runtime_config(&persisted_config)?.as_bytes(),
        )?;

        let label = instance_label(name);
        let url = format!("http://127.0.0.1:{}", config.server.port);
        let mut metadata = InstanceMetadata {
            schema_version: INSTANCE_SCHEMA_VERSION,
            name: name.to_string(),
            label: label.clone(),
            model: config.worker.model.clone(),
            port: config.server.port,
            url: url.clone(),
            pid: None,
            gateway_executable: gateway_executable.to_path_buf(),
            config_path: paths.config.clone(),
            plist_path: paths.plist.clone(),
            socket_path: paths.socket.clone(),
            stdout_log: paths.stdout_log.clone(),
            stderr_log: paths.stderr_log.clone(),
        };
        write_metadata(&paths.metadata, &metadata)?;
        write_private(
            &paths.plist,
            render_plist(&metadata, gateway_executable).as_bytes(),
        )?;

        let service_target = service_target(&label);
        run_required(
            self.runner,
            CommandSpec::new("launchctl")
                .args(vec![
                    OsString::from("enable"),
                    service_target.clone().into(),
                ])
                .timeout(COMMAND_TIMEOUT),
            "enable launchd restart",
        )?;
        run_required(
            self.runner,
            CommandSpec::new("launchctl")
                .args([
                    "bootstrap".into(),
                    launchd_domain().into(),
                    path_arg(&paths.plist),
                ])
                .timeout(COMMAND_TIMEOUT),
            "load launch agent",
        )?;

        match self.wait_until_ready(&metadata) {
            Ok(pid) => {
                metadata.pid = Some(pid);
                write_metadata(&paths.metadata, &metadata)?;
                Ok(StartResult {
                    name: name.to_string(),
                    url,
                    pid,
                })
            }
            Err(error) => {
                self.disable_and_unload(&label);
                Err(error)
            }
        }
    }

    pub(crate) fn list(&self) -> Result<String, InstanceError> {
        let mut views = Vec::new();
        if self.application.instances.is_dir() {
            let mut entries = fs::read_dir(&self.application.instances)
                .map_err(|err| io_error("read managed instance directory", err))?
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_dir())
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries {
                let name = entry.file_name().to_string_lossy().into_owned();
                let paths = InstancePaths::new(self.application, &name);
                match read_metadata(&paths.metadata) {
                    Ok(metadata) => views.push(self.inspect(&metadata)?),
                    Err(_) => views.push(InstanceView::invalid(name)),
                }
            }
        }
        Ok(render_views(&views))
    }

    pub(crate) fn attach(&self, name: &str, interrupted: &AtomicBool) -> Result<(), InstanceError> {
        validate_instance_name(name)?;
        let paths = InstancePaths::new(self.application, name);
        if !paths.root.is_dir() {
            return Err(InstanceError::new(
                InstanceErrorKind::Launchd,
                format!("managed instance '{name}' is not registered"),
            ));
        }
        let spec = CommandSpec::new("/usr/bin/tail")
            .args([
                "-n".into(),
                "100".into(),
                "-F".into(),
                path_arg(&paths.stdout_log),
                path_arg(&paths.stderr_log),
            ])
            .timeout(ATTACH_TIMEOUT)
            .output_mode(OutputMode::Inherit);
        let result = self
            .runner
            .run_interruptible(&spec, interrupted)
            .map_err(|err| InstanceError::new(InstanceErrorKind::Launchd, err.to_string()))?;
        if result.success {
            Ok(())
        } else {
            Err(command_failure("follow instance logs", &result))
        }
    }

    pub(crate) fn stop(&self, name: &str) -> Result<String, InstanceError> {
        validate_instance_name(name)?;
        let paths = InstancePaths::new(self.application, name);
        if !paths.root.is_dir() {
            return Err(InstanceError::new(
                InstanceErrorKind::Launchd,
                format!("managed instance '{name}' is not registered"),
            ));
        }
        let metadata = read_metadata(&paths.metadata).ok();
        let label = metadata
            .as_ref()
            .map_or_else(|| instance_label(name), |metadata| metadata.label.clone());
        let status = self.query_service(&label)?;
        let mut pid = status.pid;

        run_required(
            self.runner,
            CommandSpec::new("launchctl")
                .args(vec![
                    OsString::from("disable"),
                    service_target(&label).into(),
                ])
                .timeout(COMMAND_TIMEOUT),
            "disable launchd restart",
        )?;
        if status.loaded {
            run_required(
                self.runner,
                CommandSpec::new("launchctl")
                    .args(vec![
                        OsString::from("bootout"),
                        service_target(&label).into(),
                    ])
                    .timeout(COMMAND_TIMEOUT),
                "unload launch agent",
            )?;
        } else if let Some(metadata) = &metadata {
            if let Some(stale_pid) = metadata.pid {
                if self.process_matches(stale_pid, &metadata.gateway_executable)? {
                    run_required(
                        self.runner,
                        CommandSpec::new("/bin/kill")
                            .args(vec![OsString::from("-TERM"), stale_pid.to_string().into()])
                            .timeout(COMMAND_TIMEOUT),
                        "terminate orphaned instance process",
                    )?;
                    pid = Some(stale_pid);
                }
            }
        }

        if let Some(pid) = pid {
            self.wait_until_stopped(pid)?;
        }
        remove_socket(&paths.socket)?;
        fs::remove_dir_all(&paths.root)
            .map_err(|err| io_error("remove managed instance state", err))?;
        Ok(format!("Stopped {name}\n"))
    }

    fn reject_registered_port(&self, port: u16) -> Result<(), InstanceError> {
        if !self.application.instances.is_dir() {
            return Ok(());
        }
        for entry in fs::read_dir(&self.application.instances)
            .map_err(|err| io_error("read managed instance directory", err))?
            .filter_map(Result::ok)
        {
            let metadata_path = entry.path().join("instance.json");
            if let Ok(metadata) = read_metadata(&metadata_path) {
                if metadata.port == port {
                    return Err(InstanceError::new(
                        InstanceErrorKind::Conflict,
                        format!(
                            "port {port} is already registered to managed instance '{}'",
                            metadata.name
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn wait_until_ready(&self, metadata: &InstanceMetadata) -> Result<u32, InstanceError> {
        let deadline = Instant::now() + self.startup_timeout;
        loop {
            let status = self.query_service(&metadata.label)?;
            if let Some(pid) = status.pid {
                if status.loaded
                    && status.state == "running"
                    && self.process_is_alive(pid)?
                    && self.readiness.is_ready(metadata.port)
                {
                    return Ok(pid);
                }
            }
            if Instant::now() >= deadline {
                return Err(InstanceError::new(
                    InstanceErrorKind::Serving,
                    format!(
                        "managed instance '{}' did not become ready within {} seconds",
                        metadata.name,
                        self.startup_timeout.as_secs()
                    ),
                ));
            }
            thread::sleep(self.poll_interval);
        }
    }

    fn inspect(&self, metadata: &InstanceMetadata) -> Result<InstanceView, InstanceError> {
        let status = self.query_service(&metadata.label)?;
        let alive = match status.pid {
            Some(pid) => self.process_is_alive(pid)?,
            None => false,
        };
        let ready = alive && self.readiness.is_ready(metadata.port);
        let stale = !status.loaded || status.state != "running" || status.pid.is_none() || !alive;
        Ok(InstanceView {
            name: metadata.name.clone(),
            pid: status.pid,
            model: metadata.model.clone(),
            url: metadata.url.clone(),
            launchd_state: status.state,
            ready,
            stale,
        })
    }

    fn query_service(&self, label: &str) -> Result<ServiceStatus, InstanceError> {
        let result = self
            .runner
            .run(
                &CommandSpec::new("launchctl")
                    .args(vec![OsString::from("print"), service_target(label).into()])
                    .timeout(COMMAND_TIMEOUT),
            )
            .map_err(|err| InstanceError::new(InstanceErrorKind::Launchd, err.to_string()))?;
        if !result.success {
            return Ok(ServiceStatus {
                loaded: false,
                state: "unloaded".to_string(),
                pid: None,
            });
        }
        Ok(parse_service_status(&result.stdout))
    }

    fn process_is_alive(&self, pid: u32) -> Result<bool, InstanceError> {
        let result = self
            .runner
            .run(
                &CommandSpec::new("/bin/kill")
                    .args(vec![OsString::from("-0"), pid.to_string().into()])
                    .timeout(COMMAND_TIMEOUT),
            )
            .map_err(|err| InstanceError::new(InstanceErrorKind::Launchd, err.to_string()))?;
        Ok(result.success)
    }

    fn process_matches(&self, pid: u32, executable: &Path) -> Result<bool, InstanceError> {
        let result = self
            .runner
            .run(
                &CommandSpec::new("/bin/ps")
                    .args(vec![
                        OsString::from("-p"),
                        pid.to_string().into(),
                        OsString::from("-o"),
                        OsString::from("command="),
                    ])
                    .timeout(COMMAND_TIMEOUT),
            )
            .map_err(|err| InstanceError::new(InstanceErrorKind::Launchd, err.to_string()))?;
        Ok(result.success && result.stdout.trim() == executable.to_string_lossy())
    }

    fn wait_until_stopped(&self, pid: u32) -> Result<(), InstanceError> {
        let deadline = Instant::now() + self.stop_timeout;
        while self.process_is_alive(pid)? {
            if Instant::now() >= deadline {
                return Err(InstanceError::new(
                    InstanceErrorKind::Launchd,
                    format!("instance process {pid} did not stop within the shutdown timeout"),
                ));
            }
            thread::sleep(self.poll_interval);
        }
        Ok(())
    }

    fn disable_and_unload(&self, label: &str) {
        let _ = self.runner.run(
            &CommandSpec::new("launchctl")
                .args(vec![
                    OsString::from("disable"),
                    service_target(label).into(),
                ])
                .timeout(COMMAND_TIMEOUT),
        );
        let _ = self.runner.run(
            &CommandSpec::new("launchctl")
                .args(vec![
                    OsString::from("bootout"),
                    service_target(label).into(),
                ])
                .timeout(COMMAND_TIMEOUT),
        );
    }
}

#[derive(Debug)]
struct InstancePaths {
    root: PathBuf,
    config: PathBuf,
    metadata: PathBuf,
    plist: PathBuf,
    socket: PathBuf,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
}

impl InstancePaths {
    fn new(application: &ApplicationPaths, name: &str) -> Self {
        let root = application.instances.join(name);
        Self {
            config: root.join("config.toml"),
            metadata: root.join("instance.json"),
            plist: root.join("launch-agent.plist"),
            socket: application.sockets.join(format!("instance-{name}.sock")),
            stdout_log: application.logs.join(format!("{name}.stdout.log")),
            stderr_log: application.logs.join(format!("{name}.stderr.log")),
            root,
        }
    }
}

#[derive(Debug)]
struct ServiceStatus {
    loaded: bool,
    state: String,
    pid: Option<u32>,
}

#[derive(Debug)]
struct InstanceView {
    name: String,
    pid: Option<u32>,
    model: String,
    url: String,
    launchd_state: String,
    ready: bool,
    stale: bool,
}

impl InstanceView {
    fn invalid(name: String) -> Self {
        Self {
            name,
            pid: None,
            model: "-".to_string(),
            url: "-".to_string(),
            launchd_state: "invalid-metadata".to_string(),
            ready: false,
            stale: true,
        }
    }
}

pub(crate) fn validate_instance_name(name: &str) -> Result<(), InstanceError> {
    let valid = !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'));
    if valid {
        Ok(())
    } else {
        Err(InstanceError::new(
            InstanceErrorKind::Conflict,
            "instance names must contain only ASCII letters, numbers, dots, dashes, and underscores",
        ))
    }
}

fn ensure_port_available(port: u16) -> Result<(), InstanceError> {
    TcpListener::bind(("127.0.0.1", port))
        .map(drop)
        .map_err(|err| {
            InstanceError::new(
                InstanceErrorKind::Conflict,
                format!("port {port} is unavailable: {err}"),
            )
        })
}

fn prepare_instance_directories(
    application: &ApplicationPaths,
    paths: &InstancePaths,
) -> Result<(), InstanceError> {
    for directory in [
        &application.instances,
        &application.logs,
        &application.sockets,
        &paths.root,
    ] {
        fs::create_dir_all(directory)
            .map_err(|err| io_error("create managed instance directory", err))?;
    }
    fs::set_permissions(&application.sockets, fs::Permissions::from_mode(0o700))
        .map_err(|err| io_error("secure socket directory", err))?;
    fs::set_permissions(&paths.root, fs::Permissions::from_mode(0o700))
        .map_err(|err| io_error("secure instance directory", err))?;
    Ok(())
}

fn create_log_file(path: &Path) -> Result<(), InstanceError> {
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| io_error("create instance log", err))?;
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|err| io_error("secure instance log", err))
}

fn write_private(path: &Path, contents: &[u8]) -> Result<(), InstanceError> {
    let temporary = path.with_extension("tmp");
    let mut file = fs::File::create(&temporary)
        .map_err(|err| io_error("create managed instance file", err))?;
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|err| io_error("secure managed instance file", err))?;
    file.write_all(contents)
        .map_err(|err| io_error("write managed instance file", err))?;
    fs::rename(&temporary, path).map_err(|err| io_error("install managed instance file", err))
}

fn write_metadata(path: &Path, metadata: &InstanceMetadata) -> Result<(), InstanceError> {
    let bytes = serde_json::to_vec_pretty(metadata).map_err(|err| {
        InstanceError::new(
            InstanceErrorKind::Launchd,
            format!("serialize instance metadata: {err}"),
        )
    })?;
    write_private(path, &bytes)
}

fn read_metadata(path: &Path) -> Result<InstanceMetadata, InstanceError> {
    let bytes = fs::read(path).map_err(|err| io_error("read instance metadata", err))?;
    serde_json::from_slice(&bytes).map_err(|err| {
        InstanceError::new(
            InstanceErrorKind::Launchd,
            format!("parse instance metadata at {}: {err}", path.display()),
        )
    })
}

fn render_runtime_config(config: &RuntimeConfig) -> Result<String, InstanceError> {
    let mut output = String::new();
    writeln!(output, "[server]").ok();
    writeln!(output, "host = {}", toml_string(&config.server.host)?).ok();
    writeln!(output, "port = {}\n", config.server.port).ok();
    writeln!(output, "[worker]").ok();
    writeln!(output, "python = {}", toml_string(&config.worker.python)?).ok();
    writeln!(output, "module = {}", toml_string(&config.worker.module)?).ok();
    writeln!(
        output,
        "backend = {}",
        toml_string(config.worker.backend.as_str())?
    )
    .ok();
    writeln!(output, "model = {}", toml_string(&config.worker.model)?).ok();
    if let Some(vlm_model) = &config.worker.vlm_model {
        writeln!(output, "vlm_model = {}", toml_string(vlm_model)?).ok();
    }
    writeln!(
        output,
        "ipc_path = {}\n",
        toml_string(&config.worker.ipc_path)?
    )
    .ok();
    writeln!(output, "[generation]").ok();
    writeln!(output, "temperature = {}", config.generation.temperature).ok();
    writeln!(output, "top_p = {}", config.generation.top_p).ok();
    writeln!(output, "max_tokens = {}\n", config.generation.max_tokens).ok();
    writeln!(output, "[limits]").ok();
    writeln!(
        output,
        "max_pending_requests = {}",
        config.limits.max_pending_requests
    )
    .ok();
    writeln!(
        output,
        "max_active_requests = {}",
        config.limits.max_active_requests
    )
    .ok();
    writeln!(
        output,
        "max_prompt_tokens = {}",
        config.limits.max_prompt_tokens
    )
    .ok();
    writeln!(
        output,
        "max_completion_tokens = {}",
        config.limits.max_completion_tokens
    )
    .ok();
    writeln!(
        output,
        "max_total_tokens_per_request = {}",
        config.limits.max_total_tokens_per_request
    )
    .ok();
    writeln!(
        output,
        "request_timeout_seconds = {}",
        config.limits.request_timeout_seconds
    )
    .ok();
    writeln!(
        output,
        "max_vlm_images = {}\n",
        config.limits.max_vlm_images
    )
    .ok();
    writeln!(output, "[telemetry]").ok();
    writeln!(
        output,
        "enable_prometheus = {}",
        config.telemetry.enable_prometheus
    )
    .ok();
    writeln!(
        output,
        "metrics_path = {}",
        toml_string(&config.telemetry.metrics_path)?
    )
    .ok();
    Ok(output)
}

fn toml_string(value: &str) -> Result<String, InstanceError> {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                return Err(InstanceError::new(
                    InstanceErrorKind::Conflict,
                    "configuration values cannot contain control characters",
                ));
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    Ok(escaped)
}

fn render_plist(metadata: &InstanceMetadata, gateway_executable: &Path) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n<dict>\n\
  <key>Label</key>\n  <string>{}</string>\n\
  <key>ProgramArguments</key>\n  <array>\n    <string>{}</string>\n  </array>\n\
  <key>EnvironmentVariables</key>\n  <dict>\n    <key>MLX_RUNTIME_CONFIG</key>\n    <string>{}</string>\n  </dict>\n\
  <key>RunAtLoad</key>\n  <true/>\n\
  <key>KeepAlive</key>\n  <true/>\n\
  <key>ThrottleInterval</key>\n  <integer>5</integer>\n\
  <key>StandardOutPath</key>\n  <string>{}</string>\n\
  <key>StandardErrorPath</key>\n  <string>{}</string>\n\
</dict>\n</plist>\n",
        xml_escape(&metadata.label),
        xml_escape(&gateway_executable.to_string_lossy()),
        xml_escape(&metadata.config_path.to_string_lossy()),
        xml_escape(&metadata.stdout_log.to_string_lossy()),
        xml_escape(&metadata.stderr_log.to_string_lossy()),
    )
}

fn xml_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            character => escaped.push(character),
        }
    }
    escaped
}

fn parse_service_status(output: &str) -> ServiceStatus {
    let mut state = "loaded".to_string();
    let mut pid = None;
    for line in output.lines() {
        let Some((key, value)) = line.trim().split_once('=') else {
            continue;
        };
        match key.trim() {
            "state" if state == "loaded" => state = value.trim().to_string(),
            "pid" if pid.is_none() => pid = value.trim().parse().ok(),
            _ => {}
        }
    }
    ServiceStatus {
        loaded: true,
        state,
        pid,
    }
}

fn render_views(views: &[InstanceView]) -> String {
    let mut rendered = format!(
        "{:<20} {:<8} {:<32} {:<30} {:<12} {:<6} {}\n",
        "NAME", "PID", "MODEL", "URL", "LAUNCHD", "READY", "STALE"
    );
    for view in views {
        let _ = writeln!(
            rendered,
            "{:<20} {:<8} {:<32} {:<30} {:<12} {:<6} {}",
            view.name,
            view.pid
                .map_or_else(|| "-".to_string(), |pid| pid.to_string()),
            view.model,
            view.url,
            view.launchd_state,
            yes_no(view.ready),
            yes_no(view.stale)
        );
    }
    rendered
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn http_ready(port: u16) -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(250)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(250)));
    if stream
        .write_all(b"GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    stream.read_to_string(&mut response).is_ok()
        && response
            .lines()
            .next()
            .is_some_and(|line| line.contains(" 200 "))
}

fn run_required(
    runner: &dyn CommandRunner,
    spec: CommandSpec,
    action: &str,
) -> Result<(), InstanceError> {
    let result = runner
        .run(&spec)
        .map_err(|err| InstanceError::new(InstanceErrorKind::Launchd, err.to_string()))?;
    if result.success {
        Ok(())
    } else {
        Err(command_failure(action, &result))
    }
}

fn command_failure(action: &str, result: &crate::command_runner::CommandResult) -> InstanceError {
    let detail = if result.stderr.trim().is_empty() {
        result.stdout.trim()
    } else {
        result.stderr.trim()
    };
    let detail = if detail.is_empty() {
        "no output"
    } else {
        detail
    };
    InstanceError::new(
        InstanceErrorKind::Launchd,
        format!("failed to {action}: {detail}"),
    )
}

fn io_error(action: &str, error: std::io::Error) -> InstanceError {
    InstanceError::new(InstanceErrorKind::Launchd, format!("{action}: {error}"))
}

fn remove_socket(path: &Path) -> Result<(), InstanceError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(io_error("remove instance socket", err)),
    }
}

fn instance_label(name: &str) -> String {
    format!("{LABEL_PREFIX}.{name}")
}

fn launchd_domain() -> String {
    format!("gui/{}", effective_user_id())
}

fn service_target(label: &str) -> String {
    format!("{}/{}", launchd_domain(), label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_runner::{CommandError, CommandResult};
    use crate::distribution::tests::temp_path;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct FakeRunner {
        calls: RefCell<Vec<CommandSpec>>,
        results: RefCell<VecDeque<Result<CommandResult, CommandError>>>,
    }

    impl FakeRunner {
        fn new(results: Vec<CommandResult>) -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                results: RefCell::new(results.into_iter().map(Ok).collect()),
            }
        }
    }

    impl CommandRunner for FakeRunner {
        fn run(&self, spec: &CommandSpec) -> Result<CommandResult, CommandError> {
            self.calls.borrow_mut().push(spec.clone());
            self.results
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| Ok(success("")))
        }

        fn run_interruptible(
            &self,
            spec: &CommandSpec,
            _interrupted: &AtomicBool,
        ) -> Result<CommandResult, CommandError> {
            self.run(spec)
        }
    }

    struct FakeReadiness(bool);

    impl ReadinessProbe for FakeReadiness {
        fn is_ready(&self, _port: u16) -> bool {
            self.0
        }
    }

    fn success(stdout: &str) -> CommandResult {
        CommandResult {
            success: true,
            stdout: stdout.to_string(),
            stderr: String::new(),
        }
    }

    fn failure(stderr: &str) -> CommandResult {
        CommandResult {
            success: false,
            stdout: String::new(),
            stderr: stderr.to_string(),
        }
    }

    fn application(label: &str) -> (PathBuf, ApplicationPaths) {
        let root = temp_path(label);
        let mut application = ApplicationPaths::from_home(&root);
        application.sockets = root.join("sockets");
        (root, application)
    }

    fn unused_port() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.local_addr().unwrap().port()
    }

    fn metadata(application: &ApplicationPaths, name: &str, port: u16) -> InstanceMetadata {
        let paths = InstancePaths::new(application, name);
        InstanceMetadata {
            schema_version: INSTANCE_SCHEMA_VERSION,
            name: name.to_string(),
            label: instance_label(name),
            model: "test-model".to_string(),
            port,
            url: format!("http://127.0.0.1:{port}"),
            pid: Some(10),
            gateway_executable: PathBuf::from("/distribution/bin/gateway"),
            config_path: paths.config,
            plist_path: paths.plist,
            socket_path: paths.socket,
            stdout_log: paths.stdout_log,
            stderr_log: paths.stderr_log,
        }
    }

    fn register_metadata(application: &ApplicationPaths, metadata: &InstanceMetadata) {
        let paths = InstancePaths::new(application, &metadata.name);
        fs::create_dir_all(&paths.root).unwrap();
        fs::create_dir_all(&application.logs).unwrap();
        write_metadata(&paths.metadata, metadata).unwrap();
        create_log_file(&paths.stdout_log).unwrap();
        create_log_file(&paths.stderr_log).unwrap();
    }

    #[test]
    fn instance_name_accepts_the_documented_grammar() {
        assert!(validate_instance_name("model_1.test-name").is_ok());
    }

    #[test]
    fn instance_name_rejects_path_and_shell_characters() {
        for name in ["", ".", "..", "a/b", "a b", "a;rm"] {
            assert!(validate_instance_name(name).is_err(), "accepted {name}");
        }
    }

    #[test]
    fn plist_escapes_paths_and_uses_a_fixed_argument_array() {
        let (root, application) = application("plist");
        let metadata = metadata(&application, "safe", 9000);

        let plist = render_plist(&metadata, Path::new("/Applications/A&B/<gateway>"));

        assert!(plist.contains("/Applications/A&amp;B/&lt;gateway&gt;"));
        assert!(plist.contains("<key>KeepAlive</key>\n  <true/>"));
        assert_eq!(plist.matches("<key>ProgramArguments</key>").count(), 1);
        assert!(!plist.contains("/bin/sh"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_round_trips_through_instance_json() {
        let (root, application) = application("metadata");
        let metadata = metadata(&application, "json", 9001);
        let path = root.join("instance.json");
        fs::create_dir_all(&root).unwrap();

        write_metadata(&path, &metadata).unwrap();

        assert_eq!(read_metadata(&path).unwrap(), metadata);
        if root.exists() {
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn start_writes_state_and_returns_ready_pid() {
        let (root, application) = application("start-ready");
        let runner = FakeRunner::new(vec![
            success(""),
            success(""),
            success("state = running\npid = 4321\n"),
            success(""),
        ]);
        let readiness = FakeReadiness(true);
        let manager = InstanceManager::new(&application, &runner, &readiness)
            .with_timeouts(Duration::ZERO, Duration::ZERO);
        let mut config = RuntimeConfig::default();
        config.server.port = unused_port();

        let result = manager
            .start("ready", &config, Path::new("/distribution/bin/gateway"))
            .unwrap();

        assert_eq!(result.pid, 4321);
        assert!(application.instances.join("ready/config.toml").is_file());
        assert!(application.instances.join("ready/instance.json").is_file());
        assert!(application
            .instances
            .join("ready/launch-agent.plist")
            .is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn start_rejects_duplicate_name_without_launchctl() {
        let (root, application) = application("duplicate-name");
        fs::create_dir_all(application.instances.join("duplicate")).unwrap();
        let runner = FakeRunner::new(vec![]);
        let readiness = FakeReadiness(true);
        let manager = InstanceManager::new(&application, &runner, &readiness);
        let mut config = RuntimeConfig::default();
        config.server.port = unused_port();

        let error = manager
            .start("duplicate", &config, Path::new("/gateway"))
            .unwrap_err();

        assert_eq!(error.kind(), InstanceErrorKind::Conflict);
        assert!(runner.calls.borrow().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn start_rejects_duplicate_registered_port() {
        let (root, application) = application("duplicate-port");
        let port = unused_port();
        register_metadata(&application, &metadata(&application, "existing", port));
        let runner = FakeRunner::new(vec![]);
        let readiness = FakeReadiness(true);
        let manager = InstanceManager::new(&application, &runner, &readiness);
        let mut config = RuntimeConfig::default();
        config.server.port = port;

        let error = manager
            .start("second", &config, Path::new("/gateway"))
            .unwrap_err();

        assert!(error.to_string().contains("already registered"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn start_rejects_occupied_port() {
        let (root, application) = application("occupied-port");
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let runner = FakeRunner::new(vec![]);
        let readiness = FakeReadiness(true);
        let manager = InstanceManager::new(&application, &runner, &readiness);
        let mut config = RuntimeConfig::default();
        config.server.port = listener.local_addr().unwrap().port();

        let error = manager
            .start("occupied", &config, Path::new("/gateway"))
            .unwrap_err();

        assert_eq!(error.kind(), InstanceErrorKind::Conflict);
        if root.exists() {
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn failed_launchctl_load_preserves_registration_evidence() {
        let (root, application) = application("failed-load");
        let runner = FakeRunner::new(vec![success(""), failure("load failed")]);
        let readiness = FakeReadiness(true);
        let manager = InstanceManager::new(&application, &runner, &readiness);
        let mut config = RuntimeConfig::default();
        config.server.port = unused_port();

        let error = manager
            .start("failed", &config, Path::new("/gateway"))
            .unwrap_err();

        assert_eq!(error.kind(), InstanceErrorKind::Launchd);
        assert!(application.instances.join("failed/instance.json").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_readiness_disables_and_unloads_service() {
        let (root, application) = application("failed-readiness");
        let runner = FakeRunner::new(vec![
            success(""),
            success(""),
            success("state = running\npid = 22\n"),
            success(""),
            success(""),
            success(""),
        ]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness)
            .with_timeouts(Duration::ZERO, Duration::ZERO);
        let mut config = RuntimeConfig::default();
        config.server.port = unused_port();

        let error = manager
            .start("timeout", &config, Path::new("/gateway"))
            .unwrap_err();

        assert_eq!(error.kind(), InstanceErrorKind::Serving);
        let calls = runner.calls.borrow();
        assert_eq!(calls[calls.len() - 2].args[0], "disable");
        assert_eq!(calls[calls.len() - 1].args[0], "bootout");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_reports_restart_pid_without_marking_it_stale() {
        let (root, application) = application("restart");
        register_metadata(&application, &metadata(&application, "restarted", 9002));
        let runner = FakeRunner::new(vec![success("state = running\npid = 99\n"), success("")]);
        let readiness = FakeReadiness(true);
        let manager = InstanceManager::new(&application, &runner, &readiness);

        let output = manager.list().unwrap();

        assert!(output.contains("restarted"));
        assert!(output.contains("99"));
        assert!(output.contains("running"));
        assert!(output.contains("yes    no"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_marks_unloaded_registration_as_stale() {
        let (root, application) = application("stale");
        register_metadata(&application, &metadata(&application, "stale", 9003));
        let runner = FakeRunner::new(vec![failure("not found")]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness);

        let output = manager.list().unwrap();

        assert!(output.contains("unloaded"));
        assert!(output.contains("no     yes"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_marks_missing_running_process_as_stale() {
        let (root, application) = application("missing-process");
        register_metadata(&application, &metadata(&application, "missing", 9004));
        let runner = FakeRunner::new(vec![
            success("state = running\npid = 44\n"),
            failure("missing"),
        ]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness);

        let output = manager.list().unwrap();

        assert!(output.contains("running"));
        assert!(output.contains("no     yes"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn attach_uses_fixed_tail_arguments_and_accepts_interruption() {
        let (root, application) = application("attach");
        register_metadata(&application, &metadata(&application, "logs", 9005));
        let runner = FakeRunner::new(vec![success("")]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness);
        let interrupted = AtomicBool::new(true);

        manager.attach("logs", &interrupted).unwrap();

        let calls = runner.calls.borrow();
        assert_eq!(calls[0].program, "/usr/bin/tail");
        assert_eq!(&calls[0].args[..3], ["-n", "100", "-F"]);
        assert_eq!(calls[0].output_mode, OutputMode::Inherit);
        assert!(interrupted.load(Ordering::Relaxed));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stop_removes_registration_and_preserves_logs() {
        let (root, application) = application("stop");
        let metadata = metadata(&application, "service", 9006);
        register_metadata(&application, &metadata);
        let runner = FakeRunner::new(vec![
            success("state = running\npid = 10\n"),
            success(""),
            success(""),
            failure("not running"),
        ]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness)
            .with_timeouts(Duration::ZERO, Duration::ZERO);

        manager.stop("service").unwrap();

        assert!(!application.instances.join("service").exists());
        assert!(application.logs.join("service.stdout.log").is_file());
        assert!(application.logs.join("service.stderr.log").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stop_terminates_a_verified_orphan_process() {
        let (root, application) = application("stop-orphan");
        let metadata = metadata(&application, "orphan", 9007);
        register_metadata(&application, &metadata);
        let runner = FakeRunner::new(vec![
            failure("not loaded"),
            success(""),
            success("/distribution/bin/gateway\n"),
            success(""),
            failure("not running"),
        ]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness)
            .with_timeouts(Duration::ZERO, Duration::ZERO);

        manager.stop("orphan").unwrap();

        let calls = runner.calls.borrow();
        assert!(calls
            .iter()
            .any(|call| call.program == "/bin/kill" && call.args[0] == "-TERM"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stop_does_not_signal_a_reused_stale_pid() {
        let (root, application) = application("stop-reused-pid");
        let metadata = metadata(&application, "reused", 9008);
        register_metadata(&application, &metadata);
        let runner = FakeRunner::new(vec![
            failure("not loaded"),
            success(""),
            success("/unrelated/process\n"),
        ]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness);

        manager.stop("reused").unwrap();

        assert!(!runner
            .calls
            .borrow()
            .iter()
            .any(|call| call.program == "/bin/kill"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_renders_required_columns() {
        let (_root, application) = application("columns");
        let runner = FakeRunner::new(vec![]);
        let readiness = FakeReadiness(false);
        let manager = InstanceManager::new(&application, &runner, &readiness);

        let output = manager.list().unwrap();

        assert!(output.starts_with("NAME"));
        for column in ["PID", "MODEL", "URL", "LAUNCHD", "READY", "STALE"] {
            assert!(output.contains(column));
        }
    }

    #[test]
    fn launchctl_parser_ignores_nested_coalition_states() {
        let status = parse_service_status(
            "state = running\npid = 321\nresource coalition = {\nstate = active\n}\n",
        );

        assert_eq!(status.state, "running");
        assert_eq!(status.pid, Some(321));
    }
}
