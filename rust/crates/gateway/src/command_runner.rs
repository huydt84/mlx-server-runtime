use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputMode {
    Capture,
    Inherit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandSpec {
    pub(crate) program: OsString,
    pub(crate) args: Vec<OsString>,
    pub(crate) env: BTreeMap<OsString, OsString>,
    pub(crate) current_dir: Option<PathBuf>,
    pub(crate) timeout: Duration,
    pub(crate) output_mode: OutputMode,
}

impl CommandSpec {
    pub(crate) fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            current_dir: None,
            timeout: Duration::from_secs(30),
            output_mode: OutputMode::Capture,
        }
    }

    pub(crate) fn args<I, T>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub(crate) fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub(crate) fn current_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(path.into());
        self
    }

    pub(crate) fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub(crate) fn output_mode(mut self, output_mode: OutputMode) -> Self {
        self.output_mode = output_mode;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandResult {
    pub(crate) success: bool,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

#[derive(Debug)]
pub(crate) struct CommandError(String);

impl CommandError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CommandError {}

pub(crate) trait CommandRunner {
    fn run(&self, spec: &CommandSpec) -> Result<CommandResult, CommandError>;
}

pub(crate) struct ProcessCommandRunner;

impl CommandRunner for ProcessCommandRunner {
    fn run(&self, spec: &CommandSpec) -> Result<CommandResult, CommandError> {
        let mut command = Command::new(&spec.program);
        command.args(&spec.args).envs(&spec.env);
        if let Some(current_dir) = &spec.current_dir {
            command.current_dir(current_dir);
        }

        match spec.output_mode {
            OutputMode::Capture => {
                command.stdout(Stdio::piped()).stderr(Stdio::piped());
            }
            OutputMode::Inherit => {
                command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
            }
        }

        let mut child = command.spawn().map_err(|err| {
            CommandError::new(format!(
                "failed to start {}: {err}",
                spec.program.to_string_lossy()
            ))
        })?;

        let stdout_reader = child.stdout.take().map(read_pipe);
        let stderr_reader = child.stderr.take().map(read_pipe);
        let deadline = Instant::now() + spec.timeout;

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(CommandError::new(format!(
                        "{} timed out after {} seconds",
                        spec.program.to_string_lossy(),
                        spec.timeout.as_secs()
                    )));
                }
                Err(err) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(CommandError::new(format!(
                        "failed while waiting for {}: {err}",
                        spec.program.to_string_lossy()
                    )));
                }
            }
        };

        Ok(CommandResult {
            success: status.success(),
            stdout: join_pipe(stdout_reader, "stdout")?,
            stderr: join_pipe(stderr_reader, "stderr")?,
        })
    }
}

fn read_pipe<R>(mut pipe: R) -> thread::JoinHandle<std::io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn join_pipe(
    reader: Option<thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    stream_name: &str,
) -> Result<String, CommandError> {
    let Some(reader) = reader else {
        return Ok(String::new());
    };
    let bytes = reader
        .join()
        .map_err(|_| CommandError::new(format!("failed to join {stream_name} reader")))?
        .map_err(|err| CommandError::new(format!("failed to read {stream_name}: {err}")))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

pub(crate) fn display_command(spec: &CommandSpec) -> String {
    std::iter::once(spec.program.as_os_str())
        .chain(spec.args.iter().map(OsString::as_os_str))
        .map(OsStr::to_string_lossy)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn path_arg(path: &Path) -> OsString {
    path.as_os_str().to_owned()
}
