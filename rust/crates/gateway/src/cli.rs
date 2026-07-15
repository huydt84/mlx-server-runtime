//! Parser and dispatch for the public `mlx-air` CLI.

use crate::command_runner::ProcessCommandRunner;
use crate::configuration::{resolve_runtime_config, CliConfigOverrides};
use crate::distribution::{ApplicationPaths, DistributionPaths};
use crate::doctor::{run_doctor, DoctorReport, PlatformInfo};
use crate::environment::{ensure_runtime_environment, select_runtime_environment};
use crate::{BackendKind, GatewayError};
use clap::error::ErrorKind;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::PathBuf;
use std::process;

/// Stable process exit categories used by `mlx-air` commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCode {
    /// The command completed or terminated normally.
    NormalTermination = 0,
    /// Command-line arguments were invalid.
    InvalidArguments = 2,
    /// Runtime or benchmark environment setup failed.
    EnvironmentSetupFailure = 10,
    /// Serving failed during startup or readiness checks.
    ServingStartupOrReadinessFailure = 20,
    /// An instance name or port conflicts with existing state.
    InstanceNameOrPortConflict = 30,
    /// A `launchd` operation failed.
    LaunchdFailure = 40,
    /// Benchmark setup or execution failed.
    BenchmarkExecutionFailure = 50,
}

impl From<ExitCode> for process::ExitCode {
    fn from(value: ExitCode) -> Self {
        Self::from(value as u8)
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "mlx-air",
    version,
    about = "Native MLX model serving and benchmarking",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve a model through the OpenAI-compatible HTTP API.
    Serve(ServeArgs),
    /// List managed model-server instances.
    Ps,
    /// Follow logs for a managed instance.
    Attach {
        /// Managed instance name.
        name: String,
    },
    /// Stop a managed instance.
    Stop {
        /// Managed instance name.
        name: String,
    },
    /// Check platform and runtime dependencies.
    Doctor,
    /// Print the MLX Air version.
    Version,
    /// Print root or command help.
    Help {
        /// Command whose help should be displayed.
        #[arg(value_enum)]
        command: Option<HelpTopic>,
    },
    /// Run benchmarks and benchmark diagnostics.
    Bench {
        #[command(subcommand)]
        command: Option<BenchCommand>,
    },
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Model identifier or local model path.
    #[arg(long)]
    model: Option<String>,
    /// Runtime configuration file.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Worker backend.
    #[arg(long, value_enum)]
    backend: Option<Backend>,
    /// HTTP listen port.
    #[arg(long)]
    port: Option<u16>,
    /// Start a managed detached instance.
    #[arg(long, requires_all = ["name", "port"])]
    detach: bool,
    /// Managed instance name; valid only with `--detach`.
    #[arg(long, requires = "detach")]
    name: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    V1,
    NativeMlx,
}

impl From<Backend> for BackendKind {
    fn from(value: Backend) -> Self {
        match value {
            Backend::V1 => Self::V1,
            Backend::NativeMlx => Self::NativeMlx,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum HelpTopic {
    Serve,
    Ps,
    Attach,
    Stop,
    Doctor,
    Version,
    Help,
    Bench,
}

impl HelpTopic {
    fn as_str(self) -> &'static str {
        match self {
            Self::Serve => "serve",
            Self::Ps => "ps",
            Self::Attach => "attach",
            Self::Stop => "stop",
            Self::Doctor => "doctor",
            Self::Version => "version",
            Self::Help => "help",
            Self::Bench => "bench",
        }
    }
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    /// Run a benchmark suite.
    Run(BenchmarkArgs),
    /// Collect diagnostics for an existing benchmark result.
    Diagnose(BenchmarkArgs),
    /// Calibrate benchmark repetition counts.
    Calibrate(BenchmarkArgs),
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    /// Arguments forwarded unchanged to the benchmark implementation.
    #[arg(
        value_name = "ARG",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    args: Vec<OsString>,
}

struct CliOutput {
    code: ExitCode,
    stdout: String,
    stderr: String,
}

impl CliOutput {
    fn stdout(code: ExitCode, stdout: String) -> Self {
        Self {
            code,
            stdout,
            stderr: String::new(),
        }
    }

    fn stderr(code: ExitCode, stderr: String) -> Self {
        Self {
            code,
            stdout: String::new(),
            stderr,
        }
    }
}

/// Parses and dispatches the public CLI using the current process arguments.
pub fn main_entry() -> process::ExitCode {
    let output = execute(std::env::args_os(), &ProductionRuntime);
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    let _ = stdout.write_all(output.stdout.as_bytes());
    let _ = stderr.write_all(output.stderr.as_bytes());
    output.code.into()
}

fn execute<I, T>(args: I, runtime: &dyn RuntimeOperations) -> CliOutput
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => dispatch(cli, runtime),
        Err(error) => clap_output(error),
    }
}

fn dispatch(cli: Cli, runtime: &dyn RuntimeOperations) -> CliOutput {
    match cli.command {
        None => help_output(None),
        Some(Command::Help { command }) => help_output(command.map(HelpTopic::as_str)),
        Some(Command::Bench { command: None }) => help_output(Some("bench")),
        Some(Command::Version) => CliOutput::stdout(
            ExitCode::NormalTermination,
            format!("mlx-air {}\n", env!("CARGO_PKG_VERSION")),
        ),
        Some(Command::Serve(args)) if args.detach => {
            stub_output("serve --detach", ExitCode::LaunchdFailure)
        }
        Some(Command::Serve(args)) => match runtime.serve(&args) {
            Ok(()) => CliOutput::stdout(ExitCode::NormalTermination, String::new()),
            Err(failure) => {
                CliOutput::stderr(failure.code, format!("error: {}\n", failure.message))
            }
        },
        Some(Command::Doctor) => match runtime.doctor() {
            Ok(report) => doctor_output(&report),
            Err(message) => CliOutput::stderr(
                ExitCode::EnvironmentSetupFailure,
                format!("error: {message}\n"),
            ),
        },
        Some(Command::Ps) => stub_output("ps", ExitCode::LaunchdFailure),
        Some(Command::Attach { name }) => {
            let _ = name;
            stub_output("attach", ExitCode::LaunchdFailure)
        }
        Some(Command::Stop { name }) => {
            let _ = name;
            stub_output("stop", ExitCode::LaunchdFailure)
        }
        Some(Command::Bench {
            command: Some(command),
        }) => {
            let action = match command {
                BenchCommand::Run(args) => {
                    let _ = args.args;
                    "bench run"
                }
                BenchCommand::Diagnose(args) => {
                    let _ = args.args;
                    "bench diagnose"
                }
                BenchCommand::Calibrate(args) => {
                    let _ = args.args;
                    "bench calibrate"
                }
            };
            stub_output(action, ExitCode::BenchmarkExecutionFailure)
        }
    }
}

trait RuntimeOperations {
    fn serve(&self, args: &ServeArgs) -> Result<(), RuntimeFailure>;
    fn doctor(&self) -> Result<DoctorReport, String>;
}

struct RuntimeFailure {
    code: ExitCode,
    message: String,
}

impl RuntimeFailure {
    fn environment(message: impl Into<String>) -> Self {
        Self {
            code: ExitCode::EnvironmentSetupFailure,
            message: message.into(),
        }
    }

    fn serving(error: GatewayError) -> Self {
        let code = match &error {
            GatewayError::Io(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
                ExitCode::InstanceNameOrPortConflict
            }
            _ => ExitCode::ServingStartupOrReadinessFailure,
        };
        Self {
            code,
            message: error.to_string(),
        }
    }
}

struct ProductionRuntime;

impl RuntimeOperations for ProductionRuntime {
    fn serve(&self, args: &ServeArgs) -> Result<(), RuntimeFailure> {
        let distribution = DistributionPaths::from_current_executable()
            .map_err(|err| RuntimeFailure::environment(err.to_string()))?;
        let application = ApplicationPaths::from_environment()
            .map_err(|err| RuntimeFailure::environment(err.to_string()))?;
        let selected = select_runtime_environment(&distribution, &application)
            .map_err(|err| RuntimeFailure::environment(err.to_string()))?;
        let environment_config = std::env::var_os("MLX_RUNTIME_CONFIG").map(PathBuf::from);
        let overrides = CliConfigOverrides {
            model: args.model.clone(),
            backend: args.backend.map(Into::into),
            port: args.port,
        };
        let mut config = resolve_runtime_config(
            &distribution.default_config,
            environment_config.as_deref(),
            args.config.as_deref(),
            &overrides,
            &selected.python,
        )
        .map_err(|err| RuntimeFailure::environment(err.to_string()))?;
        if config.server.host != "127.0.0.1" {
            return Err(RuntimeFailure::serving(GatewayError::WorkerStartup(
                format!(
                    "mlx-air serves only on 127.0.0.1; configured host is {}",
                    config.server.host
                ),
            )));
        }
        ensure_runtime_environment(&distribution, &application, &ProcessCommandRunner)
            .map_err(|err| RuntimeFailure::environment(err.to_string()))?;
        let socket_path = application
            .create_foreground_socket_path()
            .map_err(|err| RuntimeFailure::environment(err.to_string()))?;
        config.worker.ipc_path = socket_path.to_string_lossy().into_owned();
        crate::run(config).map_err(RuntimeFailure::serving)
    }

    fn doctor(&self) -> Result<DoctorReport, String> {
        let distribution =
            DistributionPaths::from_current_executable().map_err(|err| err.to_string())?;
        let application = ApplicationPaths::from_environment().map_err(|err| err.to_string())?;
        let selected = select_runtime_environment(&distribution, &application)
            .map_err(|err| err.to_string())?;
        let environment_config = std::env::var_os("MLX_RUNTIME_CONFIG").map(PathBuf::from);
        let config = resolve_runtime_config(
            &distribution.default_config,
            environment_config.as_deref(),
            None,
            &CliConfigOverrides::default(),
            &selected.python,
        )
        .map_err(|err| err.to_string())?;
        Ok(run_doctor(
            &distribution,
            &application,
            &config,
            &ProcessCommandRunner,
            PlatformInfo::current(),
        ))
    }
}

fn doctor_output(report: &DoctorReport) -> CliOutput {
    let code = if report.passed() {
        ExitCode::NormalTermination
    } else {
        ExitCode::EnvironmentSetupFailure
    };
    CliOutput::stdout(code, report.render())
}

fn help_output(command: Option<&str>) -> CliOutput {
    let mut args = vec![OsString::from("mlx-air")];
    if let Some(command) = command {
        args.push(OsString::from(command));
    }
    args.push(OsString::from("--help"));

    match Cli::try_parse_from(args) {
        Ok(_) => CliOutput::stderr(
            ExitCode::InvalidArguments,
            "error: failed to render command help\n".to_string(),
        ),
        Err(error) => clap_output(error),
    }
}

fn clap_output(error: clap::Error) -> CliOutput {
    let rendered = error.render().to_string();
    match error.kind() {
        ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
            CliOutput::stdout(ExitCode::NormalTermination, rendered)
        }
        _ => CliOutput::stderr(ExitCode::InvalidArguments, rendered),
    }
}

fn stub_output(command: &str, code: ExitCode) -> CliOutput {
    CliOutput::stderr(
        code,
        format!("error: 'mlx-air {command}' is not implemented yet\n"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::DoctorCheck;
    use std::cell::Cell;

    struct FakeRuntime {
        calls: Cell<usize>,
    }

    impl FakeRuntime {
        fn new() -> Self {
            Self {
                calls: Cell::new(0),
            }
        }
    }

    impl RuntimeOperations for FakeRuntime {
        fn serve(&self, _args: &ServeArgs) -> Result<(), RuntimeFailure> {
            self.calls.set(self.calls.get() + 1);
            Ok(())
        }

        fn doctor(&self) -> Result<DoctorReport, String> {
            self.calls.set(self.calls.get() + 1);
            Ok(DoctorReport {
                checks: vec![DoctorCheck {
                    name: "fake",
                    passed: true,
                    detail: "ok".to_string(),
                }],
            })
        }
    }

    fn output(args: &[&str]) -> CliOutput {
        execute(args.iter().copied(), &FakeRuntime::new())
    }

    #[test]
    fn exit_codes_remain_stable() {
        let actual = [
            ExitCode::NormalTermination as u8,
            ExitCode::InvalidArguments as u8,
            ExitCode::EnvironmentSetupFailure as u8,
            ExitCode::ServingStartupOrReadinessFailure as u8,
            ExitCode::InstanceNameOrPortConflict as u8,
            ExitCode::LaunchdFailure as u8,
            ExitCode::BenchmarkExecutionFailure as u8,
        ];

        assert_eq!(actual, [0, 2, 10, 20, 30, 40, 50]);
    }

    #[test]
    fn root_invocations_render_identical_help() {
        let runtime = FakeRuntime::new();
        let no_args = execute(["mlx-air"], &runtime);
        let help_command = execute(["mlx-air", "help"], &runtime);
        let help_flag = execute(["mlx-air", "--help"], &runtime);

        assert_eq!(
            (no_args.code, &no_args.stdout),
            (help_command.code, &help_command.stdout)
        );
        assert_eq!(help_command.stdout, help_flag.stdout);
        assert_eq!(runtime.calls.get(), 0);
    }

    #[test]
    fn serve_help_invocations_render_identical_arguments() {
        let help_command = output(&["mlx-air", "help", "serve"]);
        let help_flag = output(&["mlx-air", "serve", "--help"]);

        assert_eq!(help_command.stdout, help_flag.stdout);
    }

    #[test]
    fn bench_help_invocations_render_identical_output() {
        let runtime = FakeRuntime::new();
        let no_action = execute(["mlx-air", "bench"], &runtime);
        let help_command = execute(["mlx-air", "help", "bench"], &runtime);
        let help_flag = execute(["mlx-air", "bench", "--help"], &runtime);

        assert_eq!(
            (no_action.code, &no_action.stdout),
            (help_command.code, &help_command.stdout)
        );
        assert_eq!(help_command.stdout, help_flag.stdout);
        assert_eq!(runtime.calls.get(), 0);
    }

    #[test]
    fn parser_accepts_foreground_serve_arguments() {
        let result = Cli::try_parse_from([
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--config",
            "/tmp/runtime.toml",
            "--backend",
            "native-mlx",
            "--port",
            "9000",
        ]);

        assert!(result.is_ok(), "unexpected parse error: {result:?}");
    }

    #[test]
    fn parser_accepts_complete_detached_serve_arguments() {
        let result = Cli::try_parse_from([
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--detach",
            "--name",
            "test-instance",
            "--port",
            "9000",
        ]);

        assert!(result.is_ok(), "unexpected parse error: {result:?}");
    }

    #[test]
    fn parser_rejects_detach_without_name() {
        let error = Cli::try_parse_from([
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--detach",
            "--port",
            "9000",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn parser_rejects_detach_without_explicit_port() {
        let error = Cli::try_parse_from([
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--detach",
            "--name",
            "test-instance",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn parser_rejects_name_without_detach() {
        let error = Cli::try_parse_from([
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--name",
            "test-instance",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn parser_rejects_unknown_backend() {
        let error = Cli::try_parse_from([
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--backend",
            "unknown",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn parser_rejects_unknown_command() {
        let error = Cli::try_parse_from(["mlx-air", "unknown"]).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn parser_accepts_benchmark_leaf_arguments_without_interpreting_them() {
        let result = Cli::try_parse_from([
            "mlx-air", "bench", "run", "--suite", "smoke", "--focus", "latency",
        ]);

        assert!(result.is_ok(), "unexpected parse error: {result:?}");
    }

    #[test]
    fn foreground_serve_returns_normal_termination_after_runtime_stops() {
        let result = output(&["mlx-air", "serve", "--model", "test-model"]);

        assert_eq!(result.code, ExitCode::NormalTermination);
        assert!(result.stderr.is_empty());
    }

    #[test]
    fn detached_serve_remains_an_explicit_stub() {
        let result = output(&[
            "mlx-air",
            "serve",
            "--model",
            "test-model",
            "--detach",
            "--name",
            "test",
            "--port",
            "9000",
        ]);

        assert_eq!(result.code, ExitCode::LaunchdFailure);
        assert_eq!(
            result.stderr,
            "error: 'mlx-air serve --detach' is not implemented yet\n"
        );
    }

    #[test]
    fn doctor_uses_normal_exit_for_successful_report() {
        let result = output(&["mlx-air", "doctor"]);

        assert_eq!(result.code, ExitCode::NormalTermination);
        assert_eq!(result.stdout, "[PASS] fake: ok\n");
    }

    #[test]
    fn doctor_uses_environment_exit_for_failed_report() {
        let report = DoctorReport {
            checks: vec![DoctorCheck {
                name: "fake",
                passed: false,
                detail: "failed".to_string(),
            }],
        };

        let result = doctor_output(&report);

        assert_eq!(result.code, ExitCode::EnvironmentSetupFailure);
        assert_eq!(result.stdout, "[FAIL] fake: failed\n");
    }

    #[test]
    fn version_prints_package_version() {
        let runtime = FakeRuntime::new();
        let result = execute(["mlx-air", "version"], &runtime);

        assert_eq!(result.code, ExitCode::NormalTermination);
        assert_eq!(
            result.stdout,
            format!("mlx-air {}\n", env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(runtime.calls.get(), 0);
    }
}
