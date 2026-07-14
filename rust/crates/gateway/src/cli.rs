//! Parser and side-effect-free Phase 1 dispatch for the public `mlx-air` CLI.

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
    model: String,
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
///
/// Operational commands intentionally remain side-effect-free stubs in Phase 1.
pub fn main_entry() -> process::ExitCode {
    let output = execute(std::env::args_os());
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    let _ = stdout.write_all(output.stdout.as_bytes());
    let _ = stderr.write_all(output.stderr.as_bytes());
    output.code.into()
}

fn execute<I, T>(args: I) -> CliOutput
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => dispatch(cli),
        Err(error) => clap_output(error),
    }
}

fn dispatch(cli: Cli) -> CliOutput {
    match cli.command {
        None => help_output(None),
        Some(Command::Help { command }) => help_output(command.map(HelpTopic::as_str)),
        Some(Command::Bench { command: None }) => help_output(Some("bench")),
        Some(Command::Version) => CliOutput::stdout(
            ExitCode::NormalTermination,
            format!("mlx-air {}\n", env!("CARGO_PKG_VERSION")),
        ),
        Some(Command::Serve(args)) => {
            let _ = args;
            stub_output("serve", ExitCode::ServingStartupOrReadinessFailure)
        }
        Some(Command::Doctor) => stub_output("doctor", ExitCode::EnvironmentSetupFailure),
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

    fn output(args: &[&str]) -> CliOutput {
        execute(args.iter().copied())
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
        let no_args = output(&["mlx-air"]);
        let help_command = output(&["mlx-air", "help"]);
        let help_flag = output(&["mlx-air", "--help"]);

        assert_eq!(
            (no_args.code, &no_args.stdout),
            (help_command.code, &help_command.stdout)
        );
        assert_eq!(help_command.stdout, help_flag.stdout);
    }

    #[test]
    fn serve_help_invocations_render_identical_arguments() {
        let help_command = output(&["mlx-air", "help", "serve"]);
        let help_flag = output(&["mlx-air", "serve", "--help"]);

        assert_eq!(help_command.stdout, help_flag.stdout);
    }

    #[test]
    fn bench_help_invocations_render_identical_output() {
        let no_action = output(&["mlx-air", "bench"]);
        let help_command = output(&["mlx-air", "help", "bench"]);
        let help_flag = output(&["mlx-air", "bench", "--help"]);

        assert_eq!(
            (no_action.code, &no_action.stdout),
            (help_command.code, &help_command.stdout)
        );
        assert_eq!(help_command.stdout, help_flag.stdout);
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
    fn operational_command_is_an_explicit_side_effect_free_stub() {
        let result = output(&["mlx-air", "serve", "--model", "test-model"]);

        assert_eq!(result.code, ExitCode::ServingStartupOrReadinessFailure);
        assert_eq!(
            result.stderr,
            "error: 'mlx-air serve' is not implemented yet\n"
        );
    }

    #[test]
    fn version_prints_package_version() {
        let result = output(&["mlx-air", "version"]);

        assert_eq!(result.code, ExitCode::NormalTermination);
        assert_eq!(
            result.stdout,
            format!("mlx-air {}\n", env!("CARGO_PKG_VERSION"))
        );
    }
}
