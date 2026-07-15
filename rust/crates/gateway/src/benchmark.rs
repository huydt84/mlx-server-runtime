//! Benchmark-only environment selection and Python command delegation.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::command_runner::{display_command, path_arg, CommandRunner, CommandSpec, OutputMode};
use crate::distribution::{ApplicationPaths, DistributionPaths};
use crate::environment::output_detail;

const BENCHMARK_EXTRA: &str = "bench";
const DISTRIBUTION_VERSION: &str = env!("CARGO_PKG_VERSION");
pub(crate) const GATEWAY_EXECUTABLE_ENV: &str = "MLX_AIR_GATEWAY_EXECUTABLE";
pub(crate) const INVOCATION_DIRECTORY_ENV: &str = "MLX_AIR_INVOCATION_DIRECTORY";
pub(crate) const MLX_AIR_VERSION_ENV: &str = "MLX_AIR_VERSION";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BenchmarkEnvironment {
    pub(crate) root: PathBuf,
    pub(crate) python: PathBuf,
    pub(crate) setup_record: PathBuf,
    pub(crate) lockfile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SetupRecord {
    distribution_version: String,
    lockfile_sha256: String,
    python_executable: PathBuf,
    python_version: String,
    installed_extras: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct BenchmarkError(String);

impl BenchmarkError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BenchmarkError {}

pub(crate) fn select_benchmark_environment(
    distribution: &DistributionPaths,
    application: &ApplicationPaths,
) -> Result<BenchmarkEnvironment, BenchmarkError> {
    let lockfile_sha256 = distribution
        .lockfile_sha256()
        .map_err(|err| BenchmarkError::new(err.to_string()))?;
    let root = application.benchmark_environment(DISTRIBUTION_VERSION, &lockfile_sha256);
    Ok(BenchmarkEnvironment {
        python: root.join("bin/python"),
        setup_record: root.join("setup.json"),
        root,
        lockfile_sha256,
    })
}

pub(crate) fn ensure_benchmark_environment(
    distribution: &DistributionPaths,
    application: &ApplicationPaths,
    runner: &dyn CommandRunner,
) -> Result<BenchmarkEnvironment, BenchmarkError> {
    distribution
        .validate_benchmark_resources()
        .map_err(|err| BenchmarkError::new(err.to_string()))?;
    let environment = select_benchmark_environment(distribution, application)?;
    if setup_is_current(&environment) {
        return Ok(environment);
    }

    fs::create_dir_all(&environment.root).map_err(|err| {
        BenchmarkError::new(format!(
            "failed to create benchmark environment directory {}: {err}",
            environment.root.display()
        ))
    })?;

    eprintln!(
        "Setting up MLX Air benchmark environment at {}",
        environment.root.display()
    );
    let sync = CommandSpec::new("uv")
        .args([
            "--directory".into(),
            path_arg(&distribution.python_project),
            "sync".into(),
            "--frozen".into(),
            "--no-dev".into(),
            "--extra".into(),
            BENCHMARK_EXTRA.into(),
        ])
        .env(
            "UV_PROJECT_ENVIRONMENT",
            environment.root.as_os_str().to_owned(),
        )
        .current_dir(&distribution.root)
        .timeout(Duration::from_secs(1_800))
        .output_mode(OutputMode::Inherit);
    let result = runner
        .run(&sync)
        .map_err(|err| BenchmarkError::new(format!("{}: {err}", display_command(&sync))))?;
    if !result.success {
        return Err(BenchmarkError::new(format!(
            "benchmark environment setup failed: {}",
            display_command(&sync)
        )));
    }
    if !environment.python.is_file() {
        return Err(BenchmarkError::new(format!(
            "uv completed without creating {}",
            environment.python.display()
        )));
    }

    let version_spec = CommandSpec::new(environment.python.as_os_str().to_owned())
        .args(["--version"])
        .timeout(Duration::from_secs(30));
    let version = runner
        .run(&version_spec)
        .map_err(|err| BenchmarkError::new(format!("failed to inspect managed Python: {err}")))?;
    if !version.success {
        return Err(BenchmarkError::new(format!(
            "managed Python version check failed: {}",
            output_detail(&version.stdout, &version.stderr)
        )));
    }

    write_setup_record(
        &environment.setup_record,
        &SetupRecord {
            distribution_version: DISTRIBUTION_VERSION.to_string(),
            lockfile_sha256: environment.lockfile_sha256.clone(),
            python_executable: environment.python.clone(),
            python_version: output_detail(&version.stdout, &version.stderr),
            installed_extras: vec![BENCHMARK_EXTRA.to_string()],
        },
    )?;
    eprintln!("MLX Air benchmark environment is ready");
    Ok(environment)
}

pub(crate) fn benchmark_command(
    distribution: &DistributionPaths,
    environment: &BenchmarkEnvironment,
    invocation_directory: &Path,
    action: &str,
    args: &[OsString],
) -> CommandSpec {
    CommandSpec::new("uv")
        .args(
            [
                OsString::from("--directory"),
                path_arg(&distribution.python_project),
                OsString::from("run"),
                OsString::from("--extra"),
                OsString::from(BENCHMARK_EXTRA),
                OsString::from("python"),
                OsString::from("-m"),
                OsString::from("mlx_benchmark"),
                OsString::from(action),
            ]
            .into_iter()
            .chain(args.iter().cloned()),
        )
        .env(
            "UV_PROJECT_ENVIRONMENT",
            environment.root.as_os_str().to_owned(),
        )
        .env(
            GATEWAY_EXECUTABLE_ENV,
            distribution.gateway_executable.as_os_str().to_owned(),
        )
        .env(
            INVOCATION_DIRECTORY_ENV,
            invocation_directory.as_os_str().to_owned(),
        )
        .env(MLX_AIR_VERSION_ENV, DISTRIBUTION_VERSION)
        .current_dir(&distribution.root)
        .output_mode(OutputMode::Inherit)
}

fn setup_is_current(environment: &BenchmarkEnvironment) -> bool {
    if !environment.python.is_file() {
        return false;
    }
    let Ok(contents) = fs::read(&environment.setup_record) else {
        return false;
    };
    let Ok(record) = serde_json::from_slice::<SetupRecord>(&contents) else {
        return false;
    };
    record.distribution_version == DISTRIBUTION_VERSION
        && record.lockfile_sha256 == environment.lockfile_sha256
        && record.python_executable == environment.python
        && record.installed_extras == [BENCHMARK_EXTRA]
}

fn write_setup_record(path: &Path, record: &SetupRecord) -> Result<(), BenchmarkError> {
    let bytes = serde_json::to_vec_pretty(record).map_err(|err| {
        BenchmarkError::new(format!("failed to serialize benchmark setup record: {err}"))
    })?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes).map_err(|err| {
        BenchmarkError::new(format!(
            "failed to write benchmark setup record {}: {err}",
            temporary.display()
        ))
    })?;
    fs::rename(&temporary, path).map_err(|err| {
        BenchmarkError::new(format!(
            "failed to install benchmark setup record {}: {err}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::ffi::OsStr;

    use super::*;
    use crate::command_runner::{CommandError, CommandResult};
    use crate::distribution::tests::{staged_distribution, temp_path};

    struct FakeRunner {
        calls: RefCell<Vec<CommandSpec>>,
        results: RefCell<VecDeque<CommandResult>>,
    }

    impl FakeRunner {
        fn successful_setup() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                results: RefCell::new(VecDeque::from([
                    CommandResult {
                        success: true,
                        stdout: String::new(),
                        stderr: String::new(),
                    },
                    CommandResult {
                        success: true,
                        stdout: "Python 3.12.1\n".to_string(),
                        stderr: String::new(),
                    },
                ])),
            }
        }
    }

    impl CommandRunner for FakeRunner {
        fn run(&self, spec: &CommandSpec) -> Result<CommandResult, CommandError> {
            self.calls.borrow_mut().push(spec.clone());
            self.results
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| CommandError::new("missing fake command result"))
        }
    }

    #[test]
    fn setup_uses_exact_uv_arguments_and_benchmark_environment() {
        let (distribution_root, distribution) = staged_distribution("benchmark-sync");
        let home = temp_path("benchmark-home");
        let application = ApplicationPaths::from_home(&home);
        let selected = select_benchmark_environment(&distribution, &application).unwrap();
        fs::create_dir_all(selected.python.parent().unwrap()).unwrap();
        fs::write(&selected.python, "python").unwrap();
        let runner = FakeRunner::successful_setup();

        ensure_benchmark_environment(&distribution, &application, &runner).unwrap();

        let calls = runner.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0].args,
            [
                "--directory".into(),
                distribution.python_project.as_os_str().to_owned(),
                "sync".into(),
                "--frozen".into(),
                "--no-dev".into(),
                "--extra".into(),
                "bench".into(),
            ]
        );
        assert_eq!(
            calls[0].env.get(OsStr::new("UV_PROJECT_ENVIRONMENT")),
            Some(&selected.root.as_os_str().to_owned())
        );

        fs::remove_dir_all(distribution_root).unwrap();
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn delegation_uses_exact_uv_arguments_and_distribution_gateway() {
        let (distribution_root, distribution) = staged_distribution("benchmark-run");
        let home = temp_path("benchmark-run-home");
        let application = ApplicationPaths::from_home(&home);
        let selected = select_benchmark_environment(&distribution, &application).unwrap();
        let command = benchmark_command(
            &distribution,
            &selected,
            Path::new("/invocation"),
            "run",
            &["--suite".into(), "smoke".into()],
        );

        assert_eq!(
            command.args,
            [
                "--directory".into(),
                distribution.python_project.as_os_str().to_owned(),
                "run".into(),
                "--extra".into(),
                "bench".into(),
                "python".into(),
                "-m".into(),
                "mlx_benchmark".into(),
                "run".into(),
                "--suite".into(),
                "smoke".into(),
            ]
        );
        assert_eq!(
            command.env.get(OsStr::new(GATEWAY_EXECUTABLE_ENV)),
            Some(&distribution.gateway_executable.as_os_str().to_owned())
        );
        assert_eq!(
            command.env.get(OsStr::new(INVOCATION_DIRECTORY_ENV)),
            Some(&OsString::from("/invocation"))
        );
        assert_eq!(
            command.env.get(OsStr::new(MLX_AIR_VERSION_ENV)),
            Some(&OsString::from(DISTRIBUTION_VERSION))
        );
        assert_eq!(command.current_dir, Some(distribution.root.clone()));
        assert_eq!(command.output_mode, OutputMode::Inherit);

        fs::remove_dir_all(distribution_root).unwrap();
    }

    #[test]
    fn current_benchmark_setup_skips_uv_sync() {
        let (distribution_root, distribution) = staged_distribution("benchmark-current");
        let home = temp_path("benchmark-current-home");
        let application = ApplicationPaths::from_home(&home);
        let selected = select_benchmark_environment(&distribution, &application).unwrap();
        fs::create_dir_all(selected.python.parent().unwrap()).unwrap();
        fs::write(&selected.python, "python").unwrap();
        write_setup_record(
            &selected.setup_record,
            &SetupRecord {
                distribution_version: DISTRIBUTION_VERSION.to_string(),
                lockfile_sha256: selected.lockfile_sha256.clone(),
                python_executable: selected.python.clone(),
                python_version: "Python 3.12.1".to_string(),
                installed_extras: vec![BENCHMARK_EXTRA.to_string()],
            },
        )
        .unwrap();
        let runner = FakeRunner {
            calls: RefCell::new(Vec::new()),
            results: RefCell::new(VecDeque::new()),
        };

        ensure_benchmark_environment(&distribution, &application, &runner).unwrap();

        assert!(runner.calls.borrow().is_empty());
        fs::remove_dir_all(distribution_root).unwrap();
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn setup_rejects_a_distribution_without_benchmark_package() {
        let (distribution_root, distribution) = staged_distribution("benchmark-missing");
        let home = temp_path("benchmark-missing-home");
        fs::remove_dir_all(&distribution.benchmark_package).unwrap();
        let runner = FakeRunner::successful_setup();

        let error = ensure_benchmark_environment(
            &distribution,
            &ApplicationPaths::from_home(&home),
            &runner,
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("missing bundled mlx_benchmark package"));
        assert!(runner.calls.borrow().is_empty());

        fs::remove_dir_all(distribution_root).unwrap();
    }
}
