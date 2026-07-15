use crate::command_runner::{display_command, path_arg, CommandRunner, CommandSpec, OutputMode};
use crate::distribution::{ApplicationPaths, DistributionPaths};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

const DISTRIBUTION_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeEnvironment {
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
    project_install_mode: String,
    installed_dependency_groups: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct EnvironmentError(String);

impl EnvironmentError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EnvironmentError {}

pub(crate) fn select_runtime_environment(
    distribution: &DistributionPaths,
    application: &ApplicationPaths,
) -> Result<RuntimeEnvironment, EnvironmentError> {
    let lockfile_sha256 = distribution
        .lockfile_sha256()
        .map_err(|err| EnvironmentError::new(err.to_string()))?;
    let root = application.runtime_environment(DISTRIBUTION_VERSION, &lockfile_sha256);
    Ok(RuntimeEnvironment {
        python: root.join("bin/python"),
        setup_record: root.join("setup.json"),
        root,
        lockfile_sha256,
    })
}

pub(crate) fn ensure_runtime_environment(
    distribution: &DistributionPaths,
    application: &ApplicationPaths,
    runner: &dyn CommandRunner,
) -> Result<RuntimeEnvironment, EnvironmentError> {
    let environment = select_runtime_environment(distribution, application)?;
    if setup_is_current(&environment) {
        return Ok(environment);
    }

    fs::create_dir_all(&environment.root).map_err(|err| {
        EnvironmentError::new(format!(
            "failed to create runtime environment directory {}: {err}",
            environment.root.display()
        ))
    })?;

    eprintln!(
        "Setting up MLX Air runtime environment at {}",
        environment.root.display()
    );
    let sync = CommandSpec::new("uv")
        .args([
            "--directory".into(),
            path_arg(&distribution.python_project),
            "sync".into(),
            "--frozen".into(),
            "--no-dev".into(),
            "--no-editable".into(),
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
        .map_err(|err| EnvironmentError::new(format!("{}: {err}", display_command(&sync))))?;
    if !result.success {
        return Err(EnvironmentError::new(format!(
            "runtime environment setup failed: {}",
            display_command(&sync)
        )));
    }
    if !environment.python.is_file() {
        return Err(EnvironmentError::new(format!(
            "uv completed without creating {}",
            environment.python.display()
        )));
    }

    let version_spec = CommandSpec::new(environment.python.as_os_str().to_owned())
        .args(["--version"])
        .timeout(Duration::from_secs(30));
    let version = runner
        .run(&version_spec)
        .map_err(|err| EnvironmentError::new(format!("failed to inspect managed Python: {err}")))?;
    if !version.success {
        return Err(EnvironmentError::new(format!(
            "managed Python version check failed: {}",
            output_detail(&version.stdout, &version.stderr)
        )));
    }
    let record = SetupRecord {
        distribution_version: DISTRIBUTION_VERSION.to_string(),
        lockfile_sha256: environment.lockfile_sha256.clone(),
        python_executable: environment.python.clone(),
        python_version: output_detail(&version.stdout, &version.stderr),
        project_install_mode: "non-editable".to_string(),
        installed_dependency_groups: Vec::new(),
    };
    write_setup_record(&environment.setup_record, &record)?;
    eprintln!("MLX Air runtime environment is ready");
    Ok(environment)
}

fn setup_is_current(environment: &RuntimeEnvironment) -> bool {
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
        && record.project_install_mode == "non-editable"
        && record.installed_dependency_groups.is_empty()
}

fn write_setup_record(path: &Path, record: &SetupRecord) -> Result<(), EnvironmentError> {
    let bytes = serde_json::to_vec_pretty(record)
        .map_err(|err| EnvironmentError::new(format!("failed to serialize setup record: {err}")))?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes).map_err(|err| {
        EnvironmentError::new(format!(
            "failed to write setup record {}: {err}",
            temporary.display()
        ))
    })?;
    fs::rename(&temporary, path).map_err(|err| {
        EnvironmentError::new(format!(
            "failed to install setup record {}: {err}",
            path.display()
        ))
    })
}

pub(crate) fn output_detail(stdout: &str, stderr: &str) -> String {
    let stdout = stdout.trim();
    let stderr = stderr.trim();
    match (stdout.is_empty(), stderr.is_empty()) {
        (false, _) => stdout.to_string(),
        (true, false) => stderr.to_string(),
        (true, true) => "no output".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_runner::{CommandError, CommandResult};
    use crate::distribution::tests::{staged_distribution, temp_path};
    use std::cell::RefCell;
    use std::collections::VecDeque;

    struct FakeRunner {
        calls: RefCell<Vec<CommandSpec>>,
        results: RefCell<VecDeque<CommandResult>>,
    }

    impl FakeRunner {
        fn successful() -> Self {
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
    fn setup_uses_exact_uv_arguments_and_runtime_environment() {
        let (distribution_root, distribution) = staged_distribution("environment-sync");
        let home = temp_path("environment-home");
        let application = ApplicationPaths::from_home(&home);
        let selected = select_runtime_environment(&distribution, &application).unwrap();
        fs::create_dir_all(selected.python.parent().unwrap()).unwrap();
        fs::write(&selected.python, "python").unwrap();
        let runner = FakeRunner::successful();

        ensure_runtime_environment(&distribution, &application, &runner).unwrap();

        let calls = runner.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].program, "uv");
        assert_eq!(
            calls[0].args,
            [
                "--directory".into(),
                distribution.python_project.as_os_str().to_owned(),
                "sync".into(),
                "--frozen".into(),
                "--no-dev".into(),
                "--no-editable".into(),
            ]
        );
        assert_eq!(
            calls[0]
                .env
                .get(std::ffi::OsStr::new("UV_PROJECT_ENVIRONMENT")),
            Some(&selected.root.as_os_str().to_owned())
        );
        assert_eq!(calls[0].current_dir, Some(distribution.root.clone()));
        assert!(!calls[0].args.iter().any(|arg| arg == "bench"));

        fs::remove_dir_all(distribution_root).unwrap();
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn current_setup_record_skips_all_commands() {
        let (distribution_root, distribution) = staged_distribution("environment-current");
        let home = temp_path("environment-current-home");
        let application = ApplicationPaths::from_home(&home);
        let selected = select_runtime_environment(&distribution, &application).unwrap();
        fs::create_dir_all(selected.python.parent().unwrap()).unwrap();
        fs::write(&selected.python, "python").unwrap();
        write_setup_record(
            &selected.setup_record,
            &SetupRecord {
                distribution_version: DISTRIBUTION_VERSION.to_string(),
                lockfile_sha256: selected.lockfile_sha256.clone(),
                python_executable: selected.python.clone(),
                python_version: "Python 3.12.1".to_string(),
                project_install_mode: "non-editable".to_string(),
                installed_dependency_groups: Vec::new(),
            },
        )
        .unwrap();
        let runner = FakeRunner {
            calls: RefCell::new(Vec::new()),
            results: RefCell::new(VecDeque::new()),
        };

        ensure_runtime_environment(&distribution, &application, &runner).unwrap();

        assert!(runner.calls.borrow().is_empty());
        fs::remove_dir_all(distribution_root).unwrap();
        fs::remove_dir_all(home).unwrap();
    }
}
