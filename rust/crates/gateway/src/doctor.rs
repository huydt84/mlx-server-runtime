use crate::command_runner::{path_arg, CommandRunner, CommandSpec};
use crate::distribution::{ApplicationPaths, DistributionPaths};
use crate::environment::{ensure_runtime_environment, output_detail, RuntimeEnvironment};
use crate::RuntimeConfig;
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::net::TcpListener;
use std::path::Path;
use std::time::Duration;

const REQUIRED_IMPORTS: &str =
    "import mlx.core; import mlx_lm; import mlx_vlm; import mlx_worker; print('imports_ok')";
const METAL_PROBE: &str = "import mlx.core as mx; value = mx.array([1.0]); mx.eval(value); assert value.item() == 1.0; print('metal_ok')";

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlatformInfo<'a> {
    pub(crate) os: &'a str,
    pub(crate) arch: &'a str,
}

impl PlatformInfo<'static> {
    pub(crate) fn current() -> Self {
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DoctorCheck {
    pub(crate) name: &'static str,
    pub(crate) passed: bool,
    pub(crate) detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DoctorReport {
    pub(crate) checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    pub(crate) fn passed(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }

    pub(crate) fn render(&self) -> String {
        let mut rendered = String::new();
        for check in &self.checks {
            let status = if check.passed { "PASS" } else { "FAIL" };
            let _ = writeln!(rendered, "[{status}] {}: {}", check.name, check.detail);
        }
        rendered
    }
}

pub(crate) fn run_doctor(
    distribution: &DistributionPaths,
    application: &ApplicationPaths,
    config: &RuntimeConfig,
    runner: &dyn CommandRunner,
    platform: PlatformInfo<'_>,
) -> DoctorReport {
    let mut checks = Vec::new();
    checks.push(check_platform(platform));
    checks.push(check_macos_version(runner, platform.os));
    let uv = check_uv(runner);
    let uv_available = uv.passed;
    checks.push(uv);
    checks.push(check_writable(
        "runtime directory",
        &application.runtime_environments,
    ));
    checks.push(check_writable("log directory", &application.logs));
    checks.push(check_writable("instance directory", &application.instances));
    checks.push(check_writable("socket directory", &application.sockets));

    let environment = if uv_available {
        match ensure_runtime_environment(distribution, application, runner) {
            Ok(environment) => {
                checks.push(pass(
                    "runtime environment",
                    environment.root.display().to_string(),
                ));
                Some(environment)
            }
            Err(err) => {
                checks.push(fail("runtime environment", err.to_string()));
                None
            }
        }
    } else {
        checks.push(fail(
            "runtime environment",
            "not checked because uv is unavailable",
        ));
        None
    };

    if let Some(environment) = environment {
        checks.push(check_python_imports(runner, &environment));
        checks.push(check_metal(runner, &environment));
    } else {
        checks.push(fail(
            "Python imports",
            "not checked because the runtime environment is unavailable",
        ));
        checks.push(fail(
            "Metal execution",
            "not checked because the runtime environment is unavailable",
        ));
    }
    checks.push(check_port(config.server.port));
    DoctorReport { checks }
}

fn check_platform(platform: PlatformInfo<'_>) -> DoctorCheck {
    if platform.os == "macos" && platform.arch == "aarch64" {
        pass("Apple Silicon", "macOS arm64")
    } else {
        fail(
            "Apple Silicon",
            format!(
                "requires macOS arm64, found {} {}",
                platform.os, platform.arch
            ),
        )
    }
}

fn check_macos_version(runner: &dyn CommandRunner, os: &str) -> DoctorCheck {
    if os != "macos" {
        return fail(
            "macOS version",
            format!("requires macOS 14 or newer, found {os}"),
        );
    }
    let spec = CommandSpec::new("sw_vers")
        .args(["-productVersion"])
        .timeout(Duration::from_secs(10));
    match runner.run(&spec) {
        Ok(result) if result.success => {
            let version = output_detail(&result.stdout, &result.stderr);
            let major = version
                .split('.')
                .next()
                .and_then(|value| value.parse::<u32>().ok());
            match major {
                Some(major) if major >= 14 => pass("macOS version", version),
                Some(_) => fail(
                    "macOS version",
                    format!("requires 14 or newer, found {version}"),
                ),
                None => fail(
                    "macOS version",
                    format!("could not parse version: {version}"),
                ),
            }
        }
        Ok(result) => fail(
            "macOS version",
            output_detail(&result.stdout, &result.stderr),
        ),
        Err(err) => fail("macOS version", err.to_string()),
    }
}

fn check_uv(runner: &dyn CommandRunner) -> DoctorCheck {
    let spec = CommandSpec::new("uv")
        .args(["--version"])
        .timeout(Duration::from_secs(10));
    match runner.run(&spec) {
        Ok(result) if result.success => pass("uv", output_detail(&result.stdout, &result.stderr)),
        Ok(result) => fail("uv", output_detail(&result.stdout, &result.stderr)),
        Err(err) => fail("uv", err.to_string()),
    }
}

fn check_writable(name: &'static str, path: &Path) -> DoctorCheck {
    if let Err(err) = fs::create_dir_all(path) {
        return fail(name, format!("{}: {err}", path.display()));
    }
    let probe = path.join(format!(".mlx-air-write-probe-{}", std::process::id()));
    let result = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .and_then(|mut file| file.write_all(b"ok"));
    match result {
        Ok(()) => match fs::remove_file(&probe) {
            Ok(()) => pass(name, path.display().to_string()),
            Err(err) => fail(name, format!("failed to remove {}: {err}", probe.display())),
        },
        Err(err) => fail(name, format!("{}: {err}", path.display())),
    }
}

fn check_python_imports(
    runner: &dyn CommandRunner,
    environment: &RuntimeEnvironment,
) -> DoctorCheck {
    let spec = CommandSpec::new(path_arg(&environment.python))
        .args(["-c", REQUIRED_IMPORTS])
        .timeout(Duration::from_secs(30));
    match runner.run(&spec) {
        Ok(result) if result.success => pass(
            "Python imports",
            output_detail(&result.stdout, &result.stderr),
        ),
        Ok(result) => fail(
            "Python imports",
            output_detail(&result.stdout, &result.stderr),
        ),
        Err(err) => fail("Python imports", err.to_string()),
    }
}

fn check_metal(runner: &dyn CommandRunner, environment: &RuntimeEnvironment) -> DoctorCheck {
    let spec = CommandSpec::new(path_arg(&environment.python))
        .args(["-c", METAL_PROBE])
        .timeout(Duration::from_secs(30));
    match runner.run(&spec) {
        Ok(result) if result.success => pass(
            "Metal execution",
            output_detail(&result.stdout, &result.stderr),
        ),
        Ok(result) => fail(
            "Metal execution",
            output_detail(&result.stdout, &result.stderr),
        ),
        Err(err) => fail("Metal execution", err.to_string()),
    }
}

fn check_port(port: u16) -> DoctorCheck {
    match TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => {
            drop(listener);
            pass("port availability", format!("127.0.0.1:{port}"))
        }
        Err(err) => fail("port availability", format!("127.0.0.1:{port}: {err}")),
    }
}

fn pass(name: &'static str, detail: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        passed: true,
        detail: detail.into(),
    }
}

fn fail(name: &'static str, detail: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        passed: false,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_runner::{CommandError, CommandResult};
    use crate::distribution::tests::{staged_distribution, temp_path};
    use crate::environment::select_runtime_environment;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    struct FakeRunner {
        results: RefCell<VecDeque<CommandResult>>,
    }

    impl CommandRunner for FakeRunner {
        fn run(&self, _spec: &CommandSpec) -> Result<CommandResult, CommandError> {
            self.results
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| CommandError::new("missing doctor fake result"))
        }
    }

    fn success(output: &str) -> CommandResult {
        CommandResult {
            success: true,
            stdout: format!("{output}\n"),
            stderr: String::new(),
        }
    }

    #[test]
    fn doctor_reports_each_successful_check() {
        let (distribution_root, distribution) = staged_distribution("doctor-success");
        let home = temp_path("doctor-home");
        let application = ApplicationPaths::from_home(&home);
        let selected = select_runtime_environment(&distribution, &application).unwrap();
        fs::create_dir_all(selected.python.parent().unwrap()).unwrap();
        fs::write(&selected.python, "python").unwrap();
        let runner = FakeRunner {
            results: RefCell::new(VecDeque::from([
                success("14.6.1"),
                success("uv 0.8.0"),
                success(""),
                success("Python 3.12.1"),
                success("imports_ok"),
                success("metal_ok"),
            ])),
        };
        let mut config = RuntimeConfig::default();
        config.server.port = 0;

        let report = run_doctor(
            &distribution,
            &application,
            &config,
            &runner,
            PlatformInfo {
                os: "macos",
                arch: "aarch64",
            },
        );

        assert!(report.passed(), "{}", report.render());
        assert_eq!(report.checks.len(), 11);
        fs::remove_dir_all(distribution_root).unwrap();
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn doctor_returns_failure_when_platform_is_unsupported() {
        let (distribution_root, distribution) = staged_distribution("doctor-platform");
        let home = temp_path("doctor-platform-home");
        let application = ApplicationPaths::from_home(&home);
        let runner = FakeRunner {
            results: RefCell::new(VecDeque::from([success("uv 0.8.0")])),
        };

        let report = run_doctor(
            &distribution,
            &application,
            &RuntimeConfig::default(),
            &runner,
            PlatformInfo {
                os: "linux",
                arch: "x86_64",
            },
        );

        assert!(!report.passed());
        assert!(report.render().contains("[FAIL] Apple Silicon"));
        fs::remove_dir_all(distribution_root).unwrap();
        fs::remove_dir_all(home).unwrap();
    }
}
