use std::env;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct StagedCli {
    base: PathBuf,
    root: PathBuf,
    home: PathBuf,
    fake_bin: PathBuf,
    executable: PathBuf,
}

impl StagedCli {
    fn new(label: &str) -> Self {
        let base = temp_path(label);
        let root = base.join("distribution");
        let home = base.join("home");
        let fake_bin = base.join("fake-bin");
        let executable = root.join("bin/mlx-air");

        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir_all(root.join("config")).unwrap();
        fs::create_dir_all(root.join("licenses")).unwrap();
        fs::create_dir_all(root.join("python/mlx_worker")).unwrap();
        fs::create_dir_all(root.join("python/mlx_benchmark")).unwrap();
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&fake_bin).unwrap();

        fs::copy(env!("CARGO_BIN_EXE_mlx-air"), &executable).unwrap();
        fs::write(root.join("bin/mlx_runtime_gateway"), "gateway").unwrap();
        fs::write(root.join("config/runtime.toml"), "[worker]\n").unwrap();
        fs::write(root.join("licenses/LICENSE"), "license").unwrap();
        fs::write(
            root.join("python/pyproject.toml"),
            "[project]\nname = \"mlx-worker\"\nversion = \"0.1.0\"\nrequires-python = \">=3.10\"\n[project.optional-dependencies]\nbench = []\n",
        )
        .unwrap();
        fs::write(root.join("python/uv.lock"), "lock").unwrap();
        fs::write(root.join("python/mlx_worker/__init__.py"), "").unwrap();
        fs::write(root.join("python/mlx_benchmark/__init__.py"), "").unwrap();
        fs::write(
            root.join("python/mlx_benchmark/__main__.py"),
            include_str!("../../../../python/mlx_benchmark/__main__.py"),
        )
        .unwrap();

        let fake_uv = fake_bin.join("uv");
        fs::write(&fake_uv, fake_uv_script()).unwrap();
        fs::set_permissions(&fake_uv, fs::Permissions::from_mode(0o755)).unwrap();

        Self {
            base,
            root,
            home,
            fake_bin,
            executable,
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command.env("HOME", &self.home).env("PATH", self.path());
        command
    }

    fn output(&self, args: &[&str]) -> Output {
        self.command().args(args).output().unwrap()
    }

    fn path(&self) -> OsString {
        let mut paths = vec![self.fake_bin.clone()];
        if let Some(current) = env::var_os("PATH") {
            paths.extend(env::split_paths(&current));
        }
        env::join_paths(paths).unwrap()
    }
}

impl Drop for StagedCli {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

#[test]
fn rust_delegation_reaches_python_leaf_help() {
    let staged = StagedCli::new("leaf-help");

    let output = staged.output(&["bench", "run", "--help"]);

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("--suite"));
}

#[test]
fn external_server_mode_requires_base_url() {
    let staged = StagedCli::new("external-requires-url");

    let output = staged.output(&[
        "bench",
        "run",
        "--suite",
        "smoke",
        "--server-mode",
        "external",
    ]);

    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--base-url is required with --server-mode external"));
}

#[test]
fn self_launched_server_mode_rejects_base_url() {
    let staged = StagedCli::new("self-launched-rejects-url");

    let output = staged.output(&[
        "bench",
        "run",
        "--suite",
        "smoke",
        "--base-url",
        "http://127.0.0.1:9000",
    ]);

    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--base-url is not valid with --server-mode self-launched"));
}

#[test]
fn delegated_uv_exit_status_becomes_cli_exit_status() {
    let staged = StagedCli::new("exit-status");

    let output = staged
        .command()
        .env("FAKE_UV_RUN_MODE", "exit-37")
        .args(["bench", "run", "--suite", "smoke"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(37));
}

#[test]
fn delegated_process_receives_the_cli_signal() {
    let staged = StagedCli::new("signal");
    let ready = staged.base.join("uv-ready");
    let mut child = staged
        .command()
        .env("FAKE_UV_RUN_MODE", "sleep")
        .env("FAKE_UV_READY_FILE", &ready)
        .args(["bench", "run", "--suite", "smoke"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_file(&ready);

    let result = unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    assert_eq!(result, 0);
    let status = child.wait().unwrap();

    assert_eq!(status.signal(), Some(libc::SIGTERM));
}

#[test]
fn missing_sibling_gateway_fails_before_uv_delegation() {
    let staged = StagedCli::new("missing-gateway");
    fs::remove_file(staged.root.join("bin/mlx_runtime_gateway")).unwrap();

    let output = staged.output(&["bench", "run", "--suite", "smoke"]);

    assert_eq!(output.status.code(), Some(10));
    assert!(stderr(&output).contains("missing sibling mlx_runtime_gateway"));
}

#[test]
fn missing_bundled_python_project_fails_before_uv_delegation() {
    let staged = StagedCli::new("missing-python-project");
    fs::remove_dir_all(staged.root.join("python")).unwrap();

    let output = staged.output(&["bench", "run", "--suite", "smoke"]);

    assert_eq!(output.status.code(), Some(10));
    assert!(stderr(&output).contains("missing bundled Python project"));
}

fn temp_path(label: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    env::temp_dir().join(format!(
        "mlx-air-bench-delegation-{label}-{}-{nanos}-{sequence}",
        std::process::id()
    ))
}

fn fake_uv_script() -> &'static str {
    r#"#!/bin/sh
set -eu
project=""
action=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --directory)
            project="$2"
            shift 2
            ;;
        sync|run)
            action="$1"
            shift
            break
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$action" = "sync" ]; then
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
    ln -sf "$(command -v python3)" "$UV_PROJECT_ENVIRONMENT/bin/python"
    exit 0
fi

case "${FAKE_UV_RUN_MODE:-python}" in
    exit-37)
        exit 37
        ;;
    sleep)
        : > "$FAKE_UV_READY_FILE"
        exec /bin/sleep 60
        ;;
esac

if [ "$1" = "--extra" ]; then
    shift 2
fi
if [ "$1" = "python" ]; then
    shift
fi
export PYTHONPATH="$project"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"
"#
}

fn wait_for_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !path.is_file() {
        assert!(Instant::now() < deadline, "timed out waiting for fake uv");
        thread::sleep(Duration::from_millis(20));
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
