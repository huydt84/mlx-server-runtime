use std::env;
use std::ffi::OsString;
use std::fs;
use std::net::{TcpListener, TcpStream};
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
        fs::write(
            root.join("bin/mlx_runtime_gateway"),
            include_str!("fixtures/fake_benchmark_gateway.py"),
        )
        .unwrap();
        fs::set_permissions(
            root.join("bin/mlx_runtime_gateway"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        fs::write(root.join("config/runtime.toml"), "[worker]\n").unwrap();
        fs::write(
            root.join("config/benchmark.toml"),
            include_str!("fixtures/phase7_benchmark.toml"),
        )
        .unwrap();
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
        fs::write(
            root.join("python/mlx_benchmark/runner.py"),
            include_str!("../../../../python/mlx_benchmark/runner.py"),
        )
        .unwrap();
        fs::write(
            root.join("python/mlx_benchmark/configuration.py"),
            include_str!("../../../../python/mlx_benchmark/configuration.py"),
        )
        .unwrap();
        fs::write(
            root.join("python/mlx_benchmark/loadgen.py"),
            include_str!("../../../../python/mlx_benchmark/loadgen.py"),
        )
        .unwrap();
        fs::write(
            root.join("python/mlx_benchmark/prompts.py"),
            include_str!("../../../../python/mlx_benchmark/prompts.py"),
        )
        .unwrap();
        fs::write(
            root.join("python/mlx_benchmark/report.py"),
            include_str!("../../../../python/mlx_benchmark/report.py"),
        )
        .unwrap();
        fs::write(
            root.join("python/mlx_benchmark/statistics.py"),
            include_str!("../../../../python/mlx_benchmark/statistics.py"),
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

#[test]
fn missing_bundled_benchmark_config_fails_before_uv_delegation() {
    let staged = StagedCli::new("missing-benchmark-config");
    fs::remove_file(staged.root.join("config/benchmark.toml")).unwrap();

    let output = staged.output(&["bench", "run", "--suite", "smoke"]);

    assert_eq!(output.status.code(), Some(10));
    assert!(stderr(&output).contains("missing bundled default benchmark configuration"));
}

#[test]
fn invalid_configuration_values_report_exact_fields_before_server_startup() {
    let staged = StagedCli::new("invalid-config");
    let pid_file = staged.base.join("invalid-pids");
    let cases = [
        (
            "cache_state = \"cold\"",
            "cache_state = \"missing\"",
            "workloads.sequential_stream.cache_state",
        ),
        (
            "metric_unit = \"ms\"",
            "metric_unit = \"furlongs\"",
            "workloads.sequential_stream.metric_unit",
        ),
        (
            "metric_direction = \"lower\"",
            "metric_direction = \"sideways\"",
            "workloads.sequential_stream.metric_direction",
        ),
        (
            "requests_per_trial = 2",
            "requests_per_trial = 0",
            "workloads.sequential_stream.requests_per_trial",
        ),
        (
            "load_mode = \"sequential\"",
            "load_mode = \"ramp\"",
            "workloads.sequential_stream.load_mode",
        ),
        (
            "[warmup_groups.short]\nprompt_group = \"warmup_short\"\nconcurrency = 1\noutput_tokens = 4",
            "[warmup_groups.short]\nprompt_group = \"warmup_short\"\nconcurrency = 1",
            "warmup_groups.short.output_tokens",
        ),
    ];

    for (index, (original, replacement, field)) in cases.into_iter().enumerate() {
        let invalid = staged.base.join(format!("invalid-{index}.toml"));
        fs::write(
            &invalid,
            include_str!("fixtures/phase7_benchmark.toml").replacen(original, replacement, 1),
        )
        .unwrap();
        let output = staged
            .command()
            .env("FAKE_GATEWAY_PID_FILE", &pid_file)
            .args([
                "bench",
                "run",
                "--suite",
                "smoke",
                "--benchmark-config",
                invalid.to_str().unwrap(),
            ])
            .output()
            .unwrap();

        assert_eq!(
            output.status.code(),
            Some(50),
            "stderr: {}",
            stderr(&output)
        );
        assert!(
            stderr(&output).contains(field),
            "stderr: {}",
            stderr(&output)
        );
    }
    assert!(!pid_file.exists());
}

#[test]
fn warmup_requirements_size_the_self_launched_server() {
    let staged = StagedCli::new("warmup-server-limits");
    let output_directory = staged.base.join("warmup-server-limits-artifacts");
    let config = staged.base.join("warmup-server-limits.toml");
    let configuration = include_str!("fixtures/phase7_benchmark.toml")
        .replacen(
            "count = 2\ntarget_tokens = 16",
            "count = 4\ntarget_tokens = 32",
            1,
        )
        .replacen(
            "concurrency = 1\noutput_tokens = 4",
            "concurrency = 4\noutput_tokens = 9",
            1,
        );
    fs::write(&config, configuration).unwrap();

    let output = staged
        .command()
        .args([
            "bench",
            "run",
            "--suite",
            "smoke",
            "--benchmark-config",
            config.to_str().unwrap(),
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let results = read_results(&output_directory);
    assert_eq!(
        results["server"]["runtime_configuration"]["limits"]["max_active_requests"],
        4
    );
    assert_eq!(
        results["server"]["runtime_configuration"]["limits"]["max_prompt_tokens"],
        320
    );
    assert_eq!(
        results["server"]["runtime_configuration"]["limits"]["max_completion_tokens"],
        9
    );
}

#[test]
fn configuration_order_cannot_schedule_a_runtime_outside_the_suite() {
    let staged = StagedCli::new("order-runtime-outside-suite");
    let config = staged.base.join("order-runtime-outside-suite.toml");
    let configuration = include_str!("fixtures/phase7_benchmark.toml")
        .replacen(
            "[runtime_configurations.serial]",
            "[runtime_configurations.overlap]\nbackend = \"v1\"\nenvironment = {}\n\n[runtime_configurations.serial]",
            1,
        )
        .replacen(
            "runtime_configurations = [\"serial\"]\n\n[tail_sets.none]",
            "runtime_configurations = [\"serial\", \"overlap\"]\n\n[tail_sets.none]",
            1,
        );
    fs::write(&config, configuration).unwrap();

    let output = staged.output(&[
        "bench",
        "run",
        "--suite",
        "smoke",
        "--benchmark-config",
        config.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(50));
    assert!(
        stderr(&output).contains(
            "suites.smoke.configuration_orders: order 'primary' uses runtime configurations outside the suite: ['overlap']"
        ),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn every_scheduled_model_runtime_pair_must_execute_a_workload() {
    let staged = StagedCli::new("scheduled-runtime-without-workload");
    let config = staged.base.join("scheduled-runtime-without-workload.toml");
    let configuration = include_str!("fixtures/phase7_benchmark.toml")
        .replacen(
            "[runtime_configurations.serial]",
            "[runtime_configurations.overlap]\nbackend = \"v1\"\nenvironment = {}\n\n[runtime_configurations.serial]",
            1,
        )
        .replacen(
            "runtime_configurations = [\"serial\"]\n\n[tail_sets.none]",
            "runtime_configurations = [\"serial\", \"overlap\"]\n\n[tail_sets.none]",
            1,
        )
        .replacen(
            "runtime_configurations = [\"serial\"]\nwarmup_groups",
            "runtime_configurations = [\"serial\", \"overlap\"]\nwarmup_groups",
            1,
        )
        .replacen("max_model_starts = 1", "max_model_starts = 2", 1);
    fs::write(&config, configuration).unwrap();

    let output = staged.output(&[
        "bench",
        "run",
        "--suite",
        "smoke",
        "--benchmark-config",
        config.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(50));
    assert!(
        stderr(&output).contains(
            "suites.smoke.coverage[0].workloads: does not assign workloads to scheduled runtime configurations ['overlap']"
        ),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn external_mode_rejects_a_multi_execution_selection() {
    let staged = StagedCli::new("external-multi-execution");
    let config = staged.base.join("external-multi-execution.toml");
    let configuration = include_str!("fixtures/phase7_benchmark.toml")
        .replacen(
            "[runtime_configurations.serial]",
            "[runtime_configurations.overlap]\nbackend = \"v1\"\nenvironment = {}\n\n[runtime_configurations.serial]",
            1,
        )
        .replace(
            "runtime_configurations = [\"serial\"]",
            "runtime_configurations = [\"serial\", \"overlap\"]",
        )
        .replacen("max_model_starts = 1", "max_model_starts = 2", 1);
    fs::write(&config, configuration).unwrap();

    let output = staged.output(&[
        "bench",
        "run",
        "--suite",
        "smoke",
        "--server-mode",
        "external",
        "--base-url",
        "http://127.0.0.1:9",
        "--benchmark-config",
        config.to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(50));
    assert!(
        stderr(&output).contains("external server mode requires exactly one declared execution"),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn self_launched_run_writes_exact_trials_and_reaps_process_group() {
    let staged = StagedCli::new("successful-run");
    let output_directory = staged.base.join("successful-artifacts");
    let pid_file = staged.base.join("successful-pids");

    let output = staged
        .command()
        .env("FAKE_GATEWAY_PID_FILE", &pid_file)
        .args([
            "bench",
            "run",
            "--suite",
            "smoke",
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let results = read_results(&output_directory);
    assert_eq!(results["status"], "succeeded");
    assert_eq!(results["configuration"]["suite"], "smoke");
    assert_eq!(
        results["configuration"]["benchmark_config"],
        staged
            .root
            .join("config/benchmark.toml")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
    );
    assert_eq!(results["configuration"]["sampling"]["seed"], 7);
    assert_eq!(
        results["configuration"]["workloads"][2]["load_mode"],
        "closed-loop"
    );
    assert_eq!(results["trials"].as_array().unwrap().len(), 3);
    assert_eq!(results["trials"][0]["request_count"], 2);
    assert_eq!(results["trials"][1]["request_count"], 3);
    assert_eq!(results["trials"][2]["request_count"], 6);
    assert_eq!(results["trials"][0]["maximum_observed_in_flight"], 1);
    assert_eq!(results["trials"][1]["maximum_observed_in_flight"], 3);
    assert_eq!(results["trials"][2]["maximum_observed_in_flight"], 2);
    assert_eq!(
        results["trials"][0]["submission_policy"],
        "submit-one-after-previous-completes"
    );
    assert_eq!(
        results["trials"][1]["submission_policy"],
        "submit-declared-count-at-once"
    );
    assert_eq!(
        results["trials"][2]["submission_policy"],
        "replace-each-completed-request"
    );
    assert_eq!(all_requests(&results).len(), 11);
    assert!(all_requests(&results).iter().all(|request| {
        request["first_byte_monotonic_ns"].is_number()
            && request["first_token_monotonic_ns"].is_number()
            && request["final_token_monotonic_ns"].is_number()
            && request["completed_monotonic_ns"].is_number()
            && request["prompt_tokens"] == 5
            && request["completion_tokens"] == 2
            && request["total_tokens"] == 7
            && request["output_sha256"].as_str().is_some()
    }));
    assert!(results["trials"][0]["requests"]
        .as_array()
        .unwrap()
        .iter()
        .all(|request| request["streaming"] == true));
    assert!(results["trials"][1]["requests"]
        .as_array()
        .unwrap()
        .iter()
        .all(|request| request["streaming"] == false));
    assert!(output_directory.join("report.md").is_file());
    assert!(output_directory.join("logs/gateway.log").is_file());
    assert!(output_directory.join("logs/worker.log").is_file());
    assert_processes_reaped(&pid_file);
}

#[test]
fn phase8_run_applies_order_cache_state_and_trial_metric_deltas() {
    let staged = StagedCli::new("phase8-state");
    let output_directory = staged.base.join("phase8-artifacts");
    let config = staged.base.join("phase8.toml");
    fs::write(&config, include_str!("fixtures/phase8_benchmark.toml")).unwrap();

    let output = staged
        .command()
        .args([
            "bench",
            "run",
            "--suite",
            "phase8",
            "--benchmark-config",
            config.to_str().unwrap(),
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let results = read_results(&output_directory);
    assert_eq!(results["applied_order"].as_array().unwrap().len(), 2);
    assert_eq!(
        results["applied_order"][0]["runtime_configuration"],
        "serial"
    );
    assert_eq!(
        results["applied_order"][1]["runtime_configuration"],
        "overlap"
    );
    assert_eq!(results["warmups"].as_array().unwrap().len(), 2);
    assert!(results["warmups"]
        .as_array()
        .unwrap()
        .iter()
        .all(|warmup| warmup["measured"] == false));
    assert_eq!(results["trials"].as_array().unwrap().len(), 6);
    for trial in results["trials"].as_array().unwrap() {
        assert_eq!(trial["configuration_order"], "round");
        assert!(trial["runtime_metrics"]
            .as_object()
            .unwrap()
            .contains_key("mlx_requests_total"));
        if trial["workload_name"] == "cold" {
            assert!(trial["requests"]
                .as_array()
                .unwrap()
                .iter()
                .all(|request| request["cached_tokens"] == 0));
        } else if trial["workload_name"] == "shared" {
            assert!(trial["requests"]
                .as_array()
                .unwrap()
                .iter()
                .all(|request| request["cached_tokens"] == 3));
            assert_eq!(metric_delta(trial, "mlx_prefix_cache_hits_by_backend"), 2.0);
        } else if trial["workload_name"] == "pressure" {
            assert_eq!(
                metric_delta(trial, "mlx_prefix_cache_evictions_by_backend"),
                2.0
            );
        }
    }
}

#[test]
fn phase9_run_reports_trial_statistics_configured_tails_and_controlled_delays() {
    let staged = StagedCli::new("phase9-statistics");
    let output_directory = staged.base.join("phase9-artifacts");
    let config = staged.base.join("phase9.toml");
    fs::write(&config, include_str!("fixtures/phase9_benchmark.toml")).unwrap();

    let output = staged
        .command()
        .args([
            "bench",
            "run",
            "--suite",
            "phase9",
            "--benchmark-config",
            config.to_str().unwrap(),
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let results = read_results(&output_directory);
    let primary = results["analysis"]["primary_metrics"].as_array().unwrap();
    assert_eq!(primary.len(), 3);
    assert!(primary.iter().all(|summary| {
        summary["independent_trial_count"] == 2
            && summary["request_count"] == 4
            && summary["bootstrap_95_interval"]["resampling_unit"] == "trial"
    }));
    let gateway = primary
        .iter()
        .find(|summary| summary["workload"] == "gateway_latency")
        .unwrap();
    assert_eq!(gateway["better_direction"], "lower");
    assert!(gateway["mean"].as_f64().unwrap() >= 70.0);
    let prefill = primary
        .iter()
        .find(|summary| summary["workload"] == "prefill_ttft")
        .unwrap();
    assert!(prefill["mean"].as_f64().unwrap() >= 40.0);
    let decode = primary
        .iter()
        .find(|summary| summary["workload"] == "decode_tpot")
        .unwrap();
    assert!(decode["mean"].as_f64().unwrap() >= 35.0);
    let tails = results["analysis"]["tails"].as_array().unwrap();
    assert_eq!(tails.len(), 1);
    assert_eq!(tails[0]["workload"], "prefill_ttft");
    assert_eq!(tails[0]["request_ttft_p95"]["sample_count"], 4);
    assert_eq!(tails[0]["trial_wall_time_p95"]["sample_count"], 2);
    let report = fs::read_to_string(output_directory.join("report.md")).unwrap();
    assert!(report.contains("Lower latency, TTFT, and TPOT are better"));
    assert!(report.contains("Requests within a trial are correlated"));
}

#[test]
fn phase9_calibration_repeats_unchanged_configuration_and_reports_variation() {
    let staged = StagedCli::new("phase9-calibration");
    let config = staged.base.join("phase9.toml");
    let pid_file = staged.base.join("phase9-calibration-pids");
    fs::write(&config, include_str!("fixtures/phase9_benchmark.toml")).unwrap();

    let output = staged
        .command()
        .current_dir(&staged.base)
        .env("FAKE_GATEWAY_PID_FILE", &pid_file)
        .args([
            "bench",
            "calibrate",
            "--suite",
            "phase9",
            "--benchmark-config",
            config.to_str().unwrap(),
            "--repetitions",
            "2",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let result_path = PathBuf::from(stdout(&output).trim());
    let calibration_directory = result_path.parent().unwrap();
    let results = read_results(calibration_directory);
    assert_eq!(results["command"], "calibrate");
    assert_eq!(results["status"], "succeeded");
    assert_eq!(results["repetitions"]["requested"], 2);
    assert_eq!(results["repetitions"]["completed"], 2);
    assert_eq!(results["configuration"]["suite"], "phase9");
    assert_eq!(results["configuration"]["workloads"][0]["trials"], 2);
    assert_eq!(results["runs"].as_array().unwrap().len(), 2);
    assert_eq!(results["host_observations"].as_array().unwrap().len(), 4);
    assert!(results["host_observations"]
        .as_array()
        .unwrap()
        .iter()
        .all(|observation| {
            observation["thermal_state"].is_string()
                && observation["power_state"].is_string()
                && observation["memory_pressure"].is_string()
        }));
    let repeated = results["repeated_measurements"].as_array().unwrap();
    assert_eq!(repeated.len(), 3);
    assert!(repeated.iter().all(|summary| {
        summary["completed_repetition_count"] == 2
            && summary["repetition_values"].as_array().unwrap().len() == 2
            && summary["bootstrap_95_interval"]["resampling_unit"] == "run"
            && summary["run_to_run_range"]["unit"].is_string()
    }));
    assert!(!results["unstable_workloads"].as_array().unwrap().is_empty());
    assert!(calibration_directory.join("report.md").is_file());
    assert_processes_reaped(&pid_file);
}

#[test]
fn repeated_runs_preserve_workload_prompt_trial_and_request_order() {
    let staged = StagedCli::new("deterministic-order");
    let first_directory = staged.base.join("first");
    let second_directory = staged.base.join("second");

    for directory in [&first_directory, &second_directory] {
        let output = staged
            .command()
            .args([
                "bench",
                "run",
                "--suite",
                "smoke",
                "--output-dir",
                directory.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "stderr: {}", stderr(&output));
    }

    let first = request_identity(&read_results(&first_directory));
    let second = request_identity(&read_results(&second_directory));
    assert_eq!(first, second);
}

#[test]
fn request_failure_preserves_measurements_and_reaps_process_group() {
    let staged = StagedCli::new("request-failure");
    let output_directory = staged.base.join("failure-artifacts");
    let pid_file = staged.base.join("failure-pids");

    let output = staged
        .command()
        .env("FAKE_GATEWAY_PID_FILE", &pid_file)
        .env("FAKE_GATEWAY_RUN_MODE", "request-failure")
        .args([
            "bench",
            "run",
            "--suite",
            "smoke",
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(50));
    let results = read_results(&output_directory);
    assert_eq!(results["status"], "failed");
    assert_eq!(results["failure_stage"], "measurement");
    assert_eq!(all_requests(&results).len(), 11);
    assert!(all_requests(&results)
        .iter()
        .all(|request| request["status"] == "failed"));
    assert_processes_reaped(&pid_file);
}

#[test]
fn ambiguous_transport_failure_does_not_duplicate_an_inference_post() {
    let staged = StagedCli::new("no-duplicate-post");
    let output_directory = staged.base.join("no-duplicate-post-artifacts");
    let pid_file = staged.base.join("no-duplicate-post-pids");
    let request_count_file = staged.base.join("wire-request-count");

    let output = staged
        .command()
        .env("FAKE_GATEWAY_PID_FILE", &pid_file)
        .env("FAKE_GATEWAY_REQUEST_COUNT_FILE", &request_count_file)
        .env("FAKE_GATEWAY_RUN_MODE", "drop-first-request")
        .args([
            "bench",
            "run",
            "--suite",
            "smoke",
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(50));
    assert_eq!(fs::read_to_string(request_count_file).unwrap().trim(), "12");
    let results = read_results(&output_directory);
    assert_eq!(all_requests(&results).len(), 11);
    assert_processes_reaped(&pid_file);
}

#[test]
fn sigint_preserves_failure_stage_and_reaps_process_group() {
    let staged = StagedCli::new("interrupt-run");
    let output_directory = staged.base.join("interrupt-artifacts");
    let pid_file = staged.base.join("interrupt-pids");
    let mut child = staged
        .command()
        .env("FAKE_GATEWAY_PID_FILE", &pid_file)
        .env("FAKE_GATEWAY_DELAY_READY", "1")
        .args([
            "bench",
            "run",
            "--suite",
            "smoke",
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_file(&pid_file);

    let result = unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
    assert_eq!(result, 0);
    let status = child.wait().unwrap();

    assert_eq!(status.code(), Some(130));
    let results = read_results(&output_directory);
    assert_eq!(results["status"], "interrupted");
    assert_eq!(results["failure_stage"], "readiness");
    assert!(output_directory.join("logs/gateway.log").is_file());
    assert!(output_directory.join("logs/worker.log").is_file());
    assert_processes_reaped(&pid_file);
}

#[test]
fn external_mode_records_server_identity_without_stopping_server() {
    let staged = StagedCli::new("external-server");
    let output_directory = staged.base.join("external-artifacts");
    let port = reserve_port();
    let config = staged.base.join("external.toml");
    fs::write(
        &config,
        format!(
            "[server]\nport = {port}\n[worker]\nmodel = \"mlx-community/Qwen3-4B-Instruct-2507-4bit\"\n"
        ),
    )
    .unwrap();
    let mut server = Command::new(staged.root.join("bin/mlx_runtime_gateway"))
        .env("MLX_RUNTIME_CONFIG", &config)
        .env("MLX_AIR_BENCHMARK_ENABLED", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_port(port);

    let output = staged
        .command()
        .args([
            "bench",
            "run",
            "--suite",
            "smoke",
            "--server-mode",
            "external",
            "--base-url",
            &format!("http://127.0.0.1:{port}"),
            "--output-dir",
            output_directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(server.try_wait().unwrap().is_none());
    let results = read_results(&output_directory);
    assert_eq!(results["server"]["mode"], "external");
    assert_eq!(results["versions"]["gateway"], "0.1.0");
    unsafe { libc::kill(server.id() as i32, libc::SIGTERM) };
    assert!(server.wait().unwrap().success());
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

fn reserve_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap().port()
}

fn wait_for_port(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for port {port}"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn read_results(directory: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(directory.join("results.json")).unwrap()).unwrap()
}

fn all_requests(results: &serde_json::Value) -> Vec<&serde_json::Value> {
    results["trials"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|trial| trial["requests"].as_array().unwrap())
        .collect()
}

fn request_identity(results: &serde_json::Value) -> Vec<(String, u64, u64, u64, u64, String)> {
    all_requests(results)
        .into_iter()
        .map(|request| {
            (
                request["workload_name"].as_str().unwrap().to_string(),
                request["trial_index"].as_u64().unwrap(),
                request["request_index"].as_u64().unwrap(),
                request["prompt_index"].as_u64().unwrap(),
                request["prompt_target_tokens"].as_u64().unwrap(),
                request["prompt_sha256"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn metric_delta(trial: &serde_json::Value, prefix: &str) -> f64 {
    trial["runtime_metrics"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(name, _)| name.starts_with(prefix))
        .and_then(|(_, value)| value["delta"].as_f64())
        .unwrap()
}

fn assert_processes_reaped(pid_file: &Path) {
    let pids = fs::read_to_string(pid_file)
        .unwrap()
        .lines()
        .map(|line| line.parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(pids.len(), 2);
    for pid in pids {
        let result = unsafe { libc::kill(pid, 0) };
        assert_eq!(result, -1, "process {pid} was not reaped");
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
