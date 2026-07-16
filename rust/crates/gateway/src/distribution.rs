use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fmt;
use std::fs;
use std::io::Read as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DistributionPaths {
    pub(crate) root: PathBuf,
    pub(crate) python_project: PathBuf,
    pub(crate) pyproject: PathBuf,
    pub(crate) uv_lock: PathBuf,
    pub(crate) python_package: PathBuf,
    pub(crate) benchmark_package: PathBuf,
    pub(crate) benchmark_config: PathBuf,
    pub(crate) default_config: PathBuf,
    pub(crate) licenses: PathBuf,
    pub(crate) version_metadata: PathBuf,
    pub(crate) layout_metadata: PathBuf,
    pub(crate) gateway_executable: PathBuf,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct LayoutMetadata {
    schema_version: u32,
    distribution: String,
    version: String,
    platform: String,
    paths: LayoutPaths,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct LayoutPaths {
    cli: String,
    gateway: String,
    python_project: String,
    runtime_config: String,
    benchmark_config: String,
    licenses: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApplicationPaths {
    pub(crate) root: PathBuf,
    pub(crate) runtime_environments: PathBuf,
    pub(crate) benchmark_environments: PathBuf,
    pub(crate) instances: PathBuf,
    pub(crate) logs: PathBuf,
    pub(crate) sockets: PathBuf,
}

#[derive(Debug)]
pub(crate) struct PathError(String);

impl PathError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PathError {}

impl DistributionPaths {
    pub(crate) fn from_current_executable() -> Result<Self, PathError> {
        let executable = env::current_exe().map_err(|err| {
            PathError::new(format!("failed to resolve current executable: {err}"))
        })?;
        Self::from_executable(&executable)
    }

    pub(crate) fn from_executable(executable: &Path) -> Result<Self, PathError> {
        let bin_dir = executable.parent().ok_or_else(|| {
            PathError::new(format!(
                "executable has no parent directory: {}",
                executable.display()
            ))
        })?;
        let root = bin_dir.parent().ok_or_else(|| {
            PathError::new(format!(
                "executable is not inside a distribution bin directory: {}",
                executable.display()
            ))
        })?;

        let paths = Self {
            root: root.to_path_buf(),
            python_project: root.join("python"),
            pyproject: root.join("python/pyproject.toml"),
            uv_lock: root.join("python/uv.lock"),
            python_package: root.join("python/mlx_worker"),
            benchmark_package: root.join("python/mlx_benchmark"),
            benchmark_config: root.join("config/benchmark.toml"),
            default_config: root.join("config/runtime.toml"),
            licenses: root.join("licenses"),
            version_metadata: root.join("metadata/version.txt"),
            layout_metadata: root.join("metadata/layout.json"),
            gateway_executable: bin_dir.join("mlx_runtime_gateway"),
        };
        paths.validate()?;
        Ok(paths)
    }

    fn validate(&self) -> Result<(), PathError> {
        require_directory(&self.python_project, "bundled Python project")?;
        require_file(&self.pyproject, "bundled pyproject.toml")?;
        require_file(&self.uv_lock, "bundled uv.lock")?;
        require_directory(&self.python_package, "bundled mlx_worker package")?;
        require_file(&self.default_config, "bundled default configuration")?;
        require_directory(&self.licenses, "bundled licenses")?;
        require_file(&self.version_metadata, "bundled version metadata")?;
        require_file(&self.layout_metadata, "bundled layout metadata")?;
        require_file(&self.gateway_executable, "sibling mlx_runtime_gateway")?;

        let has_license = fs::read_dir(&self.licenses)
            .map_err(|err| {
                PathError::new(format!(
                    "failed to inspect bundled licenses at {}: {err}",
                    self.licenses.display()
                ))
            })?
            .filter_map(Result::ok)
            .any(|entry| entry.path().is_file());
        if !has_license {
            return Err(PathError::new(format!(
                "bundled licenses directory is empty: {}",
                self.licenses.display()
            )));
        }
        self.validate_metadata()?;
        Ok(())
    }

    fn validate_metadata(&self) -> Result<(), PathError> {
        let version = fs::read_to_string(&self.version_metadata).map_err(|err| {
            PathError::new(format!(
                "failed to read bundled version metadata at {}: {err}",
                self.version_metadata.display()
            ))
        })?;
        if version.trim() != env!("CARGO_PKG_VERSION") {
            return Err(PathError::new(format!(
                "bundled version metadata {} does not match executable version {}",
                version.trim(),
                env!("CARGO_PKG_VERSION")
            )));
        }

        let layout_bytes = fs::read(&self.layout_metadata).map_err(|err| {
            PathError::new(format!(
                "failed to read bundled layout metadata at {}: {err}",
                self.layout_metadata.display()
            ))
        })?;
        let layout: LayoutMetadata = serde_json::from_slice(&layout_bytes).map_err(|err| {
            PathError::new(format!(
                "failed to parse bundled layout metadata at {}: {err}",
                self.layout_metadata.display()
            ))
        })?;
        let expected = LayoutMetadata {
            schema_version: 1,
            distribution: "mlx-air".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            platform: "darwin-arm64".to_string(),
            paths: LayoutPaths {
                cli: "bin/mlx-air".to_string(),
                gateway: "bin/mlx_runtime_gateway".to_string(),
                python_project: "python".to_string(),
                runtime_config: "config/runtime.toml".to_string(),
                benchmark_config: "config/benchmark.toml".to_string(),
                licenses: "licenses".to_string(),
            },
        };
        if layout != expected {
            return Err(PathError::new(format!(
                "bundled layout metadata does not match executable layout: {}",
                self.layout_metadata.display()
            )));
        }
        Ok(())
    }

    pub(crate) fn lockfile_sha256(&self) -> Result<String, PathError> {
        let bytes = fs::read(&self.uv_lock).map_err(|err| {
            PathError::new(format!(
                "failed to read bundled uv.lock at {}: {err}",
                self.uv_lock.display()
            ))
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub(crate) fn validate_benchmark_resources(&self) -> Result<(), PathError> {
        require_directory(&self.benchmark_package, "bundled mlx_benchmark package")?;
        require_file(
            &self.benchmark_config,
            "bundled default benchmark configuration",
        )
    }
}

impl ApplicationPaths {
    pub(crate) fn from_environment() -> Result<Self, PathError> {
        let home = env::var_os("HOME")
            .ok_or_else(|| PathError::new("HOME is not set; cannot resolve MLX Air paths"))?;
        Ok(Self::from_home(Path::new(&home)))
    }

    pub(crate) fn from_home(home: &Path) -> Self {
        let root = home.join("Library/Application Support/mlx-air");
        Self {
            runtime_environments: root.join("environments/runtime"),
            benchmark_environments: root.join("environments/benchmark"),
            instances: root.join("instances"),
            logs: home.join("Library/Logs/mlx-air"),
            sockets: PathBuf::from(format!("/tmp/mlx-air-{}", effective_user_id())),
            root,
        }
    }

    pub(crate) fn runtime_environment(&self, version: &str, lock_hash: &str) -> PathBuf {
        self.runtime_environments.join(version).join(lock_hash)
    }

    pub(crate) fn benchmark_environment(&self, version: &str, lock_hash: &str) -> PathBuf {
        self.benchmark_environments.join(version).join(lock_hash)
    }

    pub(crate) fn create_foreground_socket_path(&self) -> Result<PathBuf, PathError> {
        fs::create_dir_all(&self.sockets).map_err(|err| {
            PathError::new(format!(
                "failed to create private socket directory {}: {err}",
                self.sockets.display()
            ))
        })?;
        fs::set_permissions(&self.sockets, fs::Permissions::from_mode(0o700)).map_err(|err| {
            PathError::new(format!(
                "failed to secure socket directory {}: {err}",
                self.sockets.display()
            ))
        })?;

        let mut random = [0_u8; 8];
        fs::File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut random))
            .map_err(|err| PathError::new(format!("failed to generate socket suffix: {err}")))?;
        let suffix = u64::from_ne_bytes(random);
        Ok(self.sockets.join(format!(
            "foreground-{}-{suffix:016x}.sock",
            std::process::id()
        )))
    }
}

fn require_file(path: &Path, label: &str) -> Result<(), PathError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(PathError::new(format!(
            "missing {label}: {}",
            path.display()
        )))
    }
}

fn require_directory(path: &Path, label: &str) -> Result<(), PathError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(PathError::new(format!(
            "missing {label}: {}",
            path.display()
        )))
    }
}

#[cfg(unix)]
pub(crate) fn effective_user_id() -> u32 {
    // SAFETY: `geteuid` takes no pointers and has no preconditions.
    unsafe { libc::geteuid() }
}

#[cfg(not(unix))]
pub(crate) fn effective_user_id() -> u32 {
    0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(crate) fn temp_path(label: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "mlx-air-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    pub(crate) fn staged_distribution(label: &str) -> (PathBuf, DistributionPaths) {
        let root = temp_path(label);
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir_all(root.join("python/mlx_worker")).unwrap();
        fs::create_dir_all(root.join("python/mlx_benchmark")).unwrap();
        fs::create_dir_all(root.join("config")).unwrap();
        fs::create_dir_all(root.join("licenses")).unwrap();
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(root.join("bin/mlx-air"), "binary").unwrap();
        fs::write(root.join("bin/mlx_runtime_gateway"), "binary").unwrap();
        fs::write(root.join("python/pyproject.toml"), "[project]\n").unwrap();
        fs::write(root.join("python/uv.lock"), "lock").unwrap();
        fs::write(root.join("python/mlx_worker/__init__.py"), "").unwrap();
        fs::write(root.join("python/mlx_benchmark/__init__.py"), "").unwrap();
        fs::write(root.join("config/benchmark.toml"), "schema_version = 1\n").unwrap();
        fs::write(
            root.join("config/runtime.toml"),
            "[worker]\nmodel = \"default-model\"\n",
        )
        .unwrap();
        fs::write(root.join("licenses/LICENSE"), "license").unwrap();
        fs::write(
            root.join("metadata/version.txt"),
            concat!(env!("CARGO_PKG_VERSION"), "\n"),
        )
        .unwrap();
        fs::write(
            root.join("metadata/layout.json"),
            format!(
                concat!(
                    "{{\n",
                    "  \"distribution\": \"mlx-air\",\n",
                    "  \"paths\": {{\n",
                    "    \"benchmark_config\": \"config/benchmark.toml\",\n",
                    "    \"cli\": \"bin/mlx-air\",\n",
                    "    \"gateway\": \"bin/mlx_runtime_gateway\",\n",
                    "    \"licenses\": \"licenses\",\n",
                    "    \"python_project\": \"python\",\n",
                    "    \"runtime_config\": \"config/runtime.toml\"\n",
                    "  }},\n",
                    "  \"platform\": \"darwin-arm64\",\n",
                    "  \"schema_version\": 1,\n",
                    "  \"version\": \"{}\"\n",
                    "}}\n"
                ),
                env!("CARGO_PKG_VERSION")
            ),
        )
        .unwrap();
        let paths = DistributionPaths::from_executable(&root.join("bin/mlx-air")).unwrap();
        (root, paths)
    }

    #[test]
    fn resolver_finds_every_resource_relative_to_executable() {
        let (root, paths) = staged_distribution("paths-complete");

        assert_eq!(paths.root, root);
        assert_eq!(
            paths.gateway_executable,
            root.join("bin/mlx_runtime_gateway")
        );
        assert_eq!(paths.python_package, root.join("python/mlx_worker"));
        assert_eq!(paths.benchmark_package, root.join("python/mlx_benchmark"));
        assert_eq!(paths.benchmark_config, root.join("config/benchmark.toml"));
        assert_eq!(paths.default_config, root.join("config/runtime.toml"));
        assert_eq!(paths.version_metadata, root.join("metadata/version.txt"));
        assert_eq!(paths.layout_metadata, root.join("metadata/layout.json"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolver_rejects_missing_bundled_resource_without_fallback() {
        let (root, _) = staged_distribution("paths-missing");
        fs::remove_file(root.join("python/uv.lock")).unwrap();

        let error = DistributionPaths::from_executable(&root.join("bin/mlx-air")).unwrap_err();

        assert!(error.to_string().contains("missing bundled uv.lock"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolver_rejects_version_metadata_that_disagrees_with_executable() {
        let (root, _) = staged_distribution("version-mismatch");
        fs::write(root.join("metadata/version.txt"), "9.9.9\n").unwrap();

        let error = DistributionPaths::from_executable(&root.join("bin/mlx-air")).unwrap_err();

        assert!(error
            .to_string()
            .contains("does not match executable version"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn runtime_environment_path_uses_version_and_lock_hash() {
        let home = temp_path("application-paths");
        let paths = ApplicationPaths::from_home(&home);

        assert_eq!(
            paths.runtime_environment("1.2.3", "abc123"),
            home.join("Library/Application Support/mlx-air/environments/runtime/1.2.3/abc123")
        );
    }

    #[test]
    fn benchmark_environment_path_uses_version_and_lock_hash() {
        let home = temp_path("benchmark-application-paths");
        let paths = ApplicationPaths::from_home(&home);

        assert_eq!(
            paths.benchmark_environment("1.2.3", "abc123"),
            home.join("Library/Application Support/mlx-air/environments/benchmark/1.2.3/abc123")
        );
    }

    #[test]
    fn foreground_socket_path_is_unique_and_directory_is_private() {
        let home = temp_path("foreground-socket");
        let mut paths = ApplicationPaths::from_home(&home);
        paths.sockets = home.join("sockets");

        let first = paths.create_foreground_socket_path().unwrap();
        let second = paths.create_foreground_socket_path().unwrap();
        let mode = fs::metadata(&paths.sockets).unwrap().permissions().mode() & 0o777;

        assert_ne!(first, second);
        assert_eq!(mode, 0o700);
        assert!(first.starts_with(&paths.sockets));
        fs::remove_dir_all(home).unwrap();
    }
}
