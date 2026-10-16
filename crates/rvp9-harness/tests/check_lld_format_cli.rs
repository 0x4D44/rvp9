#[path = "../src/bounded_child.rs"]
mod bounded_child;

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static TEST_DIR_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        loop {
            let sequence = TEST_DIR_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "rvp9-check-lld-format-{label}-{}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("could not create test dir {}: {error}", path.display()),
            }
        }
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct CheckerOutput {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_checker(dir: &Path, capture_dir: &Path) -> CheckerOutput {
    let stdout_path = capture_dir.join("stdout.log");
    let stderr_path = capture_dir.join("stderr.log");
    let mut command = Command::new(env!("CARGO_BIN_EXE_check-lld-format"));
    command
        .arg(dir)
        .stdout(Stdio::from(File::create(&stdout_path).unwrap()))
        .stderr(Stdio::from(File::create(&stderr_path).unwrap()));
    let status = bounded_child::run_with_timeout(&mut command, Duration::from_secs(5))
        .expect("check-lld-format should finish within the test deadline");

    CheckerOutput {
        status,
        stdout: fs::read_to_string(stdout_path).unwrap(),
        stderr: fs::read_to_string(stderr_path).unwrap(),
    }
}

fn valid_lld() -> &'static str {
    "## 1. Purpose\n\
     ## 2. libvpx origin\n\
     ## 3. Public API\n\
     ## 4. Types\n\
     ## 5. Internal algorithm\n\
     ## 6. SIMD dispatch\n\
     ## 7. Deliberate divergences\n\
     **Where:** src/example.rs:12\n\
     ## 8. Open questions\n\
     ## 9. Test strategy\n"
}

#[test]
fn unreadable_only_input_fails_and_names_the_file() {
    let temp = TestDir::new("unreadable-only");
    let bad = temp.path().join("bad.md");
    fs::write(&bad, [0xff]).unwrap();

    let output = run_checker(temp.path(), temp.path());

    assert_eq!(output.status.code(), Some(1), "stderr: {}", output.stderr);
    assert!(
        output.stderr.contains("bad.md"),
        "stderr: {}",
        output.stderr
    );
}

#[test]
fn unreadable_input_fails_while_valid_report_is_preserved() {
    let temp = TestDir::new("mixed-inputs");
    fs::write(temp.path().join("valid.md"), valid_lld()).unwrap();
    fs::write(temp.path().join("bad.md"), [0xff]).unwrap();

    let output = run_checker(temp.path(), temp.path());

    assert_eq!(output.status.code(), Some(1), "stderr: {}", output.stderr);
    assert!(
        output.stdout.contains("valid.md"),
        "stdout: {}",
        output.stdout
    );
    assert!(
        output.stderr.contains("bad.md"),
        "stderr: {}",
        output.stderr
    );
}

#[test]
fn valid_only_input_keeps_success_and_report_shape() {
    let temp = TestDir::new("valid-only");
    fs::write(temp.path().join("valid.md"), valid_lld()).unwrap();

    let output = run_checker(temp.path(), temp.path());

    assert!(output.status.success(), "stderr: {}", output.stderr);
    assert!(
        output.stdout.contains("valid.md"),
        "stdout: {}",
        output.stdout
    );
    assert!(
        output.stdout.contains(
            "summary: 1 file(s); 1 structurally OK, 0 with missing sections, 0 total TBDs"
        ),
        "stdout: {}",
        output.stdout
    );
}
