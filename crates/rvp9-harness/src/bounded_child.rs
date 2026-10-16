//! Shared bounded child-process support for diagnostic harness binaries.

use std::io;
use std::process::{Child, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};

const REAP_TIMEOUT: Duration = Duration::from_secs(1);
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Spawn a vpxenc child with stdin closed and enforce a finite deadline.
///
/// On timeout the child is killed and polled for reaping for at most one
/// second. The function never blocks indefinitely in `wait`.
pub fn run_with_timeout(command: &mut Command, timeout: Duration) -> Result<ExitStatus, String> {
    command.stdin(std::process::Stdio::null());
    let mut child = command
        .spawn()
        .map_err(|error| format!("vpxenc spawn: {error}"))?;
    let deadline = Instant::now() + timeout;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                let reap = kill_and_reap(&mut child);
                return Err(match reap {
                    Ok(()) => format!("vpxenc timed out after {:.1}s", timeout.as_secs_f64()),
                    Err(ReapError::Kill(error)) => format!(
                        "vpxenc timed out after {:.1}s and could not be killed (pid {}): {error}",
                        timeout.as_secs_f64(),
                        child.id()
                    ),
                    Err(ReapError::Reap(error)) => format!(
                        "vpxenc timed out after {:.1}s and could not be reaped within {:.1}s (pid {}): {error}",
                        timeout.as_secs_f64(),
                        REAP_TIMEOUT.as_secs_f64(),
                        child.id()
                    ),
                });
            }
            Ok(None) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                thread::sleep(remaining.min(POLL_INTERVAL));
            }
            Err(error) => {
                let cleanup = kill_and_reap(&mut child);
                return Err(match cleanup {
                    Ok(()) => format!("vpxenc status: {error}"),
                    Err(ReapError::Kill(kill_error)) => format!(
                        "vpxenc status: {error}; child {} could not be killed: {kill_error}",
                        child.id()
                    ),
                    Err(ReapError::Reap(reap_error)) => format!(
                        "vpxenc status: {error}; child {} could not be reaped: {reap_error}",
                        child.id()
                    ),
                });
            }
        }
    }
}

enum ReapError {
    Kill(io::Error),
    Reap(io::Error),
}

fn kill_and_reap(child: &mut Child) -> Result<(), ReapError> {
    let kill_result = child.kill();
    let deadline = Instant::now() + REAP_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return kill_result.map_err(ReapError::Kill),
            Ok(None) if Instant::now() >= deadline => {
                return match kill_result {
                    Err(error) => Err(ReapError::Kill(error)),
                    Ok(()) => Err(ReapError::Reap(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "child remained running after kill",
                    ))),
                };
            }
            Ok(None) => thread::sleep(POLL_INTERVAL),
            Err(error) => return Err(ReapError::Reap(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_MARKER_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestMarker(PathBuf);

    impl TestMarker {
        fn new() -> Self {
            loop {
                let sequence = TEST_MARKER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "rvp9-bounded-child-marker-{}-{sequence}",
                    std::process::id()
                ));
                match OpenOptions::new().write(true).create_new(true).open(&path) {
                    Ok(file) => {
                        drop(file);
                        return Self(path);
                    }
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => {
                        panic!("could not reserve child marker {}: {error}", path.display())
                    }
                }
            }
        }
    }

    impl Drop for TestMarker {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn child_process_helper() {
        if let Ok(mode) = std::env::var("RVP9_BOUNDED_CHILD_TEST_MODE") {
            match mode.as_str() {
                "stdin-eof" => {
                    let mut bytes = Vec::new();
                    io::stdin().read_to_end(&mut bytes).unwrap();
                    assert!(bytes.is_empty(), "child stdin must reach EOF immediately");
                    let marker = std::env::var_os("RVP9_BOUNDED_CHILD_TEST_MARKER").unwrap();
                    std::fs::write(marker, b"selected").unwrap();
                }
                "sleep" => thread::sleep(Duration::from_millis(500)),
                _ => panic!("unknown child mode: {mode}"),
            }
        }
    }

    fn helper_command(mode: &str, marker: Option<&Path>) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "bounded_child::tests::child_process_helper"])
            .env("RVP9_BOUNDED_CHILD_TEST_MODE", mode)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        if let Some(marker) = marker {
            command.env("RVP9_BOUNDED_CHILD_TEST_MARKER", marker);
        }
        command
    }

    #[test]
    fn successful_child_observes_closed_stdin() {
        let marker = TestMarker::new();
        let status = run_with_timeout(
            &mut helper_command("stdin-eof", Some(&marker.0)),
            Duration::from_secs(2),
        )
        .expect("helper process should exit successfully");
        assert!(status.success(), "helper process failed: {status}");
        assert_eq!(std::fs::read(&marker.0).unwrap(), b"selected");
    }

    #[test]
    fn timeout_kills_and_reaps_a_finite_child_before_its_natural_exit() {
        let started = Instant::now();
        let error = run_with_timeout(
            &mut helper_command("sleep", None),
            Duration::from_millis(30),
        )
        .expect_err("child that exceeds the injected deadline must time out");
        assert!(error.contains("timed out"), "unexpected error: {error}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
