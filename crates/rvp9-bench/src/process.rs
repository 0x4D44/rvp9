//! Bounded external commands. File-backed capture cannot fill a pipe or wait for
//! a descendant to close an inherited pipe after the direct child is killed.
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
const CAPTURE_LIMIT: u64 = 64 * 1024;
static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(0);

struct Capture {
    file: Option<File>,
    path: PathBuf,
}
impl Capture {
    fn new() -> std::io::Result<Self> {
        loop {
            let path = std::env::temp_dir().join(format!(
                "rvp9-bench-{}-{}.log",
                std::process::id(),
                NEXT_CAPTURE.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        file: Some(file),
                        path,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }
    fn tail(&mut self) -> std::io::Result<Vec<u8>> {
        let file = self.file.as_mut().expect("capture is open");
        let length = file.metadata()?.len();
        file.seek(SeekFrom::Start(length.saturating_sub(CAPTURE_LIMIT)))?;
        let mut bytes = Vec::new();
        // Limit reads even if a descendant still writes to the file.
        file.take(CAPTURE_LIMIT).read_to_end(&mut bytes)?;
        if length > CAPTURE_LIMIT {
            let mut marked = b"[earlier output truncated]\n".to_vec();
            marked.extend(bytes);
            return Ok(marked);
        }
        Ok(bytes)
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        // Close the parent handle before unlinking on Windows.
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

// Never turn a failed kill into an unbounded blocking wait. try_wait also
// reaps on success; retain the PID if the OS cannot complete termination.
fn terminate(child: &mut Child) -> String {
    let pid = child.id();
    let kill = child.kill();
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return format!("pid={pid}; kill={kill:?}; reaped={status}"),
            Ok(None) if started.elapsed() < Duration::from_secs(5) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            result => return format!("pid={pid}; kill={kill:?}; reap incomplete: {result:?}"),
        }
    }
}

pub fn run(command: &mut Command, role: &str, timeout: Duration) -> Result<Output, String> {
    let context = format!("{role} ({:?})", command.get_program());
    let mut execute = || -> Result<Output, String> {
        let mut stdout = Capture::new().map_err(|e| format!("capture stdout: {e}"))?;
        let mut stderr = Capture::new().map_err(|e| format!("capture stderr: {e}"))?;
        let stdout_handle = stdout
            .file
            .as_ref()
            .unwrap()
            .try_clone()
            .map_err(|e| e.to_string())?;
        let stderr_handle = stderr
            .file
            .as_ref()
            .unwrap()
            .try_clone()
            .map_err(|e| e.to_string())?;
        command
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout_handle))
            .stderr(Stdio::from(stderr_handle));
        let child = command.spawn();
        // Drop Command's capture handles on both spawn success and failure.
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let mut child = child.map_err(|e| format!("spawn: {e}"))?;
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) if started.elapsed() >= timeout => {
                    let termination = terminate(&mut child);
                    break Err(format!(
                        "timed out after {:.3}s; {termination}",
                        timeout.as_secs_f64()
                    ));
                }
                Ok(None) => std::thread::sleep(
                    Duration::from_millis(10).min(timeout.saturating_sub(started.elapsed())),
                ),
                Err(error) => {
                    let termination = terminate(&mut child);
                    break Err(format!("status: {error}; {termination}"));
                }
            }
        };
        let stdout = stdout.tail();
        let stderr = stderr.tail();
        match status {
            Ok(status) => Ok(Output {
                status,
                stdout: stdout.map_err(|e| format!("read stdout: {e}"))?,
                stderr: stderr.map_err(|e| format!("read stderr: {e}"))?,
            }),
            Err(error) => {
                let stdout =
                    stdout.unwrap_or_else(|e| format!("[could not read stdout: {e}]").into_bytes());
                let stderr =
                    stderr.unwrap_or_else(|e| format!("[could not read stderr: {e}]").into_bytes());
                Err(format!(
                    "{error}\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&stdout),
                    String::from_utf8_lossy(&stderr)
                ))
            }
        }
    };
    execute().map_err(|e| format!("{context}: {e}"))
}

pub fn run_status(
    command: &mut Command,
    role: &str,
    timeout: Duration,
) -> Result<ExitStatus, String> {
    let output = run(command, role, timeout)?;
    let _ = std::io::stdout().write_all(&output.stdout);
    let _ = std::io::stderr().write_all(&output.stderr);
    Ok(output.status)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .stdin(Stdio::piped())
            .args(["--exact", "process::tests::fake_child", "--nocapture"])
            .env("BENCH_RUNNER_CHILD", mode);
        command
    }

    #[test]
    fn fake_child() {
        let Ok(mode) = std::env::var("BENCH_RUNNER_CHILD") else {
            return;
        };
        let mut input = Vec::new();
        std::io::stdin().read_to_end(&mut input).unwrap();
        assert!(input.is_empty(), "stdin must be closed");
        println!("retained stdout");
        eprintln!("retained stderr");
        std::io::stdout().flush().unwrap();
        std::io::stderr().flush().unwrap();
        match mode.as_str() {
            "timeout" => std::thread::sleep(Duration::from_secs(2)),
            "loud" => {
                std::io::stdout()
                    .write_all(&vec![b'x'; 256 * 1024])
                    .unwrap();
                std::io::stderr()
                    .write_all(&vec![b'y'; 256 * 1024])
                    .unwrap();
            }
            "fail" => std::process::exit(7),
            _ => {}
        }
    }

    #[test]
    fn timeout_kills_reaps_and_retains_diagnostics() {
        let mut command = child("timeout");
        let started = Instant::now();
        let error = run(&mut command, "test encoder", Duration::from_secs(1)).unwrap_err();
        assert!(
            error.contains("test encoder") && error.contains("timed out"),
            "{error}"
        );
        assert!(
            error.contains("retained stdout") && error.contains("retained stderr"),
            "{error}"
        );
        assert!(
            error.contains("pid=") && error.contains("reaped="),
            "{error}"
        );
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn large_output_does_not_fill_pipes_and_has_bounded_tails() {
        let output = run(&mut child("loud"), "loud tool", Duration::from_secs(3)).unwrap();
        assert!(output.status.success());
        assert!(output.stdout.len() < 66 * 1024 && output.stderr.len() < 66 * 1024);
        assert!(output.stdout.starts_with(b"[earlier output truncated]"));
        assert!(output.stderr.starts_with(b"[earlier output truncated]"));
    }

    #[test]
    fn nonzero_exit_keeps_status_and_diagnostics() {
        let output = run(&mut child("fail"), "failing tool", Duration::from_secs(3)).unwrap();
        assert_eq!(output.status.code(), Some(7));
        assert!(String::from_utf8_lossy(&output.stderr).contains("retained stderr"));
    }
}
