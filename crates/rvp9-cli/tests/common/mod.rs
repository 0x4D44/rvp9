use std::io::{self, Read};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(10);

pub trait CommandExt {
    fn bounded_output(&mut self) -> io::Result<Output>;
    fn bounded_status(&mut self) -> io::Result<ExitStatus>;
}

impl CommandExt for Command {
    fn bounded_output(&mut self) -> io::Result<Output> {
        run_with_timeout(self, COMMAND_TIMEOUT)
    }

    fn bounded_status(&mut self) -> io::Result<ExitStatus> {
        self.bounded_output().map(|output| output.status)
    }
}

pub(crate) fn run_with_timeout(command: &mut Command, timeout: Duration) -> io::Result<Output> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("child stdout pipe was not created"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("child stderr pipe was not created"))?;
    let stdout_reader = thread::spawn(move || read_capped(stdout));
    let stderr_reader = thread::spawn(move || read_capped(stderr));
    let deadline = Instant::now() + timeout;

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(POLL_INTERVAL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let stdout = join_reader(stdout_reader, "stdout")?;
                let stderr = join_reader(stderr_reader, "stderr")?;
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "child exceeded {timeout:?}; captured {} stdout bytes and {} stderr bytes",
                        stdout.len(),
                        stderr.len()
                    ),
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_reader(stdout_reader, "stdout");
                let _ = join_reader(stderr_reader, "stderr");
                return Err(error);
            }
        }
    };

    Ok(Output {
        status,
        stdout: join_reader(stdout_reader, "stdout")?,
        stderr: join_reader(stderr_reader, "stderr")?,
    })
}

fn read_capped(mut reader: impl Read) -> io::Result<Vec<u8>> {
    let mut captured = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let remaining = MAX_CAPTURE_BYTES.saturating_sub(captured.len());
        captured.extend_from_slice(&chunk[..count.min(remaining)]);
    }
    Ok(captured)
}

fn join_reader(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    stream: &'static str,
) -> io::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| io::Error::other(format!("child {stream} reader panicked")))?
}
