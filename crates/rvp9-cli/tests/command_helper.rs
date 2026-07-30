mod common;

use std::io::{self, Write};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant};

use common::{CommandExt, MAX_CAPTURE_BYTES, run_with_timeout};

#[test]
fn timeout_child() {
    if std::env::var_os("RVP9_CLI_TIMEOUT_CHILD").is_some() {
        thread::sleep(Duration::from_secs(5));
    }
}

#[test]
fn bounded_output_kills_a_timed_out_child() {
    let started = Instant::now();
    let error = run_helper_test(
        "timeout_child",
        "RVP9_CLI_TIMEOUT_CHILD",
        Duration::from_millis(50),
    )
    .expect_err("sleeping child must time out");

    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn output_child() {
    if std::env::var_os("RVP9_CLI_OUTPUT_CHILD").is_some() {
        std::io::stdout()
            .write_all(&vec![b'x'; MAX_CAPTURE_BYTES * 2])
            .expect("write oversized output");
    }
}

#[test]
fn bounded_output_caps_but_drains_each_stream() {
    let mut command = helper_test_command("output_child", "RVP9_CLI_OUTPUT_CHILD")
        .expect("build output-child command");
    let output = command.bounded_output().expect("output child must finish");

    assert!(output.status.success());
    assert_eq!(output.stdout.len(), MAX_CAPTURE_BYTES);
    assert!(output.stderr.len() <= MAX_CAPTURE_BYTES);
}

#[test]
fn bounded_status_uses_the_same_runner() {
    let mut command =
        helper_test_command("output_child", "RVP9_CLI_UNUSED").expect("build status command");
    assert!(
        command
            .bounded_status()
            .expect("status child must finish")
            .success()
    );
}

fn run_helper_test(test_name: &str, environment: &str, timeout: Duration) -> io::Result<Output> {
    let mut command = helper_test_command(test_name, environment)?;
    run_with_timeout(&mut command, timeout)
}

fn helper_test_command(test_name: &str, environment: &str) -> io::Result<Command> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .args(["--exact", test_name, "--nocapture"])
        .env(environment, "1");
    Ok(command)
}
