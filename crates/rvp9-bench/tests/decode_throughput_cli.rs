use std::process::Command;

#[test]
fn missing_explicit_fixture_exits_nonzero() {
    let missing = std::env::temp_dir().join(format!(
        "rvp9-decode-throughput-missing-{}.ivf",
        std::process::id()
    ));
    let csv = std::env::temp_dir().join(format!(
        "rvp9-decode-throughput-unused-{}.csv",
        std::process::id()
    ));

    let output = Command::new(env!("CARGO_BIN_EXE_decode-throughput"))
        .args(["--fixture", missing.to_str().unwrap()])
        .args(["--csv", csv.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "missing fixture unexpectedly exited successfully\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no fixtures produced timed samples"),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn csv_append_failure_exits_nonzero() {
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/corpus/profile1/foreman_c444_kfonly.ivf");
    let csv_directory = std::env::temp_dir().join(format!(
        "rvp9-decode-throughput-csv-directory-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&csv_directory).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_decode-throughput"))
        .args(["--fixture", fixture.to_str().unwrap()])
        .args(["--csv", csv_directory.to_str().unwrap()])
        .output()
        .unwrap();
    std::fs::remove_dir(&csv_directory).unwrap();

    assert!(
        !output.status.success(),
        "CSV failure unexpectedly exited successfully\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("failed to append CSV"),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
