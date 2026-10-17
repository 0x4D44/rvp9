use std::process::Command;

#[test]
fn requested_cross_decode_requires_existing_vpxdec() {
    let missing =
        std::env::temp_dir().join(format!("rvp9-soak-missing-vpxdec-{}", std::process::id()));

    let output = Command::new(env!("CARGO_BIN_EXE_soak"))
        .args(["--cross-decode", "--vpxdec", missing.to_str().unwrap()])
        .args(["--duration-secs", "0", "--iterations", "1"])
        .args(["--frames-per-iter", "1"])
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "missing vpxdec unexpectedly disabled the requested oracle\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--cross-decode requires vpxdec"),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
