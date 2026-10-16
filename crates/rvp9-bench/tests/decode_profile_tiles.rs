//! Exercise profiling evidence through the shipped driver process.

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn scratch(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
    let path = root.join(format!("profile-tiles-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn run(name: &str, args: &[&str]) -> Output {
    let dir = scratch(name);
    let mut command = Command::new(env!("CARGO_BIN_EXE_decode-profile-tiles"));
    if !args.contains(&"--out") {
        command.args(["--out", dir.join("out.csv").to_str().unwrap()]);
    }
    command.args(args).stdin(Stdio::null()).output().unwrap()
}

#[test]
fn zero_runs_is_an_argument_error() {
    let output = run("zero-runs", &["--runs", "0"]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--runs"));
}

#[test]
fn zero_min_timed_is_an_argument_error() {
    let output = run("zero-timed", &["--runs", "1", "--min-timed", "0"]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--min-timed"));
}

#[test]
fn invalid_numeric_arguments_do_not_fall_back_to_defaults() {
    for flag in ["--runs", "--min-timed", "--threads", "--warmup"] {
        for value in [Some("nope"), None] {
            let mut args = vec![flag];
            if let Some(value) = value {
                args.push(value);
            }
            let output = run("invalid", &args);
            assert_eq!(output.status.code(), Some(2), "{flag}: {output:?}");
            assert!(String::from_utf8_lossy(&output.stderr).contains(flag));
        }
    }
}

#[test]
fn unsupported_instrumentation_is_fatal() {
    // The decoder owns the effective feature; direct dependency feature
    // activation deliberately need not enable the bench's own feature.
    if rvp9_decoder::tile_profile_output_status().is_none() {
        let output = run("unsupported", &["--runs", "1", "--min-timed", "1"]);
        assert!(!output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("unsupported"),
            "{output:?}"
        );
        assert!(!String::from_utf8_lossy(&output.stdout).contains("wrote "));
    }
}

fn fixture(name: &str, tiled: bool) -> PathBuf {
    use rvp9_encoder::{EncoderConfig, EncoderParallelism, StreamEncoder, YuvFrameRef};
    use rvp9_harness::container::{IvfHeader, write_ivf};
    const WIDTH: usize = 512;
    const HEIGHT: usize = 64;
    let mut encoder = StreamEncoder::new_with_parallelism(
        EncoderConfig::new(WIDTH as u32, HEIGHT as u32),
        EncoderParallelism {
            max_threads: 2,
            requested_log2_tile_cols: u8::from(tiled),
        },
    )
    .unwrap();
    let y = vec![96; WIDTH * HEIGHT];
    let uv = vec![128; WIDTH * HEIGHT / 4];
    let encoded = encoder
        .encode_frame(&YuvFrameRef {
            y: &y,
            u: &uv,
            v: &uv,
            y_stride: WIDTH,
            u_stride: WIDTH / 2,
            v_stride: WIDTH / 2,
        })
        .unwrap();
    let mut ivf = Vec::new();
    write_ivf(
        &mut ivf,
        &IvfHeader::vp9(WIDTH as u16, HEIGHT as u16, 30, 1),
        &[&encoded.data],
    )
    .unwrap();
    let path = scratch(name).join("fixture.ivf");
    std::fs::write(&path, ivf).unwrap();
    path
}

#[test]
fn empty_fixture_cannot_claim_success() {
    use rvp9_harness::container::{IvfHeader, write_ivf};
    let dir = scratch("empty");
    let mut ivf = Vec::new();
    write_ivf(&mut ivf, &IvfHeader::vp9(16, 16, 30, 1), &[]).unwrap();
    let path = dir.join("empty.ivf");
    std::fs::write(&path, ivf).unwrap();
    let output = run(
        "empty",
        &[
            "--fixture",
            path.to_str().unwrap(),
            "--runs",
            "1",
            "--warmup",
            "0",
            "--min-timed",
            "1",
        ],
    );
    assert!(!output.status.success(), "{output:?}");
    assert!(
        !String::from_utf8_lossy(&output.stdout).contains("wrote "),
        "{output:?}"
    );
}

#[test]
fn serial_decode_without_profile_rows_is_fatal() {
    if rvp9_decoder::tile_profile_output_status().is_none() {
        return;
    }
    let path = fixture("serial", false);
    let output = run(
        "serial",
        &[
            "--fixture",
            path.to_str().unwrap(),
            "--threads",
            "2",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--min-timed",
            "1",
        ],
    );
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no timed profile CSV rows"),
        "{output:?}"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("wrote "));
}

#[test]
fn tiled_decode_emits_positive_timed_rows_excluding_warmup() {
    if rvp9_decoder::tile_profile_output_status().is_none() {
        return;
    }
    let path = fixture("tiled", true);
    // One-frame fixture forces replay, including passes spent wholly in
    // warmup. Across two runs: (3 warmup + 2 timed) * 2 = 10 CSV rows.
    let output = run(
        "tiled",
        &[
            "--fixture",
            path.to_str().unwrap(),
            "--threads",
            "2",
            "--runs",
            "2",
            "--warmup",
            "3",
            "--min-timed",
            "2",
        ],
    );
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("4 timed CSV rows (warmup rows excluded)"),
        "{stdout}"
    );
    let mhz: f64 = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("cpu MHz:"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(mhz.is_finite() && mhz > 0.0, "{stdout}");
    let csv = std::fs::read_to_string(scratch("tiled").join("out.csv")).unwrap();
    assert_eq!(csv.lines().count(), 11, "{csv}");
    assert!(
        csv.lines()
            .next()
            .unwrap()
            .starts_with("schema,frame_idx,num_tiles,")
    );
    for row in csv.lines().skip(1) {
        let columns: Vec<_> = row.split(',').collect();
        assert_eq!(columns.len(), 36, "{row}");
        assert_eq!(columns[2], "2", "positive control must have two tiles");
        assert!(columns[4].parse::<f64>().unwrap() > 0.0, "{row}");
    }
}

#[test]
fn warmup_boundary_replays_before_rejecting_missing_timed_rows() {
    use rvp9_harness::container::{IvfHeader, read_packets, write_ivf};
    if rvp9_decoder::tile_profile_output_status().is_none() {
        return;
    }
    let path = fixture("warmup-boundary", true);
    let (data, packets) = read_packets(&path).unwrap();
    // Profile-0 show_existing_frame references slot zero refreshed by the
    // keyframe. It displays a frame without entering the tile profiler.
    let show_existing = [0x88];
    let mut ivf = Vec::new();
    write_ivf(
        &mut ivf,
        &IvfHeader::vp9(512, 64, 30, 1),
        &[packets[0].slice(&data), &show_existing],
    )
    .unwrap();
    std::fs::write(&path, ivf).unwrap();
    let output = run(
        "warmup-boundary",
        &[
            "--fixture",
            path.to_str().unwrap(),
            "--threads",
            "2",
            "--runs",
            "1",
            "--warmup",
            "1",
            "--min-timed",
            "1",
        ],
    );
    assert!(output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 timed CSV rows (warmup rows excluded)"),
        "{output:?}"
    );
    let csv = std::fs::read_to_string(scratch("warmup-boundary").join("out.csv")).unwrap();
    assert_eq!(
        csv.lines().count(),
        3,
        "one warmup and one timed row: {csv}"
    );
}

#[test]
fn csv_open_failure_is_fatal() {
    if rvp9_decoder::tile_profile_output_status().is_none() {
        return;
    }
    let path = fixture("write-failure", true);
    let dir = scratch("write-failure");
    let output = run(
        "write-failure",
        &[
            "--fixture",
            path.to_str().unwrap(),
            "--threads",
            "2",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--min-timed",
            "1",
            "--out",
            dir.to_str().unwrap(),
        ],
    );
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("CSV output failed"),
        "{output:?}"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("wrote "));
}
