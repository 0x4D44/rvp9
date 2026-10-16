use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rvp9_bench::throughput::DEFAULT_FIXTURES;
use rvp9_encoder::{EncoderConfig, StreamEncoder, YuvFrameRef};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);

        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!(
                "rvp9-decode-throughput-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory {}: {error}", path.display()),
            }
        }
        panic!("could not allocate a unique decode-throughput test directory");
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn element(id: &[u8], payload: &[u8]) -> Vec<u8> {
    assert!(
        payload.len() < 0x3fff,
        "test WebM element exceeds two-byte VINT"
    );
    let mut bytes = id.to_vec();
    if payload.len() < 0x7f {
        bytes.push(0x80 | payload.len() as u8);
    } else {
        let size = 0x4000u16 | payload.len() as u16;
        bytes.extend_from_slice(&size.to_be_bytes());
    }
    bytes.extend_from_slice(payload);
    bytes
}

fn uint_element(id: &[u8], value: u64) -> Vec<u8> {
    let encoded = value.to_be_bytes();
    let first_byte = encoded.iter().position(|byte| *byte != 0).unwrap_or(7);
    element(id, &encoded[first_byte..])
}

fn encoded_keyframe() -> Vec<u8> {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let mut encoder =
        StreamEncoder::new(EncoderConfig::new(WIDTH, HEIGHT)).expect("create fixture encoder");

    let y = vec![128; WIDTH as usize * HEIGHT as usize];
    let u = vec![128; WIDTH as usize * HEIGHT as usize / 4];
    let v = vec![128; WIDTH as usize * HEIGHT as usize / 4];
    let frame = YuvFrameRef {
        y: &y,
        u: &u,
        v: &v,
        y_stride: WIDTH as usize,
        u_stride: WIDTH as usize / 2,
        v_stride: WIDTH as usize / 2,
    };
    encoder
        .encode_frame(&frame)
        .expect("encode fixture keyframe")
        .data
}

fn minimal_vp9_webm(packet: &[u8]) -> Vec<u8> {
    const WIDTH: u64 = 64;
    const HEIGHT: u64 = 64;

    let mut ebml_header = Vec::new();
    ebml_header.extend(uint_element(&[0x42, 0x86], 1)); // EBMLVersion
    ebml_header.extend(uint_element(&[0x42, 0xf7], 1)); // EBMLReadVersion
    ebml_header.extend(uint_element(&[0x42, 0xf2], 4)); // EBMLMaxIDLength
    ebml_header.extend(uint_element(&[0x42, 0xf3], 8)); // EBMLMaxSizeLength
    ebml_header.extend(element(&[0x42, 0x82], b"webm")); // DocType
    ebml_header.extend(uint_element(&[0x42, 0x87], 4)); // DocTypeVersion
    ebml_header.extend(uint_element(&[0x42, 0x85], 2)); // DocTypeReadVersion
    let mut webm = element(&[0x1a, 0x45, 0xdf, 0xa3], &ebml_header);

    let mut info = uint_element(&[0x2a, 0xd7, 0xb1], 1_000_000); // TimecodeScale
    info.extend(element(&[0x4d, 0x80], b"rvp9 test")); // MuxingApp
    info.extend(element(&[0x57, 0x41], b"rvp9 test")); // WritingApp

    let mut video = uint_element(&[0xb0], WIDTH); // PixelWidth
    video.extend(uint_element(&[0xba], HEIGHT)); // PixelHeight
    let mut track = uint_element(&[0xd7], 1); // TrackNumber
    track.extend(uint_element(&[0x73, 0xc5], 1)); // TrackUID
    track.extend(uint_element(&[0x83], 1)); // TrackType: video
    track.extend(element(&[0x86], b"V_VP9")); // CodecID
    track.extend(element(&[0xe0], &video)); // Video
    let tracks = element(&[0x16, 0x54, 0xae, 0x6b], &element(&[0xae], &track));

    let mut block = vec![0x81, 0x00, 0x00, 0x80]; // track 1, timecode 0, keyframe
    block.extend_from_slice(packet);
    let mut cluster = uint_element(&[0xe7], 0); // Timecode
    cluster.extend(element(&[0xa3], &block)); // SimpleBlock
    let cluster = element(&[0x1f, 0x43, 0xb6, 0x75], &cluster);

    let mut segment = element(&[0x15, 0x49, 0xa9, 0x66], &info); // Info
    segment.extend(tracks);
    segment.extend(cluster);
    webm.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment)); // Segment
    webm
}

fn write_fixture_dir(dir: &Path, names: &[&str], bytes: &[u8]) {
    for name in names {
        fs::write(dir.join(name), bytes)
            .unwrap_or_else(|error| panic!("write fixture {}: {error}", dir.join(name).display()));
    }
}

fn run_default_sweep(vectors_dir: &Path, csv: &Path) -> Output {
    run_cli(
        Command::new(env!("CARGO_BIN_EXE_decode-throughput"))
            .env("RVP9_VECTORS_DIR", vectors_dir)
            .arg("--csv")
            .arg(csv),
        "decode-throughput default sweep",
    )
}

fn run_cli(command: &mut Command, role: &str) -> Output {
    rvp9_bench::process::run(command, role, Duration::from_secs(30))
        .unwrap_or_else(|error| panic!("{error}"))
}

fn csv_rows(csv: &Path) -> Vec<Vec<String>> {
    let contents = fs::read_to_string(csv).expect("read decode-throughput CSV");
    let mut lines = contents.lines();
    let header = lines.next().expect("CSV header");
    assert!(header.starts_with("schema_version,timestamp_unix,host,unpinned"));
    lines
        .map(|line| line.split(',').map(str::to_owned).collect())
        .collect()
}

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

    let output = run_cli(
        Command::new(env!("CARGO_BIN_EXE_decode-throughput"))
            .args(["--fixture", missing.to_str().unwrap()])
            .args(["--csv", csv.to_str().unwrap()]),
        "decode-throughput missing explicit fixture",
    );

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
    fs::create_dir_all(&csv_directory).unwrap();

    let output = run_cli(
        Command::new(env!("CARGO_BIN_EXE_decode-throughput"))
            .args(["--fixture", fixture.to_str().unwrap()])
            .args(["--csv", csv_directory.to_str().unwrap()]),
        "decode-throughput CSV append failure",
    );
    fs::remove_dir(&csv_directory).unwrap();

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

#[test]
fn default_sweep_with_failed_fixtures_exits_nonzero_and_keeps_success_row() {
    assert!(DEFAULT_FIXTURES.len() >= 2);
    let temp = TestDir::new();
    let valid = minimal_vp9_webm(&encoded_keyframe());
    fs::write(temp.0.join(DEFAULT_FIXTURES[0]), valid).unwrap();
    fs::write(temp.0.join(DEFAULT_FIXTURES[1]), [0x00]).unwrap();
    let csv = temp.0.join("partial.csv");

    let output = run_default_sweep(&temp.0, &csv);
    assert!(
        !output.status.success(),
        "partial default sweep unexpectedly exited successfully\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ERROR:"),
        "failure diagnostic missing:\n{stderr}"
    );
    assert!(
        stderr.contains("one or more fixtures failed"),
        "sweep summary diagnostic missing:\n{stderr}"
    );

    let rows = csv_rows(&csv);
    assert_eq!(rows.len(), 1, "CSV rows: {rows:?}");
    assert_eq!(rows[0].len(), 12, "CSV row: {:?}", rows[0]);
    assert_eq!(rows[0][5], DEFAULT_FIXTURES[0]);
    assert!(
        rows[0][6].parse::<usize>().unwrap() > 0,
        "CSV timed frame count must be positive: {:?}",
        rows[0]
    );
}

#[test]
fn successful_default_sweep_records_all_four_fixtures() {
    let temp = TestDir::new();
    let valid = minimal_vp9_webm(&encoded_keyframe());
    write_fixture_dir(&temp.0, DEFAULT_FIXTURES, &valid);
    let csv = temp.0.join("complete.csv");

    let output = run_default_sweep(&temp.0, &csv);
    assert!(
        output.status.success(),
        "complete default sweep failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let rows = csv_rows(&csv);
    let names: Vec<&str> = rows.iter().map(|row| row[5].as_str()).collect();
    assert_eq!(
        rows.len(),
        4,
        "configured default fixture count changed: {rows:?}"
    );
    assert_eq!(names, DEFAULT_FIXTURES);
    for row in &rows {
        assert_eq!(row.len(), 12, "CSV row: {row:?}");
        assert!(
            row[6].parse::<usize>().unwrap() > 0,
            "CSV timed frame count must be positive: {row:?}"
        );
    }
}
