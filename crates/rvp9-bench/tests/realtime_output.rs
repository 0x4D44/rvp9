#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use std::os::unix::fs::{PermissionsExt, symlink};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let scratch_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
        fs::create_dir_all(&scratch_root).expect("create repository scratch root");
        loop {
            let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path =
                scratch_root.join(format!("realtime-output-{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("creating test directory {}: {error}", path.display()),
            }
        }
    }

    fn child_dir(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir(&path).expect("create isolated child TMPDIR");
        path
    }

    fn write_encoder(&self) -> PathBuf {
        let path = self.0.join("fake-encoder.sh");
        fs::write(
            &path,
            "#!/bin/sh\n\
output=\n\
while [ \"$#\" -gt 0 ]; do\n\
    if [ \"$1\" = \"-o\" ]; then\n\
        shift\n\
        output=$1\n\
    fi\n\
    shift\n\
done\n\
if [ -n \"${RVP9_TEST_CAPTURE-}\" ]; then printf '%s\\n' \"$output\" > \"$RVP9_TEST_CAPTURE\"; fi\n\
if [ -n \"${RVP9_TEST_MARKER-}\" ]; then printf 'ran\\n' > \"$RVP9_TEST_MARKER\"; fi\n\
if [ -n \"${RVP9_TEST_SLEEP-}\" ]; then sleep \"$RVP9_TEST_SLEEP\"; fi\n\
if [ \"${RVP9_TEST_EXIT-}\" != \"\" ]; then exit \"$RVP9_TEST_EXIT\"; fi\n\
if [ \"${RVP9_TEST_NO_WRITE-}\" = \"1\" ]; then exit 0; fi\n\
cp \"$RVP9_TEST_FIXTURE\" \"$output\"\n",
        )
        .expect("write fake encoder");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))
            .expect("make fake encoder executable");
        path
    }

    fn write_ivf(&self, name: &str, frames: u32) -> PathBuf {
        let path = self.0.join(name);
        let mut data = Vec::new();
        data.extend_from_slice(b"DKIF");
        data.extend_from_slice(&0_u16.to_le_bytes());
        data.extend_from_slice(&32_u16.to_le_bytes());
        data.extend_from_slice(b"VP90");
        data.extend_from_slice(&16_u16.to_le_bytes());
        data.extend_from_slice(&16_u16.to_le_bytes());
        data.extend_from_slice(&30_u32.to_le_bytes());
        data.extend_from_slice(&1_u32.to_le_bytes());
        data.extend_from_slice(&frames.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());
        for index in 0..frames {
            data.extend_from_slice(&1_u32.to_le_bytes());
            data.extend_from_slice(&u64::from(index).to_le_bytes());
            data.push(index as u8);
        }
        fs::write(&path, data).expect("write IVF fixture");
        assert_eq!(rvp9_bench::count_ivf_frames(&path), frames);
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn realtime_command(encoder: &Path, tmpdir: &Path, fixture: &Path, capture: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_realtime"));
    command
        .arg("--encoder")
        .arg(encoder)
        .args(["--input", "unused.y4m", "--budget-ms", "10000"])
        .env("TMPDIR", tmpdir)
        .env("RVP9_TEST_FIXTURE", fixture)
        .env("RVP9_TEST_CAPTURE", capture);
    command
}

fn run_realtime(command: &mut Command) -> Output {
    rvp9_bench::process::run(
        command,
        "realtime output integration test",
        Duration::from_secs(10),
    )
    .expect("bounded realtime subprocess")
}

fn captured_output_path(capture: &Path) -> PathBuf {
    PathBuf::from(
        fs::read_to_string(capture)
            .expect("fake encoder captured output path")
            .trim(),
    )
}

fn reported_frames(output: &Output) -> Option<u32> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| line.trim_start().starts_with("frames:"))?
        .split_once(':')?
        .1
        .trim()
        .parse()
        .ok()
}

#[test]
fn concurrent_default_runs_use_distinct_outputs_and_count_their_own_ivf() {
    let scratch = TestDirectory::new();
    let tmpdir = scratch.child_dir("tmp");
    let encoder = scratch.write_encoder();
    let first_fixture = scratch.write_ivf("first.ivf", 2);
    let second_fixture = scratch.write_ivf("second.ivf", 5);
    let first_capture = scratch.0.join("first-output.txt");
    let second_capture = scratch.0.join("second-output.txt");

    let (first_result, second_result) = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let mut command = realtime_command(&encoder, &tmpdir, &first_fixture, &first_capture);
            command.env("RVP9_TEST_SLEEP", "0.15");
            run_realtime(&mut command)
        });
        let second = scope.spawn(|| {
            let mut command = realtime_command(&encoder, &tmpdir, &second_fixture, &second_capture);
            command.env("RVP9_TEST_SLEEP", "0.15");
            run_realtime(&mut command)
        });
        (
            first.join().expect("join first realtime process"),
            second.join().expect("join second realtime process"),
        )
    });

    assert!(
        first_result.status.success(),
        "first run failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&first_result.stdout),
        String::from_utf8_lossy(&first_result.stderr)
    );
    assert!(
        second_result.status.success(),
        "second run failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&second_result.stdout),
        String::from_utf8_lossy(&second_result.stderr)
    );

    let first_output = captured_output_path(&first_capture);
    let second_output = captured_output_path(&second_capture);
    assert_ne!(
        first_output, second_output,
        "each run needs its own IVF path"
    );
    assert_eq!(reported_frames(&first_result), Some(2));
    assert_eq!(reported_frames(&second_result), Some(5));
    assert!(
        !first_output.exists(),
        "successful default output was retained"
    );
    assert!(
        !second_output.exists(),
        "successful default output was retained"
    );
    assert!(
        fs::read_dir(&tmpdir)
            .expect("read isolated TMPDIR")
            .next()
            .is_none(),
        "successful default output directories were not removed"
    );
}

#[test]
fn stale_default_output_is_ignored_and_zero_frame_run_cleans_owned_directory() {
    let scratch = TestDirectory::new();
    let tmpdir = scratch.child_dir("tmp");
    let encoder = scratch.write_encoder();
    let fixture = scratch.write_ivf("fixture.ivf", 3);
    let stale_path = tmpdir.join("rvp9_realtime_bench.ivf");
    fs::copy(&fixture, &stale_path).expect("create stale default artifact");
    let stale_contents = fs::read(&stale_path).expect("read stale artifact");
    let capture = scratch.0.join("captured-output.txt");

    let mut command = realtime_command(&encoder, &tmpdir, &fixture, &capture);
    command.env("RVP9_TEST_NO_WRITE", "1");
    let output = run_realtime(&mut command);
    let output_path = captured_output_path(&capture);

    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no frames in output IVF"),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_ne!(output_path, stale_path);
    assert!(!output_path.exists(), "failed default output was retained");
    assert_eq!(
        fs::read(&stale_path).expect("read stale artifact"),
        stale_contents
    );
    assert_eq!(
        fs::read_dir(&tmpdir).expect("read isolated TMPDIR").count(),
        1,
        "stale file should be the only entry in TMPDIR"
    );
}

fn explicit_existing_path_is_rejected_before_running_encoder(use_symlink: bool) {
    let scratch = TestDirectory::new();
    let tmpdir = scratch.child_dir("tmp");
    let encoder = scratch.write_encoder();
    let fixture = scratch.write_ivf("fixture.ivf", 1);
    let sentinel_path = scratch.0.join("sentinel.ivf");
    let sentinel = b"preserve this explicit artifact";
    fs::write(&sentinel_path, sentinel).expect("write sentinel output");
    let output_path = if use_symlink {
        let path = scratch.0.join("output-link.ivf");
        symlink(&sentinel_path, &path).expect("create output symlink");
        path
    } else {
        sentinel_path.clone()
    };
    let marker = scratch.0.join("encoder-ran.txt");

    let capture = scratch.0.join("captured-output.txt");
    let mut command = realtime_command(&encoder, &tmpdir, &fixture, &capture);
    command
        .arg("--output")
        .arg(&output_path)
        .env("RVP9_TEST_MARKER", &marker);
    let result = run_realtime(&mut command);

    assert_eq!(result.status.code(), Some(1));
    assert!(!marker.exists(), "encoder ran for an existing output path");
    assert!(
        !capture.exists(),
        "encoder captured an existing output path"
    );
    assert_eq!(
        fs::read(&sentinel_path).expect("read sentinel target"),
        sentinel
    );
    if use_symlink {
        assert!(
            fs::symlink_metadata(&output_path)
                .expect("inspect output symlink")
                .file_type()
                .is_symlink()
        );
    }
}

#[test]
fn existing_explicit_file_is_rejected_before_running_encoder() {
    explicit_existing_path_is_rejected_before_running_encoder(false);
}

#[test]
fn existing_explicit_symlink_is_rejected_before_running_encoder() {
    explicit_existing_path_is_rejected_before_running_encoder(true);
}

#[test]
fn fresh_explicit_output_is_used_and_retained_after_success() {
    let scratch = TestDirectory::new();
    let tmpdir = scratch.child_dir("tmp");
    let encoder = scratch.write_encoder();
    let fixture = scratch.write_ivf("fixture.ivf", 4);
    let output_path = scratch.0.join("requested.ivf");
    let capture = scratch.0.join("captured-output.txt");
    assert!(!output_path.exists());

    let mut command = realtime_command(&encoder, &tmpdir, &fixture, &capture);
    command.arg("--output").arg(&output_path);
    let output = run_realtime(&mut command);

    assert!(
        output.status.success(),
        "run failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(captured_output_path(&capture), output_path);
    assert_eq!(reported_frames(&output), Some(4));
    assert!(
        output_path.is_file(),
        "explicit output was removed after success"
    );
    assert_eq!(rvp9_bench::count_ivf_frames(&output_path), 4);
}

#[test]
fn default_output_is_cleaned_after_encoder_and_timing_failures() {
    let scratch = TestDirectory::new();
    let tmpdir = scratch.child_dir("tmp");
    let encoder = scratch.write_encoder();
    let fixture = scratch.write_ivf("fixture.ivf", 2);

    let failed_capture = scratch.0.join("failed-output.txt");
    let mut failed_command = realtime_command(&encoder, &tmpdir, &fixture, &failed_capture);
    failed_command.env("RVP9_TEST_EXIT", "9");
    let encoder_failure = run_realtime(&mut failed_command);
    let failed_output_path = captured_output_path(&failed_capture);
    assert_eq!(encoder_failure.status.code(), Some(1));
    assert!(
        !failed_output_path.exists(),
        "encoder failure output directory leaked"
    );

    let timing = scratch.0.join("invalid-timing.csv");
    fs::write(&timing, "wrong,header\n").expect("write malformed timing CSV");
    let timing_capture = scratch.0.join("timing-output.txt");
    let mut timing_command = realtime_command(&encoder, &tmpdir, &fixture, &timing_capture);
    timing_command.args(["--frame-time-csv"]).arg(&timing);
    let timing_failure = run_realtime(&mut timing_command);
    let timing_output_path = captured_output_path(&timing_capture);
    assert_eq!(timing_failure.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&timing_failure.stderr).contains("invalid frame timing CSV"),
        "stderr:\n{}",
        String::from_utf8_lossy(&timing_failure.stderr)
    );
    assert!(
        !timing_output_path.exists(),
        "timing failure output directory leaked"
    );
    assert!(
        fs::read_dir(&tmpdir)
            .expect("read isolated TMPDIR")
            .next()
            .is_none(),
        "failed default output directories were not removed"
    );
}
