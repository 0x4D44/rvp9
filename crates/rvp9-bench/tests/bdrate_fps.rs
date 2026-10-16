#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TARGETS: [u32; 4] = [100, 200, 400, 800];
static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    encoder: PathBuf,
    decoder: PathBuf,
    input: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let worktree = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("bench crate has a workspace root");
        let scratch = worktree.join(".scratch");
        fs::create_dir_all(&scratch).unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let root = scratch.join(format!(
            "bdrate-fps-{}-{nonce}-{fixture_id}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();

        let encoder = root.join("fake-vpxenc");
        let decoder = root.join("fake-vpxdec");
        let input = root.join("source.y4m");
        fs::write(&input, source_y4m()).unwrap();
        write_executable(&encoder, ENCODER_SCRIPT);
        write_executable(&decoder, DECODER_SCRIPT);

        Self {
            root,
            encoder,
            decoder,
            input,
        }
    }

    fn run(&self, index: usize, cli_fps: &str, anchor_fps: &str, test_fps: &str) {
        let work_dir = self.root.join(format!("run-{index}"));
        fs::create_dir(&work_dir).unwrap();
        let csv = self.root.join(format!("run-{index}.csv"));
        let mut command = self.command(&work_dir, &csv, cli_fps);
        if anchor_fps != cli_fps {
            command
                .arg("--anchor-arg")
                .arg(format!("--fps={anchor_fps}"));
        }
        if test_fps != cli_fps {
            command.arg("--test-arg").arg(format!("--fps={test_fps}"));
        }

        let output = rvp9_bench::process::run(
            &mut command,
            "bdrate fps regression",
            Duration::from_secs(30),
        )
        .unwrap();
        assert!(
            output.status.success(),
            "bdrate failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).unwrap();
        let csv_text = fs::read_to_string(csv).unwrap();
        for role in ["anchor", "test"] {
            let fps = if role == "anchor" {
                parse_fps(anchor_fps)
            } else {
                parse_fps(test_fps)
            };
            for target in TARGETS {
                let artifact = work_dir.join(format!("{role}_{target}k.ivf"));
                let bytes = fs::metadata(artifact).unwrap().len() as f64;
                let expected_kbps = bytes * 8.0 * fps.0 as f64 / (1000.0 * fps.1 as f64);
                let row = csv_text
                    .lines()
                    .find(|line| line.starts_with(&format!("{role},{target},")))
                    .unwrap_or_else(|| panic!("missing {role}/{target} CSV row:\n{csv_text}"));
                let measured_kbps: f64 = row.split(',').nth(2).unwrap().parse().unwrap();
                assert!(
                    (measured_kbps - expected_kbps).abs() <= 0.0005,
                    "{role}/{target} measured {measured_kbps:.3} kbps; expected {expected_kbps:.6} from {bytes} bytes at {}/{} fps\nCSV:\n{csv_text}",
                    fps.0,
                    fps.1
                );

                let line = stdout
                    .lines()
                    .find(|line| line.starts_with(&format!("target {target:>5}k")))
                    .unwrap_or_else(|| panic!("missing {target}k stdout row:\n{stdout}"));
                let displayed = format!("{role} {:>6.0}k", expected_kbps);
                assert!(
                    line.contains(&displayed),
                    "stdout row for {role}/{target} did not contain {displayed:?}: {line}"
                );
            }
        }
    }

    fn command(&self, work_dir: &Path, csv: &Path, fps: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_bdrate"));
        command
            .args([
                "--anchor-encoder",
                self.encoder.to_str().unwrap(),
                "--test-encoder",
                self.encoder.to_str().unwrap(),
                "--decoder",
                self.decoder.to_str().unwrap(),
                "--input",
                self.input.to_str().unwrap(),
                "--fps",
                fps,
                "--bitrates",
                "100,200,400,800",
                "--work-dir",
                work_dir.to_str().unwrap(),
                "--csv",
                csv.to_str().unwrap(),
            ])
            .env("TMPDIR", &self.root);
        command
    }

    fn run_zero_timebase(&self, index: usize, fps: &str) -> Result<(), String> {
        let work_dir = self.root.join(format!("invalid-{index}"));
        fs::create_dir(&work_dir).unwrap();
        let csv = self.root.join(format!("invalid-{index}.csv"));
        let mut command = self.command(&work_dir, &csv, fps);
        let output = rvp9_bench::process::run(
            &mut command,
            "bdrate zero IVF timebase regression",
            Duration::from_secs(30),
        )
        .unwrap();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() {
            return Err(format!(
                "bdrate accepted IVF timebase {fps}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ));
        }
        if csv.exists() {
            return Err(format!("bdrate wrote CSV for IVF timebase {fps}"));
        }
        if !stderr.contains("framerate") {
            return Err(format!(
                "missing contextual framerate error for {fps}:\n{stderr}"
            ));
        }
        Ok(())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn reports_each_ivfs_effective_frame_rate() {
    let fixture = Fixture::new();

    // The source is 50 fps. The output containers follow these explicit CLI
    // rates, and the role-specific arguments exercise later FPS overrides.
    fixture.run(0, "30/1", "30/1", "30/1");
    fixture.run(1, "30000/1001", "30000/1001", "30000/1001");
    fixture.run(2, "30/1", "24/1", "60/1");
}

#[test]
fn rejects_zero_ivf_timebase_components() {
    let fixture = Fixture::new();
    let failures: Vec<String> = [(0, "0/1"), (1, "30/0")]
        .into_iter()
        .filter_map(|(index, fps)| fixture.run_zero_timebase(index, fps).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

fn parse_fps(fps: &str) -> (u32, u32) {
    let (num, den) = fps.split_once('/').unwrap();
    (num.parse().unwrap(), den.parse().unwrap())
}

fn source_y4m() -> Vec<u8> {
    let mut data = b"YUV4MPEG2 W2 H2 F50:1 Ip A1:1 C420jpeg\nFRAME\n".to_vec();
    data.extend_from_slice(&[0, 0, 0, 0, 128, 128]);
    data
}

fn write_executable(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions).unwrap();
}

const ENCODER_SCRIPT: &str = r#"#!/bin/sh
set -eu
fps=
bitrate=
output=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fps=*) fps=$(printf '%s\n' "$1" | cut -d= -f2-) ;;
        --target-bitrate=*) bitrate=$(printf '%s\n' "$1" | cut -d= -f2-) ;;
        -o) shift; output=$1 ;;
    esac
    shift
done
case "$fps" in
    0/1) rate='\000\000\000\000'; scale='\001\000\000\000' ;;
    30/1) rate='\036\000\000\000'; scale='\001\000\000\000' ;;
    30000/1001) rate='\060\165\000\000'; scale='\351\003\000\000' ;;
    30/0) rate='\036\000\000\000'; scale='\000\000\000\000' ;;
    24/1) rate='\030\000\000\000'; scale='\001\000\000\000' ;;
    60/1) rate='\074\000\000\000'; scale='\001\000\000\000' ;;
    *) echo "unsupported fake fps: $fps" >&2; exit 2 ;;
esac
case "$bitrate" in
    100) packet='\144\000\000\000'; payload=100 ;;
    200) packet='\310\000\000\000'; payload=200 ;;
    400) packet='\220\001\000\000'; payload=400 ;;
    800) packet='\040\003\000\000'; payload=800 ;;
    *) echo "unsupported fake bitrate: $bitrate" >&2; exit 2 ;;
esac
printf 'DKIF\000\000\040\000VP90\002\000\002\000' > "$output"
printf '%b%b' "$rate" "$scale" >> "$output"
printf '\001\000\000\000\000\000\000\000' >> "$output"
printf '%b\000\000\000\000\000\000\000\000' "$packet" >> "$output"
dd if=/dev/zero bs=1 count="$payload" 2>/dev/null >> "$output"
"#;

const DECODER_SCRIPT: &str = r#"#!/bin/sh
set -eu
output=
input=
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) shift; output=$1 ;;
        *.ivf) input=$1 ;;
    esac
    shift
done
case "$input" in
    *100k.ivf) frame='\004\004\004\004' ;;
    *200k.ivf) frame='\003\003\003\003' ;;
    *400k.ivf) frame='\002\002\002\002' ;;
    *800k.ivf) frame='\001\001\001\001' ;;
    *) echo "unknown fake IVF: $input" >&2; exit 2 ;;
esac
printf 'YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\nFRAME\n%b\200\200' "$frame" > "$output"
"#;
