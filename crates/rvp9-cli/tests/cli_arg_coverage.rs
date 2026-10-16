//! CLI argument-surface coverage for `rvp9enc` / `rvp9dec`.
//!
//! Sister file to `cli_roundtrip.rs`. The roundtrip file pins the happy
//! path; this file pins:
//!
//! * unknown / unsupported flag rejection
//! * output-format selection (`--y4m`, `--i420`, `--yv12`, `--rawvideo`,
//!   `--md5`, `--framemd5`, `--noblit`)
//! * `--limit` / `--skip` flow control
//! * `--summary` / `--progress` user-visible side channels
//! * container demux failures (truncated / bad-magic IVF, unknown ext)
//! * `--codec` / `--end-usage` / `--passes` / `--aq-mode` / `--fps`
//!   value-validation contracts
//! * raw-input dimension validation
//!
//! Each test asserts a *user-visible contract* (exit code, stderr
//! substring, stdout shape) — never internal state.
//!
//! These tests are extracted from `cli_roundtrip.rs` rather than added
//! there so the contract surface is browsable in one file as the CLIs
//! grow.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::CommandExt;
use rvp9_io::container::{IvfHeader, read_packets, write_ivf};

// ---- shared fixtures ----------------------------------------------------

const W: usize = 64;
const H: usize = 64;
const FRAME_BYTES: usize = W * H * 3 / 2;
const N_FRAMES: usize = 6;

fn enc_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rvp9enc"))
}
fn dec_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rvp9dec"))
}

/// Unique temp path for this test process + a tag. Distinct prefix from
/// `cli_roundtrip.rs` so parallel test runs don't trample each other.
fn tmp(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("rvp9cli_argcov_{}_{id}_{tag}", std::process::id()))
}

/// Write `N_FRAMES` synthetic 64×64 I420 frames with per-frame motion
/// so the encoder produces non-trivial inter frames.
fn write_synthetic_raw(path: &Path) {
    write_synthetic_raw_count(path, N_FRAMES);
}

fn write_synthetic_raw_count(path: &Path, frame_count: usize) {
    let mut buf = Vec::with_capacity(FRAME_BYTES * frame_count);
    for f in 0..frame_count {
        for y in 0..H {
            for x in 0..W {
                buf.push(((x + y + f * 4) & 0xff) as u8);
            }
        }
        for plane in 0..2 {
            for y in 0..H / 2 {
                for x in 0..W / 2 {
                    let sample = if plane == 0 {
                        x * 2 + y + f
                    } else {
                        128 + x + y + f
                    };
                    buf.push((sample & 0xff) as u8);
                }
            }
        }
    }
    assert_eq!(buf.len(), FRAME_BYTES * frame_count);
    std::fs::write(path, &buf).expect("write synthetic raw");
}

/// Encode a `N_FRAMES` 64×64 IVF clip. Caller owns the returned path.
fn encode_synthetic_ivf(tag: &str) -> PathBuf {
    let raw = tmp(&format!("{tag}_in.yuv"));
    let ivf = tmp(&format!("{tag}_out.ivf"));
    write_synthetic_raw(&raw);
    let status = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_status()
        .expect("run rvp9enc");
    assert!(status.success(), "rvp9enc failed for tag {tag}");
    let _ = std::fs::remove_file(&raw);
    ivf
}

/// Repack the first two frames from an IVF as one two-component VP9
/// superframe packet. The original encode order is preserved, so the second
/// inter frame can reference the first keyframe normally.
fn repack_first_two_frames_as_superframe(input: &Path, tag: &str) -> PathBuf {
    let (data, packets) = read_packets(input).expect("read source IVF");
    assert!(packets.len() >= 2, "source IVF needs at least two frames");
    let frames = [packets[0].slice(&data), packets[1].slice(&data)];
    let max_size = frames.iter().map(|frame| frame.len()).max().unwrap();
    let mag = match max_size {
        0..=0xff => 1usize,
        0x100..=0xffff => 2,
        0x1_0000..=0xff_ffff => 3,
        _ => 4,
    };
    let marker = 0xc0 | (((mag - 1) as u8) << 3) | 1;
    let mut superframe =
        Vec::with_capacity(frames.iter().map(|frame| frame.len()).sum::<usize>() + 6);
    for frame in frames {
        superframe.extend_from_slice(frame);
    }
    superframe.push(marker);
    for frame in frames {
        let encoded = (frame.len() as u32).to_le_bytes();
        superframe.extend_from_slice(&encoded[..mag]);
    }
    superframe.push(marker);

    let output = tmp(&format!("{tag}_superframe.ivf"));
    let mut file = std::fs::File::create(&output).expect("create superframe IVF");
    write_ivf(
        &mut file,
        &IvfHeader::vp9(W as u16, H as u16, 30, 1),
        &[&superframe],
    )
    .expect("write superframe IVF");
    output
}

// ========================================================================
// Unknown / unsupported flag rejection
// ========================================================================

/// rvp9dec must reject unknown long options with a clear message that
/// names the option (mirrors the encoder-side `unsupported_flag_fails_loudly`
/// in cli_roundtrip.rs).
#[test]
fn dec_unknown_long_flag_rejected() {
    let out = Command::new(dec_bin())
        .args(["--bogus-flag-xyz", "irrelevant.ivf"])
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown option") && stderr.contains("--bogus-flag-xyz"),
        "stderr must name unknown option, got: {stderr}"
    );
}

/// rvp9enc must reject unknown long options the same way.
#[test]
fn enc_unknown_long_flag_rejected() {
    let out = Command::new(enc_bin())
        .args(["--bogus-flag-xyz", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown option") && stderr.contains("--bogus-flag-xyz"),
        "stderr must name unknown option, got: {stderr}"
    );
}

/// rvp9enc must reject unknown short options.
#[test]
fn enc_unknown_short_flag_rejected() {
    let out = Command::new(enc_bin())
        .args(["-z"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown option") && stderr.contains("-z"),
        "stderr must name unknown short option, got: {stderr}"
    );
}

#[test]
fn enc_webm_rejected_before_creating_ivf_output() {
    let raw = tmp("webm_rejected_in.yuv");
    let output = tmp("webm_rejected_out.webm");
    write_synthetic_raw(&raw);

    let result = Command::new(enc_bin())
        .args(["--webm", "-w", "64", "-h", "64", "-o"])
        .arg(&output)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");

    let _ = std::fs::remove_file(&raw);
    let output_was_created = output.exists();
    let _ = std::fs::remove_file(&output);
    assert!(!result.status.success(), "--webm must fail until muxed");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("--webm") && stderr.contains("not supported yet"),
        "stderr must explain that WebM output is unsupported, got: {stderr}"
    );
    assert!(
        !output_was_created,
        "unsupported --webm must not create an IVF file"
    );
}

// The parser's `apply: None` path keeps recognized but unavailable
// features fail-loud. `--webm` exercises that contract above until a
// WebM muxer exists; other unimplemented vpxenc parity flags remain
// explicit `None` entries until their values affect encoding.

// ========================================================================
// --codec validation (both binaries)
// ========================================================================

/// rvp9enc accepts only `--codec vp9`; other values are rejected with a
/// message that names both the requested value and the supported one.
#[test]
fn enc_codec_vp8_rejected() {
    let out = Command::new(enc_bin())
        .args(["--codec", "vp8", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--codec") && stderr.contains("vp8") && stderr.contains("vp9"),
        "stderr must name both the bad and supported codec, got: {stderr}"
    );
}

/// rvp9dec accepts only `--codec vp9`.
#[test]
fn dec_codec_vp8_rejected() {
    let out = Command::new(dec_bin())
        .args(["--codec", "vp8", "in.ivf"])
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--codec") && stderr.contains("vp8") && stderr.contains("vp9"),
        "stderr must reject --codec=vp8, got: {stderr}"
    );
}

// ========================================================================
// Encoder enum-value flags
// ========================================================================

/// `--end-usage` accepts the four one-pass modes (`cbr`/`vbr`/`cq`/`q`)
/// since re-audit pass 3 wired `rc_mode` through to the controller. An
/// unknown mode is rejected (not silently treated as `cbr`).
#[test]
fn enc_end_usage_unknown_rejected() {
    let out = Command::new(enc_bin())
        .args(["--end-usage", "bogus", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--end-usage") && stderr.contains("cbr/vbr/cq/q"),
        "stderr must name --end-usage and the accepted modes, got: {stderr}"
    );
}

/// `--passes 2` is now ACCEPTED at parse time — two-pass VBR is
/// implemented (`rvp9_encoder::encode_two_pass`). The encode here fails
/// only because no input/output is supplied; the `--passes` flag itself
/// must NOT be the rejection reason (it parses fine). A regression that
/// reverted `--passes 2` to "not supported" would surface the old
/// "--passes" error string here.
#[test]
fn enc_passes_two_accepted_at_parse_time() {
    let out = Command::new(enc_bin())
        .args(["--passes", "2", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    // Run still fails (no -o / missing file), but NOT because --passes
    // was rejected.
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("--passes=2 not understood")
            && !stderr.contains("--passes is recognised but not supported"),
        "--passes 2 must be accepted (two-pass VBR is implemented); got: {stderr}"
    );
}

#[test]
fn enc_passes_two_still_writes_complete_ivf() {
    let cases: [(&str, &[&str]); 5] = [
        ("defaults", &[]),
        (
            "threads_zero_tile_zero",
            &["--threads=0", "--tile-columns=0"],
        ),
        (
            "threads_one_tile_zero",
            &["--threads", "1", "--tile-columns", "0"],
        ),
        ("explicit_zero_lag", &["--lag-in-frames=0"]),
        ("explicit_zero_alt_ref", &["--auto-alt-ref=0"]),
    ];
    let mut violations = Vec::new();

    for (tag, controls) in cases {
        let raw = tmp(&format!("two_pass_{tag}_in.yuv"));
        let ivf = tmp(&format!("two_pass_{tag}_out.ivf"));
        write_synthetic_raw(&raw);
        let out = Command::new(enc_bin())
            .args(["--passes", "2"])
            .args(controls)
            .args(["-w", "64", "-h", "64", "--target-bitrate", "500", "-o"])
            .arg(&ivf)
            .arg(&raw)
            .bounded_output()
            .expect("run two-pass rvp9enc");
        if !out.status.success() {
            violations.push(format!(
                "{tag}: encode failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        } else {
            match std::fs::read(&ivf) {
                Ok(bytes) if bytes.len() >= 28 => {
                    let frame_count =
                        u32::from_le_bytes(bytes[24..28].try_into().expect("frame count")) as usize;
                    if frame_count != N_FRAMES {
                        violations.push(format!(
                            "{tag}: expected {N_FRAMES} frames, got {frame_count}"
                        ));
                    }
                    match read_packets(&ivf) {
                        Ok((data, packets)) => {
                            if packets.len() != N_FRAMES {
                                violations.push(format!(
                                    "{tag}: expected {N_FRAMES} parsed packets, got {}",
                                    packets.len()
                                ));
                            }
                            if packets.iter().any(|packet| packet.slice(&data).is_empty()) {
                                violations.push(format!("{tag}: IVF contains an empty packet"));
                            }
                        }
                        Err(error) => {
                            violations.push(format!("{tag}: output is not a valid IVF: {error}"));
                        }
                    }
                }
                Ok(_) => violations.push(format!("{tag}: output is shorter than an IVF header")),
                Err(error) => violations.push(format!("{tag}: could not read IVF output: {error}")),
            }
        }
        let _ = std::fs::remove_file(&raw);
        let _ = std::fs::remove_file(&ivf);
    }

    assert!(
        violations.is_empty(),
        "two-pass defaults and supported explicit parallelism must write complete IVFs: {}",
        violations.join("; ")
    );
}

#[test]
fn enc_two_pass_rejects_explicit_parallelism_before_file_io() {
    let input = tmp("two_pass_parallelism_in.yuv");
    write_synthetic_raw(&input);
    let input_before = std::fs::read(&input).expect("read synthetic raw input");
    let mut violations = Vec::new();

    let cases: [(&str, &[&str], &str, bool); 4] = [
        (
            "threads_equals",
            &["--passes=2", "--threads=2"],
            "--threads",
            false,
        ),
        (
            "threads_split",
            &["--passes", "2", "--threads", "8"],
            "--threads",
            false,
        ),
        (
            "tiles_equals",
            &["--passes=2", "--tile-columns=1"],
            "--tile-columns",
            true,
        ),
        (
            "tiles_split",
            &["--passes", "2", "--tile-columns", "2"],
            "--tile-columns",
            false,
        ),
    ];

    for (tag, args, unsupported_flag, preserve_existing_output) in cases {
        let output = tmp(&format!("two_pass_parallelism_{tag}.ivf"));
        let sentinel = b"existing output must survive rejected options";
        if preserve_existing_output {
            std::fs::write(&output, sentinel).expect("write sentinel output");
        }

        let result = Command::new(enc_bin())
            .args(args)
            .args([
                "-w",
                "64",
                "-h",
                "64",
                "--fps",
                "30/1",
                "--target-bitrate",
                "500",
                "-o",
            ])
            .arg(&output)
            .arg(&input)
            .bounded_output()
            .expect("run two-pass parallelism rvp9enc");

        let stderr = String::from_utf8_lossy(&result.stderr).to_ascii_lowercase();
        if result.status.success() {
            violations.push(format!("{tag}: command succeeded"));
        }
        if !(stderr.contains("--passes")
            && stderr.contains(unsupported_flag)
            && stderr.contains("one thread")
            && stderr.contains("one tile"))
        {
            violations.push(format!("{tag}: missing parallelism diagnostic: {stderr}"));
        }
        match std::fs::read(&output) {
            Ok(bytes) if preserve_existing_output && bytes == sentinel => {}
            Err(_) if !preserve_existing_output => {}
            Ok(_) if !preserve_existing_output => {
                violations.push(format!("{tag}: rejected command created output"));
            }
            _ => violations.push(format!("{tag}: rejected command changed sentinel output")),
        }
        match std::fs::read(&input) {
            Ok(bytes) if bytes == input_before => {}
            _ => violations.push(format!("{tag}: rejected command changed input")),
        }

        let _ = std::fs::remove_file(&output);
    }

    let _ = std::fs::remove_file(&input);
    assert!(
        violations.is_empty(),
        "two-pass explicit parallelism must be rejected before file I/O: {}",
        violations.join("; ")
    );
}

#[test]
fn enc_two_pass_rejects_positive_lag_before_file_io() {
    let input = tmp("two_pass_lag_missing_input.yuv");
    let cases: [(&str, &[&str]); 4] = [
        ("lag_one_inline", &["--passes=2", "--lag-in-frames=1"]),
        ("lag_one_split", &["--passes=2", "--lag-in-frames", "1"]),
        ("lag_max_inline", &["--passes=2", "--lag-in-frames=25"]),
        ("lag_max_split", &["--passes=2", "--lag-in-frames", "25"]),
    ];
    let mut violations = Vec::new();

    for (tag, args) in cases {
        let output = tmp(&format!("two_pass_lag_{tag}.ivf"));
        let sentinel = b"existing output must survive rejected options";
        std::fs::write(&output, sentinel).expect("write output sentinel");

        let result = Command::new(enc_bin())
            .args(args)
            .args(["-w", "64", "-h", "64", "-o"])
            .arg(&output)
            .arg(&input)
            .bounded_output()
            .expect("run two-pass lag rvp9enc");

        let stderr = String::from_utf8_lossy(&result.stderr).to_ascii_lowercase();
        if result.status.success() {
            violations.push(format!("{tag}: command succeeded"));
        }
        if !(stderr.contains("--passes 2")
            && stderr.contains("--lag-in-frames")
            && stderr.contains("delayed scheduling")
            && stderr.contains("omitted or 0"))
        {
            violations.push(format!(
                "{tag}: missing delayed-scheduling diagnostic: {stderr}"
            ));
        }
        if stderr.contains("opening ") {
            violations.push(format!(
                "{tag}: missing-input error took precedence: {stderr}"
            ));
        }
        match std::fs::read(&output) {
            Ok(bytes) if bytes == sentinel => {}
            Ok(_) => violations.push(format!("{tag}: rejected command changed sentinel output")),
            Err(error) => {
                violations.push(format!("{tag}: could not read output sentinel: {error}"))
            }
        }

        let _ = std::fs::remove_file(&output);
    }

    assert!(
        violations.is_empty(),
        "two-pass positive lag must be rejected before input/output I/O: {}",
        violations.join("; ")
    );
}

#[test]
fn enc_two_pass_rejects_positive_alt_ref_before_file_io() {
    let input = tmp("two_pass_alt_ref_missing_input.yuv");
    let cases: [(&str, &[&str]); 4] = [
        ("alt_ref_one_inline", &["--passes=2", "--auto-alt-ref=1"]),
        ("alt_ref_one_split", &["--passes=2", "--auto-alt-ref", "1"]),
        ("alt_ref_max_inline", &["--passes=2", "--auto-alt-ref=6"]),
        ("alt_ref_max_split", &["--passes=2", "--auto-alt-ref", "6"]),
    ];
    let mut violations = Vec::new();

    for (tag, args) in cases {
        let output = tmp(&format!("two_pass_alt_ref_{tag}.ivf"));
        let sentinel = b"existing output must survive rejected options";
        std::fs::write(&output, sentinel).expect("write output sentinel");

        let result = Command::new(enc_bin())
            .args(args)
            .args(["-w", "64", "-h", "64", "-o"])
            .arg(&output)
            .arg(&input)
            .bounded_output()
            .expect("run two-pass ALTREF rvp9enc");

        let stderr = String::from_utf8_lossy(&result.stderr).to_ascii_lowercase();
        if result.status.success() {
            violations.push(format!("{tag}: command succeeded"));
        }
        if !(stderr.contains("--passes 2")
            && stderr.contains("--auto-alt-ref")
            && stderr.contains("delayed scheduling")
            && stderr.contains("omitted or 0"))
        {
            violations.push(format!(
                "{tag}: missing delayed-scheduling diagnostic: {stderr}"
            ));
        }
        if stderr.contains("opening ") {
            violations.push(format!(
                "{tag}: missing-input error took precedence: {stderr}"
            ));
        }
        match std::fs::read(&output) {
            Ok(bytes) if bytes == sentinel => {}
            Ok(_) => violations.push(format!("{tag}: rejected command changed sentinel output")),
            Err(error) => {
                violations.push(format!("{tag}: could not read output sentinel: {error}"))
            }
        }

        let _ = std::fs::remove_file(&output);
    }

    assert!(
        violations.is_empty(),
        "two-pass positive ALTREF must be rejected before input/output I/O: {}",
        violations.join("; ")
    );
}

#[test]
fn enc_two_pass_positive_lag_is_rejected_with_valid_input() {
    let input = tmp("two_pass_lag_valid_input.yuv");
    let output = tmp("two_pass_lag_valid_output.ivf");
    let sentinel = b"existing output must survive rejected options";
    write_synthetic_raw(&input);
    std::fs::write(&output, sentinel).expect("write output sentinel");

    let result = Command::new(enc_bin())
        .args([
            "--passes=2",
            "--lag-in-frames=1",
            "-w",
            "64",
            "-h",
            "64",
            "--target-bitrate",
            "500",
            "-o",
        ])
        .arg(&output)
        .arg(&input)
        .bounded_output()
        .expect("run two-pass lag rvp9enc with valid input");

    let stderr = String::from_utf8_lossy(&result.stderr).to_ascii_lowercase();
    let mut violations = Vec::new();
    if result.status.success() {
        violations.push("command succeeded".to_string());
    }
    if !(stderr.contains("--passes 2")
        && stderr.contains("--lag-in-frames")
        && stderr.contains("delayed scheduling")
        && stderr.contains("omitted or 0"))
    {
        violations.push(format!("missing delayed-scheduling diagnostic: {stderr}"));
    }
    match std::fs::read(&output) {
        Ok(bytes) if bytes == sentinel => {}
        Ok(_) => violations.push("rejected request changed output sentinel".to_string()),
        Err(error) => violations.push(format!("could not read output sentinel: {error}")),
    }

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    assert!(
        violations.is_empty(),
        "two-pass positive lag must be rejected instead of silently ignored: {}",
        violations.join("; ")
    );
}

#[test]
fn enc_two_pass_temporal_svc_rejected_before_output_creation() {
    let mut violations = Vec::new();

    for temporal_layers in [2, 3] {
        let input = tmp(&format!("two_pass_svc_{temporal_layers}_in.y4m"));
        let output = tmp(&format!("two_pass_svc_{temporal_layers}_out.ivf"));
        write_synthetic_y4m(&input, 64, 64, 3, 30, 1);
        let input_before = std::fs::read(&input).expect("read synthetic Y4M input");

        let result = Command::new(enc_bin())
            .args(["--passes", "2", "--temporal-layers"])
            .arg(temporal_layers.to_string())
            .args(["--target-bitrate", "500", "-o"])
            .arg(&output)
            .arg(&input)
            .bounded_output()
            .expect("run two-pass temporal SVC rvp9enc");

        let stderr = String::from_utf8_lossy(&result.stderr).to_ascii_lowercase();
        if result.status.success() {
            violations.push(format!(
                "{temporal_layers} temporal layers: command succeeded"
            ));
        }
        if !(stderr.contains("two-pass") && stderr.contains("temporal")) {
            violations.push(format!(
                "{temporal_layers} temporal layers: missing two-pass/temporal diagnostic: {stderr}"
            ));
        }
        if output.exists() {
            violations.push(format!(
                "{temporal_layers} temporal layers: output file was created"
            ));
        }
        match std::fs::read(&input) {
            Ok(after) if after == input_before => {}
            _ => violations.push(format!("{temporal_layers} temporal layers: input changed")),
        }

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    assert!(
        violations.is_empty(),
        "two-pass temporal SVC must be rejected without creating output or changing input: {}",
        violations.join("; ")
    );
}

/// `--passes 3` is rejected — only 1 or 2 passes are valid.
#[test]
fn enc_passes_three_rejected() {
    let out = Command::new(enc_bin())
        .args(["--passes", "3", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--passes") && stderr.contains("1 or 2"),
        "stderr must reject --passes=3 naming the valid range, got: {stderr}"
    );
}

#[test]
fn enc_frame_time_csv_writes_one_row_per_temporal_svc_frame() {
    let raw = tmp("frame_time_csv_in.yuv");
    let ivf = tmp("frame_time_csv_out.ivf");
    let csv = tmp("frame_time_csv.csv");
    write_synthetic_raw_count(&raw, 4);
    let out = Command::new(enc_bin())
        .args([
            "--temporal-layers",
            "3",
            "--lag-in-frames",
            "0",
            "--frame-time-csv",
        ])
        .arg(&csv)
        .args(["-w", "64", "-h", "64", "--target-bitrate", "500", "-o"])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc with frame timing diagnostic");
    assert!(
        out.status.success(),
        "frame timing encode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let csv_text = std::fs::read_to_string(&csv).expect("read frame timing CSV");
    let lines: Vec<&str> = csv_text.lines().collect();
    assert_eq!(
        lines.first().copied(),
        Some("frame_index,temporal_layer_id,encode_ns")
    );
    assert_eq!(lines.len(), 5, "header plus one row per source frame");
    let rows: Vec<(usize, u8, u128)> = lines[1..]
        .iter()
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 3, "CSV row must have three fields: {line}");
            (
                fields[0].parse().expect("frame index"),
                fields[1].parse().expect("temporal layer id"),
                fields[2].parse().expect("encode time"),
            )
        })
        .collect();
    assert_eq!(
        rows.iter().map(|row| row.0).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        rows.iter().map(|row| row.1).collect::<Vec<_>>(),
        [0, 2, 1, 2],
        "T3 timing rows must carry the emitted temporal layer IDs"
    );
    assert!(rows.iter().all(|row| row.2 > 0), "timings must be nonzero");

    let ivf_bytes = std::fs::read(&ivf).expect("read timed IVF");
    assert_eq!(
        u32::from_le_bytes(ivf_bytes[24..28].try_into().expect("frame count")),
        4,
        "timed encode must still emit one IVF frame per source frame"
    );
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn enc_frame_time_csv_rejects_two_pass() {
    let csv = tmp("frame_time_csv_two_pass.csv");
    let out = Command::new(enc_bin())
        .args(["--frame-time-csv"])
        .arg(&csv)
        .args(["--passes", "2", "input.y4m"])
        .bounded_output()
        .expect("run invalid timed two-pass command");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--frame-time-csv")
            && stderr.contains("one-pass")
            && stderr.contains("two-pass"),
        "timing diagnostic must reject two-pass encoding clearly, got: {stderr}"
    );
    assert!(!csv.exists(), "rejected timing command must not create CSV");
}

#[test]
fn enc_frame_time_csv_rejects_nonzero_lag() {
    let csv = tmp("frame_time_csv_lag.csv");
    let out = Command::new(enc_bin())
        .args(["--frame-time-csv"])
        .arg(&csv)
        .args(["--lag-in-frames", "1", "input.y4m"])
        .bounded_output()
        .expect("run invalid timed lag command");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--frame-time-csv")
            && stderr.contains("lag-in-frames=0")
            && stderr.contains("nonzero"),
        "timing diagnostic must reject nonzero lag clearly, got: {stderr}"
    );
    assert!(!csv.exists(), "rejected timing command must not create CSV");
}

/// `--aq-mode` accepts 0 and 3 only — 2 must be rejected with the menu
/// in the message.
#[test]
fn enc_aq_mode_unsupported_value_rejected() {
    let out = Command::new(enc_bin())
        .args(["--aq-mode", "2", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--aq-mode") && stderr.contains("cyclic"),
        "stderr must name --aq-mode and 'cyclic', got: {stderr}"
    );
}

/// `--aq-mode` requires a value.
#[test]
fn enc_aq_mode_missing_value_rejected() {
    let out = Command::new(enc_bin())
        .args(["--aq-mode"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--aq-mode") && stderr.contains("requires a value"),
        "stderr must require --aq-mode value, got: {stderr}"
    );
}

/// `--aq-mode 3` is accepted (cyclic-refresh wired). Encode succeeds end-to-end.
#[test]
fn enc_aq_mode_three_accepted_and_encodes() {
    let raw = tmp("aq3_in.yuv");
    let ivf = tmp("aq3_out.ivf");
    write_synthetic_raw(&raw);
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "--aq-mode",
            "3",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "encode must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ivf_bytes = std::fs::read(&ivf).expect("read ivf");
    assert_eq!(&ivf_bytes[0..4], b"DKIF", "bad IVF magic with --aq-mode 3");
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// --fps parsing branches
// ========================================================================

/// `--fps N` (single integer, no slash) must be accepted as `N/1`. The
/// only observable from the CLI is "encode succeeds with this fps".
#[test]
fn enc_fps_single_integer_accepted() {
    let raw = tmp("fps_int_in.yuv");
    let ivf = tmp("fps_int_out.ivf");
    write_synthetic_raw(&raw);
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "60",
            "--target-bitrate",
            "500",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "--fps 60 must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // IVF stores fps as (num,den) at offset 16/20. Bytes 16-19 = num le u32.
    let ivf_bytes = std::fs::read(&ivf).expect("read ivf");
    let num = u32::from_le_bytes([ivf_bytes[16], ivf_bytes[17], ivf_bytes[18], ivf_bytes[19]]);
    let den = u32::from_le_bytes([ivf_bytes[20], ivf_bytes[21], ivf_bytes[22], ivf_bytes[23]]);
    assert_eq!((num, den), (60, 1), "--fps 60 must encode as 60/1");
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

/// `--fps abc` (bad single value) rejected with the bad token in the message.
#[test]
fn enc_fps_bad_single_value_rejected() {
    let out = Command::new(enc_bin())
        .args(["--fps", "abc", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--fps") && stderr.contains("abc"),
        "stderr must echo --fps and 'abc', got: {stderr}"
    );
}

/// `--fps abc/1` rejected with mention of the numerator.
#[test]
fn enc_fps_bad_numerator_rejected() {
    let out = Command::new(enc_bin())
        .args(["--fps", "abc/1", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--fps") && stderr.contains("numerator"),
        "stderr must mention --fps numerator, got: {stderr}"
    );
}

/// `--fps 30/abc` rejected with mention of the denominator.
#[test]
fn enc_fps_bad_denominator_rejected() {
    let out = Command::new(enc_bin())
        .args(["--fps", "30/abc", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--fps") && stderr.contains("denominator"),
        "stderr must mention --fps denominator, got: {stderr}"
    );
}

/// `--fps 30/0` rejected with a zero-denominator-specific message.
#[test]
fn enc_fps_zero_denominator_rejected() {
    let out = Command::new(enc_bin())
        .args(["--fps", "30/0", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--fps") && stderr.contains("non-zero"),
        "stderr must mention non-zero denominator, got: {stderr}"
    );
}

// ========================================================================
// Positional / output argument validation
// ========================================================================

#[test]
fn enc_no_input_with_flags_errors() {
    let out = Command::new(enc_bin())
        .args(["-w", "64", "-h", "64", "-o", "x.ivf"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no input file given"),
        "stderr must say 'no input file given', got: {stderr}"
    );
}

#[test]
fn enc_two_inputs_errors() {
    let out = Command::new(enc_bin())
        .args(["-w", "64", "-h", "64", "-o", "x.ivf", "a.yuv", "b.yuv"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("expected exactly one input file"),
        "stderr must say 'expected exactly one input file', got: {stderr}"
    );
}

#[test]
fn enc_missing_output_errors() {
    let raw = tmp("noout_in.yuv");
    write_synthetic_raw(&raw);
    let out = Command::new(enc_bin())
        .args(["-w", "64", "-h", "64"])
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no output file"),
        "stderr must report missing output, got: {stderr}"
    );
    let _ = std::fs::remove_file(&raw);
}

#[test]
fn dec_two_inputs_errors() {
    let out = Command::new(dec_bin())
        .args(["a.ivf", "b.ivf"])
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("expected exactly one input file"),
        "stderr must say 'expected exactly one input file', got: {stderr}"
    );
}

// ========================================================================
// Raw-input dimension validation (encoder)
// ========================================================================

#[test]
fn enc_raw_input_requires_width() {
    let raw = tmp("raw_no_w.yuv");
    std::fs::write(&raw, vec![0u8; 100]).expect("write");
    let out = Command::new(enc_bin())
        .args(["-o", "x.ivf"])
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("requires -w") || stderr.contains("--width"),
        "stderr must require -w/--width, got: {stderr}"
    );
    let _ = std::fs::remove_file(&raw);
}

#[test]
fn enc_raw_input_requires_height() {
    let raw = tmp("raw_no_h.yuv");
    std::fs::write(&raw, vec![0u8; 100]).expect("write");
    let out = Command::new(enc_bin())
        .args(["-w", "64", "-o", "x.ivf"])
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("requires -h") || stderr.contains("--height"),
        "stderr must require -h/--height, got: {stderr}"
    );
    let _ = std::fs::remove_file(&raw);
}

#[test]
fn enc_raw_invalid_geometry_precedes_frame_errors_and_preserves_outputs() {
    let raw = tmp("raw_invalid_geometry.yuv");
    let output = tmp("raw_invalid_geometry.ivf");
    let csv = tmp("raw_invalid_geometry.csv");
    let output_sentinel = b"existing IVF must survive invalid raw geometry";
    let csv_sentinel = b"existing CSV must survive invalid raw geometry";
    std::fs::write(&raw, b"tiny partial raw input").expect("write tiny raw input");
    std::fs::write(&output, output_sentinel).expect("write output sentinel");
    std::fs::write(&csv, csv_sentinel).expect("write CSV sentinel");

    let out = Command::new(enc_bin())
        .args(["--lag-in-frames", "0", "--frame-time-csv"])
        .arg(&csv)
        .args(["-w", "16448", "-h", "64", "--target-bitrate", "500", "-o"])
        .arg(&output)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc with invalid raw geometry");
    assert!(!out.status.success(), "invalid geometry must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("raw input geometry 16448x64")
            && stderr.contains("is outside supported encoder limits")
            && stderr.contains("16384")
            && stderr.contains("67108864"),
        "stderr must identify the geometry and supported limits before frame I/O errors, got: {stderr}"
    );
    assert!(
        !stderr.contains("whole number") && !stderr.contains("reading raw"),
        "geometry rejection must precede partial-file and frame-read errors, got: {stderr}"
    );
    assert_eq!(
        std::fs::read(&output).expect("read protected output"),
        output_sentinel,
        "invalid geometry must preserve an existing output"
    );
    assert_eq!(
        std::fs::read(&csv).expect("read protected CSV"),
        csv_sentinel,
        "invalid geometry must preserve an existing CSV"
    );
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&output);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn enc_raw_partial_frame_rejected() {
    let raw = tmp("raw_partial.yuv");
    // 100 bytes is not a whole multiple of 6144 (a 64x64 I420 frame).
    std::fs::write(&raw, vec![0u8; 100]).expect("write");
    let out = Command::new(enc_bin())
        .args(["-w", "64", "-h", "64", "-o", "x.ivf"])
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("whole number") && stderr.contains("I420"),
        "stderr must explain partial-frame error, got: {stderr}"
    );
    let _ = std::fs::remove_file(&raw);
}

#[test]
fn enc_raw_empty_input_rejected() {
    let raw = tmp("raw_empty.yuv");
    std::fs::write(&raw, b"").expect("write");
    let ivf = tmp("raw_empty_out.ivf");
    let out = Command::new(enc_bin())
        .args(["-w", "64", "-h", "64", "-o"])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no frames"),
        "stderr must say input had no frames, got: {stderr}"
    );
    let _ = std::fs::remove_file(&raw);
}

// ========================================================================
// Encoder --verbose / --limit / --ivf wiring
// ========================================================================

/// `--verbose` (alias `-v`) must produce a recognisable progress line on
/// stderr that mentions frame count and dimensions.
#[test]
fn enc_verbose_emits_summary_line() {
    let raw = tmp("verb_in.yuv");
    let ivf = tmp("verb_out.ivf");
    write_synthetic_raw(&raw);
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "-v",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("encoded") && stderr.contains("64x64"),
        "stderr must contain 'encoded' + '64x64', got: {stderr}"
    );
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

#[test]
fn enc_parse_only_flags_reject_and_are_tagged_not_yet() {
    let unplumbed = [
        ("--profile", Some("2")),
        ("--tile-rows", Some("1")),
        ("--kf-min-dist", Some("8")),
        ("--disable-kf", None),
        ("--buf-sz", Some("6000")),
        ("--buf-initial-sz", Some("4000")),
        ("--buf-optimal-sz", Some("5000")),
        ("--undershoot-pct", Some("25")),
        ("--overshoot-pct", Some("50")),
        ("--lossless", Some("1")),
        ("--frame-parallel", Some("1")),
        ("--error-resilient", Some("1")),
        ("--tune", Some("psnr")),
        ("--svc", None),
        ("--bit-depth", Some("10")),
        ("--psnr", None),
        ("--arnr-maxframes", Some("7")),
        ("--arnr-strength", Some("5")),
        ("--pass", Some("2")),
        ("--fpf", Some("stats.fpf")),
        ("--rate-hist", Some("64")),
        ("--q-hist", Some("32")),
        ("--test-decode", Some("warn")),
        ("--frame-boost", Some("1")),
        ("--max-inter-rate", Some("120")),
        ("--min-gf-interval", Some("4")),
        ("--max-gf-interval", Some("16")),
        ("--color-space", Some("bt709")),
        ("--alt-ref-aq", Some("1")),
        ("--input-bit-depth", Some("8")),
        ("--target-level", Some("41")),
        ("--row-mt", Some("1")),
        ("--resize-allowed", Some("1")),
        ("--resize-width", Some("640")),
        ("--resize-height", Some("360")),
    ];

    let help = Command::new(enc_bin())
        .arg("--help")
        .bounded_output()
        .expect("run rvp9enc --help");
    assert!(help.status.success());
    let help = String::from_utf8_lossy(&help.stdout);

    for (flag, value) in unplumbed {
        let mut command = Command::new(enc_bin());
        command.arg(flag);
        if let Some(value) = value {
            command.arg(value);
        }
        command.arg("in.y4m");
        let output = command.bounded_output().expect("run rvp9enc");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success()
                && stderr.contains(flag)
                && stderr.contains("not supported yet"),
            "{flag} must fail as unimplemented, got: {stderr}"
        );

        let line = help
            .lines()
            .find(|line| {
                line.split_whitespace()
                    .any(|word| word.trim_end_matches(',') == flag)
            })
            .unwrap_or_else(|| panic!("{flag} missing from help"));
        assert!(
            line.contains("[not yet]") && !line.contains("[supported]"),
            "{flag} help tag is dishonest: {line}"
        );
    }
}

/// `--limit N` truncates the encode at N frames — the produced IVF must
/// declare N frames in its header, not the full source.
#[test]
fn enc_limit_truncates_output() {
    let raw = tmp("limit_in.yuv");
    let ivf = tmp("limit_out.ivf");
    write_synthetic_raw(&raw);
    let limit = 3;
    assert!(limit < N_FRAMES, "test setup");
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "--limit",
            "3",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ivf_bytes = std::fs::read(&ivf).expect("read ivf");
    let num_frames =
        u32::from_le_bytes([ivf_bytes[24], ivf_bytes[25], ivf_bytes[26], ivf_bytes[27]]);
    assert_eq!(
        num_frames as usize, limit,
        "IVF frame count must reflect --limit"
    );
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

/// A raw encode with `--limit` must not read or validate bytes after the
/// requested frames.
#[test]
fn enc_raw_limit_does_not_read_trailing_partial_frame() {
    let raw = tmp("limit_partial_in.yuv");
    let ivf = tmp("limit_partial_out.ivf");
    let frame_size = 64 * 64 * 3 / 2;
    let mut input = vec![128u8; frame_size];
    input.extend_from_slice(b"unread trailing partial frame");
    std::fs::write(&raw, input).expect("write raw fixture");

    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--target-bitrate",
            "500",
            "--limit",
            "1",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run limited rvp9enc");
    assert!(
        out.status.success(),
        "limited raw encode read the trailing partial frame: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

/// `--ivf` is a no-op (the only container) but accepting it must not
/// break the encode pipeline.
#[test]
fn enc_explicit_ivf_flag_is_noop() {
    let raw = tmp("ivf_in.yuv");
    let ivf = tmp("ivf_out.ivf");
    write_synthetic_raw(&raw);
    let out = Command::new(enc_bin())
        .args([
            "--ivf",
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ivf_bytes = std::fs::read(&ivf).expect("read ivf");
    assert_eq!(&ivf_bytes[0..4], b"DKIF");
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

/// `--kf-max-dist N` is wired — encoding with a tiny GOP succeeds. We
/// don't decode the bitstream to count keyframes here; the encoder's
/// internal accounting is exercised by the conformance suite. What this
/// test pins is that the *flag plumbing* accepts the value.
#[test]
fn enc_kf_max_dist_accepted() {
    let raw = tmp("kf_in.yuv");
    let ivf = tmp("kf_out.ivf");
    write_synthetic_raw(&raw);
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "--kf-max-dist",
            "2",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "encode must accept --kf-max-dist=2: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ivf_bytes = std::fs::read(&ivf).expect("read ivf");
    assert_eq!(&ivf_bytes[0..4], b"DKIF");
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// Encoder --version / no-args
// ========================================================================

#[test]
fn enc_version_prints_and_exits() {
    let out = Command::new(enc_bin())
        .arg("--version")
        .bounded_output()
        .expect("run rvp9enc --version");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("rvp9enc "),
        "stdout must start with 'rvp9enc ', got: {stdout}"
    );
}

#[test]
fn enc_no_args_emits_usage() {
    let out = Command::new(enc_bin())
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no input") && stderr.contains("Usage:"),
        "stderr must show 'no input' + usage, got: {stderr}"
    );
}

#[test]
fn dec_no_args_emits_usage() {
    let out = Command::new(dec_bin())
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no input") && stderr.contains("Usage:"),
        "stderr must show 'no input' + usage, got: {stderr}"
    );
}

#[test]
fn dec_version_prints_and_exits() {
    let out = Command::new(dec_bin())
        .arg("--version")
        .bounded_output()
        .expect("run rvp9dec --version");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("rvp9dec "),
        "stdout must start with 'rvp9dec ', got: {stdout}"
    );
}

// ========================================================================
// Decoder output-format selection
// ========================================================================

/// `--y4m -o file.y4m` produces a Y4M container starting with the
/// YUV4MPEG2 magic.
#[test]
fn dec_y4m_output_has_yuv4mpeg2_magic() {
    let ivf = encode_synthetic_ivf("y4m");
    let out_path = tmp("y4m.y4m");
    let out = Command::new(dec_bin())
        .args(["--y4m", "-o"])
        .arg(&out_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "rvp9dec --y4m failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&out_path).expect("read y4m out");
    assert!(
        bytes.starts_with(b"YUV4MPEG2"),
        "y4m output must start with YUV4MPEG2, got first 16: {:?}",
        &bytes[..16.min(bytes.len())]
    );
    // FRAME tag must appear at least once per displayed frame.
    let frame_count = bytes.windows(6).filter(|w| *w == b"FRAME\n").count();
    assert_eq!(frame_count, N_FRAMES, "one FRAME tag per displayed frame");
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out_path);
}

/// Output extension `.y4m` auto-selects Y4M (no `--y4m` flag needed).
#[test]
fn dec_y4m_inferred_from_extension() {
    let ivf = encode_synthetic_ivf("y4m_ext");
    let out_path = tmp("inferred.y4m");
    let out = Command::new(dec_bin())
        .arg("-o")
        .arg(&out_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&out_path).expect("read y4m out");
    assert!(
        bytes.starts_with(b"YUV4MPEG2"),
        ".y4m extension must auto-select Y4M output"
    );
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out_path);
}

/// `--i420` produces tight-packed I420 planes (Y then U then V), no
/// container. The output file length must equal `N_FRAMES * FRAME_BYTES`.
#[test]
fn dec_i420_output_has_no_container() {
    let ivf = encode_synthetic_ivf("i420");
    let out_path = tmp("i420.yuv");
    let out = Command::new(dec_bin())
        .args(["--i420", "-o"])
        .arg(&out_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let len = std::fs::metadata(&out_path).expect("stat").len() as usize;
    assert_eq!(
        len,
        N_FRAMES * FRAME_BYTES,
        "i420 output must be exactly N_FRAMES * frame_bytes"
    );
    // No "YUV4MPEG2", no "FRAME\n".
    let bytes = std::fs::read(&out_path).expect("read");
    assert!(
        !bytes.starts_with(b"YUV4MPEG2"),
        "i420 must NOT have Y4M magic"
    );
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out_path);
}

/// `--rawvideo` behaves identically to `--i420`.
#[test]
fn dec_rawvideo_equivalent_to_i420() {
    let ivf = encode_synthetic_ivf("rawvideo");
    let raw_path = tmp("rawvideo.yuv");
    let i420_path = tmp("rawvideo_i420.yuv");

    let out_raw = Command::new(dec_bin())
        .args(["--rawvideo", "-o"])
        .arg(&raw_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --rawvideo");
    assert!(
        out_raw.status.success(),
        "{}",
        String::from_utf8_lossy(&out_raw.stderr)
    );

    let out_i = Command::new(dec_bin())
        .args(["--i420", "-o"])
        .arg(&i420_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --i420");
    assert!(out_i.status.success());

    let raw_bytes = std::fs::read(&raw_path).expect("read rawvideo");
    let i_bytes = std::fs::read(&i420_path).expect("read i420");
    assert_eq!(
        raw_bytes, i_bytes,
        "--rawvideo and --i420 must produce identical bytes"
    );
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&raw_path);
    let _ = std::fs::remove_file(&i420_path);
}

/// `--yv12` swaps the U and V planes vs `--i420`. The total byte length
/// must match; the Y plane bytes must match; the U/V plane bytes must
/// be swapped.
#[test]
fn dec_yv12_swaps_u_and_v_vs_i420() {
    let ivf = encode_synthetic_ivf("yv12");
    let i_path = tmp("yv12_i420.yuv");
    let yv_path = tmp("yv12_yv12.yuv");

    let s1 = Command::new(dec_bin())
        .args(["--i420", "-o"])
        .arg(&i_path)
        .arg(&ivf)
        .bounded_status()
        .expect("run --i420");
    assert!(s1.success());
    let s2 = Command::new(dec_bin())
        .args(["--yv12", "-o"])
        .arg(&yv_path)
        .arg(&ivf)
        .bounded_status()
        .expect("run --yv12");
    assert!(s2.success());

    let i = std::fs::read(&i_path).expect("read i420");
    let yv = std::fs::read(&yv_path).expect("read yv12");
    assert_eq!(i.len(), yv.len(), "total byte length must match");

    let y_size = W * H;
    let uv_size = (W / 2) * (H / 2);
    let frame_size = y_size + 2 * uv_size;
    assert_eq!(i.len(), frame_size * N_FRAMES, "test setup");

    for f in 0..N_FRAMES {
        let off = f * frame_size;
        let i_y = &i[off..off + y_size];
        let yv_y = &yv[off..off + y_size];
        assert_eq!(i_y, yv_y, "frame {f}: Y plane must match");

        let i_u = &i[off + y_size..off + y_size + uv_size];
        let i_v = &i[off + y_size + uv_size..off + frame_size];
        let yv_v = &yv[off + y_size..off + y_size + uv_size];
        let yv_u = &yv[off + y_size + uv_size..off + frame_size];
        assert_ne!(
            i_u, i_v,
            "frame {f}: fixture must decode distinct U and V planes"
        );
        assert_eq!(
            i_u, yv_u,
            "frame {f}: i420 U bytes must equal yv12's second-half (U) bytes"
        );
        assert_eq!(
            i_v, yv_v,
            "frame {f}: i420 V bytes must equal yv12's first-half (V) bytes"
        );
    }
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&i_path);
    let _ = std::fs::remove_file(&yv_path);
}

/// Multiple format flags: last-wins (the apply functions just overwrite
/// the format slot). The decoder must accept the combination without
/// erroring and produce output in the last-specified format.
#[test]
fn dec_format_flags_last_wins() {
    let ivf = encode_synthetic_ivf("conflict");
    let out_path = tmp("conflict.out");
    // --y4m then --i420 → I420 (not Y4M).
    let out = Command::new(dec_bin())
        .args(["--y4m", "--i420", "-o"])
        .arg(&out_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&out_path).expect("read");
    assert!(
        !bytes.starts_with(b"YUV4MPEG2"),
        "last-wins: --i420 after --y4m must produce I420 (no Y4M magic)"
    );
    assert_eq!(bytes.len(), N_FRAMES * FRAME_BYTES);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out_path);
}

// ========================================================================
// Decoder --md5 / --framemd5 / --noblit shapes
// ========================================================================

/// `--framemd5` emits one `index: hex32` line per displayed frame, indexed
/// from 0.
#[test]
fn dec_framemd5_indexed_lines() {
    let ivf = encode_synthetic_ivf("fmd5");
    let out = Command::new(dec_bin())
        .arg("--framemd5")
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --framemd5");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), N_FRAMES, "one indexed md5 per displayed frame");
    for (i, l) in lines.iter().enumerate() {
        let parts: Vec<&str> = l.splitn(2, ": ").collect();
        assert_eq!(parts.len(), 2, "framemd5 line must be 'index: hex': {l}");
        assert_eq!(
            parts[0].parse::<usize>().expect("index parses"),
            i,
            "framemd5 indices must be 0..N: {l}"
        );
        assert_eq!(parts[1].len(), 32, "hex digest must be 32 chars: {l}");
        assert!(
            parts[1].bytes().all(|b| b.is_ascii_hexdigit()),
            "hex digest must be all hex: {l}"
        );
    }
    let _ = std::fs::remove_file(&ivf);
}

/// One decoder call may contain several shown VP9 components (spatial-SVC
/// layers). Match libvpx by decoding all components but emitting only the
/// final component from the packet.
#[test]
fn dec_superframe_emits_only_final_shown_component() {
    let source = encode_synthetic_ivf("shown_superframe_source");
    let source_out = Command::new(dec_bin())
        .args(["--md5", "--limit", "2"])
        .arg(&source)
        .bounded_output()
        .expect("decode source frames");
    assert!(
        source_out.status.success(),
        "{}",
        String::from_utf8_lossy(&source_out.stderr)
    );
    let source_md5s: Vec<&str> = std::str::from_utf8(&source_out.stdout)
        .expect("source MD5 output is UTF-8")
        .lines()
        .collect();
    assert_eq!(source_md5s.len(), 2);

    let packed = repack_first_two_frames_as_superframe(&source, "shown_superframe");
    let packed_out = Command::new(dec_bin())
        .args(["--md5", "--summary"])
        .arg(&packed)
        .bounded_output()
        .expect("decode packed superframe");
    assert!(
        packed_out.status.success(),
        "{}",
        String::from_utf8_lossy(&packed_out.stderr)
    );
    let packed_md5s: Vec<&str> = std::str::from_utf8(&packed_out.stdout)
        .expect("packed MD5 output is UTF-8")
        .lines()
        .collect();
    assert_eq!(packed_md5s, [source_md5s[1]]);
    let summary = String::from_utf8_lossy(&packed_out.stderr);
    assert!(
        summary.contains("1 decoded frames/1 showed frames"),
        "unexpected summary: {summary}"
    );

    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&packed);
}

/// `--noblit` decodes silently — no stdout output, exit 0.
#[test]
fn dec_noblit_emits_no_output() {
    let ivf = encode_synthetic_ivf("noblit");
    let out = Command::new(dec_bin())
        .arg("--noblit")
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --noblit");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "--noblit must emit no stdout, got {} bytes",
        out.stdout.len()
    );
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// Decoder --limit / --skip flow control
// ========================================================================

/// `--limit N` caps the displayed frame count to N (md5 line count is the
/// observable).
#[test]
fn dec_limit_caps_output_frame_count() {
    let ivf = encode_synthetic_ivf("dec_limit");
    let out = Command::new(dec_bin())
        .args(["--md5", "--limit", "2"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --limit");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 2, "--limit 2 must produce exactly 2 md5 lines");
    let _ = std::fs::remove_file(&ivf);
}

/// A limited decode must stop pulling container packets after the requested
/// displayed frame, leaving malformed unread tail records untouched.
#[test]
fn dec_limit_does_not_read_trailing_bad_ivf_record() {
    let ivf = encode_synthetic_ivf("dec_limit_bad_tail");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&ivf)
        .expect("open IVF fixture");
    use std::io::Write as _;
    file.write_all(&100u32.to_le_bytes())
        .expect("append bad frame size");
    file.write_all(&99u64.to_le_bytes())
        .expect("append bad frame pts");
    drop(file);

    let limited = Command::new(dec_bin())
        .args(["--md5", "--limit", "1"])
        .arg(&ivf)
        .bounded_output()
        .expect("run limited rvp9dec");
    assert!(
        limited.status.success(),
        "limited decode read the malformed tail: {}",
        String::from_utf8_lossy(&limited.stderr)
    );

    let unlimited = Command::new(dec_bin())
        .arg("--md5")
        .arg(&ivf)
        .bounded_output()
        .expect("run unlimited rvp9dec");
    assert!(
        !unlimited.status.success()
            && String::from_utf8_lossy(&unlimited.stderr).contains("declared 100 bytes"),
        "unlimited decode must still validate the tail: {}",
        String::from_utf8_lossy(&unlimited.stderr)
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--skip N` drops the first N displayed frames; remaining count is
/// `total - N`.
#[test]
fn dec_skip_drops_leading_frames() {
    let ivf = encode_synthetic_ivf("dec_skip");
    let out = Command::new(dec_bin())
        .args(["--md5", "--skip", "4"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --skip");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        N_FRAMES - 4,
        "--skip 4 with {N_FRAMES} frames must leave {} lines",
        N_FRAMES - 4
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--skip` + `--limit` combine: skip first K, then take L.
#[test]
fn dec_skip_then_limit_combines() {
    let ivf = encode_synthetic_ivf("dec_skip_lim");
    let out = Command::new(dec_bin())
        .args(["--md5", "--skip", "2", "--limit", "2"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --skip --limit");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        2,
        "--skip 2 --limit 2 must produce 2 md5 lines (got: {stdout})"
    );

    // The skipped lines must match the equivalent indexed-md5 frames 2..4.
    let full = Command::new(dec_bin())
        .arg("--framemd5")
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --framemd5");
    assert!(full.status.success());
    let full_text = String::from_utf8_lossy(&full.stdout);
    let frame_2_md5 = full_text
        .lines()
        .find(|l| l.starts_with("2: "))
        .expect("frame 2");
    let frame_3_md5 = full_text
        .lines()
        .find(|l| l.starts_with("3: "))
        .expect("frame 3");
    let expected_2 = &frame_2_md5[3..]; // strip "N: "
    let expected_3 = &frame_3_md5[3..];
    assert_eq!(lines[0], expected_2, "first --md5 line must match frame 2");
    assert_eq!(lines[1], expected_3, "second --md5 line must match frame 3");

    let _ = std::fs::remove_file(&ivf);
}

/// `--skip` value larger than total displayed frames → zero output, exit 0.
#[test]
fn dec_skip_more_than_available_yields_empty() {
    let ivf = encode_synthetic_ivf("dec_skip_big");
    let out = Command::new(dec_bin())
        .args(["--md5", "--skip", "100"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(out.status.success());
    assert!(
        out.stdout.is_empty(),
        "--skip beyond stream length must emit no md5 lines"
    );
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// Decoder --summary / --progress side channels
// ========================================================================

/// `--summary` prints a recognisable single-line summary on stderr.
#[test]
fn dec_summary_emits_summary_line() {
    let ivf = encode_synthetic_ivf("summary");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--summary"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --summary");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // vpxdec-compatible format (rvp9dec.rs:340):
    //   "<frame_in> decoded frames/<frame_out> showed frames in <us> us (<fps> fps)"
    assert!(
        stderr.contains("decoded frames") && stderr.contains("showed frames"),
        "stderr must contain summary line, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--progress` emits per-frame progress markers on stderr.
#[test]
fn dec_progress_emits_per_frame_marker() {
    let ivf = encode_synthetic_ivf("progress");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--progress"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --progress");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("rvp9dec") && stderr.contains("frames decoded"),
        "stderr must contain progress marker, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// Decoder accepted-with-warning flags (--scale, --output-bit-depth,
// --error-concealment, --frame-buffers)
// ========================================================================

/// `--scale` is accepted but emits a "not yet implemented" warning on
/// stderr.
#[test]
fn dec_scale_accepted_with_warning() {
    let ivf = encode_synthetic_ivf("scale");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--scale"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --scale");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--scale") && stderr.contains("not yet"),
        "stderr must warn about --scale, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `-S` short form for --scale works identically.
#[test]
fn dec_scale_short_form_accepted() {
    let ivf = encode_synthetic_ivf("scale_s");
    let out = Command::new(dec_bin())
        .args(["--noblit", "-S"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec -S");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--scale"),
        "stderr must mention --scale for -S, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--output-bit-depth 8` is accepted silently (no warning), exit 0.
#[test]
fn dec_output_bit_depth_eight_silent() {
    let ivf = encode_synthetic_ivf("obd8");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--output-bit-depth", "8"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --output-bit-depth 8");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // No "only 8-bit" warning for the supported case.
    assert!(
        !stderr.contains("only 8-bit"),
        "--output-bit-depth 8 should not warn, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--output-bit-depth 10` is accepted but warns that only 8-bit is
/// supported.
#[test]
fn dec_output_bit_depth_ten_warns() {
    let ivf = encode_synthetic_ivf("obd10");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--output-bit-depth", "10"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --output-bit-depth 10");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--output-bit-depth") && stderr.contains("8-bit"),
        "stderr must warn about 10-bit, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--output-bit-depth` without value rejected.
#[test]
fn dec_output_bit_depth_missing_value_rejected() {
    let out = Command::new(dec_bin())
        .arg("--output-bit-depth")
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--output-bit-depth") && stderr.contains("requires a value"),
        "stderr must report missing value, got: {stderr}"
    );
}

/// `--error-concealment` accepted with a warning.
#[test]
fn dec_error_concealment_accepted_with_warning() {
    let ivf = encode_synthetic_ivf("ec");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--error-concealment"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --error-concealment");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--error-concealment") && stderr.contains("not yet"),
        "stderr must warn about --error-concealment, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--frame-buffers N` accepted with a warning (internal pool used).
#[test]
fn dec_frame_buffers_accepted_with_warning() {
    let ivf = encode_synthetic_ivf("fb");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--frame-buffers", "4"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --frame-buffers");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--frame-buffers") && stderr.contains("internal buffer pool"),
        "stderr must mention internal buffer pool, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--frame-buffers` without value rejected.
#[test]
fn dec_frame_buffers_missing_value_rejected() {
    let out = Command::new(dec_bin())
        .arg("--frame-buffers")
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--frame-buffers") && stderr.contains("requires a value"),
        "stderr must report missing value, got: {stderr}"
    );
}

/// The VP8-only postproc flags are silently accepted as no-ops (no
/// warning, exit 0). We pin one — `--postproc` — to ensure the silent
/// no-op branch is exercised.
#[test]
fn dec_postproc_silent_noop() {
    let ivf = encode_synthetic_ivf("pp");
    let baseline_stderr = {
        let o = Command::new(dec_bin())
            .args(["--noblit"])
            .arg(&ivf)
            .bounded_output()
            .expect("run rvp9dec");
        assert!(o.status.success());
        String::from_utf8_lossy(&o.stderr).into_owned()
    };
    let out = Command::new(dec_bin())
        .args([
            "--noblit",
            "--postproc",
            "--deblock",
            "--demacroblock",
            "--mfqe",
        ])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        stderr, baseline_stderr,
        "VP8 postproc flags must be silent no-ops"
    );
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// Decoder container demux failure surfaces
// ========================================================================

/// Truncated IVF (< 32-byte file header) → exit non-zero, stderr names
/// the truncation.
#[test]
fn dec_truncated_ivf_header_rejected() {
    let path = tmp("trunc_hdr.ivf");
    std::fs::write(&path, b"DKIF\x00\x00").expect("write");
    let out = Command::new(dec_bin())
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("truncated") || stderr.contains("IVF"),
        "stderr must explain truncation, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// 32-byte IVF header with wrong magic → exit non-zero, stderr mentions
/// DKIF.
#[test]
fn dec_bad_ivf_magic_rejected() {
    let path = tmp("badmagic.ivf");
    let mut bytes = vec![0u8; 32];
    bytes[0..4].copy_from_slice(b"BAAD");
    std::fs::write(&path, &bytes).expect("write");
    let out = Command::new(dec_bin())
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("DKIF") || stderr.contains("IVF"),
        "stderr must mention DKIF/IVF magic, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// IVF with a frame header declaring a payload that runs past EOF →
/// rejected.
#[test]
fn dec_ivf_frame_size_past_eof_rejected() {
    let path = tmp("badframe.ivf");
    // 32-byte valid header + a frame-header claiming 0xFFFF payload bytes
    // but no actual payload follows.
    let mut bytes = vec![0u8; 32];
    bytes[0..4].copy_from_slice(b"DKIF");
    bytes[8..12].copy_from_slice(b"VP90"); // codec fourcc
    bytes[12..14].copy_from_slice(&64u16.to_le_bytes()); // width
    bytes[14..16].copy_from_slice(&64u16.to_le_bytes()); // height
    bytes[16..20].copy_from_slice(&30u32.to_le_bytes()); // fps num
    bytes[20..24].copy_from_slice(&1u32.to_le_bytes()); // fps den
    // Frame header: u32 size_le = 0xFFFF, u64 pts_le = 0, no payload.
    let frame_hdr = [
        0xFFu8, 0xFFu8, 0u8, 0u8, // size_le = 65535
        0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, // pts
    ];
    bytes.extend_from_slice(&frame_hdr);
    std::fs::write(&path, &bytes).expect("write");
    let out = Command::new(dec_bin())
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.is_empty(),
        "stderr must explain the failure (got empty)"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dec_invalid_vp9_preserves_existing_output_file() {
    let input = tmp("invalid_vp9.ivf");
    let output = tmp("invalid_vp9_output.yuv");
    let sentinel = b"keep existing output";

    let mut bytes = vec![0u8; 32];
    bytes[0..4].copy_from_slice(b"DKIF");
    bytes[8..12].copy_from_slice(b"VP90");
    bytes[12..14].copy_from_slice(&64u16.to_le_bytes());
    bytes[14..16].copy_from_slice(&64u16.to_le_bytes());
    bytes[16..20].copy_from_slice(&30u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.push(0);
    std::fs::write(&input, &bytes).expect("write invalid IVF");
    std::fs::write(&output, sentinel).expect("write sentinel output");

    let result = Command::new(dec_bin())
        .args(["-o"])
        .arg(&output)
        .arg(&input)
        .bounded_output()
        .expect("run rvp9dec");

    let preserved = std::fs::read(&output).expect("read preserved output");
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    assert!(!result.status.success(), "invalid VP9 must fail");
    assert_eq!(
        preserved, sentinel,
        "decode failure before the first frame must not truncate output"
    );
}

#[test]
fn dec_i420_and_yv12_reject_profile1_444() {
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/corpus/profile1/foreman_c444_kfonly.ivf");
    assert!(
        input.is_file(),
        "missing Profile-1 fixture: {}",
        input.display()
    );

    for flag in ["--i420", "--yv12"] {
        let output = tmp(&format!("non420_{}.yuv", &flag[2..]));
        let result = Command::new(dec_bin())
            .args([flag, "-o"])
            .arg(&output)
            .arg(&input)
            .bounded_output()
            .expect("run rvp9dec");
        let _ = std::fs::remove_file(&output);

        assert!(!result.status.success(), "{flag} accepted 4:4:4 input");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains(flag) && stderr.contains("4:2:0"),
            "{flag} rejection must explain its 4:2:0 contract, got: {stderr}"
        );
    }
}

/// Decoding a file with an unknown extension → exit non-zero with a
/// recognisable error from the container layer.
#[test]
fn dec_unknown_extension_rejected() {
    let path = tmp("mystery.xyz");
    std::fs::write(&path, b"X").expect("write");
    let out = Command::new(dec_bin())
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The container layer surfaces this via ContainerError; the CLI
    // forwards the underlying message. We just check the file path
    // appears and stderr is non-empty (the precise wording is owned by
    // rvp9-io and pinned by tests there).
    assert!(
        stderr.contains("mystery.xyz") || stderr.contains("xyz"),
        "stderr must mention the offending file/extension, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Decoding a path that doesn't exist → exit non-zero with stderr that
/// names the path.
#[test]
fn dec_missing_input_rejected() {
    let path = tmp("does_not_exist.ivf");
    let _ = std::fs::remove_file(&path);
    let out = Command::new(dec_bin())
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("does_not_exist") && stderr.contains("reading"),
        "stderr must report 'reading <path>', got: {stderr}"
    );
}

// ========================================================================
// Decoder --limit / --skip numeric validation
// ========================================================================

#[test]
fn dec_limit_non_numeric_rejected() {
    let out = Command::new(dec_bin())
        .args(["--limit", "abc"])
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--limit") && stderr.contains("abc"),
        "stderr must echo --limit and 'abc', got: {stderr}"
    );
}

#[test]
fn dec_skip_negative_rejected() {
    // --skip parses as `usize`, so '-1' is rejected by parse_value.
    let out = Command::new(dec_bin())
        .args(["--skip", "-1"])
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--skip") && stderr.contains("-1"),
        "stderr must echo --skip and '-1', got: {stderr}"
    );
}

// ========================================================================
// Help output sanity: every wired flag appears in --help
// ========================================================================

#[test]
fn enc_help_lists_every_wired_flag() {
    let out = Command::new(enc_bin())
        .arg("--help")
        .bounded_output()
        .expect("run rvp9enc --help");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--help",
        "--version",
        "--codec",
        "--output",
        "--width",
        "--height",
        "--fps",
        "--target-bitrate",
        "--end-usage",
        "--passes",
        "--frame-time-csv",
        "--kf-max-dist",
        "--aq-mode",
        "--limit",
        "--ivf",
        "--verbose",
    ] {
        assert!(
            text.contains(flag),
            "enc --help must list {flag}, got:\n{text}"
        );
    }
    assert!(
        text.contains("delayed scheduling")
            && text.contains("--lag-in-frames")
            && text.contains("--auto-alt-ref")
            && text.contains("omitted or 0"),
        "enc --help must describe the two-pass lookahead contract, got:\n{text}"
    );
}

// ========================================================================
// Y4M input path for the encoder
// ========================================================================

/// Write a minimal Y4M (4:2:0, C420jpeg) clip with `n` frames at the
/// given dimensions. Used to exercise the encoder's Y4M input path.
fn write_synthetic_y4m(path: &Path, w: usize, h: usize, n: usize, fps_num: u32, fps_den: u32) {
    let mut buf = Vec::new();
    buf.extend_from_slice(
        format!("YUV4MPEG2 W{w} H{h} F{fps_num}:{fps_den} Ip A1:1 C420jpeg\n").as_bytes(),
    );
    for f in 0..n {
        buf.extend_from_slice(b"FRAME\n");
        // Y plane
        for y in 0..h {
            for x in 0..w {
                buf.push(((x + y + f * 3) & 0xff) as u8);
            }
        }
        // U/V planes (4:2:0)
        for _ in 0..2 {
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    buf.push(((x + y + f) & 0xff) as u8);
                }
            }
        }
    }
    std::fs::write(path, &buf).expect("write y4m");
}

/// `.y4m` extension triggers the y4m reader path. Successful encode of
/// a small 64x64 y4m clip is the contract.
#[test]
fn enc_y4m_input_encodes_successfully() {
    let y4m = tmp("y4m_in.y4m");
    let ivf = tmp("y4m_out.ivf");
    write_synthetic_y4m(&y4m, 64, 64, 4, 30, 1);
    let out = Command::new(enc_bin())
        .args(["--target-bitrate", "500", "-o"])
        .arg(&ivf)
        .arg(&y4m)
        .bounded_output()
        .expect("run rvp9enc on y4m");
    assert!(
        out.status.success(),
        "encode of y4m must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&ivf).expect("read ivf");
    assert_eq!(&bytes[0..4], b"DKIF");
    // The Y4M header declared 30/1 — IVF must carry that.
    let num = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let den = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    assert_eq!((num, den), (30, 1), "fps should come from y4m header");
    // Frame count.
    let n = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert_eq!(n, 4, "y4m frame count must propagate");

    let _ = std::fs::remove_file(&y4m);
    let _ = std::fs::remove_file(&ivf);
}

/// Y4M input with `--fps` override: the CLI flag wins over the Y4M
/// header's framerate.
#[test]
fn enc_y4m_fps_override_wins() {
    let y4m = tmp("y4m_fps_in.y4m");
    let ivf = tmp("y4m_fps_out.ivf");
    write_synthetic_y4m(&y4m, 64, 64, 3, 30, 1);
    let out = Command::new(enc_bin())
        .args(["--target-bitrate", "500", "--fps", "60/1", "-o"])
        .arg(&ivf)
        .arg(&y4m)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&ivf).expect("read ivf");
    let num = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let den = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    assert_eq!((num, den), (60, 1), "--fps must override y4m header");
    let _ = std::fs::remove_file(&y4m);
    let _ = std::fs::remove_file(&ivf);
}

/// Y4M with `--limit` shorter than the source: only `limit` frames
/// encoded.
#[test]
fn enc_y4m_limit_truncates() {
    let y4m = tmp("y4m_lim_in.y4m");
    let ivf = tmp("y4m_lim_out.ivf");
    write_synthetic_y4m(&y4m, 64, 64, 6, 30, 1);
    let out = Command::new(enc_bin())
        .args(["--target-bitrate", "500", "--limit", "2", "-o"])
        .arg(&ivf)
        .arg(&y4m)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&ivf).expect("read ivf");
    let n = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert_eq!(n, 2, "--limit must truncate y4m encode to 2 frames");
    let _ = std::fs::remove_file(&y4m);
    let _ = std::fs::remove_file(&ivf);
}

/// Y4M with coded size *larger* than the source must be rejected.
#[test]
fn enc_y4m_coded_larger_than_source_rejected() {
    let y4m = tmp("y4m_big_in.y4m");
    write_synthetic_y4m(&y4m, 64, 64, 1, 30, 1);
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "128",
            "-h",
            "128",
            "--target-bitrate",
            "500",
            "-o",
            "x.ivf",
        ])
        .arg(&y4m)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success(), "must reject oversized coded dims");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("exceeds source"),
        "stderr must mention 'exceeds source', got: {stderr}"
    );
    let _ = std::fs::remove_file(&y4m);
}

// ========================================================================
// Decoder happy-path --codec / IVF framerate edge cases
// ========================================================================

/// `--codec vp9` is the happy-path arm — accepted silently.
#[test]
fn dec_codec_vp9_accepted() {
    let ivf = encode_synthetic_ivf("codec_vp9");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--codec", "vp9"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --codec vp9");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_file(&ivf);
}

/// `--codec` without value rejected.
#[test]
fn dec_codec_missing_value_rejected() {
    let out = Command::new(dec_bin())
        .arg("--codec")
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--codec") && stderr.contains("requires a value"),
        "stderr must report missing --codec value, got: {stderr}"
    );
}

/// `--codec` without value (encoder side).
#[test]
fn enc_codec_missing_value_rejected() {
    let out = Command::new(enc_bin())
        .arg("--codec")
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--codec") && stderr.contains("requires a value"),
        "stderr must report missing --codec value, got: {stderr}"
    );
}

/// Y4M output across multiple frames writes exactly ONE YUV4MPEG2
/// header (the per-frame `FRAME\n` tag repeats). This pins the
/// `if !*y4m_header_written` false-branch.
#[test]
fn dec_y4m_single_header_then_per_frame_tags() {
    let ivf = encode_synthetic_ivf("y4m_multi");
    let out_path = tmp("y4m_multi.y4m");
    let out = Command::new(dec_bin())
        .args(["--y4m", "-o"])
        .arg(&out_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --y4m");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&out_path).expect("read y4m");
    let header_count = bytes
        .windows(b"YUV4MPEG2".len())
        .filter(|w| *w == b"YUV4MPEG2")
        .count();
    assert_eq!(
        header_count, 1,
        "exactly one YUV4MPEG2 header for the stream"
    );
    let frame_count = bytes.windows(6).filter(|w| *w == b"FRAME\n").count();
    assert_eq!(frame_count, N_FRAMES, "one FRAME per displayed frame");
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out_path);
}

/// Decoder must run without `-o` (stdout sink) — exercises the
/// `open_output(None)` branch. The captured stdout is the I420 bytes.
#[test]
fn dec_no_output_uses_stdout() {
    let ivf = encode_synthetic_ivf("stdout_dec");
    let out = Command::new(dec_bin())
        .args(["--i420"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec --i420 (stdout)");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        out.stdout.len(),
        N_FRAMES * FRAME_BYTES,
        "stdout must carry the full I420 stream when no -o given"
    );
    let _ = std::fs::remove_file(&ivf);
}

/// Output path without recognised extension (and no format flag)
/// defaults to raw I420. This pins the `else => RawI420` arm of
/// `resolve_format`.
#[test]
fn dec_default_format_for_unknown_extension() {
    let ivf = encode_synthetic_ivf("unk_ext");
    let out_path = tmp("output_no_ext.bin");
    let out = Command::new(dec_bin())
        .arg("-o")
        .arg(&out_path)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&out_path).expect("read");
    assert!(
        !bytes.starts_with(b"YUV4MPEG2"),
        "unknown extension must default to raw I420 (no Y4M magic)"
    );
    assert_eq!(bytes.len(), N_FRAMES * FRAME_BYTES);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out_path);
}

/// IVF with framerate fields = 0 — the decoder must fall back to 30/1
/// rather than panic. We craft an IVF with a single zero-payload frame
/// to exercise `ivf_framerate`'s `num == 0 || den == 0` -> `None` branch.
#[test]
fn dec_ivf_zero_framerate_uses_default() {
    let path = tmp("zerofps.ivf");
    let mut bytes = vec![0u8; 32];
    bytes[0..4].copy_from_slice(b"DKIF");
    bytes[8..12].copy_from_slice(b"VP90");
    bytes[12..14].copy_from_slice(&64u16.to_le_bytes());
    bytes[14..16].copy_from_slice(&64u16.to_le_bytes());
    bytes[16..20].copy_from_slice(&0u32.to_le_bytes()); // fps_num = 0
    bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // fps_den = 0
    // No frames — read_packets returns an empty packet list, decode loop
    // runs zero iterations, ivf_framerate is called and returns None
    // (taking the zero-fps branch); decoder exits cleanly with summary 0.
    std::fs::write(&path, &bytes).expect("write");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--summary"])
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "must accept empty IVF with zero fps: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("0 decoded frames"),
        "summary must report 0 frames, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// `--summary` reports decoded + showed counts on a single vpxdec-compatible
/// line. Note: the vpxdec-compatible format intentionally omits the `emitted`
/// (post-skip-and-limit) count to stay byte-compatible with vpxdec's
/// `show_progress` output (see rvp9dec.rs:342-345).
#[test]
fn dec_summary_reports_skipped_and_emitted_counts() {
    let ivf = encode_synthetic_ivf("sum_counts");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--summary", "--skip", "1", "--limit", "2"])
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Summary line: "<frame_in> decoded frames/<frame_out> showed frames in ...".
    // The reader stops after one skipped and two emitted displayed frames.
    // Both counters therefore stop at 3 rather than scanning the remaining
    // three packets.
    assert!(
        stderr.contains("3 decoded frames") && stderr.contains("3 showed frames"),
        "summary must report 3 decoded / 3 showed frames, got: {stderr}"
    );
    let _ = std::fs::remove_file(&ivf);
}

// ========================================================================
// Help-after-parse paths (--help mixed with other flags is handled by the
// main early-out, but the post-parse `if opts.help` branch is also
// reachable when --help comes after a value-bearing flag in some shells)
// ========================================================================

/// Encoder `--help` short-circuits before parsing positionals.
#[test]
fn enc_help_short_circuits_before_validation() {
    // Pass deliberately bogus positionals: --help must still succeed.
    let out = Command::new(enc_bin())
        .args(["--help", "--bogus", "extra"])
        .bounded_output()
        .expect("run rvp9enc --help");
    assert!(out.status.success(), "--help must short-circuit");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Usage:"), "--help must print usage");
}

/// Decoder `--help` short-circuits.
#[test]
fn dec_help_short_circuits_before_validation() {
    let out = Command::new(dec_bin())
        .args(["--help", "--bogus", "extra"])
        .bounded_output()
        .expect("run rvp9dec --help");
    assert!(out.status.success(), "--help must short-circuit");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Usage:"), "--help must print usage");
}

/// IVF with non-zero num and zero den exercises the `den == 0` clause
/// of `ivf_framerate`'s validation (the `num == 0` clause is exercised
/// by `dec_ivf_zero_framerate_uses_default`; the OR short-circuits in
/// that case, so this test pins the second clause).
#[test]
fn dec_ivf_zero_den_uses_default_framerate() {
    let path = tmp("zeroden.ivf");
    let mut bytes = vec![0u8; 32];
    bytes[0..4].copy_from_slice(b"DKIF");
    bytes[8..12].copy_from_slice(b"VP90");
    bytes[12..14].copy_from_slice(&64u16.to_le_bytes());
    bytes[14..16].copy_from_slice(&64u16.to_le_bytes());
    bytes[16..20].copy_from_slice(&30u32.to_le_bytes()); // fps_num = 30
    bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // fps_den = 0
    std::fs::write(&path, &bytes).expect("write");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--summary"])
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "must accept IVF with zero denominator: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("0 decoded frames"),
        "summary must report 0 frames, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// A `.webm` file with at least 32 bytes but no `DKIF` magic exercises
/// the *second* clause of `ivf_framerate`'s `data.len() < 32 || ... != b"DKIF"`
/// check. (The first clause is exercised by the empty-webm test below.)
/// We hand-craft a minimal valid EBML stream (header + Segment with a VP9
/// video TrackEntry and one Void element of padding) so the webm parser
/// accepts it and yields zero packets — then the decoder calls
/// `ivf_framerate` on a >32-byte non-IVF buffer, which is the missing branch.
#[test]
fn dec_webm_large_non_ivf_data_uses_default_framerate() {
    let path = tmp("large_webm.webm");
    let mut bytes: Vec<u8> = Vec::new();
    // EBML header element: id 0x1A45DFA3, size vint = 0 (no payload).
    bytes.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3, 0x80]);
    // Segment element: id 0x18538067, size vint = 48 (marker bit + value).
    bytes.extend_from_slice(&[0x18, 0x53, 0x80, 0x67, 0x80 | 48]);
    // Tracks containing one VP9 video TrackEntry:
    // TrackNumber=1, TrackType=1 (video), CodecID="V_VP9".
    bytes.extend_from_slice(&[
        0x16, 0x54, 0xAE, 0x6B, 0x8F, 0xAE, 0x8D, 0xD7, 0x81, 0x01, 0x83, 0x81, 0x01, 0x86, 0x85,
        b'V', b'_', b'V', b'P', b'9',
    ]);
    // Inside the Segment: a Void element (id 0xEC) with a 26-byte
    // zero-filled payload. This pads the file past 32 bytes without
    // introducing any block frames.
    bytes.push(0xEC);
    bytes.push(0x80 | 26);
    bytes.extend(std::iter::repeat_n(0u8, 26));
    assert!(bytes.len() >= 32, "webm fixture must be >= 32 bytes");
    std::fs::write(&path, &bytes).expect("write");

    let out = Command::new(dec_bin())
        .args(["--noblit", "--summary"])
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec on webm");
    assert!(
        out.status.success(),
        "minimal valid webm must be accepted: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("0 decoded frames"),
        "zero-block webm must report 0 frames, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Decoding an empty `.webm` succeeds with zero frames. Exercises the
/// non-IVF container path AND the `ivf_framerate` early-return for
/// non-IVF (`data.len() < 32`) data.
#[test]
fn dec_empty_webm_decodes_zero_frames() {
    let path = tmp("empty_webm.webm");
    std::fs::write(&path, b"").expect("write");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--summary"])
        .arg(&path)
        .bounded_output()
        .expect("run rvp9dec on empty webm");
    assert!(
        out.status.success(),
        "empty webm must be accepted (zero packets): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("0 decoded frames"),
        "summary must report 0 frames for empty webm, got: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// `-o` pointing to a path that can't be created (a directory) surfaces
/// the io_err path during `open_output`/file create.
#[test]
fn dec_output_path_is_directory_rejected() {
    let ivf = encode_synthetic_ivf("outdir");
    let dir = tmp("outdir_target");
    let _ = std::fs::create_dir_all(&dir);
    let out = Command::new(dec_bin())
        .args(["--i420", "-o"])
        .arg(&dir)
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        !out.status.success(),
        "writing to a directory path must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("creating") || stderr.contains("io"),
        "stderr must report an I/O failure, got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&ivf);
}

/// Encoder: Y4M with coded height larger than source (but width OK)
/// exercises the second clause of `coded_w > hdr.w || coded_h > hdr.h`.
#[test]
fn enc_y4m_height_larger_than_source_rejected() {
    let y4m = tmp("y4m_tall.y4m");
    write_synthetic_y4m(&y4m, 128, 64, 1, 30, 1);
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "64",
            "-h",
            "128",
            "--target-bitrate",
            "500",
            "-o",
            "x.ivf",
        ])
        .arg(&y4m)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(
        !out.status.success(),
        "height exceeding source must be rejected"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("exceeds source"),
        "stderr must mention 'exceeds source', got: {stderr}"
    );
    let _ = std::fs::remove_file(&y4m);
}

#[test]
fn dec_help_lists_every_wired_flag() {
    let out = Command::new(dec_bin())
        .arg("--help")
        .bounded_output()
        .expect("run rvp9dec --help");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--help",
        "--version",
        "--codec",
        "--output",
        "--y4m",
        "--i420",
        "--yv12",
        "--rawvideo",
        "--md5",
        "--framemd5",
        "--limit",
        "--skip",
        "--summary",
        "--noblit",
        "--threads",
        "--progress",
        "--scale",
        "--output-bit-depth",
        "--error-concealment",
        "--frame-buffers",
        "--postproc",
        "--deblock",
        "--demacroblock",
        "--mfqe",
    ] {
        assert!(
            text.contains(flag),
            "dec --help must list {flag}, got:\n{text}"
        );
    }
}

#[cfg(unix)]
mod invalid_unicode_argv {
    use super::{Command, CommandExt, dec_bin, enc_bin, tmp};
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::{Path, PathBuf};

    struct RemoveOnDrop(PathBuf);

    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn assert_controlled_unicode_error(
        binary: &Path,
        tool: &str,
        args: Vec<OsString>,
        argument_index: usize,
    ) {
        let out = Command::new(binary)
            .args(args)
            .bounded_output()
            .expect("run CLI with invalid-Unicode argument");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{tool} must return its controlled failure code, got status {} and stderr: {stderr}",
            out.status
        );
        assert!(
            stderr.contains(tool)
                && stderr.contains("argument")
                && stderr.contains(&argument_index.to_string())
                && stderr.contains("Unicode"),
            "{tool} must identify the invalid-Unicode argument {argument_index}, got: {stderr}"
        );
        for panic_marker in [
            "panicked",
            "backtrace",
            "failed to convert",
            "std::env::args",
        ] {
            assert!(
                !stderr.to_ascii_lowercase().contains(panic_marker),
                "{tool} must not print panic details, got: {stderr}"
            );
        }
        assert!(
            out.stdout.is_empty(),
            "{tool} must not print output before rejecting argv"
        );
    }

    #[test]
    fn both_clis_reject_invalid_unicode_in_every_argument_position() {
        let input = tmp("invalid_unicode_input_sentinel");
        let _input_cleanup = RemoveOnDrop(input.clone());
        let input_sentinel = b"input sentinel remains unchanged";
        std::fs::write(&input, input_sentinel).expect("write input sentinel");

        let cases = [
            (
                "path-like positional",
                vec![OsString::from_vec(b"missing_\xff.ivf".to_vec())],
                1,
            ),
            (
                "long option text",
                vec![
                    input.as_os_str().to_os_string(),
                    OsString::from_vec(b"--unknown_\xff".to_vec()),
                ],
                2,
            ),
            (
                "option value",
                vec![
                    input.as_os_str().to_os_string(),
                    OsString::from("--codec"),
                    OsString::from_vec(b"vp\xff".to_vec()),
                ],
                3,
            ),
        ];

        for (tool, binary) in [("rvp9enc", enc_bin()), ("rvp9dec", dec_bin())] {
            for (case, args, argument_index) in &cases {
                assert_controlled_unicode_error(&binary, tool, args.clone(), *argument_index);
                assert_eq!(
                    std::fs::read(&input).expect("read input sentinel"),
                    input_sentinel,
                    "{tool} changed input while rejecting {case}"
                );
            }
        }

        let output = tmp("invalid_unicode_output_sentinel");
        let _output_cleanup = RemoveOnDrop(output.clone());
        let output_sentinel = b"output sentinel remains unchanged";
        std::fs::write(&output, output_sentinel).expect("write output sentinel");
        for (tool, binary) in [("rvp9enc", enc_bin()), ("rvp9dec", dec_bin())] {
            let args = vec![
                OsString::from("-o"),
                output.as_os_str().to_os_string(),
                OsString::from_vec(b"missing_\xff.ivf".to_vec()),
            ];
            assert_controlled_unicode_error(&binary, tool, args, 3);
            assert_eq!(
                std::fs::read(&output).expect("read output sentinel"),
                output_sentinel,
                "{tool} changed output while rejecting invalid Unicode"
            );
        }
    }
}
