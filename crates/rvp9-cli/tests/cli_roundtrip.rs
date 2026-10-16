//! End-to-end CLI tests for `rvp9enc` / `rvp9dec`.
//!
//! Hermetic: builds synthetic raw I420 frames at 64×64 and rectangular
//! sizes with distinct chroma patterns, so the tests need no external
//! corpus. Drives the actual built binaries via the
//! `CARGO_BIN_EXE_*` paths Cargo provides to integration tests.
//!
//! Oracles:
//! - Encode succeeds and produces a well-formed IVF with the right
//!   frame count.
//! - Decode reproduces the right number of frames at the right size.
//! - `--md5` is deterministic and emits one digest per displayed frame.
//! - Unsupported flags fail loudly ("not supported yet"), never silently.

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::CommandExt;
use rvp9_decoder::Decoder;
use rvp9_io::container::read_packets;

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

/// Unique temp path for this test process + a tag.
fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rvp9cli_{}_{tag}", std::process::id()))
}

/// Write `N_FRAMES` synthetic 64×64 I420 frames with per-frame motion
/// (a diagonal gradient that shifts each frame).
fn write_synthetic_raw(path: &std::path::Path) {
    write_synthetic_raw_dims(path, W, H, N_FRAMES);
}

fn write_synthetic_raw_dims(path: &std::path::Path, width: usize, height: usize, frames: usize) {
    let frame_bytes = width * height * 3 / 2;
    let mut buf = Vec::with_capacity(frame_bytes * frames);
    for f in 0..frames {
        // Y plane.
        for y in 0..height {
            for x in 0..width {
                buf.push(((x + y + f * 4) & 0xff) as u8);
            }
        }
        // U, V planes (half resolution).
        for plane in 0..2 {
            for y in 0..height / 2 {
                for x in 0..width / 2 {
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
    assert_eq!(buf.len(), frame_bytes * frames);
    std::fs::write(path, &buf).expect("write synthetic raw");
}

#[test]
fn spatial_svc_cli_emits_three_component_access_units_and_decodes_top_layer() {
    const SPATIAL_W: usize = 256;
    const SPATIAL_H: usize = 128;
    const SPATIAL_FRAMES: usize = 2;

    let raw = tmp("spatial_in.yuv");
    let ivf = tmp("spatial_out.ivf");
    let yuv = tmp("spatial_dec.yuv");
    write_synthetic_raw_dims(&raw, SPATIAL_W, SPATIAL_H, SPATIAL_FRAMES);

    let status = Command::new(enc_bin())
        .args([
            "-w",
            "256",
            "-h",
            "128",
            "--fps",
            "30/1",
            "--target-bitrate",
            "630",
            "--end-usage",
            "cbr",
            "--cpu-used",
            "8",
            "--lag-in-frames",
            "0",
            "--spatial-layers",
            "3",
            "--inter-layer-pred",
            "0",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_status()
        .expect("run spatial rvp9enc");
    assert!(status.success(), "spatial rvp9enc failed");

    let (ivf_bytes, packets) = read_packets(&ivf).expect("read spatial IVF");
    assert_eq!(packets.len(), SPATIAL_FRAMES, "IVF access-unit count");
    let mut decoder = Decoder::new();
    let first = decoder
        .parse_packet(packets[0].slice(&ivf_bytes))
        .expect("parse first spatial access unit");
    assert_eq!(first.count(), 3, "spatial components in first access unit");
    assert_eq!(first.output_frame_index(), Some(2));

    let status = Command::new(dec_bin())
        .args(["--i420", "-o"])
        .arg(&yuv)
        .arg(&ivf)
        .bounded_status()
        .expect("run spatial rvp9dec");
    assert!(status.success(), "spatial rvp9dec failed");
    let decoded_bytes = std::fs::metadata(&yuv).expect("stat spatial decode").len() as usize;
    assert_eq!(
        decoded_bytes,
        SPATIAL_W * SPATIAL_H * 3 / 2 * SPATIAL_FRAMES,
        "only the top spatial layer is output per access unit"
    );

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&yuv);
}

#[test]
fn encode_decode_roundtrip_raw_input() {
    let raw = tmp("in.yuv");
    let ivf = tmp("out.ivf");
    let yuv = tmp("dec.yuv");
    write_synthetic_raw(&raw);

    // Encode raw I420 -> IVF.
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
            "--end-usage",
            "cbr",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_status()
        .expect("run rvp9enc");
    assert!(status.success(), "rvp9enc failed");

    // IVF sanity: magic + frame-count field.
    let ivf_bytes = std::fs::read(&ivf).expect("read ivf");
    assert!(ivf_bytes.len() > 32, "ivf too small");
    assert_eq!(&ivf_bytes[0..4], b"DKIF", "bad IVF magic");
    let num_frames =
        u32::from_le_bytes([ivf_bytes[24], ivf_bytes[25], ivf_bytes[26], ivf_bytes[27]]);
    assert_eq!(num_frames as usize, N_FRAMES, "IVF frame count");

    // Decode IVF -> raw I420.
    let status = Command::new(dec_bin())
        .args(["--i420", "-o"])
        .arg(&yuv)
        .arg(&ivf)
        .bounded_status()
        .expect("run rvp9dec");
    assert!(status.success(), "rvp9dec failed");
    let dec_len = std::fs::metadata(&yuv).expect("stat dec.yuv").len() as usize;
    assert_eq!(dec_len, FRAME_BYTES * N_FRAMES, "decoded raw size");

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&yuv);
}

#[test]
fn encode_decode_rectangular_roundtrip_preserves_geometry_and_planes() {
    const RECT_W: usize = 128;
    const RECT_H: usize = 64;
    const RECT_FRAMES: usize = 4;
    const RECT_FRAME_BYTES: usize = RECT_W * RECT_H * 3 / 2;

    let raw = tmp("rect_in.yuv");
    let ivf = tmp("rect_out.ivf");
    let i420 = tmp("rect_dec.yuv");
    let y4m = tmp("rect_dec.y4m");
    write_synthetic_raw_dims(&raw, RECT_W, RECT_H, RECT_FRAMES);

    let status = Command::new(enc_bin())
        .args([
            "-w",
            "128",
            "-h",
            "64",
            "--fps",
            "30/1",
            "--target-bitrate",
            "500",
            "--end-usage",
            "cbr",
            "-o",
        ])
        .arg(&ivf)
        .arg(&raw)
        .bounded_status()
        .expect("run rectangular rvp9enc");
    assert!(status.success(), "rectangular rvp9enc failed");

    let (ivf_bytes, packets) = read_packets(&ivf).expect("read rectangular IVF");
    assert_eq!(&ivf_bytes[0..4], b"DKIF", "bad IVF magic");
    assert_eq!(
        u16::from_le_bytes([ivf_bytes[12], ivf_bytes[13]]) as usize,
        RECT_W,
        "IVF header width"
    );
    assert_eq!(
        u16::from_le_bytes([ivf_bytes[14], ivf_bytes[15]]) as usize,
        RECT_H,
        "IVF header height"
    );
    assert_eq!(packets.len(), RECT_FRAMES, "IVF packet count");
    assert_eq!(
        u32::from_le_bytes(ivf_bytes[24..28].try_into().expect("IVF frame count")) as usize,
        RECT_FRAMES,
        "IVF header frame count"
    );

    for (format, output) in [("--i420", &i420), ("--y4m", &y4m)] {
        let status = Command::new(dec_bin())
            .arg(format)
            .arg("-o")
            .arg(output)
            .arg(&ivf)
            .bounded_status()
            .unwrap_or_else(|error| panic!("run rvp9dec {format}: {error}"));
        assert!(status.success(), "rvp9dec {format} failed");
    }

    let i420_bytes = std::fs::read(&i420).expect("read rectangular I420 output");
    assert_eq!(
        i420_bytes.len(),
        RECT_FRAME_BYTES * RECT_FRAMES,
        "I420 frame count and size"
    );

    let y4m_bytes = std::fs::read(&y4m).expect("read rectangular Y4M output");
    let header_end = y4m_bytes
        .iter()
        .position(|&byte| byte == b'\n')
        .expect("Y4M header newline");
    let header = std::str::from_utf8(&y4m_bytes[..header_end]).expect("Y4M header UTF-8");
    let header_fields: Vec<&str> = header.split_ascii_whitespace().collect();
    assert_eq!(header_fields.first(), Some(&"YUV4MPEG2"), "Y4M magic");
    assert!(
        header_fields.contains(&"W128"),
        "Y4M header width: {header}"
    );
    assert!(
        header_fields.contains(&"H64"),
        "Y4M header height: {header}"
    );

    let y_size = RECT_W * RECT_H;
    let uv_size = (RECT_W / 2) * (RECT_H / 2);
    let frame_size = y_size + 2 * uv_size;
    let frame_tag = b"FRAME\n";
    let mut y4m_frames = Vec::with_capacity(RECT_FRAMES);
    let mut cursor = header_end + 1;
    while cursor < y4m_bytes.len() {
        assert!(
            y4m_bytes[cursor..].starts_with(frame_tag),
            "Y4M frame {count} tag at byte {cursor}",
            count = y4m_frames.len()
        );
        cursor += frame_tag.len();
        let frame_end = cursor.checked_add(frame_size).expect("Y4M frame end");
        assert!(frame_end <= y4m_bytes.len(), "truncated Y4M frame body");
        y4m_frames.push(&y4m_bytes[cursor..frame_end]);
        cursor = frame_end;
    }
    assert_eq!(y4m_frames.len(), RECT_FRAMES, "Y4M frame count");

    for (frame_index, y4m_frame) in y4m_frames.iter().enumerate() {
        let offset = frame_index * frame_size;
        let i420_frame = &i420_bytes[offset..offset + frame_size];
        assert_eq!(
            &i420_frame[..y_size],
            &y4m_frame[..y_size],
            "frame {frame_index}: Y plane boundary"
        );
        assert_eq!(
            &i420_frame[y_size..y_size + uv_size],
            &y4m_frame[y_size..y_size + uv_size],
            "frame {frame_index}: U plane boundary"
        );
        assert_eq!(
            &i420_frame[y_size + uv_size..],
            &y4m_frame[y_size + uv_size..],
            "frame {frame_index}: V plane boundary"
        );
        let u_plane = &i420_frame[y_size..y_size + uv_size];
        let v_plane = &i420_frame[y_size + uv_size..];
        assert_ne!(
            u_plane, v_plane,
            "frame {frame_index}: fixture must decode distinct U and V planes"
        );
        let u_mean = u_plane.iter().map(|&sample| u64::from(sample)).sum::<u64>() / uv_size as u64;
        let v_mean = v_plane.iter().map(|&sample| u64::from(sample)).sum::<u64>() / uv_size as u64;
        assert!(
            u_mean + 64 < v_mean,
            "frame {frame_index}: decoded U mean {u_mean} must remain well below V mean {v_mean}"
        );
    }

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&i420);
    let _ = std::fs::remove_file(&y4m);
}

#[test]
fn md5_is_deterministic_and_per_frame() {
    let raw = tmp("md5_in.yuv");
    let ivf = tmp("md5_out.ivf");
    write_synthetic_raw(&raw);

    let enc = Command::new(enc_bin())
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
    assert!(enc.success());

    let run_md5 = || {
        let out = Command::new(dec_bin())
            .arg("--md5")
            .arg(&ivf)
            .bounded_output()
            .expect("run rvp9dec --md5");
        assert!(out.status.success());
        String::from_utf8(out.stdout).expect("utf8")
    };
    let first = run_md5();
    let second = run_md5();
    assert_eq!(first, second, "--md5 must be deterministic");
    let lines: Vec<&str> = first.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), N_FRAMES, "one md5 per displayed frame");
    for l in &lines {
        assert_eq!(l.len(), 32, "md5 hex length: {l}");
        assert!(l.bytes().all(|b| b.is_ascii_hexdigit()), "hex: {l}");
    }

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
}

/// Unrecognised long options must still hard-error from rvp9enc — this
/// is the unknown-flag rejection path (`args::parse` reports `unknown
/// option`). The sibling "recognised but not supported yet" path
/// (`apply: None`) was retired from rvp9enc's table by W0.1 of the
/// vpxenc parity port (every previously-rejected flag is now stored on
/// `EncOpts`). That path is still unit-tested against a synthetic table
/// at `crates/rvp9-cli/src/args.rs::tests::unsupported_flag_errors_not_silently_ignored`,
/// so it re-engages automatically the moment a new libvpx flag is added
/// to the table without an `apply` handler.
#[test]
fn unsupported_flag_fails_loudly() {
    let out = Command::new(enc_bin())
        .args(["--no-such-flag-xyz", "-o", "x.ivf", "in.y4m"])
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success(), "must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown option") && stderr.contains("--no-such-flag-xyz"),
        "expected 'unknown option' rejection, got: {stderr}"
    );
}

#[test]
fn non_multiple_of_64_rejected_clearly() {
    // 100×100 raw: not a multiple of 64. Must be rejected with guidance.
    let raw = tmp("odd.yuv");
    std::fs::write(&raw, vec![0u8; 100 * 100 * 3 / 2]).expect("write");
    let out = Command::new(enc_bin())
        .args([
            "-w",
            "100",
            "-h",
            "100",
            "--target-bitrate",
            "500",
            "-o",
            "x.ivf",
        ])
        .arg(&raw)
        .bounded_output()
        .expect("run rvp9enc");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("multiple of 64"), "got: {stderr}");
    let _ = std::fs::remove_file(&raw);
}

#[test]
fn help_lists_supported_and_not_yet_tags() {
    // Every CLI binary must mark its flags as `[supported]` in the
    // `--help` listing — that's the load-bearing affordance for users
    // scanning the table.
    //
    // The `[not yet]` tag is the catalogue marker for `apply: None`
    // entries (libvpx flags recognised by rvp9 but not yet wired). As
    // of W0.1 of the vpxenc parity port, every flag currently in the
    // encoder's table is `apply: Some(_)` (stored on `EncOpts`, ready
    // for downstream waves), so `[not yet]` does not appear today.
    // The mechanism re-engages the moment a new libvpx flag is added
    // without a handler — see args::tests for the synthetic-table
    // proof. Asserting only `[supported]` keeps the user-facing
    // affordance pinned without flagging "fully wired" as a regression.
    for bin in [enc_bin(), dec_bin()] {
        let out = Command::new(&bin)
            .arg("--help")
            .bounded_output()
            .expect("run --help");
        assert!(out.status.success());
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("[supported]"), "no supported tag in {bin:?}");
    }
}

// ---- Q.1.B: `--threads` wiring ------------------------------------------------
//
// The CLI is a thin shim over `Decoder::with_threads()`. These tests pin the
// *routing* (flag accepted, libvpx-parity `0 = 1 thread`, validation rejects
// junk, help no longer says "[not yet]"). The pool-usage oracle lives in
// `rvp9-harness/tests/tile_threading.rs::multi_tile_dispatch_uses_worker_pool`,
// which exercises the constructor at N=2 and counts dispatched jobs via the
// `RVP9_G5_1_PARALLEL_JOBS` counter. We deliberately do not re-prove that here
// — see Q.1.B HLD §6 "Coverage trade-off".

/// Encode the standard 6-frame 64×64 synthetic clip and return the IVF path.
/// Caller owns the returned path and is responsible for cleanup.
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

/// Decode `ivf` with the given `extra` flags prepended to the dec args and
/// return the captured stdout as UTF-8. Fails the test if rvp9dec exits
/// non-zero or the stdout isn't valid UTF-8.
fn decode_md5_with(ivf: &std::path::Path, extra: &[&str]) -> String {
    let mut cmd = Command::new(dec_bin());
    for a in extra {
        cmd.arg(a);
    }
    let out = cmd
        .arg("--md5")
        .arg(ivf)
        .bounded_output()
        .expect("run rvp9dec");
    assert!(
        out.status.success(),
        "rvp9dec failed with {extra:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 stdout")
}

/// Q.1.B test 1: `--threads N` is accepted and produces byte-identical
/// pixels to the auto-detect (no-flag) path.
#[test]
fn threads_flag_accepted() {
    let ivf = encode_synthetic_ivf("threads_ok");

    let baseline = decode_md5_with(&ivf, &[]);
    let threaded = decode_md5_with(&ivf, &["--threads", "4"]);
    assert_eq!(
        baseline, threaded,
        "--threads 4 must produce identical output to auto-detect"
    );

    // Equality of MD5 lines also proves the decoded byte stream matched —
    // md5_of_yv12_visible covers the whole displayed YV12. No need for a
    // separate `--i420` byte-compare.

    let _ = std::fs::remove_file(&ivf);
}

/// Q.1.B test 2: `--threads 0` exits 0 and matches `--threads 1` exactly
/// (libvpx parity: 0 means 1 thread, NOT auto-detect).
#[test]
fn threads_zero_means_one_thread() {
    let ivf = encode_synthetic_ivf("threads_zero");

    let zero = decode_md5_with(&ivf, &["--threads", "0"]);
    let one = decode_md5_with(&ivf, &["--threads", "1"]);
    assert_eq!(
        zero, one,
        "--threads 0 must behave identically to --threads 1 (libvpx parity)"
    );

    let _ = std::fs::remove_file(&ivf);
}

/// Resource-allocation failures from a user-supplied thread count are normal
/// CLI errors, not panics.
#[test]
fn threads_unallocatable_count_returns_error_without_panicking() {
    let ivf = encode_synthetic_ivf("threads_unallocatable");
    let out = Command::new(dec_bin())
        .args(["--noblit", "--threads"])
        .arg(usize::MAX.to_string())
        .arg(&ivf)
        .bounded_output()
        .expect("run rvp9dec");

    assert!(
        !out.status.success(),
        "unallocatable thread count succeeded"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("creating decoder worker threads"),
        "stderr must report decoder construction, got: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "thread allocation must not panic, got: {stderr}"
    );

    let _ = std::fs::remove_file(&ivf);
}

/// Q.1.B test 3: invalid `--threads` values are rejected with a clear
/// message that names `--threads`. Three sub-cases: negative, non-numeric,
/// missing value.
#[test]
fn threads_invalid_rejected() {
    let ivf = encode_synthetic_ivf("threads_bad");

    // Helper: run rvp9dec with the given threads token + the ivf and assert
    // it exits non-zero with a stderr that mentions `--threads`.
    let assert_rejects = |threads_tokens: &[&str], label: &str| {
        let mut cmd = Command::new(dec_bin());
        for t in threads_tokens {
            cmd.arg(t);
        }
        cmd.arg(&ivf);
        let out = cmd.bounded_output().expect("run rvp9dec");
        assert!(
            !out.status.success(),
            "{label}: must exit non-zero, stderr=`{}`",
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--threads"),
            "{label}: stderr must mention --threads, got: {stderr}"
        );
    };

    // Negative: parse_value::<usize> rejects '-1'.
    assert_rejects(&["--threads", "-1"], "negative");
    // Non-numeric: parse_value::<usize> rejects 'abc'.
    assert_rejects(&["--threads", "abc"], "non-numeric");
    // Missing value: --threads at EOL with nothing following (the ivf is
    // consumed as the --threads value, then no positional input remains
    // → "no input file given" — that's a separate error path. The truly
    // missing-value case is `--threads` as the *last* token.
    let out = Command::new(dec_bin())
        .arg("--threads")
        .bounded_output()
        .expect("run rvp9dec");
    assert!(!out.status.success(), "missing-value: must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--threads") && stderr.contains("requires a value"),
        "missing-value: expected '--threads requires a value', got: {stderr}"
    );

    let _ = std::fs::remove_file(&ivf);
}

/// Q.1.B test 4: `--help` lists `--threads` as `[supported]`, not `[not yet]`.
#[test]
fn threads_help_tag_supported() {
    let out = Command::new(dec_bin())
        .arg("--help")
        .bounded_output()
        .expect("run rvp9dec --help");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find(|l| l.contains("--threads"))
        .expect("--threads must appear in --help output");
    assert!(
        line.contains("[supported]"),
        "--threads must be tagged [supported], got: {line}"
    );
    assert!(
        !line.contains("[not yet]"),
        "--threads must no longer be tagged [not yet], got: {line}"
    );
}
