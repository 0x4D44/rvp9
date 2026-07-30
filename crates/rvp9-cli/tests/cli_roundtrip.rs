//! End-to-end CLI tests for `rvp9enc` / `rvp9dec`.
//!
//! Hermetic: builds synthetic raw I420 frames at 64×64 (a single
//! super-block, the smallest legal coded size), so the tests need no
//! external corpus. Drives the actual built binaries via the
//! `CARGO_BIN_EXE_*` paths Cargo provides to integration tests.
//!
//! Oracles:
//! - Encode succeeds and produces a well-formed IVF with the right
//!   frame count.
//! - Decode reproduces the right number of frames at the right size.
//! - `--md5` is deterministic and emits one digest per displayed frame.
//! - Unsupported flags fail loudly ("not supported yet"), never silently.

use std::path::PathBuf;
use std::process::Command;

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
    let mut buf = Vec::with_capacity(FRAME_BYTES * N_FRAMES);
    for f in 0..N_FRAMES {
        // Y plane.
        for y in 0..H {
            for x in 0..W {
                buf.push(((x + y + f * 4) & 0xff) as u8);
            }
        }
        // U, V planes (half resolution).
        for _plane in 0..2 {
            for y in 0..H / 2 {
                for x in 0..W / 2 {
                    buf.push(((x * 2 + y + f) & 0xff) as u8);
                }
            }
        }
    }
    assert_eq!(buf.len(), FRAME_BYTES * N_FRAMES);
    std::fs::write(path, &buf).expect("write synthetic raw");
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
        .status()
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
        .status()
        .expect("run rvp9dec");
    assert!(status.success(), "rvp9dec failed");
    let dec_len = std::fs::metadata(&yuv).expect("stat dec.yuv").len() as usize;
    assert_eq!(dec_len, FRAME_BYTES * N_FRAMES, "decoded raw size");

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&yuv);
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
        .status()
        .expect("run rvp9enc");
    assert!(enc.success());

    let run_md5 = || {
        let out = Command::new(dec_bin())
            .arg("--md5")
            .arg(&ivf)
            .output()
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
        .output()
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
        .output()
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
            .output()
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
        .status()
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
    let out = cmd.arg("--md5").arg(ivf).output().expect("run rvp9dec");
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
        .output()
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
        let out = cmd.output().expect("run rvp9dec");
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
        .output()
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
        .output()
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
