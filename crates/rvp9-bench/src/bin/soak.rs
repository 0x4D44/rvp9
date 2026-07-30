// Soak binary: stderr/stdout IS the output mechanism. The workspace's
// `print_stderr` / `print_stdout` lints target library code; for a CLI
// soak harness those lints are the wrong rule.
#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

//! D3 — 72-hour soak test harness (PLAN.md §Stage 5 hardening).
//!
//! Runs an encode → decode → optional cross-decode loop indefinitely (or
//! for a configured duration / iteration count). Designed to surface
//! rare bugs and memory growth that single-shot tests don't catch.
//!
//! Each iteration:
//! 1. Loads N frames from a source clip (rotating through the configured
//!    sources — y4m corpus on disk, plus a synthetic fallback if none
//!    are available).
//! 2. Encodes the N frames through `rvp9-encoder::StreamEncoder` at a
//!    chosen target bitrate.
//! 3. Wraps the encoded packets in an IVF container.
//! 4. Decodes the IVF through `rvp9-decoder::Decoder`, computing a
//!    cheap Y-plane checksum.
//! 5. Optionally pipes the IVF to `vpxdec.exe` and confirms exit 0.
//!
//! Failures (encode panic / decode panic / size mismatch / cross-decoder
//! exit-non-zero) print a clear `SOAK FAILURE at iteration N: <reason>`
//! line to stderr and exit non-zero.
//!
//! ## Invocation
//!
//! ```text
//! cargo run -p rvp9-bench --release --bin soak -- \
//!     --duration-secs 30 \
//!     --frames-per-iter 8 \
//!     [--report-every-secs 10] \
//!     [--cross-decode] \
//!     [--vpxdec D:/language/libvpx-stock/vpxdec.exe]
//! ```
//!
//! Pass `--duration-secs 0` to run forever until Ctrl-C; pass
//! `--iterations N` to stop after exactly N iterations.

use rvp9_decoder::Decoder;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{YuvFrameBuf, probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const VPXDEC_TIMEOUT: Duration = Duration::from_secs(30);
static TEMP_DIR_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> std::io::Result<Self> {
        loop {
            let sequence = TEMP_DIR_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("{prefix}-{}-{sequence}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn run_with_timeout(command: &mut Command, timeout: Duration) -> Result<ExitStatus, String> {
    command.stdin(Stdio::null());
    let mut child = command
        .spawn()
        .map_err(|error| format!("vpxdec spawn: {error}"))?;
    let deadline = Instant::now() + timeout;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                let kill_result = child.kill();
                let wait_result = child.wait();
                if let Err(error) = kill_result {
                    return Err(format!(
                        "vpxdec timed out after {:.1}s and could not be killed: {error}",
                        timeout.as_secs_f64()
                    ));
                }
                if let Err(error) = wait_result {
                    return Err(format!(
                        "vpxdec timed out after {:.1}s and could not be reaped: {error}",
                        timeout.as_secs_f64()
                    ));
                }
                return Err(format!(
                    "vpxdec timed out after {:.1}s",
                    timeout.as_secs_f64()
                ));
            }
            Ok(None) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                std::thread::sleep(remaining.min(Duration::from_millis(10)));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("vpxdec status: {error}"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    /// Stop after this many seconds. `0` = run forever.
    duration_secs: u64,
    /// Hard cap on iterations. `0` = unlimited (subject to `duration_secs`).
    iterations: u64,
    /// Number of frames encoded per iteration.
    frames_per_iter: u32,
    /// How often to print metric lines (seconds).
    report_every_secs: u64,
    /// If set, also pipe IVF through vpxdec and check exit-0.
    cross_decode: bool,
    /// Path to vpxdec. Defaults to the canonical sibling-tree path.
    vpxdec_path: PathBuf,
}

impl Args {
    fn parse() -> Self {
        let argv: Vec<String> = std::env::args().collect();
        let get = |flag: &str| -> Option<String> {
            argv.iter()
                .position(|a| a == flag)
                .and_then(|i| argv.get(i + 1).cloned())
        };
        let has_flag = |flag: &str| argv.iter().any(|a| a == flag);
        Self {
            duration_secs: get("--duration-secs")
                .map(|s| s.parse().expect("--duration-secs u64"))
                .unwrap_or(60),
            iterations: get("--iterations")
                .map(|s| s.parse().expect("--iterations u64"))
                .unwrap_or(0),
            frames_per_iter: get("--frames-per-iter")
                .map(|s| s.parse().expect("--frames-per-iter u32"))
                .unwrap_or(30),
            report_every_secs: get("--report-every-secs")
                .map(|s| s.parse().expect("--report-every-secs u64"))
                .unwrap_or(30),
            cross_decode: has_flag("--cross-decode"),
            vpxdec_path: get("--vpxdec")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(r"D:\language\libvpx-stock\vpxdec.exe")),
        }
    }
}

// ---------------------------------------------------------------------------
// Frame sources.
// ---------------------------------------------------------------------------

/// A loadable source of frames for the soak loop.
#[derive(Debug, Clone)]
enum Source {
    /// y4m file on disk; centre-crop to (w, h).
    Y4m(PathBuf, u32, u32),
    /// Synthetic animated gradient at (w, h).
    SyntheticGradient(u32, u32),
}

impl Source {
    fn label(&self) -> String {
        match self {
            Source::Y4m(p, w, h) => format!("y4m({}, {w}x{h})", p.display()),
            Source::SyntheticGradient(w, h) => format!("synthetic-gradient({w}x{h})"),
        }
    }

    fn dims(&self) -> (u32, u32) {
        match *self {
            Source::Y4m(_, w, h) | Source::SyntheticGradient(w, h) => (w, h),
        }
    }

    /// Load N frames + the framerate.
    fn load(&self, n: u32) -> Result<(Vec<YuvFrameBuf>, f64), String> {
        match self {
            Source::Y4m(path, w, h) => match y4m_read_cropped(path, *w, *h, n as usize) {
                Ok((frames, fr)) if frames.len() == n as usize => Ok((frames, fr)),
                Ok((frames, _)) => Err(format!(
                    "y4m source {} only yielded {}/{n} frames",
                    path.display(),
                    frames.len()
                )),
                Err(e) => Err(format!("y4m read failed: {e}")),
            },
            Source::SyntheticGradient(w, h) => {
                let frames = (0..n)
                    .map(|i| synthetic_gradient(*w as usize, *h as usize, i))
                    .collect();
                Ok((frames, 30.0))
            }
        }
    }
}

/// Synthetic animated gradient — diagonal shift per frame so inter has
/// real motion. Same generator pattern as the bdrate gate's
/// `animated_gradient`, replicated locally to avoid pulling the test
/// crate as a dep.
fn synthetic_gradient(w: usize, h: usize, idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let v = 64i32 + (r as i32 / 2) + (c as i32 / 2) + (idx as i32 * 3);
            y[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// Probe canonical paths for the Foreman CIF y4m corpus file.
///
/// The `probe_foreman_cif` helper isn't available in this worktree's
/// `rvp9-harness` (predates main); inline the path check so the soak
/// harness can still pick up the clip when present.
fn probe_foreman_cif_local() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("tests/corpus/foreman_cif.y4m"),
        PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_cif.y4m"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Build the round-robin list of sources for this run. Probes the
/// y4m corpus paths and always includes a synthetic fallback so the
/// harness can run on a bare CI checkout.
fn build_source_list() -> Vec<Source> {
    let mut sources = Vec::new();
    if let Some(p) = probe_testsrc_720p() {
        sources.push(Source::Y4m(p, 256, 256));
    }
    if let Some(p) = probe_foreman_cif_local() {
        sources.push(Source::Y4m(p, 256, 256));
    }
    // Always include synthetic so the loop runs even with no corpus.
    sources.push(Source::SyntheticGradient(256, 256));
    sources
}

// ---------------------------------------------------------------------------
// RSS (Resident Set Size) on Windows.
//
// Calls GetProcessMemoryInfo via direct FFI to avoid adding a new
// crate dependency. Returns 0 if the call fails (best-effort
// diagnostic — soak still runs without RSS).
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod rss {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    // `GetCurrentProcess` returns a pseudo-handle that needs no Close.
    // `GetProcessMemoryInfo` is in psapi.dll on Windows 7+ (it's also
    // forwarded by kernel32.dll on modern Windows but linking against
    // psapi is the documented contract).
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }

    /// Current working-set size in bytes. Returns 0 on failure.
    pub fn current_rss_bytes() -> usize {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..ProcessMemoryCounters::default()
        };
        // SAFETY: `GetCurrentProcess` returns a valid pseudo-handle.
        // `GetProcessMemoryInfo` writes at most `cb` bytes into
        // `counters` (which is fully owned, properly sized, and not
        // aliased). Both functions are reentrant and thread-safe.
        let ok = unsafe {
            let proc = GetCurrentProcess();
            GetProcessMemoryInfo(proc, &mut counters, counters.cb)
        };
        if ok != 0 {
            counters.working_set_size
        } else {
            0
        }
    }
}

#[cfg(not(windows))]
mod rss {
    pub fn current_rss_bytes() -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Loop body helpers.
// ---------------------------------------------------------------------------

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: w,
        u_stride: w / 2,
        v_stride: w / 2,
    }
}

/// Cheap deterministic Y-plane fingerprint for "did decode produce
/// the bytes we expected" assertions. Wyhash-ish 64-bit xor-mul.
/// We don't need cryptographic strength; we need fast and stable.
fn checksum(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Encode + decode one iteration. Returns the encoded byte count
/// (for bitrate metrics) and a checksum of the decoded Y planes.
///
/// Errors are returned as `Err(reason)` so the caller can format the
/// SOAK FAILURE line with the iteration index.
fn run_one_iteration(
    iter_idx: u64,
    source: &Source,
    frames_per_iter: u32,
    cross_decode: Option<&Path>,
) -> Result<IterStats, String> {
    let (w, h) = source.dims();
    let (frames, framerate) = source
        .load(frames_per_iter)
        .map_err(|e| format!("source load: {e}"))?;

    // ---- Encode ----------------------------------------------------------
    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps: 1_000_000,
        gop_length: frames_per_iter + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 0,
        deadline: rvp9_encoder::Deadline::Good,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).map_err(|e| format!("StreamEncoder::new: {e:?}"))?;

    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    let mut total_encoded_bytes: u64 = 0;
    for (frame_idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w as usize))
            .map_err(|e| format!("encode_frame[{frame_idx}]: {e:?}"))?;
        total_encoded_bytes += pkt.data.len() as u64;
        packets.push(pkt.data);
    }

    // ---- Decode through rvp9-decoder -------------------------------------
    let mut dec = Decoder::new();
    let mut y_checksum: u64 = 0;
    let mut decoded_frames: u32 = 0;
    for (pkt_idx, pkt) in packets.iter().enumerate() {
        // Re-materialise ParsedFrames as owned-of-`&pkt` so the inner
        // mutable borrow on `dec` (via decode_frame) doesn't conflict
        // with the immutable borrow held inside PacketFrames.
        let owned: Vec<rvp9_decoder::ParsedFrame> = {
            let parsed = dec
                .parse_packet(pkt)
                .map_err(|e| format!("parse_packet[{pkt_idx}]: {e:?}"))?;
            parsed
                .frames()
                .iter()
                .map(|pf| rvp9_decoder::ParsedFrame {
                    header: pf.header.clone(),
                    header_end_byte: pf.header_end_byte,
                    full_frame_data: pf.full_frame_data,
                })
                .collect()
        };
        for (sub, pf) in owned.iter().enumerate() {
            let buf = dec
                .decode_frame(pf)
                .map_err(|e| format!("decode_frame[{pkt_idx}/{sub}]: {e:?}"))?;
            let yp = buf.y_plane();
            // Hash just the first row; cheap and still catches divergence.
            // Hashing the whole plane every iteration would dominate
            // runtime at high frame counts.
            let row0 = &yp.data()[..w as usize];
            y_checksum ^= checksum(row0).rotate_left(decoded_frames & 63);
            decoded_frames += 1;
        }
    }
    if decoded_frames != frames_per_iter {
        return Err(format!(
            "decode produced {decoded_frames} frames, expected {frames_per_iter}"
        ));
    }

    // ---- Optional vpxdec cross-decode ------------------------------------
    if let Some(vpxdec) = cross_decode {
        let tmp_dir = ScopedTempDir::new("rvp9-soak-cross-decode")
            .map_err(|e| format!("create cross-decode temp dir: {e}"))?;
        let ivf_path = tmp_dir.path().join(format!("iter{iter_idx}.ivf"));
        {
            let mut f = std::fs::File::create(&ivf_path).map_err(|e| format!("create ivf: {e}"))?;
            let header = IvfHeader::vp9(w as u16, h as u16, framerate.round() as u32, 1);
            let packet_refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
            write_ivf(&mut f, &header, &packet_refs).map_err(|e| format!("write_ivf: {e}"))?;
        }
        let out_path = tmp_dir.path().join(format!("iter{iter_idx}.yuv"));
        let stderr_path = tmp_dir.path().join("vpxdec.stderr");
        let stderr_file = std::fs::File::create(&stderr_path)
            .map_err(|e| format!("create vpxdec stderr file: {e}"))?;
        let mut command = Command::new(vpxdec);
        command
            .args(["--codec=vp9", "--i420", "-o"])
            .arg(&out_path)
            .arg(&ivf_path)
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file));
        let status = run_with_timeout(&mut command, VPXDEC_TIMEOUT)?;
        if !status.success() {
            let stderr = std::fs::read(&stderr_path).unwrap_or_default();
            return Err(format!(
                "vpxdec exited non-zero: {} stderr={}",
                status,
                String::from_utf8_lossy(&stderr)
            ));
        }
    }

    Ok(IterStats {
        encoded_bytes: total_encoded_bytes,
        y_checksum,
        decoded_frames,
    })
}

#[derive(Debug, Clone, Copy)]
struct IterStats {
    encoded_bytes: u64,
    y_checksum: u64,
    decoded_frames: u32,
}

// ---------------------------------------------------------------------------
// Main loop.
// ---------------------------------------------------------------------------

fn fmt_bytes(b: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.1} MiB", b as f64 / MIB)
}

fn main() {
    let args = Args::parse();
    let cross_decode_path = if args.cross_decode {
        if !args.vpxdec_path.is_file() {
            eprintln!(
                "error: --cross-decode requires vpxdec at {}",
                args.vpxdec_path.display()
            );
            std::process::exit(2);
        }
        Some(args.vpxdec_path.clone())
    } else {
        None
    };
    let sources = build_source_list();

    eprintln!("=== rvp9 soak harness (D3) ===");
    eprintln!("duration_secs    : {}", args.duration_secs);
    eprintln!("iterations       : {}", args.iterations);
    eprintln!("frames_per_iter  : {}", args.frames_per_iter);
    eprintln!("report_every_secs: {}", args.report_every_secs);
    eprintln!("cross_decode     : {}", args.cross_decode);
    eprintln!("vpxdec_path      : {}", args.vpxdec_path.display());
    eprintln!("sources:");
    for s in &sources {
        eprintln!("  - {}", s.label());
    }
    eprintln!();

    let start = Instant::now();
    let deadline = if args.duration_secs == 0 {
        None
    } else {
        Some(start + Duration::from_secs(args.duration_secs))
    };
    let report_interval = Duration::from_secs(args.report_every_secs.max(1));
    let mut last_report = start;

    let mut iter_idx: u64 = 0;
    // `errors` stays at 0 today (we exit fast on first failure). Kept as
    // a field so the periodic report has a permanent slot for it; a
    // future `--keep-going` flag would accumulate into this.
    let errors: u64 = 0;
    let mut total_encoded_bytes: u64 = 0;
    let mut total_decoded_frames: u64 = 0;
    let mut acc_checksum: u64 = 0;

    eprintln!(
        "[start] wall=0.0s iter=0 rss={}",
        fmt_bytes(rss::current_rss_bytes())
    );

    loop {
        // Stop conditions.
        if let Some(dl) = deadline
            && Instant::now() >= dl
        {
            break;
        }
        if args.iterations != 0 && iter_idx >= args.iterations {
            break;
        }

        let source = &sources[(iter_idx as usize) % sources.len()];
        let iter_start = Instant::now();
        match run_one_iteration(
            iter_idx,
            source,
            args.frames_per_iter,
            cross_decode_path.as_deref(),
        ) {
            Ok(stats) => {
                total_encoded_bytes += stats.encoded_bytes;
                total_decoded_frames += u64::from(stats.decoded_frames);
                acc_checksum ^= stats.y_checksum;
            }
            Err(reason) => {
                // First failure exits non-zero — the entire point of a
                // soak is to surface rare divergences for inspection.
                // (A future `--keep-going` flag could accumulate
                // `errors` and emit a final summary; today we fail fast.)
                eprintln!(
                    "SOAK FAILURE at iteration {iter_idx} (source {}): {reason}",
                    source.label()
                );
                std::process::exit(2);
            }
        }
        let iter_elapsed = iter_start.elapsed();
        iter_idx += 1;

        // Periodic report.
        let now = Instant::now();
        if now.duration_since(last_report) >= report_interval {
            print_report(
                start,
                iter_idx,
                total_encoded_bytes,
                total_decoded_frames,
                errors,
                iter_elapsed,
                acc_checksum,
            );
            last_report = now;
        }
    }

    // Final report (always emit, even if no periodic report fired).
    eprintln!();
    eprintln!("=== soak complete ===");
    print_report(
        start,
        iter_idx,
        total_encoded_bytes,
        total_decoded_frames,
        errors,
        Duration::from_secs(0),
        acc_checksum,
    );
    if errors > 0 {
        std::process::exit(2);
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    start: Instant,
    iter_idx: u64,
    total_encoded_bytes: u64,
    total_decoded_frames: u64,
    errors: u64,
    last_iter_elapsed: Duration,
    acc_checksum: u64,
) {
    let wall = start.elapsed();
    let wall_s = wall.as_secs_f64();
    let avg_ms_per_iter = if iter_idx > 0 {
        (wall_s * 1000.0) / iter_idx as f64
    } else {
        0.0
    };
    let bitrate_kbps = if wall_s > 0.0 {
        (total_encoded_bytes as f64 * 8.0 / 1000.0) / wall_s
    } else {
        0.0
    };
    let fps = if wall_s > 0.0 {
        total_decoded_frames as f64 / wall_s
    } else {
        0.0
    };
    eprintln!(
        "[report] wall={wall_s:7.1}s iter={iter_idx:>6} avg={avg_ms_per_iter:7.1} ms/iter \
         frames={total_decoded_frames:>7} ({fps:6.1} fps) br={bitrate_kbps:7.1} kbps \
         last_iter={last_iter_ms:7.1} ms rss={rss} errors={errors} cksum={acc_checksum:016x}",
        last_iter_ms = last_iter_elapsed.as_secs_f64() * 1000.0,
        rss = fmt_bytes(rss::current_rss_bytes()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_decode_temp_dirs_are_isolated_and_cleaned() {
        let first = ScopedTempDir::new("rvp9-soak-test").unwrap();
        let second = ScopedTempDir::new("rvp9-soak-test").unwrap();
        let first_path = first.path().to_owned();

        assert_ne!(first.path(), second.path());
        std::fs::write(first.path().join("marker"), b"owned").unwrap();
        drop(first);
        assert!(!first_path.exists());
    }

    #[test]
    fn cross_decoder_timeout_kills_child() {
        const CHILD_ENV: &str = "RVP9_SOAK_TIMEOUT_TEST_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            std::thread::sleep(Duration::from_secs(60));
            return;
        }

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "tests::cross_decoder_timeout_kills_child",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1");

        let started = Instant::now();
        let error = run_with_timeout(&mut command, Duration::from_millis(100)).unwrap_err();
        assert!(error.contains("timed out"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
