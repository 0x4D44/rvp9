//! BD-rate (Bjontegaard delta rate) measurement.
//!
//! Runs two encoders (`anchor` and `test`) at a target-bitrate sweep, decodes
//! each output, computes PSNR vs the source, builds two RD curves, and
//! reports BD-rate (the percentage bitrate difference between curves at
//! matched PSNR).
//!
//! In Stage 0 this runs `libvpx vpxenc` as both anchor and test. Expected
//! BD-rate: ~0% (any non-zero is harness noise — informative as a baseline).
//! In Stage 4 we point `--test-encoder` at `rvp9enc` and require BD-rate
//! within the limits stated in `PLAN.md` §2.
//!
//! Stage 0 status: SKELETON. Performs the encode sweep and produces an RD
//! table; the BD-rate curve fit is implemented but exercised end-to-end only
//! once decode+PSNR machinery lands (TODO below).
//!
//! Usage:
//!   cargo run --release --bin bdrate -- \
//!       --anchor-encoder D:/language/libvpx-stock/vpxenc.exe \
//!       --test-encoder   D:/language/libvpx-stock/vpxenc.exe \
//!       --decoder        D:/language/libvpx-stock/vpxdec.exe \
//!       --input          tests/corpus/parkrun_720p30.y4m \
//!       --cpu-used       7 \
//!       --bitrates       500,1000,1500,2000,3000,4500

use std::path::{Path, PathBuf};
use std::process::Command;

struct Args {
    anchor_encoder: PathBuf,
    test_encoder: PathBuf,
    decoder: PathBuf,
    input: PathBuf,
    cpu_used: u32,
    bitrates_kbps: Vec<u32>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };

    Args {
        anchor_encoder: get("--anchor-encoder")
            .map(PathBuf::from)
            .expect("--anchor-encoder required"),
        test_encoder: get("--test-encoder")
            .map(PathBuf::from)
            .expect("--test-encoder required"),
        decoder: get("--decoder")
            .map(PathBuf::from)
            .expect("--decoder required"),
        input: get("--input").map(PathBuf::from).expect("--input required"),
        cpu_used: get("--cpu-used")
            .map(|s| s.parse().expect("--cpu-used u32"))
            .unwrap_or(7),
        bitrates_kbps: get("--bitrates")
            .map(|s| {
                s.split(',')
                    .map(|t| t.parse::<u32>().expect("bitrate u32"))
                    .collect()
            })
            .unwrap_or_else(|| vec![500, 1000, 1500, 2000, 3000, 4500]),
    }
}

#[derive(Debug, Clone, Copy)]
struct RdPoint {
    /// Realised bitrate in kbps (from the IVF, not the requested target).
    bitrate_kbps: f64,
    /// PSNR vs source in dB.
    psnr_db: f64,
}

/// Encode `input` with `encoder` at the given target bitrate. Returns the
/// path to the resulting IVF.
fn encode(encoder: &Path, input: &Path, cpu_used: u32, bitrate_kbps: u32, tag: &str) -> PathBuf {
    let out = std::env::temp_dir().join(format!("rvp9_bdrate_{tag}_{bitrate_kbps}k.ivf"));
    let status = Command::new(encoder)
        .args([
            "--codec=vp9",
            "--passes=1",
            "--rt",
            &format!("--cpu-used={cpu_used}"),
            "--end-usage=cbr",
            &format!("--target-bitrate={bitrate_kbps}"),
            "--lag-in-frames=0",
            "--ivf",
            "-o",
        ])
        .arg(&out)
        .arg(input)
        .status()
        .expect("encoder failed");
    assert!(status.success(), "encoder {} failed at {bitrate_kbps}k", encoder.display());
    out
}

/// TODO(Stage 0 follow-up): decode `ivf` with `decoder`, run pixel-wise PSNR
/// against `source_y4m`, return PSNR_Y in dB. Stub for now.
fn decode_and_psnr(_decoder: &Path, _ivf: &Path, _source_y4m: &Path) -> f64 {
    // Stage 0 leaves this stubbed because the PSNR machinery needs a Y4M
    // reader and YUV math. Will be filled in alongside the first round of
    // rvp9 vs libvpx BD-rate runs in Stage 4a.
    f64::NAN
}

/// File size of `ivf` in kilobits, divided by clip duration in seconds.
fn measured_bitrate_kbps(ivf: &Path, frames: u32, fps: f64) -> f64 {
    let bytes = std::fs::metadata(ivf).expect("ivf metadata").len() as f64;
    let duration_s = f64::from(frames) / fps;
    (bytes * 8.0 / 1000.0) / duration_s
}

/// Bjontegaard delta rate, cubic polynomial fit on log-bitrate vs PSNR.
/// Returns percent (positive = test costs more bits for same quality).
///
/// Reference: Bjontegaard, "Calculation of Average PSNR Differences between
/// RD-curves", ITU-T VCEG Q6/16, Doc. VCEG-M33, 2001.
fn bd_rate(anchor: &[RdPoint], test: &[RdPoint]) -> Option<f64> {
    if anchor.len() < 4 || test.len() < 4 {
        return None;
    }
    // Convert bitrate to log-scale for the fit.
    let to_xy = |p: &RdPoint| (p.psnr_db, p.bitrate_kbps.ln());
    let anchor_xy: Vec<(f64, f64)> = anchor.iter().map(to_xy).collect();
    let test_xy: Vec<(f64, f64)> = test.iter().map(to_xy).collect();

    // Cubic least-squares fit of log_bitrate as a function of PSNR.
    let a = polyfit3(&anchor_xy);
    let b = polyfit3(&test_xy);

    // Integration range: the OVERLAP of the two PSNR ranges (Bjontegaard
    // integrates only over PSNR values where both curves are defined).
    let anchor_min = anchor_xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let anchor_max = anchor_xy.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let test_min = test_xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let test_max = test_xy.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let range_lo = anchor_min.max(test_min);
    let range_hi = anchor_max.min(test_max);
    if range_hi <= range_lo {
        return None;
    }

    // Definite integral of a cubic on [lo, hi].
    let integrate = |c: &[f64; 4]| -> f64 {
        let f = |x: f64| {
            c[0] * x + c[1] * x * x / 2.0 + c[2] * x * x * x / 3.0 + c[3] * x * x * x * x / 4.0
        };
        f(range_hi) - f(range_lo)
    };
    let int_a = integrate(&a);
    let int_b = integrate(&b);
    let avg_diff = (int_b - int_a) / (range_hi - range_lo);
    Some((avg_diff.exp() - 1.0) * 100.0)
}

/// Least-squares cubic fit y = c0 + c1*x + c2*x^2 + c3*x^3.
/// Uses the normal equations (4x4 system) with Gauss-Jordan elimination.
///
/// The index-based loops below are deliberate — the matrix-elimination
/// invariants are clearer with explicit `m[i][j]` than with iterator chains
/// over rows. We allow `clippy::needless_range_loop` for that reason.
#[allow(clippy::needless_range_loop)]
fn polyfit3(pts: &[(f64, f64)]) -> [f64; 4] {
    // Build the normal-equations augmented matrix [A | b] where
    //   A[i][j] = sum(x^(i+j))
    //   b[i]    = sum(y * x^i)
    let mut m = [[0.0f64; 5]; 4];
    for &(x, y) in pts {
        let mut xp = [1.0; 7];
        for i in 1..7 {
            xp[i] = xp[i - 1] * x;
        }
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] += xp[i + j];
            }
            m[i][4] += y * xp[i];
        }
    }
    // Gauss-Jordan.
    for i in 0..4 {
        // Pivot.
        let mut max = i;
        for k in (i + 1)..4 {
            if m[k][i].abs() > m[max][i].abs() {
                max = k;
            }
        }
        m.swap(i, max);
        let pivot = m[i][i];
        if pivot.abs() < 1e-12 {
            return [0.0; 4]; // degenerate; caller should ignore.
        }
        for j in 0..5 {
            m[i][j] /= pivot;
        }
        for k in 0..4 {
            if k != i {
                let f = m[k][i];
                for j in 0..5 {
                    m[k][j] -= f * m[i][j];
                }
            }
        }
    }
    [m[0][4], m[1][4], m[2][4], m[3][4]]
}

fn main() {
    let args = parse_args();

    println!("=== BD-rate sweep ===");
    println!("anchor: {}", args.anchor_encoder.display());
    println!("test:   {}", args.test_encoder.display());
    println!("input:  {}", args.input.display());
    println!();

    // TODO(Stage 0 follow-up): parse the y4m header to learn fps + frame
    // count instead of hard-coding 30.0. Acceptable Stage 0 limitation.
    let fps = 30.0;

    let mut anchor_rd: Vec<RdPoint> = Vec::new();
    let mut test_rd: Vec<RdPoint> = Vec::new();

    for &br in &args.bitrates_kbps {
        let a_ivf = encode(&args.anchor_encoder, &args.input, args.cpu_used, br, "anchor");
        let t_ivf = encode(&args.test_encoder, &args.input, args.cpu_used, br, "test");

        let a_frames = rvp9_bench::count_ivf_frames(&a_ivf);
        let t_frames = rvp9_bench::count_ivf_frames(&t_ivf);

        let a_br = measured_bitrate_kbps(&a_ivf, a_frames, fps);
        let t_br = measured_bitrate_kbps(&t_ivf, t_frames, fps);

        let a_psnr = decode_and_psnr(&args.decoder, &a_ivf, &args.input);
        let t_psnr = decode_and_psnr(&args.decoder, &t_ivf, &args.input);

        println!("target {br:>5}k | anchor {a_br:>6.0}k psnr {a_psnr:>5.2} | test {t_br:>6.0}k psnr {t_psnr:>5.2}");

        anchor_rd.push(RdPoint { bitrate_kbps: a_br, psnr_db: a_psnr });
        test_rd.push(RdPoint { bitrate_kbps: t_br, psnr_db: t_psnr });
    }

    if anchor_rd.iter().any(|p| p.psnr_db.is_nan()) || test_rd.iter().any(|p| p.psnr_db.is_nan()) {
        println!("\nBD-rate: SKIP (PSNR not yet wired; see decode_and_psnr stub)");
        return;
    }

    match bd_rate(&anchor_rd, &test_rd) {
        Some(bd) => println!("\nBD-rate: {bd:+.3}%  (anchor → test)"),
        None => println!("\nBD-rate: insufficient points"),
    }
}
