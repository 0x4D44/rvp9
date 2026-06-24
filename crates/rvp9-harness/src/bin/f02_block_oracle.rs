// F.0.2 — per-block oracle harness binary. stdout/stderr IS the
// output mechanism (artifacts go to target/track_f_oracle/).
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! F.0.2 — per-(frame, block) libvpx-vs-rvp9 picker-decision oracle.
//!
//! Drives the rvp9 leg (via [`StreamEncoder`] with the
//! `RVP9_F02_ORACLE=1` thread-local sink) and the libvpx leg (via the
//! patched `vpxenc.exe` produced by `scripts/build_f02_libvpx.ps1`)
//! over the same foreman_cif input, then emits:
//!
//! - A per-frame summary table (markdown) per
//!   `target/track_f_oracle/<run>/summary.md`.
//! - The raw paired (rvp9, libvpx) records per
//!   `target/track_f_oracle/<run>/foreman_<kbps>kbps_raw.jsonl`.
//!
//! When the patched libvpx binary is missing, the binary errors out
//! loudly with the exact PowerShell command to bring it up (HLD §3.3).
//! When the corpus file is missing, same behaviour.
//!
//! Usage (HLD §4):
//! ```text
//! cargo run -p rvp9-harness --release --bin f02-block-oracle -- \
//!     --foreman tests/corpus/foreman_cif.y4m \
//!     --vpxenc third_party/libvpx/build_f02/vpxenc.exe \
//!     --frames 64 \
//!     --kbps 25,50,100,250 \
//!     --out target/track_f_oracle/
//! ```

use rvp9_encoder::f02_oracle::{self, BlockDecision};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::f02_schema::parse_jsonl_stream;
use rvp9_harness::y4m::read_cropped;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_W: u32 = 256;
const DEFAULT_H: u32 = 256;
const DEFAULT_FRAMES: u32 = 64;
const DEFAULT_FRAMERATE: f64 = 30.0;

fn arg(args: &[String], key: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == key {
            return iter.next().cloned();
        }
        if let Some(stripped) = a.strip_prefix(&format!("{key}=")) {
            return Some(stripped.to_string());
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let foreman_path = PathBuf::from(
        arg(&argv, "--foreman").unwrap_or_else(|| "tests/corpus/foreman_cif.y4m".into()),
    );
    let vpxenc_path = PathBuf::from(
        arg(&argv, "--vpxenc").unwrap_or_else(|| r"third_party\libvpx\build_f02\vpxenc.exe".into()),
    );
    let frames: u32 = arg(&argv, "--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_FRAMES);
    let kbps_list: Vec<u32> = arg(&argv, "--kbps")
        .unwrap_or_else(|| "100".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let out_dir =
        PathBuf::from(arg(&argv, "--out").unwrap_or_else(|| "target/track_f_oracle".into()));
    let no_libvpx = argv.iter().any(|a| a == "--no-libvpx");

    if !foreman_path.exists() {
        eprintln!(
            "ERROR: foreman corpus not found at {}",
            foreman_path.display()
        );
        eprintln!("  Run scripts/fetch-corpus.{{ps1,sh}} to fetch it, then re-run.");
        std::process::exit(1);
    }
    if !no_libvpx && !vpxenc_path.exists() {
        eprintln!(
            "ERROR: patched vpxenc.exe not found at {}",
            vpxenc_path.display()
        );
        eprintln!("  Build it with:");
        eprintln!("    pwsh -File scripts/build_f02_libvpx.ps1");
        eprintln!("  Or pass --no-libvpx to drive only the rvp9 leg.");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&out_dir)?;
    let run_dir = out_dir.join(format!("run_{}", chrono_like_timestamp()));
    std::fs::create_dir_all(&run_dir)?;

    println!("F.0.2 per-block oracle harness");
    println!("  foreman   : {}", foreman_path.display());
    if no_libvpx {
        println!("  libvpx    : SKIPPED (--no-libvpx)");
    } else {
        println!("  libvpx    : {}", vpxenc_path.display());
    }
    println!("  frames    : {}", frames);
    println!("  bitrates  : {:?} kbps", kbps_list);
    println!("  out dir   : {}", run_dir.display());

    // Enable the rvp9 leg's oracle. Once enabled at process startup
    // for THIS process; the harness driver's encoder runs are the
    // only consumers, so no measurement leakage to other tools.
    // SAFETY: process-local; no other thread races on this var.
    unsafe { std::env::set_var("RVP9_F02_ORACLE", "1") };
    f02_oracle::init_from_env();

    for &kbps in &kbps_list {
        println!("\n=== {} kbps target bitrate ===", kbps);
        let rvp9_records = drive_rvp9_leg(&foreman_path, frames, kbps)?;
        println!(
            "  rvp9: {} per-block decisions across {} frames",
            rvp9_records.len(),
            frames
        );

        let libvpx_records = if no_libvpx {
            Vec::new()
        } else {
            drive_libvpx_leg(&vpxenc_path, &foreman_path, frames, kbps, &run_dir)?
        };
        if !no_libvpx {
            println!("  libvpx: {} per-block decisions", libvpx_records.len());
        }

        // Per-frame summary table.
        let summary_path = run_dir.join(format!("foreman_{}kbps_summary.md", kbps));
        let mut sf = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&summary_path)?;
        writeln!(sf, "# F.0.2 oracle — {} kbps", kbps)?;
        writeln!(sf)?;
        writeln!(sf, "Schema liveness at this milestone (HLD §3.4.1):")?;
        writeln!(
            sf,
            "- `coef_bits_est`: **0 on both legs** at F.0.2 (forward-looking; F.1 lands the rvp9-side estimate)."
        )?;
        writeln!(
            sf,
            "- `mode_bits_est`, `mv_bits_est`: live on rvp9, **0 on libvpx leg** at F.0.2 (libvpx-side split is a future patch)."
        )?;
        writeln!(sf)?;
        writeln!(sf, "| frame_idx | rvp9 n_blocks | libvpx n_blocks |")?;
        writeln!(sf, "|---|---|---|")?;
        let mut rvp9_by_frame = std::collections::BTreeMap::<u32, usize>::new();
        for d in &rvp9_records {
            *rvp9_by_frame.entry(d.frame_idx).or_default() += 1;
        }
        let mut vpx_by_frame = std::collections::BTreeMap::<u32, usize>::new();
        for d in &libvpx_records {
            *vpx_by_frame.entry(d.frame_idx).or_default() += 1;
        }
        let max_frame = rvp9_by_frame
            .keys()
            .chain(vpx_by_frame.keys())
            .copied()
            .max()
            .unwrap_or(0);
        for f in 0..=max_frame {
            writeln!(
                sf,
                "| {} | {} | {} |",
                f,
                rvp9_by_frame.get(&f).copied().unwrap_or(0),
                vpx_by_frame.get(&f).copied().unwrap_or(0),
            )?;
        }
        println!("  summary → {}", summary_path.display());

        // Raw JSONL dump of rvp9-side records.
        let raw_path = run_dir.join(format!("foreman_{}kbps_rvp9.jsonl", kbps));
        let mut rf = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&raw_path)?;
        for d in &rvp9_records {
            d.write_jsonl(&mut rf)?;
        }
        println!("  rvp9 raw  → {}", raw_path.display());
    }

    println!("\nDone. Artifacts in {}", run_dir.display());
    Ok(())
}

/// Approximation of an ISO-8601-ish stamp without pulling in `chrono`
/// (workspace has no dep on it). Format `yyyymmdd_hhmmss` based on
/// `SystemTime::now()`.
fn chrono_like_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Crude: seconds-since-epoch in hex. Unique enough for the
    // harness's per-run subdir.
    format!("{:x}", now)
}

/// Drive the rvp9 leg: open foreman_cif, encode `frames` frames at
/// the requested target bitrate, drain the per-block sink at end of
/// each frame.
fn drive_rvp9_leg(
    foreman_path: &Path,
    frames: u32,
    kbps: u32,
) -> Result<Vec<BlockDecision>, Box<dyn std::error::Error>> {
    let cfg = EncoderConfig {
        width: DEFAULT_W,
        height: DEFAULT_H,
        framerate: DEFAULT_FRAMERATE,
        target_bitrate_bps: kbps * 1000,
        gop_length: 30,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg)?;
    let (yuv_frames, _src_fps) = read_cropped(foreman_path, DEFAULT_W, DEFAULT_H, frames as usize)?;
    let mut all_records = Vec::new();
    // Drain any prior thread-local state.
    let _ = f02_oracle::drain_frame();
    for (y, u, v) in yuv_frames.iter() {
        let view = YuvFrameRef {
            y,
            u,
            v,
            y_stride: DEFAULT_W as usize,
            u_stride: DEFAULT_W as usize / 2,
            v_stride: DEFAULT_W as usize / 2,
        };
        let _ = enc.encode_frame(&view)?;
        let mut per_frame = f02_oracle::drain_frame();
        all_records.append(&mut per_frame);
    }
    Ok(all_records)
}

/// Drive the libvpx leg: spawn `vpxenc` with `VPX_F02_ORACLE_OUT`
/// pointed at a sidecar JSONL file under `run_dir`. The harness
/// then re-reads that file.
///
/// **Stub note (F.0.2 landing):** this invocation assumes the patched
/// vpxenc is on the right libvpx commit and accepts y4m input on
/// stdin. The CLI options match the BD-rate gate's libvpx invocation
/// for parity. If the patched binary is absent, the caller already
/// errored out before reaching here.
fn drive_libvpx_leg(
    vpxenc: &Path,
    foreman: &Path,
    frames: u32,
    kbps: u32,
    run_dir: &Path,
) -> Result<Vec<BlockDecision>, Box<dyn std::error::Error>> {
    let sidecar = run_dir.join(format!("foreman_{}kbps_libvpx.jsonl", kbps));
    // libvpx CLI mirror of the BD-rate gate at e1_block_stats.rs:586+
    // (matched config so the per-block decisions are comparable).
    let mut cmd = Command::new(vpxenc);
    cmd.env("VPX_F02_ORACLE_OUT", &sidecar)
        .arg(foreman)
        .arg("-o")
        .arg(run_dir.join(format!("foreman_{}kbps_libvpx.ivf", kbps)))
        .arg("--codec=vp9")
        .arg("--rt")
        .arg("--cpu-used=7")
        .arg(format!("--target-bitrate={}", kbps))
        .arg(format!("--limit={}", frames))
        .arg("--end-usage=cbr")
        .arg("--aq-mode=0")
        .arg("--passes=1")
        // `--rt` makes lag-in-frames a no-op; libvpx still prints a
        // warning and waits for `y` on stdin. Set lag explicitly to
        // suppress the prompt (matches the BD-rate gate invocation).
        .arg("--lag-in-frames=0")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let out = cmd.output()?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "vpxenc failed (exit {}): {}",
            out.status.code().unwrap_or(-1),
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        )
        .into());
    }
    if !sidecar.exists() {
        return Err(format!(
            "libvpx sidecar at {} was NOT produced — \
             is third_party/libvpx-patches/f02_oracle.patch applied?",
            sidecar.display()
        )
        .into());
    }
    let f = File::open(&sidecar)?;
    let records = parse_jsonl_stream(BufReader::new(f))
        .map_err(|e| format!("libvpx sidecar parse failed: {e}"))?;
    Ok(records)
}
