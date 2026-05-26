// Throwaway diagnostic (E1): stdout/stderr ARE the output mechanism.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! E1 — rvp9 vs libvpx realtime encoder bit-allocation diagnostic.
//!
//! Decodes BOTH a rvp9-encoded foreman bitstream AND a libvpx-encoded
//! foreman bitstream through the SAME rvp9 decoder, dumping per-block
//! statistics so the two encoders can be compared apples-to-apples
//! (same parser, same definitions — a true oracle side-by-side).
//!
//! For each gate target bitrate (25/50/100/250 kbps) it measures, per
//! encoder, both per-frame and aggregate:
//!   1. Bit allocation (total/keyframe/inter bits per frame).
//!   2. Partition size distribution (distinct coded leaf blocks +
//!      area-weighted coverage).
//!   3. Mode distribution (ZEROMV / NEARESTMV / NEARMV / NEWMV / intra).
//!   4. Skip-flag fraction (area-weighted).
//!   5. Per-frame base_qindex.
//!   6. Reference-frame usage (LAST / GOLDEN / ALTREF).
//!
//! NO encoder/decoder logic is changed: this is additive measurement.
//!
//! Usage:
//! ```text
//! cargo run --release --bin e1-block-stats -- \
//!     --foreman D:\language\rvp9-2\tests\corpus\foreman_cif.y4m \
//!     --vpxenc  D:\language\libvpx-stock\vpxenc.exe \
//!     --frames 64 --out wrk_docs\E1_encoder_gap_analysis.md
//! ```

use rvp9_core::block::{BLOCK_SIZES, BlockSize};
use rvp9_core::bitstream::FrameKind as WireFrameKind;
use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, RefFrame};
use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::read_packets;
use rvp9_harness::y4m::{YuvFrameBuf, read_cropped};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

const W: u32 = 256;
const H: u32 = 256;

/// One frame's worth of decoded block statistics, accumulated from the
/// rvp9 decoder's `last_frame_mi_grid()`.
#[derive(Clone, Default)]
struct FrameStats {
    /// Frame size in bits (whole packet, header + payload).
    bits: u64,
    /// `true` for keyframe / intra-only.
    is_key: bool,
    /// `base_qindex` from the uncompressed header (255 if header absent).
    base_qindex: u16,
    /// Distinct coded leaf blocks per BlockSize (deduped, counts each
    /// partition leaf exactly once at its top-left mi cell).
    block_count: [u64; BLOCK_SIZES],
    /// Area in 8x8 units coded by each BlockSize (sums to the visible
    /// 8x8 cell count).
    block_area8: [u64; BLOCK_SIZES],
    /// Area-weighted (8x8 units) mode tallies.
    intra_area8: u64,
    inter_mode_area8: [u64; 4], // indexed by InterMode::idx (NEAREST/NEAR/ZERO/NEW)
    /// Area-weighted skip=1.
    skip_area8: u64,
    /// Total visible 8x8 cells this frame.
    total_area8: u64,
    /// Area-weighted reference-frame usage (inter blocks only).
    ref_last_area8: u64,
    ref_golden_area8: u64,
    ref_altref_area8: u64,
    /// `true` if the rvp9 decoder returned an error for this frame
    /// (so the mi-grid is STALE — stats must be treated as invalid).
    decode_failed: bool,
}

/// Aggregate over a sequence, per encoder.
#[derive(Default)]
struct Aggregate {
    frames: Vec<FrameStats>,
}

impl Aggregate {
    fn push(&mut self, f: FrameStats) {
        self.frames.push(f);
    }

    fn key_bits(&self) -> u64 {
        self.frames.iter().filter(|f| f.is_key).map(|f| f.bits).sum()
    }
    fn inter_bits(&self) -> u64 {
        self.frames
            .iter()
            .filter(|f| !f.is_key)
            .map(|f| f.bits)
            .sum()
    }
    fn total_bits(&self) -> u64 {
        self.frames.iter().map(|f| f.bits).sum()
    }
    fn n_inter(&self) -> usize {
        self.frames.iter().filter(|f| !f.is_key).count()
    }
    fn n_decode_failed(&self) -> usize {
        self.frames.iter().filter(|f| f.decode_failed).count()
    }
    /// Sum a per-block array across all INTER frames (the realtime path).
    fn block_count_inter(&self) -> [u64; BLOCK_SIZES] {
        let mut acc = [0u64; BLOCK_SIZES];
        for f in self.frames.iter().filter(|f| !f.is_key) {
            for i in 0..BLOCK_SIZES {
                acc[i] += f.block_count[i];
            }
        }
        acc
    }
    fn block_area8_inter(&self) -> [u64; BLOCK_SIZES] {
        let mut acc = [0u64; BLOCK_SIZES];
        for f in self.frames.iter().filter(|f| !f.is_key) {
            for i in 0..BLOCK_SIZES {
                acc[i] += f.block_area8[i];
            }
        }
        acc
    }
    fn inter_aggregate(&self) -> (u64, u64, [u64; 4], u64, u64, u64, u64) {
        // (intra_area, total_area, inter_mode_area, skip_area, last, golden, altref)
        let mut intra = 0;
        let mut total = 0;
        let mut modes = [0u64; 4];
        let mut skip = 0;
        let mut last = 0;
        let mut golden = 0;
        let mut altref = 0;
        for f in self.frames.iter().filter(|f| !f.is_key) {
            intra += f.intra_area8;
            total += f.total_area8;
            for i in 0..4 {
                modes[i] += f.inter_mode_area8[i];
            }
            skip += f.skip_area8;
            last += f.ref_last_area8;
            golden += f.ref_golden_area8;
            altref += f.ref_altref_area8;
        }
        (intra, total, modes, skip, last, golden, altref)
    }
}

fn block_label(b: usize) -> &'static str {
    match b {
        0 => "4x4",
        1 => "4x8",
        2 => "8x4",
        3 => "8x8",
        4 => "8x16",
        5 => "16x8",
        6 => "16x16",
        7 => "16x32",
        8 => "32x16",
        9 => "32x32",
        10 => "32x64",
        11 => "64x32",
        12 => "64x64",
        _ => "?",
    }
}

/// Walk the decoder mi-grid and accumulate per-block stats into `fs`.
/// Dedupes leaves: each partition leaf is counted once at its top-left
/// 8x8 cell (libvpx fills every covered 8x8 cell with identical MI, so
/// we step by the leaf's 8x8 extent to avoid double-counting).
fn accumulate_grid(grid: &[Option<ModeInfo>], mi_rows: usize, mi_cols: usize, fs: &mut FrameStats) {
    let mut visited = vec![false; mi_rows * mi_cols];
    for row in 0..mi_rows {
        for col in 0..mi_cols {
            let idx = row * mi_cols + col;
            if visited[idx] {
                continue;
            }
            let Some(mi) = &grid[idx] else {
                visited[idx] = true;
                continue;
            };
            let bs = mi.block_size;
            let bw8 = bs.n_8x8_wide() as usize;
            let bh8 = bs.n_8x8_high() as usize;
            // Number of visible 8x8 cells this leaf covers (clamped to
            // frame edge).
            let cells_w = bw8.min(mi_cols - col);
            let cells_h = bh8.min(mi_rows - row);
            let area8 = (cells_w * cells_h) as u64;

            // Mark covered cells visited (clamped to grid).
            for r in row..(row + bh8).min(mi_rows) {
                for c in col..(col + bw8).min(mi_cols) {
                    visited[r * mi_cols + c] = true;
                }
            }

            // Count the distinct leaf once.
            fs.block_count[bs.idx()] += 1;
            fs.block_area8[bs.idx()] += area8;
            fs.total_area8 += area8;

            if mi.skip {
                fs.skip_area8 += area8;
            }

            match &mi.mode {
                BlockMode::Intra { .. } => {
                    fs.intra_area8 += area8;
                }
                BlockMode::Inter {
                    mode, ref_frames, ..
                } => {
                    let m: InterMode = *mode;
                    fs.inter_mode_area8[m.idx()] += area8;
                    // Primary reference frame.
                    match ref_frames[0] {
                        RefFrame::LastFrame => fs.ref_last_area8 += area8,
                        RefFrame::GoldenFrame => fs.ref_golden_area8 += area8,
                        RefFrame::AltrefFrame => fs.ref_altref_area8 += area8,
                        _ => {}
                    }
                }
            }
        }
    }
    let _ = BlockSize::Block8x8; // keep import alive if unused above
}

/// Decode one packet through the rvp9 decoder and accumulate the
/// resulting mi-grid into a FrameStats (bits + key flag set by caller).
fn decode_and_accumulate(dec: &mut Decoder, pkt: &[u8], fs: &mut FrameStats) {
    let parsed = dec.parse_packet(pkt).expect("parse_packet");
    // Collect to owned to drop the borrow on `dec`.
    let frames: Vec<ParsedFrame> = parsed
        .frames()
        .iter()
        .map(|pf| ParsedFrame {
            header: pf.header.clone(),
            header_end_byte: pf.header_end_byte,
            full_frame_data: pf.full_frame_data,
        })
        .collect();
    for pf in &frames {
        // base_qindex + key flag from the header where available.
        if let Some(tail) = &pf.header.tail {
            fs.base_qindex = u16::from(tail.quantization.base_qindex);
        }
        fs.is_key = matches!(
            pf.header.kind,
            WireFrameKind::KeyFrame | WireFrameKind::IntraOnly { .. }
        );
        let shown = pf.header.show_frame
            || matches!(pf.header.kind, WireFrameKind::ShowExistingFrame { .. });
        match dec.decode_frame(pf) {
            Ok(_) => {}
            Err(e) => {
                fs.decode_failed = true;
                if std::env::var_os("E1_VERBOSE").is_some() {
                    eprintln!("    [decode err] kind={:?} err={e:?}", pf.header.kind);
                }
            }
        }
        if !shown {
            continue;
        }
    }
    // CRITICAL: if decode failed, `last_frame_mi_grid()` holds the STALE
    // grid from the previous successful frame. Accumulating it would
    // misattribute (e.g.) the keyframe's all-intra grid to every failed
    // inter frame. Only accumulate on success.
    if fs.decode_failed {
        return;
    }
    let (mi_rows, mi_cols) = dec.last_frame_mi_dims();
    let grid = dec.last_frame_mi_grid();
    if mi_rows > 0 && mi_cols > 0 && grid.len() == (mi_rows * mi_cols) as usize {
        accumulate_grid(grid, mi_rows as usize, mi_cols as usize, fs);
    }
}

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

/// Drive rvp9 StreamEncoder over the sequence at `target_bitrate_bps`,
/// decode each packet through `dec`, return the per-frame Aggregate.
fn run_rvp9(target_bitrate_bps: u32, framerate: f64, frames: &[YuvFrameBuf]) -> Aggregate {
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps,
        gop_length: frames.len() as u32 + 1,
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
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();
    let mut agg = Aggregate::default();
    for (y, u, v) in frames {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        let mut fs = FrameStats {
            bits: u64::from(pkt.actual_bits),
            ..Default::default()
        };
        decode_and_accumulate(&mut dec, &pkt.data, &mut fs);
        agg.push(fs);
    }
    agg
}

/// Drive libvpx vpxenc (`--rt --cpu-used=7`) over the sequence, then
/// decode the IVF packets through the rvp9 decoder for the oracle stats.
fn run_libvpx(
    vpxenc: &Path,
    target_bitrate_bps: u32,
    framerate: f64,
    frames: &[YuvFrameBuf],
) -> Option<Aggregate> {
    let tmp = std::env::temp_dir();
    let raw = tmp.join(format!("e1_{}kbps.yuv", target_bitrate_bps / 1000));
    let ivf = tmp.join(format!("e1_{}kbps.ivf", target_bitrate_bps / 1000));

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&raw).ok()?;
        for (y, u, v) in frames {
            f.write_all(y).ok()?;
            f.write_all(u).ok()?;
            f.write_all(v).ok()?;
        }
    }

    let kbps = target_bitrate_bps / 1000;
    let out = Command::new(vpxenc)
        .args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--tune-content=default",
            "--passes=1",
            "--i420",
            "--aq-mode=0",
        ])
        .arg(format!("--width={W}"))
        .arg(format!("--height={H}"))
        .arg(format!("--fps={}/1", framerate as u32))
        .arg(format!("--target-bitrate={kbps}"))
        .arg("-o")
        .arg(&ivf)
        .arg(&raw)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "[libvpx FAIL] {kbps} kbps: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    let (data, packets) = read_packets(&ivf).ok()?;
    let mut dec = Decoder::new();
    let mut agg = Aggregate::default();
    for range in &packets {
        let slice = range.slice(&data);
        let mut fs = FrameStats {
            bits: (slice.len() as u64) * 8,
            ..Default::default()
        };
        decode_and_accumulate(&mut dec, slice, &mut fs);
        agg.push(fs);
    }
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    Some(agg)
}

/// Format the aggregate comparison for one target bitrate into the
/// markdown report buffer `md`.
fn emit_report(
    md: &mut String,
    kbps: u32,
    framerate: f64,
    n_frames: usize,
    rvp9: &Aggregate,
    libvpx: Option<&Aggregate>,
) {
    let _ = writeln!(md, "\n## Operating point: {kbps} kbps target\n");

    // ---- 1. Bit allocation ----
    let _ = writeln!(md, "### 1. Bit allocation\n");
    let _ = writeln!(
        md,
        "| metric | rvp9 | libvpx |\n|---|---:|---:|"
    );
    let line = |a: &Aggregate| -> (f64, f64, f64, f64) {
        let total = a.total_bits() as f64;
        let avg_kbps = total / n_frames as f64 * framerate / 1000.0;
        let key = a.key_bits() as f64;
        let inter_avg = if a.n_inter() > 0 {
            a.inter_bits() as f64 / a.n_inter() as f64
        } else {
            0.0
        };
        (avg_kbps, key, inter_avg, total)
    };
    let (r_kbps, r_key, r_inter, _r_tot) = line(rvp9);
    let _ = writeln!(md, "| achieved bitrate (kbps) | {r_kbps:.1} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.1}", line(l).0)));
    let _ = writeln!(md, "| keyframe bits | {r_key:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.0}", line(l).1)));
    let _ = writeln!(md, "| avg inter-frame bits | {r_inter:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.0}", line(l).2)));
    let _ = writeln!(md, "| frames rvp9-decoder FAILED to decode | {} | {} |",
        rvp9.n_decode_failed(),
        libvpx.map_or("n/a".into(), |l| format!("{}", l.n_decode_failed())));

    // ---- 5. Per-frame Q (report aggregate avg here, full table later) ----
    let avg_q = |a: &Aggregate| -> (f64, u16, u16) {
        let inter: Vec<u16> = a
            .frames
            .iter()
            .filter(|f| !f.is_key)
            .map(|f| f.base_qindex)
            .collect();
        let n = inter.len().max(1);
        let avg = inter.iter().map(|&q| f64::from(q)).sum::<f64>() / n as f64;
        let min = inter.iter().copied().min().unwrap_or(0);
        let max = inter.iter().copied().max().unwrap_or(0);
        (avg, min, max)
    };
    let (rq_avg, rq_min, rq_max) = avg_q(rvp9);
    let _ = writeln!(md, "\n### 5. base_qindex (inter frames)\n");
    let _ = writeln!(md, "| metric | rvp9 | libvpx |\n|---|---:|---:|");
    let _ = writeln!(md, "| avg base_qindex | {rq_avg:.1} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.1}", avg_q(l).0)));
    let _ = writeln!(md, "| min / max | {rq_min} / {rq_max} | {} |",
        libvpx.map_or("n/a".into(), |l| { let (_, mn, mx) = avg_q(l); format!("{mn} / {mx}") }));

    // ---- 2. Partition size distribution (inter frames, area-weighted) ----
    let _ = writeln!(md, "\n### 2. Partition size distribution (inter frames)\n");
    let _ = writeln!(
        md,
        "Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).\n"
    );
    let _ = writeln!(
        md,
        "| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |\n|---|---:|---:|---:|---:|"
    );
    let r_cnt = rvp9.block_count_inter();
    let r_area = rvp9.block_area8_inter();
    let r_area_tot: u64 = r_area.iter().sum::<u64>().max(1);
    let l_cnt = libvpx.map(|l| l.block_count_inter());
    let l_area = libvpx.map(|l| l.block_area8_inter());
    let l_area_tot = l_area.as_ref().map(|a| a.iter().sum::<u64>().max(1));
    for b in 0..BLOCK_SIZES {
        // Skip rows where both are zero to keep the table readable.
        let r_has = r_cnt[b] > 0;
        let l_has = l_cnt.as_ref().is_some_and(|c| c[b] > 0);
        if !r_has && !l_has {
            continue;
        }
        let r_pct = 100.0 * r_area[b] as f64 / r_area_tot as f64;
        let (lc, lp) = match (&l_cnt, &l_area, l_area_tot) {
            (Some(c), Some(a), Some(t)) => (
                format!("{}", c[b]),
                format!("{:.1}", 100.0 * a[b] as f64 / t as f64),
            ),
            _ => ("n/a".into(), "n/a".into()),
        };
        let _ = writeln!(
            md,
            "| {} | {} | {:.1} | {} | {} |",
            block_label(b),
            r_cnt[b],
            r_pct,
            lc,
            lp
        );
    }

    // ---- 3. Mode distribution + 4. Skip + 6. Ref usage (area-weighted) ----
    let _ = writeln!(md, "\n### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)\n");
    let _ = writeln!(
        md,
        "| metric | rvp9 | libvpx |\n|---|---:|---:|"
    );
    let dist = |a: &Aggregate| -> [f64; 9] {
        let (intra, total, modes, skip, last, golden, altref) = a.inter_aggregate();
        let t = total.max(1) as f64;
        [
            100.0 * intra as f64 / t,
            100.0 * modes[InterMode::NearestMv.idx()] as f64 / t,
            100.0 * modes[InterMode::NearMv.idx()] as f64 / t,
            100.0 * modes[InterMode::ZeroMv.idx()] as f64 / t,
            100.0 * modes[InterMode::NewMv.idx()] as f64 / t,
            100.0 * skip as f64 / t,
            100.0 * last as f64 / t,
            100.0 * golden as f64 / t,
            100.0 * altref as f64 / t,
        ]
    };
    let rd = dist(rvp9);
    let ld = libvpx.map(dist);
    let names = [
        "intra %",
        "NEARESTMV %",
        "NEARMV %",
        "ZEROMV %",
        "NEWMV %",
        "skip=1 %",
        "ref LAST %",
        "ref GOLDEN %",
        "ref ALTREF %",
    ];
    for (i, name) in names.iter().enumerate() {
        let _ = writeln!(
            md,
            "| {} | {:.1} | {} |",
            name,
            rd[i],
            ld.map_or("n/a".into(), |l| format!("{:.1}", l[i]))
        );
    }
}

/// Emit the per-frame bits + q table (compact) for the 100 kbps point.
fn emit_per_frame(md: &mut String, rvp9: &Aggregate, libvpx: Option<&Aggregate>) {
    let _ = writeln!(md, "\n### Per-frame bits & base_qindex (100 kbps)\n");
    let _ = writeln!(
        md,
        "| frame | kind | rvp9 bits | rvp9 q | rvp9 intra% | libvpx bits | libvpx q | libvpx intra% |\n|---:|---|---:|---:|---:|---:|---:|---:|"
    );
    let intra_pct = |f: &FrameStats| -> f64 {
        let t = f.total_area8.max(1) as f64;
        100.0 * f.intra_area8 as f64 / t
    };
    let n = rvp9.frames.len();
    for i in 0..n {
        let r = &rvp9.frames[i];
        let kind = if r.is_key { "KEY" } else { "inter" };
        let (lb, lq, lintra) = libvpx
            .and_then(|l| l.frames.get(i))
            .map_or(("n/a".to_string(), "n/a".to_string(), "n/a".to_string()), |f| {
                (
                    format!("{}", f.bits),
                    format!("{}", f.base_qindex),
                    format!("{:.0}", intra_pct(f)),
                )
            });
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {:.0} | {} | {} | {} |",
            i, kind, r.bits, r.base_qindex, intra_pct(r), lb, lq, lintra
        );
    }
}

fn arg(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1).cloned())
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let foreman = PathBuf::from(
        arg(&argv, "--foreman")
            .unwrap_or_else(|| r"D:\language\rvp9-2\tests\corpus\foreman_cif.y4m".into()),
    );
    let vpxenc = PathBuf::from(
        arg(&argv, "--vpxenc").unwrap_or_else(|| r"D:\language\libvpx-stock\vpxenc.exe".into()),
    );
    let n_frames: usize = arg(&argv, "--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let out = PathBuf::from(
        arg(&argv, "--out").unwrap_or_else(|| r"wrk_docs\E1_encoder_gap_analysis.md".into()),
    );

    let (frames, framerate) = read_cropped(&foreman, W, H, n_frames).expect("read foreman");
    assert_eq!(frames.len(), n_frames, "short read on foreman");
    eprintln!(
        "Loaded {} frames {}x{} @ {} fps from {}",
        frames.len(),
        W,
        H,
        framerate,
        foreman.display()
    );

    let vpx_avail = vpxenc.is_file();
    if !vpx_avail {
        eprintln!("[warn] vpxenc not found at {} — libvpx leg skipped", vpxenc.display());
    }

    let targets: [u32; 4] = [25_000, 50_000, 100_000, 250_000];

    let mut md = String::new();
    let _ = writeln!(
        md,
        "# E1 — rvp9 vs libvpx realtime encoder gap analysis (foreman_cif)\n"
    );
    let _ = writeln!(
        md,
        "Source: foreman_cif centre-crop {W}x{H}, {n_frames} frames @ {framerate} fps. \
         libvpx run as `vpxenc --rt --cpu-used=7 --passes=1 --aq-mode=0`. \
         BOTH bitstreams decoded through the SAME rvp9 decoder (oracle); all \
         per-block stats use identical definitions.\n"
    );
    let _ = writeln!(
        md,
        "Area-weighted % = fraction of coded 8x8 cells (so a 64x64 leaf counts \
         64x more than an 8x8 leaf). 'count' = number of distinct partition \
         leaves. Mode/skip/ref stats are over INTER frames only.\n"
    );

    // Keep the 100kbps aggregates around for the per-frame table.
    let mut rvp9_100: Option<Aggregate> = None;
    let mut libvpx_100: Option<Aggregate> = None;

    for &t in &targets {
        eprintln!("--- {} kbps ---", t / 1000);
        let r = run_rvp9(t, framerate, &frames);
        let l = if vpx_avail {
            run_libvpx(&vpxenc, t, framerate, &frames)
        } else {
            None
        };
        emit_report(&mut md, t / 1000, framerate, n_frames, &r, l.as_ref());
        if t == 100_000 {
            rvp9_100 = Some(r);
            libvpx_100 = l;
        }
    }

    if let Some(r) = &rvp9_100 {
        emit_per_frame(&mut md, r, libvpx_100.as_ref());
    }

    std::fs::write(&out, &md).expect("write report");
    eprintln!("Wrote report to {}", out.display());
    // Also echo to stdout so the data is visible in the run log.
    println!("{md}");
}
