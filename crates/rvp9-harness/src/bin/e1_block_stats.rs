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
//! cargo run --release --bin e1_block_stats -- \
//!     --foreman tests/corpus/foreman_cif.y4m \
//!     --vpxenc /path/to/vpxenc \
//!     --frames 64 --out wrk_docs\E1_encoder_gap_analysis.md
//!
//! cargo run --release --bin e1_block_stats -- \
//!     --rvp9-ivf target\tmp\e9_q80_rvp9_fixedenv.ivf \
//!     --libvpx-ivf target\tmp\e9_q80_libvpx_minq20.ivf \
//!     --label "fixed-Q Q80 / libvpx min-q20" \
//!     --fps 30 --out target\tmp\e9_q80_block_stats.md
//! ```

use rvp9_core::bitstream::FrameKind as WireFrameKind;
use rvp9_core::block::{BLOCK_SIZES, BlockSize};
use rvp9_core::entropy::{EOB_MODEL_TOKEN, INTRA_MODES, ONE_TOKEN, TWO_TOKEN, ZERO_TOKEN};
use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, RefFrame};
use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::read_packets;
use rvp9_harness::y4m::{YuvFrameBuf, read_cropped, workspace_corpus_file};
use serde::Serialize;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const W: u32 = 256;
const H: u32 = 256;
const VPXENC_TIMEOUT: Duration = Duration::from_secs(30);
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
        .map_err(|error| format!("vpxenc spawn: {error}"))?;
    let deadline = Instant::now() + timeout;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                let kill_result = child.kill();
                let wait_result = child.wait();
                if let Err(error) = kill_result {
                    return Err(format!(
                        "vpxenc timed out after {:.1}s and could not be killed: {error}",
                        timeout.as_secs_f64()
                    ));
                }
                if let Err(error) = wait_result {
                    return Err(format!(
                        "vpxenc timed out after {:.1}s and could not be reaped: {error}",
                        timeout.as_secs_f64()
                    ));
                }
                return Err(format!(
                    "vpxenc timed out after {:.1}s",
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
                return Err(format!("vpxenc status: {error}"));
            }
        }
    }
}

/// One frame's worth of decoded block statistics, accumulated from the
/// rvp9 decoder's `last_frame_mi_grid()`.
#[derive(Clone, Default)]
struct FrameStats {
    /// Frame size in bits (whole packet, header + payload).
    bits: u64,
    /// Uncompressed-header size in bits.
    header_bits: u64,
    /// VP9 compressed-header size in bits (`first_partition_size`).
    compressed_header_bits: u64,
    /// VP9 tile-data size in bits (mode syntax + residual tokens).
    tile_data_bits: u64,
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
    intra_mode_area8: [u64; INTRA_MODES],
    inter_mode_area8: [u64; 4], // indexed by InterMode::idx (NEAREST/NEAR/ZERO/NEW)
    /// Area-weighted skip=1.
    skip_area8: u64,
    /// Total visible 8x8 cells this frame.
    total_area8: u64,
    /// Area-weighted reference-frame usage (inter blocks only).
    ref_last_area8: u64,
    ref_golden_area8: u64,
    ref_altref_area8: u64,
    /// Decoder-side coefficient model token counts.
    coef_zero_tokens: u64,
    coef_one_tokens: u64,
    coef_two_plus_tokens: u64,
    coef_eob_tokens: u64,
    coef_eob_branch_attempts: u64,
    /// Distinct coded leaves by luma TX size.
    tx_leaf_count: [u64; 4],
    /// Estimated luma TX blocks by TX size.
    tx_y_block_count: [u64; 4],
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
        self.frames
            .iter()
            .filter(|f| f.is_key)
            .map(|f| f.bits)
            .sum()
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
    fn inter_bitstream_sections(&self) -> (f64, f64, f64) {
        let mut header = 0;
        let mut compressed_header = 0;
        let mut tile_data = 0;
        let mut n = 0;
        for f in self.frames.iter().filter(|f| !f.is_key) {
            header += f.header_bits;
            compressed_header += f.compressed_header_bits;
            tile_data += f.tile_data_bits;
            n += 1;
        }
        let n = f64::from(n.max(1));
        (
            header as f64 / n,
            compressed_header as f64 / n,
            tile_data as f64 / n,
        )
    }
    fn inter_coef_model_tokens(&self) -> (f64, f64, f64, f64, f64) {
        let mut zero = 0;
        let mut one = 0;
        let mut two_plus = 0;
        let mut eob = 0;
        let mut eob_branch = 0;
        let mut n = 0;
        for f in self.frames.iter().filter(|f| !f.is_key) {
            zero += f.coef_zero_tokens;
            one += f.coef_one_tokens;
            two_plus += f.coef_two_plus_tokens;
            eob += f.coef_eob_tokens;
            eob_branch += f.coef_eob_branch_attempts;
            n += 1;
        }
        let n = f64::from(n.max(1));
        (
            zero as f64 / n,
            one as f64 / n,
            two_plus as f64 / n,
            eob as f64 / n,
            eob_branch as f64 / n,
        )
    }
    fn inter_tx_counts(&self) -> ([f64; 4], [f64; 4]) {
        let mut leaves = [0u64; 4];
        let mut y_blocks = [0u64; 4];
        let mut n = 0;
        for f in self.frames.iter().filter(|f| !f.is_key) {
            for (acc, value) in leaves.iter_mut().zip(f.tx_leaf_count.iter()) {
                *acc += *value;
            }
            for (acc, value) in y_blocks.iter_mut().zip(f.tx_y_block_count.iter()) {
                *acc += *value;
            }
            n += 1;
        }
        let n = f64::from(n.max(1));
        (leaves.map(|v| v as f64 / n), y_blocks.map(|v| v as f64 / n))
    }
    /// Sum a per-block array across all INTER frames (the realtime path).
    fn block_count_inter(&self) -> [u64; BLOCK_SIZES] {
        let mut acc = [0u64; BLOCK_SIZES];
        for f in self.frames.iter().filter(|f| !f.is_key) {
            for (acc_slot, value) in acc.iter_mut().zip(f.block_count.iter()) {
                *acc_slot += *value;
            }
        }
        acc
    }
    fn block_area8_inter(&self) -> [u64; BLOCK_SIZES] {
        let mut acc = [0u64; BLOCK_SIZES];
        for f in self.frames.iter().filter(|f| !f.is_key) {
            for (acc_slot, value) in acc.iter_mut().zip(f.block_area8.iter()) {
                *acc_slot += *value;
            }
        }
        acc
    }
    fn block_count_key(&self) -> [u64; BLOCK_SIZES] {
        let mut acc = [0u64; BLOCK_SIZES];
        for f in self.frames.iter().filter(|f| f.is_key) {
            for (acc_slot, value) in acc.iter_mut().zip(f.block_count.iter()) {
                *acc_slot += *value;
            }
        }
        acc
    }
    fn block_area8_key(&self) -> [u64; BLOCK_SIZES] {
        let mut acc = [0u64; BLOCK_SIZES];
        for f in self.frames.iter().filter(|f| f.is_key) {
            for (acc_slot, value) in acc.iter_mut().zip(f.block_area8.iter()) {
                *acc_slot += *value;
            }
        }
        acc
    }
    fn intra_mode_area8_key(&self) -> [u64; INTRA_MODES] {
        let mut acc = [0u64; INTRA_MODES];
        for f in self.frames.iter().filter(|f| f.is_key) {
            for (acc_slot, value) in acc.iter_mut().zip(f.intra_mode_area8.iter()) {
                *acc_slot += *value;
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
            for (acc, value) in modes.iter_mut().zip(f.inter_mode_area8.iter()) {
                *acc += *value;
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

fn intra_mode_label(mode: usize) -> &'static str {
    match mode {
        0 => "DC",
        1 => "V",
        2 => "H",
        3 => "D45",
        4 => "D135",
        5 => "D117",
        6 => "D153",
        7 => "D207",
        8 => "D63",
        9 => "TM",
        _ => "?",
    }
}

#[derive(Serialize)]
struct IvfJsonReport {
    label: String,
    framerate: f64,
    frames_requested: usize,
    rvp9_ivf: String,
    libvpx_ivf: Option<String>,
    rvp9: AggregateJson,
    libvpx: Option<AggregateJson>,
}

#[derive(Serialize)]
struct AggregateJson {
    frames: usize,
    inter_frames: usize,
    decode_failed: usize,
    total_bits: u64,
    key_bits: u64,
    inter_bits: u64,
    achieved_kbps: f64,
    avg_inter_frame_bits: f64,
    avg_inter_q: f64,
    partition_area_pct: Vec<BlockAreaJson>,
    mode_area_pct: ModeAreaJson,
    skip_pct: f64,
}

#[derive(Serialize)]
struct BlockAreaJson {
    block: &'static str,
    count: u64,
    area_pct: f64,
}

#[derive(Serialize)]
struct ModeAreaJson {
    intra: f64,
    nearest_mv: f64,
    near_mv: f64,
    zero_mv: f64,
    new_mv: f64,
    ref_last: f64,
    ref_golden: f64,
    ref_altref: f64,
}

fn aggregate_json(a: &Aggregate, n_frames: usize, framerate: f64) -> AggregateJson {
    let achieved_kbps = a.total_bits() as f64 / n_frames.max(1) as f64 * framerate / 1000.0;
    let avg_inter_frame_bits = if a.n_inter() == 0 {
        0.0
    } else {
        a.inter_bits() as f64 / a.n_inter() as f64
    };
    let inter_q: Vec<u16> = a
        .frames
        .iter()
        .filter(|f| !f.is_key)
        .map(|f| f.base_qindex)
        .collect();
    let avg_inter_q = if inter_q.is_empty() {
        0.0
    } else {
        inter_q.iter().map(|&q| f64::from(q)).sum::<f64>() / inter_q.len() as f64
    };

    let block_count = a.block_count_inter();
    let block_area = a.block_area8_inter();
    let block_total = block_area.iter().sum::<u64>().max(1) as f64;
    let partition_area_pct = (0..BLOCK_SIZES)
        .filter(|&i| block_count[i] > 0 || block_area[i] > 0)
        .map(|i| BlockAreaJson {
            block: block_label(i),
            count: block_count[i],
            area_pct: 100.0 * block_area[i] as f64 / block_total,
        })
        .collect();

    let (intra, total, modes, skip, last, golden, altref) = a.inter_aggregate();
    let total = total.max(1) as f64;
    AggregateJson {
        frames: a.frames.len(),
        inter_frames: a.n_inter(),
        decode_failed: a.n_decode_failed(),
        total_bits: a.total_bits(),
        key_bits: a.key_bits(),
        inter_bits: a.inter_bits(),
        achieved_kbps,
        avg_inter_frame_bits,
        avg_inter_q,
        partition_area_pct,
        mode_area_pct: ModeAreaJson {
            intra: 100.0 * intra as f64 / total,
            nearest_mv: 100.0 * modes[InterMode::NearestMv.idx()] as f64 / total,
            near_mv: 100.0 * modes[InterMode::NearMv.idx()] as f64 / total,
            zero_mv: 100.0 * modes[InterMode::ZeroMv.idx()] as f64 / total,
            new_mv: 100.0 * modes[InterMode::NewMv.idx()] as f64 / total,
            ref_last: 100.0 * last as f64 / total,
            ref_golden: 100.0 * golden as f64 / total,
            ref_altref: 100.0 * altref as f64 / total,
        },
        skip_pct: 100.0 * skip as f64 / total,
    }
}

struct IvfJsonInputs<'a> {
    label: &'a str,
    framerate: f64,
    n_frames: usize,
    rvp9_ivf: &'a Path,
    libvpx_ivf: Option<&'a Path>,
    rvp9: &'a Aggregate,
    libvpx: Option<&'a Aggregate>,
}

fn write_ivf_json_report(out: &Path, inputs: IvfJsonInputs<'_>) {
    let report = IvfJsonReport {
        label: inputs.label.to_string(),
        framerate: inputs.framerate,
        frames_requested: inputs.n_frames,
        rvp9_ivf: inputs.rvp9_ivf.display().to_string(),
        libvpx_ivf: inputs.libvpx_ivf.map(|p| p.display().to_string()),
        rvp9: aggregate_json(inputs.rvp9, inputs.n_frames, inputs.framerate),
        libvpx: inputs
            .libvpx
            .map(|l| aggregate_json(l, inputs.n_frames, inputs.framerate)),
    };
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).expect("create JSON report parent");
    }
    let json = serde_json::to_vec_pretty(&report).expect("serialize block-stat JSON");
    std::fs::write(out, json).expect("write block-stat JSON report");
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
            let tx_idx = mi.tx_size as usize;
            fs.tx_leaf_count[tx_idx] += 1;
            let tx_step_4x4 = 1usize << tx_idx;
            let y_tx_blocks = (bs.n_4x4_wide() as usize).div_ceil(tx_step_4x4)
                * (bs.n_4x4_high() as usize).div_ceil(tx_step_4x4);
            fs.tx_y_block_count[tx_idx] += y_tx_blocks as u64;

            if mi.skip {
                fs.skip_area8 += area8;
            }

            match &mi.mode {
                BlockMode::Intra { y_mode, .. } => {
                    fs.intra_area8 += area8;
                    fs.intra_mode_area8[*y_mode as usize] += area8;
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

fn accumulate_coef_counts(counts: &rvp9_core::entropy::adapt::CoefCountsBank, fs: &mut FrameStats) {
    for tx in counts.coef.iter() {
        for plane in tx.iter() {
            for ref_type in plane.iter() {
                for band in ref_type.iter() {
                    for ctx in band.iter() {
                        fs.coef_zero_tokens += u64::from(ctx[ZERO_TOKEN]);
                        fs.coef_one_tokens += u64::from(ctx[ONE_TOKEN]);
                        fs.coef_two_plus_tokens += u64::from(ctx[TWO_TOKEN]);
                        fs.coef_eob_tokens += u64::from(ctx[EOB_MODEL_TOKEN]);
                    }
                }
            }
        }
    }
    for tx in counts.eob_branch.iter() {
        for plane in tx.iter() {
            for ref_type in plane.iter() {
                for band in ref_type.iter() {
                    for &ctx in band.iter() {
                        fs.coef_eob_branch_attempts += u64::from(ctx);
                    }
                }
            }
        }
    }
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
            let compressed_header_bytes = usize::from(tail.first_partition_size);
            fs.compressed_header_bits += (compressed_header_bytes as u64) * 8;
            let tile_data_start = pf.header_end_byte.saturating_add(compressed_header_bytes);
            let tile_data_bytes = pf.full_frame_data.len().saturating_sub(tile_data_start);
            fs.tile_data_bits += (tile_data_bytes as u64) * 8;
        }
        fs.header_bits += (pf.header_end_byte as u64) * 8;
        fs.is_key = matches!(
            pf.header.kind,
            WireFrameKind::KeyFrame | WireFrameKind::IntraOnly { .. }
        );
        let shown = pf.header.show_frame
            || matches!(pf.header.kind, WireFrameKind::ShowExistingFrame { .. });
        match dec.decode_frame(pf) {
            Ok(_) => {
                if !matches!(pf.header.kind, WireFrameKind::ShowExistingFrame { .. }) {
                    accumulate_coef_counts(dec.last_frame_coef_counts(), fs);
                }
            }
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
        rc_mode: RcMode::Vbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
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
    let tmp = ScopedTempDir::new("rvp9-e1-block-stats").ok()?;
    let raw = tmp.path().join("input.yuv");
    let ivf = tmp.path().join("output.ivf");
    let stderr_path = tmp.path().join("vpxenc.stderr");

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
    let stderr = std::fs::File::create(&stderr_path).ok()?;
    let mut command = Command::new(vpxenc);
    command
        .args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--lag-in-frames=0",
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
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr));
    let status = match run_with_timeout(&mut command, VPXENC_TIMEOUT) {
        Ok(status) => status,
        Err(error) => {
            eprintln!("[libvpx FAIL] {kbps} kbps: {error}");
            return None;
        }
    };
    if !status.success() {
        use std::io::Read;

        let mut stderr = String::new();
        if let Ok(file) = std::fs::File::open(&stderr_path) {
            let _ = file.take(64 * 1024).read_to_string(&mut stderr);
        }
        eprintln!("[libvpx FAIL] {kbps} kbps: {}", stderr.trim_end());
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
    Some(agg)
}

fn decode_ivf(path: &Path) -> Result<Aggregate, String> {
    let (data, packets) =
        read_packets(path).map_err(|e| format!("read {}: {e:?}", path.display()))?;
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
    Ok(agg)
}

/// Format the aggregate comparison for one target bitrate into the
/// markdown report buffer `md`.
fn emit_report(
    md: &mut String,
    label: &str,
    framerate: f64,
    n_frames: usize,
    rvp9: &Aggregate,
    libvpx: Option<&Aggregate>,
) {
    let _ = writeln!(md, "\n## Operating point: {label}\n");

    // ---- 1. Bit allocation ----
    let _ = writeln!(md, "### 1. Bit allocation\n");
    let _ = writeln!(md, "| metric | rvp9 | libvpx |\n|---|---:|---:|");
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
    let (r_hdr, r_comp, r_tile) = rvp9.inter_bitstream_sections();
    let (r_zero, r_one, r_two_plus, r_eob, r_eob_branch) = rvp9.inter_coef_model_tokens();
    let (r_tx_leaves, r_tx_y_blocks) = rvp9.inter_tx_counts();
    let _ = writeln!(
        md,
        "| achieved bitrate (kbps) | {r_kbps:.1} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.1}", line(l).0))
    );
    let _ = writeln!(
        md,
        "| keyframe bits | {r_key:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.0}", line(l).1))
    );
    let _ = writeln!(
        md,
        "| avg inter-frame bits | {r_inter:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.0}", line(l).2))
    );
    let _ = writeln!(
        md,
        "| frames rvp9-decoder FAILED to decode | {} | {} |",
        rvp9.n_decode_failed(),
        libvpx.map_or("n/a".into(), |l| format!("{}", l.n_decode_failed()))
    );
    let _ = writeln!(
        md,
        "| avg inter uncompressed-header bits | {r_hdr:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| {
            format!("{:.0}", l.inter_bitstream_sections().0)
        })
    );
    let _ = writeln!(
        md,
        "| avg inter compressed-header bits | {r_comp:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| {
            format!("{:.0}", l.inter_bitstream_sections().1)
        })
    );
    let _ = writeln!(
        md,
        "| avg inter tile-data bits | {r_tile:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| {
            format!("{:.0}", l.inter_bitstream_sections().2)
        })
    );
    let _ = writeln!(
        md,
        "| avg inter coef model tokens zero/one/two+/eob | {r_zero:.0}/{r_one:.0}/{r_two_plus:.0}/{r_eob:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| {
            let (z, o, t, e, _) = l.inter_coef_model_tokens();
            format!("{z:.0}/{o:.0}/{t:.0}/{e:.0}")
        })
    );
    let _ = writeln!(
        md,
        "| avg inter coef eob-branch attempts | {r_eob_branch:.0} | {} |",
        libvpx.map_or("n/a".into(), |l| {
            format!("{:.0}", l.inter_coef_model_tokens().4)
        })
    );
    let _ = writeln!(
        md,
        "| avg inter luma TX leaves 4x4/8x8/16x16/32x32 | {:.0}/{:.0}/{:.0}/{:.0} | {} |",
        r_tx_leaves[0],
        r_tx_leaves[1],
        r_tx_leaves[2],
        r_tx_leaves[3],
        libvpx.map_or("n/a".into(), |l| {
            let (tx, _) = l.inter_tx_counts();
            format!("{:.0}/{:.0}/{:.0}/{:.0}", tx[0], tx[1], tx[2], tx[3])
        })
    );
    let _ = writeln!(
        md,
        "| avg inter luma TX blocks 4x4/8x8/16x16/32x32 | {:.0}/{:.0}/{:.0}/{:.0} | {} |",
        r_tx_y_blocks[0],
        r_tx_y_blocks[1],
        r_tx_y_blocks[2],
        r_tx_y_blocks[3],
        libvpx.map_or("n/a".into(), |l| {
            let (_, tx) = l.inter_tx_counts();
            format!("{:.0}/{:.0}/{:.0}/{:.0}", tx[0], tx[1], tx[2], tx[3])
        })
    );

    // ---- Keyframe predictor diagnostic ----
    let _ = writeln!(md, "\n### Keyframe partition and luma-mode distribution\n");
    let _ = writeln!(
        md,
        "Distinct keyframe leaves and area coverage. Mode percentages are area-weighted.\n"
    );
    let _ = writeln!(
        md,
        "| field | rvp9 count / area% | libvpx count / area% |\n|---|---:|---:|"
    );
    let r_key_count = rvp9.block_count_key();
    let r_key_area = rvp9.block_area8_key();
    let r_key_total = r_key_area.iter().sum::<u64>().max(1);
    let l_key_count = libvpx.map(Aggregate::block_count_key);
    let l_key_area = libvpx.map(Aggregate::block_area8_key);
    let l_key_total = l_key_area
        .as_ref()
        .map(|area| area.iter().sum::<u64>().max(1));
    for block in 0..BLOCK_SIZES {
        let r_has = r_key_count[block] > 0;
        let l_has = l_key_count.as_ref().is_some_and(|count| count[block] > 0);
        if !r_has && !l_has {
            continue;
        }
        let libvpx_value = match (&l_key_count, &l_key_area, l_key_total) {
            (Some(count), Some(area), Some(total)) => format!(
                "{} / {:.1}",
                count[block],
                100.0 * area[block] as f64 / total as f64
            ),
            _ => "n/a".into(),
        };
        let _ = writeln!(
            md,
            "| partition {} | {} / {:.1} | {} |",
            block_label(block),
            r_key_count[block],
            100.0 * r_key_area[block] as f64 / r_key_total as f64,
            libvpx_value
        );
    }
    let r_key_modes = rvp9.intra_mode_area8_key();
    let l_key_modes = libvpx.map(Aggregate::intra_mode_area8_key);
    let r_mode_total = r_key_modes.iter().sum::<u64>().max(1);
    let l_mode_total = l_key_modes
        .as_ref()
        .map(|modes| modes.iter().sum::<u64>().max(1));
    for mode in 0..INTRA_MODES {
        let r_has = r_key_modes[mode] > 0;
        let l_has = l_key_modes.as_ref().is_some_and(|modes| modes[mode] > 0);
        if !r_has && !l_has {
            continue;
        }
        let libvpx_value = match (&l_key_modes, l_mode_total) {
            (Some(modes), Some(total)) => {
                format!("{:.1}", 100.0 * modes[mode] as f64 / total as f64)
            }
            _ => "n/a".into(),
        };
        let _ = writeln!(
            md,
            "| mode {} | {:.1} | {} |",
            intra_mode_label(mode),
            100.0 * r_key_modes[mode] as f64 / r_mode_total as f64,
            libvpx_value
        );
    }

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
    let _ = writeln!(
        md,
        "| avg base_qindex | {rq_avg:.1} | {} |",
        libvpx.map_or("n/a".into(), |l| format!("{:.1}", avg_q(l).0))
    );
    let _ = writeln!(
        md,
        "| min / max | {rq_min} / {rq_max} | {} |",
        libvpx.map_or("n/a".into(), |l| {
            let (_, mn, mx) = avg_q(l);
            format!("{mn} / {mx}")
        })
    );

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
    let _ = writeln!(
        md,
        "\n### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)\n"
    );
    let _ = writeln!(md, "| metric | rvp9 | libvpx |\n|---|---:|---:|");
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

/// Emit the per-frame bits + q table (compact) for one operating point.
fn emit_per_frame(md: &mut String, label: &str, rvp9: &Aggregate, libvpx: Option<&Aggregate>) {
    let _ = writeln!(md, "\n### Per-frame bits & base_qindex ({label})\n");
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
        let (lb, lq, lintra) = libvpx.and_then(|l| l.frames.get(i)).map_or(
            ("n/a".to_string(), "n/a".to_string(), "n/a".to_string()),
            |f| {
                (
                    format!("{}", f.bits),
                    format!("{}", f.base_qindex),
                    format!("{:.0}", intra_pct(f)),
                )
            },
        );
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {:.0} | {} | {} | {} |",
            i,
            kind,
            r.bits,
            r.base_qindex,
            intra_pct(r),
            lb,
            lq,
            lintra
        );
    }
}

fn arg(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1).cloned())
}

fn vpxenc_arg(argv: &[String]) -> Option<PathBuf> {
    arg(argv, "--vpxenc").map(PathBuf::from)
}

fn report_frame_count(
    rvp9: &Aggregate,
    libvpx: Option<&Aggregate>,
    requested: Option<usize>,
) -> Result<usize, String> {
    let rvp9_frames = rvp9.frames.len();
    let libvpx_frames = libvpx.map(|aggregate| aggregate.frames.len());

    if let Some(expected) = requested {
        if expected == 0 {
            return Err("--frames must be greater than zero".into());
        }
        if rvp9_frames != expected {
            return Err(format!(
                "rvp9 IVF contains {rvp9_frames} frames; expected {expected}"
            ));
        }
        if let Some(actual) = libvpx_frames.filter(|&actual| actual != expected) {
            return Err(format!(
                "libvpx IVF contains {actual} frames; expected {expected}"
            ));
        }
        return Ok(expected);
    }

    if rvp9_frames == 0 {
        return Err("rvp9 IVF contains no frames".into());
    }
    if let Some(actual) = libvpx_frames.filter(|&actual| actual != rvp9_frames) {
        return Err(format!(
            "rvp9 and libvpx IVF frame counts differ: {rvp9_frames} versus {actual}"
        ));
    }
    Ok(rvp9_frames)
}

fn run_ivf_mode(argv: &[String]) -> Option<()> {
    let rvp9_ivf = arg(argv, "--rvp9-ivf").map(PathBuf::from)?;
    let libvpx_ivf = arg(argv, "--libvpx-ivf").map(PathBuf::from);
    let out = PathBuf::from(
        arg(argv, "--out").unwrap_or_else(|| r"target\tmp\e1_ivf_block_stats.md".into()),
    );
    let json_out = arg(argv, "--json-out").map(PathBuf::from);
    let framerate: f64 = arg(argv, "--fps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let requested_frames = arg(argv, "--frames").map(|value| {
        value
            .parse::<usize>()
            .expect("--frames must be a positive integer")
    });
    let label = arg(argv, "--label").unwrap_or_else(|| {
        rvp9_ivf
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("arbitrary IVF")
            .to_string()
    });

    let rvp9 = decode_ivf(&rvp9_ivf).expect("decode rvp9 IVF");
    let libvpx = libvpx_ivf
        .as_deref()
        .map(|path| decode_ivf(path).expect("decode libvpx IVF"));
    let n_frames = report_frame_count(&rvp9, libvpx.as_ref(), requested_frames)
        .expect("IVF frame count mismatch");

    let mut md = String::new();
    let _ = writeln!(md, "# E1 — rvp9 vs libvpx IVF block-stat comparison\n");
    let _ = writeln!(
        md,
        "rvp9 IVF: `{}`\n\nlibvpx IVF: `{}`\n\nBOTH bitstreams decoded through the SAME rvp9 decoder (oracle); all per-block stats use identical definitions.\n",
        rvp9_ivf.display(),
        libvpx_ivf
            .as_ref()
            .map_or("n/a".into(), |p| p.display().to_string())
    );
    let _ = writeln!(
        md,
        "Area-weighted % = fraction of coded 8x8 cells. Mode/skip/ref stats are over INTER frames only.\n"
    );

    emit_report(&mut md, &label, framerate, n_frames, &rvp9, libvpx.as_ref());
    emit_per_frame(&mut md, &label, &rvp9, libvpx.as_ref());
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).expect("create markdown report parent");
    }
    std::fs::write(&out, &md).expect("write report");
    eprintln!("Wrote report to {}", out.display());
    if let Some(json_out) = json_out {
        write_ivf_json_report(
            &json_out,
            IvfJsonInputs {
                label: &label,
                framerate,
                n_frames,
                rvp9_ivf: &rvp9_ivf,
                libvpx_ivf: libvpx_ivf.as_deref(),
                rvp9: &rvp9,
                libvpx: libvpx.as_ref(),
            },
        );
        eprintln!("Wrote JSON report to {}", json_out.display());
        let failed = rvp9.n_decode_failed()
            + libvpx
                .as_ref()
                .map_or(0, |aggregate| aggregate.n_decode_failed());
        assert_eq!(
            failed, 0,
            "JSON block-stat mode requires all frames to decode"
        );
    }
    println!("{md}");
    Some(())
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if run_ivf_mode(&argv).is_some() {
        return;
    }

    let foreman = arg(&argv, "--foreman")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_corpus_file("foreman_cif.y4m"));
    let vpxenc = vpxenc_arg(&argv);
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

    let vpx_avail = vpxenc.as_ref().is_some_and(|path| path.is_file());
    match vpxenc.as_deref() {
        None => eprintln!("[warn] --vpxenc not supplied — libvpx leg skipped"),
        Some(path) if !vpx_avail => eprintln!(
            "[warn] vpxenc not found at {} — libvpx leg skipped",
            path.display()
        ),
        Some(path) => eprintln!("Using vpxenc at {}", path.display()),
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
         rvp9/libvpx run as one-pass VBR (`vpxenc --rt --cpu-used=7 --lag-in-frames=0 --passes=1 --aq-mode=0`). \
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
            run_libvpx(
                vpxenc.as_deref().expect("available vpxenc has a path"),
                t,
                framerate,
                &frames,
            )
        } else {
            None
        };
        emit_report(
            &mut md,
            &format!("{} kbps target", t / 1000),
            framerate,
            n_frames,
            &r,
            l.as_ref(),
        );
        if t == 100_000 {
            rvp9_100 = Some(r);
            libvpx_100 = l;
        }
    }

    if let Some(r) = &rvp9_100 {
        emit_per_frame(&mut md, "100 kbps", r, libvpx_100.as_ref());
    }

    std::fs::write(&out, &md).expect("write report");
    eprintln!("Wrote report to {}", out.display());
    // Also echo to stdout so the data is visible in the run log.
    println!("{md}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_temp_dirs_are_unique_and_removed_on_drop() {
        let first = ScopedTempDir::new("rvp9-e1-test").unwrap();
        let second = ScopedTempDir::new("rvp9-e1-test").unwrap();
        assert_ne!(first.path(), second.path());

        let first_path = first.path().to_path_buf();
        std::fs::write(first.path().join("marker"), b"test").unwrap();
        drop(first);
        assert!(!first_path.exists());
    }

    #[test]
    fn keyframe_field_aggregates_exclude_inter_frames() {
        let mut key = FrameStats {
            is_key: true,
            ..Default::default()
        };
        key.block_count[BlockSize::Block16x16.idx()] = 3;
        key.block_area8[BlockSize::Block16x16.idx()] = 12;
        key.intra_mode_area8[0] = 8;
        key.intra_mode_area8[1] = 4;

        let mut inter = FrameStats::default();
        inter.block_count[BlockSize::Block8x8.idx()] = 64;
        inter.block_area8[BlockSize::Block8x8.idx()] = 64;
        inter.intra_mode_area8[2] = 64;

        let aggregate = Aggregate {
            frames: vec![key, inter],
        };
        assert_eq!(aggregate.block_count_key()[BlockSize::Block16x16.idx()], 3);
        assert_eq!(aggregate.block_area8_key()[BlockSize::Block16x16.idx()], 12);
        assert_eq!(aggregate.block_count_key()[BlockSize::Block8x8.idx()], 0);
        assert_eq!(aggregate.intra_mode_area8_key()[0], 8);
        assert_eq!(aggregate.intra_mode_area8_key()[1], 4);
        assert_eq!(aggregate.intra_mode_area8_key()[2], 0);
    }

    #[test]
    fn run_with_timeout_kills_a_hung_child() {
        let started = Instant::now();
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "tests::timeout_child"])
            .env("RVP9_E1_TIMEOUT_CHILD", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let error = run_with_timeout(&mut child, Duration::from_millis(50)).unwrap_err();

        assert!(error.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn timeout_child() {
        if std::env::var_os("RVP9_E1_TIMEOUT_CHILD").is_some() {
            std::thread::sleep(Duration::from_secs(30));
        }
    }

    #[test]
    fn report_frame_count_rejects_requested_count_mismatch() {
        let mut rvp9 = Aggregate::default();
        rvp9.push(FrameStats::default());
        rvp9.push(FrameStats::default());

        let error = report_frame_count(&rvp9, None, Some(1)).unwrap_err();
        assert!(error.contains("rvp9 IVF contains 2 frames; expected 1"));
    }

    #[test]
    fn report_frame_count_rejects_paired_stream_mismatch() {
        let mut rvp9 = Aggregate::default();
        rvp9.push(FrameStats::default());
        let libvpx = Aggregate::default();

        let error = report_frame_count(&rvp9, Some(&libvpx), None).unwrap_err();
        assert!(error.contains("frame counts differ"));
    }

    #[test]
    fn report_frame_count_accepts_matching_contract() {
        let mut rvp9 = Aggregate::default();
        rvp9.push(FrameStats::default());
        let mut libvpx = Aggregate::default();
        libvpx.push(FrameStats::default());

        assert_eq!(
            report_frame_count(&rvp9, Some(&libvpx), Some(1)).unwrap(),
            1
        );
    }

    #[test]
    fn vpxenc_requires_an_explicit_path() {
        assert_eq!(vpxenc_arg(&["e1_block_stats".into()]), None);
        assert_eq!(
            vpxenc_arg(&[
                "e1_block_stats".into(),
                "--vpxenc".into(),
                "tools/vpxenc".into(),
            ]),
            Some(PathBuf::from("tools/vpxenc"))
        );
    }
}
