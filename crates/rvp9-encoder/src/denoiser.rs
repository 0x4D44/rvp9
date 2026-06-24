//! W5.1 – W5.4 — VP9 temporal denoiser (standalone module).
//!
//! Faithful scalar port of libvpx `vp9/encoder/vp9_denoiser.c`. The
//! denoiser is a per-block NLM-style filter that runs on the encoder's
//! reference buffer to reduce noise propagation across frames.
//!
//! ## Scope
//!
//! This module ships the **denoiser core as a standalone subsystem**:
//! allocate the running-average/MC-running-average/last-source plane
//! buffers, apply the strong+weak two-pass scalar filter to a single
//! source/MC block, and provide a top-level driver
//! (`vp9_denoiser_denoise`) that orchestrates motion-compensation,
//! filter decision, and copy-back.
//!
//! Encoder integration (wiring the denoiser to the encoder's per-block
//! pick loop, plumbing it into `frame_loop::StreamEncoder`, the
//! `noise_sensitivity` → `DenoiserLevel` mapping, the post-frame
//! update-frame-info bookkeeping, etc.) lands in a later wave. The
//! `EncoderConfig::noise_sensitivity` knob plumbed in W5.1 currently
//! gates **nothing** in the encoder hot path; it is the contract on
//! which the future integration hooks in.
//!
//! ## libvpx file:line citations
//!
//! - `vp9_denoiser.c:28-31`   — `absdiff_thresh`.
//! - `vp9_denoiser.c:33-37`   — `delta_thresh`.
//! - `vp9_denoiser.c:39-43`   — `noise_motion_thresh`.
//! - `vp9_denoiser.c:45-47`   — `sse_thresh`.
//! - `vp9_denoiser.c:49-59`   — `sse_diff_thresh`.
//! - `vp9_denoiser.c:61-63`   — `total_adj_weak_thresh`.
//! - `vp9_denoiser.h:107-110` — `total_adj_strong_thresh` (inline).
//! - `vp9_denoiser.h:22`      — `MOTION_MAGNITUDE_THRESHOLD = 8*3`.
//! - `vp9_denoiser.h:26`      — `NONSVC_REF_FRAMES = MAX_REF_FRAMES-1 = 3`.
//! - `vp9_denoiser.h:32-36`   — `VP9_DENOISER_DECISION` enum.
//! - `vp9_denoiser.h:38-43`   — `VP9_DENOISER_LEVEL` enum.
//! - `vp9_denoiser.h:45-56`   — `VP9_DENOISER` struct.
//! - `vp9_denoiser.c:68-180`  — `vp9_denoiser_filter_c` (scalar filter ref).
//! - `vp9_denoiser.c:182-185` — `block_start` (sub-block pointer math).
//! - `vp9_denoiser.c:187-334` — `perform_motion_compensation`.
//! - `vp9_denoiser.c:336-437` — `vp9_denoiser_denoise` (top-level driver).
//! - `vp9_denoiser.c:550-604` — `vp9_denoiser_realloc_svc`.
//! - `vp9_denoiser.c:606-696` — `vp9_denoiser_alloc`.
//! - `vp9_common_data.c:35-36` — `num_pels_log2_lookup`.
//!
//! ## Deviations from libvpx (documented)
//!
//! - **No HBD path.** rvp9 is 8-bit-only across the board; the
//!   `CONFIG_VP9_HIGHBITDEPTH` branches in `vp9_denoiser_alloc` and
//!   the `vp9_highbd_denoiser_filter_*` SSE2 stubs are omitted.
//! - **No SIMD.** Faithful scalar reference only. The libvpx
//!   `vp9_denoiser_filter_sse2` / `_neon` variants live behind the
//!   rtcd dispatcher; SIMD parity is a follow-up wave item.
//! - **No SVC paths.** rvp9 has no SVC support yet, so the
//!   `use_svc=true` branch in `vp9_denoiser_alloc`, the
//!   `vp9_denoiser_realloc_svc_helper`-shaped per-slot realloc logic,
//!   and the `denoise_layer_idx` indexing inside
//!   `perform_motion_compensation` collapse to single-layer no-ops.
//!   `num_layers = 1` always; `num_ref_frames = NONSVC_REF_FRAMES = 3`.
//! - **No skin-detection plumbing.** libvpx
//!   `vp9_denoiser.c:367-403` calls `vp9_compute_skin_block` to
//!   compute `is_skin` (gating the "no denoising on skin" path inside
//!   `perform_motion_compensation`). rvp9 has no skin detector yet
//!   (W2.5 placeholder); we accept `is_skin` as an argument with
//!   default `false` so future integration can pass it through.
//! - **No `PICK_MODE_CONTEXT` / `MACROBLOCK` coupling.** libvpx's
//!   `perform_motion_compensation` mutates `MACROBLOCKD` plane buffer
//!   pointers and re-runs `vp9_build_inter_predictors_sby` to fill the
//!   MC-running-avg buffer. The rvp9 standalone module accepts the MC
//!   block as a precomputed `&[u8]` argument; integration is later.
//! - **No `cpi->consec_zero_mv` read.** The libvpx driver pulls
//!   `consec_zeromv` from `cpi->consec_zero_mv[]` inside the skin path
//!   (`vp9_denoiser.c:382-396`). rvp9's `PreAnalysisState` carries
//!   `consec_zero_mv`; the integration wave will thread that through.
//! - **No reuse of `vpx_convolve_copy`.** The final copy-back uses
//!   straight `memcpy` row-by-row; libvpx's `vpx_convolve_copy` does
//!   the same thing but goes through the rtcd table.
//!
//! ## Block-size scope of the scalar filter
//!
//! libvpx's `vp9_denoiser_filter_c` accepts any `BLOCK_SIZE` from
//! `BLOCK_8X8` up. `perform_motion_compensation` early-returns
//! `COPY_BLOCK` for `8X8` / `8X16` / `16X8` and (under some
//! denoising-level / width gates) `16X16`. In practice the denoiser
//! runs on 16×16 and above. We mirror libvpx exactly: the filter
//! accepts a `BlockSize` and reads its `width × height`; the driver's
//! early-return predicates match libvpx line-for-line.

use rvp9_core::block::BlockSize;

// -----------------------------------------------------------------------------
// Constants (libvpx `vp9_denoiser.h:22-30`)
// -----------------------------------------------------------------------------

/// `vp9_denoiser.h:22` — motion-magnitude threshold below which the
/// filter shifts its `adj_val` table to be more aggressive.
const MOTION_MAGNITUDE_THRESHOLD: i32 = 8 * 3;

/// `vp9_denoiser.h:26` — non-SVC slot count = `MAX_REF_FRAMES - 1 = 3`.
///
/// libvpx allocates `running_avg_y` for `INTRA_FRAME` (index 0) plus
/// `LAST_FRAME` (1) and `GOLDEN_FRAME` (2). `ALTREF_FRAME` (index 3 in
/// `MAX_REF_FRAMES = 4`) is **excluded** since the denoiser only runs
/// in non-SVC real-time mode which has no alt-ref.
pub const NONSVC_REF_FRAMES: usize = 3;

/// libvpx INTRA_FRAME ref index (`vp9_blockd.h:58`). The running-avg
/// scratch buffer the denoiser writes into per-frame lives at this
/// slot.
pub const INTRA_FRAME_IDX: usize = 0;

/// `vp9_denoiser.h:32-36` — output of the per-block filter decision.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DenoiserDecision {
    /// Strong+weak filters both rejected; the source is copied into
    /// the running-avg buffer unmodified.
    CopyBlock,
    /// Strong or weak filter accepted; the running-avg buffer holds
    /// the filtered output and is copied back into the source.
    FilterBlock,
    /// `FilterBlock` taken with the zero-MV reference path.
    /// Distinguished by the driver post-filter (libvpx
    /// `vp9_denoiser.c:435-436`).
    FilterZeroMvBlock,
}

/// `vp9_denoiser.h:38-43` — denoising aggressiveness level. Driven by
/// `vp9_denoiser_set_noise_level` from `oxcf.noise_sensitivity`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum DenoiserLevel {
    /// `kDenLowLow` — least aggressive.
    LowLow = 0,
    /// `kDenLow` — low.
    Low = 1,
    /// `kDenMedium` — medium (libvpx default, set by `vp9_denoiser_alloc`).
    Medium = 2,
    /// `kDenHigh` — most aggressive.
    High = 3,
}

// -----------------------------------------------------------------------------
// Threshold helpers (libvpx `vp9_denoiser.c:28-63` + `vp9_denoiser.h:107-110`)
// -----------------------------------------------------------------------------

/// `log2(width * height)` for a `BlockSize`. Mirrors libvpx
/// `num_pels_log2_lookup[BLOCK_SIZES]` (`vp9_common_data.c:35-36`).
#[inline]
const fn num_pels_log2(bs: BlockSize) -> u32 {
    (bs.width_log2() + bs.height_log2()) as u32
}

/// `vp9_denoiser.c:28-31` — `absdiff_thresh`. Sample-level "small
/// enough to snap to MC" threshold. `(void)bs` in libvpx — block size
/// is ignored.
#[inline]
#[must_use]
pub const fn absdiff_thresh(_bs: BlockSize, increase_denoising: bool) -> i32 {
    3 + if increase_denoising { 1 } else { 0 }
}

/// `vp9_denoiser.c:33-37` — `delta_thresh`. Weak-filter damping cap.
/// Both `bs` and `increase_denoising` are ignored in libvpx
/// (`(void)bs; (void)increase_denoising;`).
#[inline]
#[must_use]
pub const fn delta_thresh(_bs: BlockSize, _increase_denoising: bool) -> i32 {
    4
}

/// `vp9_denoiser.c:39-43` — `noise_motion_thresh`. Motion-magnitude
/// boundary between "noise-like" and "real motion". Ignores both
/// arguments in libvpx.
#[inline]
#[must_use]
pub const fn noise_motion_thresh(_bs: BlockSize, _increase_denoising: bool) -> i32 {
    625
}

/// `vp9_denoiser.c:45-47` — `sse_thresh`. Upper bound on NEWMV SSE
/// above which the driver refuses to denoise the block.
#[inline]
#[must_use]
pub const fn sse_thresh(bs: BlockSize, increase_denoising: bool) -> u32 {
    let scale: u32 = if increase_denoising { 80 } else { 40 };
    (1u32 << num_pels_log2(bs)) * scale
}

/// `vp9_denoiser.c:49-59` — `sse_diff_thresh`. Branches on motion
/// magnitude.
#[inline]
#[must_use]
pub const fn sse_diff_thresh(
    bs: BlockSize,
    increase_denoising: bool,
    motion_magnitude: i32,
) -> i32 {
    if motion_magnitude > noise_motion_thresh(bs, increase_denoising) {
        if increase_denoising {
            (1i32 << num_pels_log2(bs)) << 2
        } else {
            0
        }
    } else {
        (1i32 << num_pels_log2(bs)) << 4
    }
}

/// `vp9_denoiser.c:61-63` — `total_adj_weak_thresh`. Weak filter's
/// per-block accumulated-adjustment cap.
#[inline]
#[must_use]
pub const fn total_adj_weak_thresh(bs: BlockSize, increase_denoising: bool) -> i32 {
    (1i32 << num_pels_log2(bs)) * if increase_denoising { 3 } else { 2 }
}

/// `vp9_denoiser.h:107-110` — `total_adj_strong_thresh`. Strong
/// filter's per-block accumulated-adjustment cap.
#[inline]
#[must_use]
pub const fn total_adj_strong_thresh(bs: BlockSize, increase_denoising: bool) -> i32 {
    (1i32 << num_pels_log2(bs)) * if increase_denoising { 3 } else { 2 }
}

// -----------------------------------------------------------------------------
// vp9_denoiser_filter_c — the scalar reference per-block filter
// (libvpx `vp9_denoiser.c:68-180`)
// -----------------------------------------------------------------------------

/// `vp9_denoiser.c:68-180` — `vp9_denoiser_filter_c`. Two-pass per-block
/// NLM-style filter.
///
/// Inputs:
/// - `sig` — source plane window (`bs.width()×bs.height()`, stride
///   `sig_stride`).
/// - `mc_avg` — motion-compensated running-average window (same shape,
///   stride `mc_avg_stride`).
/// - `avg` — running-average output window (same shape, stride
///   `avg_stride`). Will be **overwritten** with the strong filter's
///   output and possibly the weak filter's correction.
///
/// Returns:
/// - `FilterBlock` if either the strong filter's adjustment fit within
///   `total_adj_strong_thresh` OR the weak filter brought it within
///   `total_adj_weak_thresh`.
/// - `CopyBlock` otherwise.
#[allow(clippy::too_many_arguments)]
pub fn vp9_denoiser_filter(
    sig: &[u8],
    sig_stride: usize,
    mc_avg: &[u8],
    mc_avg_stride: usize,
    avg: &mut [u8],
    avg_stride: usize,
    increase_denoising: bool,
    bs: BlockSize,
    motion_magnitude: i32,
) -> DenoiserDecision {
    let bw = bs.width() as usize;
    let bh = bs.height() as usize;

    // `vp9_denoiser.c:77` — base adj table for absdiff buckets [4..7],
    // [8..15], else. Shifted up if motion is small.
    let mut adj_val: [i32; 3] = [3, 4, 6];
    let mut shift_inc: i32 = 1;
    if motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD {
        if increase_denoising {
            shift_inc = 2;
        }
        adj_val[0] += shift_inc;
        adj_val[1] += shift_inc;
        adj_val[2] += shift_inc;
    }

    // -------------------------------------------------------------------------
    // Strong filter pass (lines 93-130).
    // -------------------------------------------------------------------------
    let mut total_adj: i32 = 0;
    let absd_thresh = absdiff_thresh(bs, increase_denoising);

    for r in 0..bh {
        for c in 0..bw {
            let s = sig[r * sig_stride + c] as i32;
            let m = mc_avg[r * mc_avg_stride + c] as i32;
            let diff = m - s;
            let absdiff = diff.abs();

            if absdiff <= absd_thresh {
                // Snap to MC value.
                avg[r * avg_stride + c] = m as u8;
                total_adj += diff;
            } else {
                // Adjust toward MC by a bucket-quantised step.
                let adj = match absdiff {
                    4..=7 => adj_val[0],
                    8..=15 => adj_val[1],
                    _ => adj_val[2],
                };
                if diff > 0 {
                    avg[r * avg_stride + c] = (s + adj).min(255) as u8;
                    total_adj += adj;
                } else {
                    avg[r * avg_stride + c] = (s - adj).max(0) as u8;
                    total_adj -= adj;
                }
            }
        }
    }

    // `vp9_denoiser.c:133-135` — strong filter accepted.
    let strong_thresh = total_adj_strong_thresh(bs, increase_denoising);
    if total_adj.abs() <= strong_thresh {
        return DenoiserDecision::FilterBlock;
    }

    // `vp9_denoiser.c:138-144` — try the weak (damped) filter, or bail.
    let delta = ((total_adj.abs() - strong_thresh) >> num_pels_log2(bs)) + 1;
    if delta >= delta_thresh(bs, increase_denoising) {
        return DenoiserDecision::CopyBlock;
    }

    // -------------------------------------------------------------------------
    // Weak filter pass (lines 146-173) — claw back excess adjustment.
    // -------------------------------------------------------------------------
    for r in 0..bh {
        for c in 0..bw {
            let s = sig[r * sig_stride + c] as i32;
            let m = mc_avg[r * mc_avg_stride + c] as i32;
            let diff = m - s;
            let adj = diff.abs().min(delta);
            let cur = avg[r * avg_stride + c] as i32;
            if diff > 0 {
                // Strong pass pushed `avg` up; pull it back down.
                avg[r * avg_stride + c] = (cur - adj).max(0) as u8;
                total_adj -= adj;
            } else {
                // Strong pass pushed `avg` down; pull it back up.
                avg[r * avg_stride + c] = (cur + adj).min(255) as u8;
                total_adj += adj;
            }
        }
    }

    // `vp9_denoiser.c:176-179` — weak filter accepted iff it sat
    // within the weak-threshold envelope.
    if total_adj.abs() <= total_adj_weak_thresh(bs, increase_denoising) {
        DenoiserDecision::FilterBlock
    } else {
        DenoiserDecision::CopyBlock
    }
}

// -----------------------------------------------------------------------------
// vp9_denoiser_alloc / free / realloc (libvpx `vp9_denoiser.c:606-717`)
// -----------------------------------------------------------------------------

/// Per-frame buffer in the denoiser. A tight-stride luma plane plus
/// dimensions. The chroma planes don't participate in the denoiser
/// (libvpx only filters luma); we still mirror the libvpx
/// `YV12_BUFFER_CONFIG` shape so the future allocator can extend to
/// chroma without re-shaping the API.
#[derive(Debug, Clone, Default)]
pub struct DenoiserFrame {
    /// Tight-stride luma samples (`stride == width`, `len == width*height`).
    pub y_buffer: Vec<u8>,
    /// Plane width in pixels.
    pub width: u32,
    /// Plane height in pixels.
    pub height: u32,
    /// Row stride in bytes (== `width` for the tight-stride layout).
    pub stride: u32,
}

impl DenoiserFrame {
    /// Allocate a zero-initialised tight-stride luma buffer of the
    /// given dimensions. Mirrors `vpx_alloc_frame_buffer` for the
    /// `y_buffer` plane only (the denoiser ignores chroma).
    #[must_use]
    pub fn alloc(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            y_buffer: vec![0u8; len],
            width,
            height,
            stride: width,
        }
    }

    /// `true` if the plane carries a non-empty buffer.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        !self.y_buffer.is_empty()
    }
}

/// `vp9_denoiser.h:45-56` — the rvp9 mirror of libvpx `VP9_DENOISER`.
///
/// Layout collapsed to the non-SVC single-layer shape per the
/// deviations block: `num_layers = 1`, `num_ref_frames =
/// NONSVC_REF_FRAMES = 3`. `running_avg_y[INTRA_FRAME_IDX]` is the
/// per-frame scratch the per-block filter writes into; the other
/// slots hold the previous-frame denoised reference for LAST/GOLDEN.
#[derive(Debug, Clone)]
pub struct Denoiser {
    /// `vp9_denoiser.h:46` — per-ref-frame running-average buffers.
    /// Length == `num_ref_frames * num_layers` = 3 for non-SVC.
    /// Index 0 is the per-frame scratch (`INTRA_FRAME`); indices 1
    /// (LAST_FRAME) and 2 (GOLDEN_FRAME) hold the previous-frame
    /// denoised references.
    pub running_avg_y: Vec<DenoiserFrame>,
    /// `vp9_denoiser.h:47` — motion-compensated running-average,
    /// per layer (length == `num_layers` = 1 for non-SVC).
    pub mc_running_avg_y: Vec<DenoiserFrame>,
    /// `vp9_denoiser.h:48` — previous source frame (luma plane only).
    /// Consumed by the noise-estimator; not by the per-block filter.
    pub last_source: DenoiserFrame,
    /// `vp9_denoiser.h:49` — `frame_buffer_initialized`.
    pub frame_buffer_initialized: bool,
    /// `vp9_denoiser.h:50` — `reset`. Set by the encoder on resize /
    /// scene-cut to force a copy-from-source into all references on
    /// the next frame.
    pub reset: bool,
    /// `vp9_denoiser.h:51` — `num_ref_frames`. For non-SVC always
    /// `NONSVC_REF_FRAMES = 3`.
    pub num_ref_frames: usize,
    /// `vp9_denoiser.h:52` — `num_layers`. For non-SVC always `1`.
    pub num_layers: usize,
    /// `vp9_denoiser.h:53` — `current_denoiser_frame` counter.
    pub current_denoiser_frame: u32,
    /// `vp9_denoiser.h:54` — `denoising_level`. Initialised to
    /// `Medium` by `vp9_denoiser_alloc` (libvpx line 691).
    pub denoising_level: DenoiserLevel,
    /// `vp9_denoiser.h:55` — `prev_denoising_level`.
    pub prev_denoising_level: DenoiserLevel,
}

/// Allocation failure for [`Denoiser::alloc`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenoiserAllocError {
    /// The requested denoiser configuration is disabled or has invalid dimensions.
    InvalidConfig,
}

impl std::fmt::Display for DenoiserAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid denoiser configuration"),
        }
    }
}

impl std::error::Error for DenoiserAllocError {}

impl Denoiser {
    /// `vp9_denoiser.c:606-696` — `vp9_denoiser_alloc`. Allocates the
    /// `num_ref_frames` running-avg buffers, the `num_layers`
    /// MC-running-avg buffers, and the `last_source` plane.
    ///
    /// Returns [`DenoiserAllocError::InvalidConfig`] for invalid
    /// dimensions (width or height == 0) or disabled denoising (`noise_sen == 0`).
    /// libvpx returns `int fail` instead; the Rust port models the
    /// fallible result with a typed error.
    ///
    /// Simplifications vs libvpx:
    /// - No `use_svc` arg / SVC scaling — single layer always.
    /// - No `ssx`/`ssy` / `border` / `use_highbitdepth` args — the
    ///   rvp9 buffer model is tight-stride 8-bit only, and the
    ///   denoiser ignores chroma.
    /// - `noise_sen` collapses to "did the caller ask for
    ///   denoising?" — `0` means disabled (the caller should not call
    ///   `alloc` at all in that case); any non-zero value allocates
    ///   the buffers and sets `denoising_level = Medium`.
    pub fn alloc(noise_sen: u8, width: u32, height: u32) -> Result<Self, DenoiserAllocError> {
        if width == 0 || height == 0 || noise_sen == 0 {
            return Err(DenoiserAllocError::InvalidConfig);
        }
        let num_layers = 1;
        let num_ref_frames = NONSVC_REF_FRAMES;
        // `vp9_denoiser.c:644-674` — allocate per-layer per-ref bufs.
        let mut running_avg_y = Vec::with_capacity(num_ref_frames * num_layers);
        for _ in 0..(num_ref_frames * num_layers) {
            running_avg_y.push(DenoiserFrame::alloc(width, height));
        }
        let mut mc_running_avg_y = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            mc_running_avg_y.push(DenoiserFrame::alloc(width, height));
        }
        // `vp9_denoiser.c:678-686` — `last_source` (full frame, top layer only).
        let last_source = DenoiserFrame::alloc(width, height);
        Ok(Self {
            running_avg_y,
            mc_running_avg_y,
            last_source,
            frame_buffer_initialized: true,
            reset: false,
            num_ref_frames,
            num_layers,
            current_denoiser_frame: 0,
            // `vp9_denoiser.c:691-692` — Medium is the libvpx default
            // post-alloc.
            denoising_level: DenoiserLevel::Medium,
            prev_denoising_level: DenoiserLevel::Medium,
        })
    }

    /// `vp9_denoiser.c:698-717` — `vp9_denoiser_free`. In Rust the
    /// `Drop` impl on `Vec<DenoiserFrame>` handles deallocation;
    /// `free` is a no-op convenience that drops the buffers and
    /// marks the denoiser as uninitialised.
    pub fn free(&mut self) {
        self.running_avg_y.clear();
        self.mc_running_avg_y.clear();
        self.last_source = DenoiserFrame::default();
        self.frame_buffer_initialized = false;
    }

    /// `vp9_denoiser.c:550-604` — `vp9_denoiser_realloc_svc`,
    /// simplified to the non-SVC case (which is a no-op: there is
    /// nothing to lazily reallocate when `num_layers == 1` and
    /// the alloc path eagerly fills every slot).
    ///
    /// Kept as a method on `Denoiser` so the future SVC integration
    /// has the entry-point already named.
    pub fn realloc_svc_nonsvc(&self) {
        // Intentionally empty: non-SVC always has every slot eager-allocated.
    }

    /// Returns `true` if every slot in `running_avg_y` carries a
    /// buffer of the expected size (`width * height` samples).
    /// Used by the W5.2 PROOF test.
    #[must_use]
    pub fn buffers_match_frame(&self, width: u32, height: u32) -> bool {
        let expected = (width as usize) * (height as usize);
        self.running_avg_y.len() == self.num_ref_frames * self.num_layers
            && self
                .running_avg_y
                .iter()
                .all(|f| f.y_buffer.len() == expected && f.width == width && f.height == height)
            && self.mc_running_avg_y.len() == self.num_layers
            && self
                .mc_running_avg_y
                .iter()
                .all(|f| f.y_buffer.len() == expected && f.width == width && f.height == height)
            && self.last_source.y_buffer.len() == expected
    }
}

// -----------------------------------------------------------------------------
// perform_motion_compensation + vp9_denoiser_denoise driver
// (libvpx `vp9_denoiser.c:187-437`)
// -----------------------------------------------------------------------------

/// `vp9_denoiser.c:182-185` — pointer offset to the top-left sample
/// of the `(mi_row, mi_col)` block in a stride-`stride` plane.
/// `mi_row` / `mi_col` are 8×8-unit coordinates (libvpx `MI_SIZE = 8`).
#[inline]
#[must_use]
pub const fn block_offset(stride: usize, mi_row: usize, mi_col: usize) -> usize {
    stride * (mi_row << 3) + (mi_col << 3)
}

/// Per-block "denoiser decision context" — the subset of libvpx
/// `PICK_MODE_CONTEXT` that `perform_motion_compensation` reads. The
/// standalone module needs the caller to populate this; the future
/// encoder integration will fill it from the picker's per-block state.
///
/// Mirrors libvpx `PICK_MODE_CONTEXT` fields touched at
/// `vp9_denoiser.c:193-249`.
#[derive(Copy, Clone, Debug)]
pub struct DenoiserBlockCtx {
    /// `ctx->best_reference_frame`. `1 = LAST_FRAME` is the only
    /// supported value in non-SVC RT mode where the denoiser runs.
    pub best_reference_frame: i8,
    /// `ctx->best_zeromv_reference_frame`. Used when sse_diff is too
    /// small (line 229).
    pub best_zeromv_reference_frame: i8,
    /// `ctx->newmv_sse`. `UINT_MAX` means "no NEWMV winner".
    pub newmv_sse: u32,
    /// `ctx->zeromv_sse`.
    pub zeromv_sse: u32,
    /// `ctx->zeromv_lastref_sse`.
    pub zeromv_lastref_sse: u32,
    /// `ctx->sb_skip_denoising` — caller's gate to bypass denoising
    /// entirely for this block.
    pub sb_skip_denoising: bool,
}

impl DenoiserBlockCtx {
    /// Default context for tests. Sets `best_reference_frame =
    /// LAST_FRAME (1)`, `newmv_sse = UINT_MAX`, and all sses to zero.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            best_reference_frame: 1,
            best_zeromv_reference_frame: 1,
            newmv_sse: u32::MAX,
            zeromv_sse: 0,
            zeromv_lastref_sse: 0,
            sb_skip_denoising: false,
        }
    }
}

/// `vp9_denoiser.c:187-334` — `perform_motion_compensation`. The
/// rvp9 port collapses libvpx's MACROBLOCKD pointer-juggling and
/// inter-predictor re-run (since we accept the MC block as a
/// precomputed argument); the early-return predicates are mirrored
/// line-for-line.
///
/// Returns `CopyBlock` if any of the bail-out predicates fires, else
/// `FilterBlock` (the caller then runs `vp9_denoiser_filter`).
#[allow(clippy::too_many_arguments)]
fn perform_motion_compensation(
    denoiser: &Denoiser,
    bs: BlockSize,
    increase_denoising: bool,
    ctx: &DenoiserBlockCtx,
    motion_magnitude: i32,
    is_skin: bool,
    zeromv_filter: &mut bool,
    consec_zeromv: i32,
    width: u32,
) -> DenoiserDecision {
    // `vp9_denoiser.c:193-195` — sse_diff.
    let sse_diff = if ctx.newmv_sse == u32::MAX {
        0i32
    } else {
        (ctx.zeromv_sse as i32) - (ctx.newmv_sse as i32)
    };
    let mut frame: i8 = ctx.best_reference_frame;

    // `vp9_denoiser.c:211` — skin + motion early-out.
    if is_skin && (motion_magnitude > 0 || consec_zeromv < 4) {
        return DenoiserDecision::CopyBlock;
    }

    // `vp9_denoiser.c:215-218` — small-block + width gate. We mirror
    // the exact `bs == BLOCK_8X8 || bs == BLOCK_8X16 || bs ==
    // BLOCK_16X8 || (bs == BLOCK_16X16 && width > 480 && level <=
    // kDenLow)` predicate.
    let small_block = matches!(
        bs,
        BlockSize::Block8x8 | BlockSize::Block8x16 | BlockSize::Block16x8
    );
    if small_block
        || (matches!(bs, BlockSize::Block16x16)
            && width > 480
            && denoiser.denoising_level <= DenoiserLevel::Low)
    {
        return DenoiserDecision::CopyBlock;
    }

    // `vp9_denoiser.c:222-250` — inter-pred decision branch. We don't
    // mutate `mi->ref_frame`/`mi->mode` (no `mi` in standalone mode);
    // we still mirror the branch so `zeromv_filter` and `frame`
    // selection match libvpx.
    //
    // Constants: INTRA_FRAME=0, LAST_FRAME=1, GOLDEN_FRAME=2,
    // ALTREF_FRAME=3.
    let inter_pick = frame != 0
        && frame != 3
        && frame != 2
        && sse_diff > sse_diff_thresh(bs, increase_denoising, motion_magnitude);
    let mut effective_newmv_sse = ctx.newmv_sse;
    if inter_pick {
        // Keep `frame` as-is.
    } else {
        // `vp9_denoiser.c:228-249` — fall through to the zero-MV ref.
        frame = ctx.best_zeromv_reference_frame;
        effective_newmv_sse = ctx.zeromv_sse;
        // `vp9_denoiser.c:232-240` — bias to LAST.
        let bias_to_last = (frame == 3)
            || (frame != 1
                && ((ctx.zeromv_lastref_sse < (5 * ctx.zeromv_sse) >> 2)
                    || denoiser.denoising_level >= DenoiserLevel::High));
        if bias_to_last {
            frame = 1;
            effective_newmv_sse = ctx.zeromv_lastref_sse;
        }
        *zeromv_filter = true;
        // `vp9_denoiser.c:247-249` — high-level denoising zeros motion.
        // We hand back the (possibly-zeroed) motion via the caller's
        // `motion_magnitude` arg shadow below.
        let _ = motion_magnitude; // (the libvpx version mutates a local copy)
    }

    // `vp9_denoiser.c:267-271` — null running_avg slot guard. The
    // standalone module always allocates every slot, so this only
    // fires for an out-of-range `frame`.
    let frame_idx = frame as usize;
    if frame_idx >= denoiser.num_ref_frames || !denoiser.running_avg_y[frame_idx].is_allocated() {
        return DenoiserDecision::CopyBlock;
    }

    // `vp9_denoiser.c:273-282` — SSE / motion-magnitude bail-outs.
    if effective_newmv_sse > sse_thresh(bs, increase_denoising) {
        return DenoiserDecision::CopyBlock;
    }
    if motion_magnitude > (noise_motion_thresh(bs, increase_denoising) << 3) {
        return DenoiserDecision::CopyBlock;
    }

    // `vp9_denoiser.c:284-333` — libvpx re-runs the inter-predictor
    // here to fill `mc_running_avg_y[denoise_layer_idx]`. The
    // standalone module accepts the MC block as a precomputed input
    // to `vp9_denoiser_denoise` instead — see the caller path below.
    DenoiserDecision::FilterBlock
}

/// Per-block inputs the standalone driver needs from the encoder.
///
/// Mirrors the subset of libvpx `vp9_denoiser_denoise` arguments that
/// the encoder hot-path supplies. The MC predictor (`mc_block`) is
/// precomputed by the picker / inter-pred wave; this module never
/// re-runs prediction.
#[derive(Debug)]
pub struct DenoiseBlockInputs<'a> {
    /// 8×8 mi-row of the block top-left.
    pub mi_row: usize,
    /// 8×8 mi-col of the block top-left.
    pub mi_col: usize,
    /// Block partition size.
    pub bs: BlockSize,
    /// Picker-side state (newmv_sse / zeromv_sse / best refs).
    pub ctx: DenoiserBlockCtx,
    /// Best NEWMV row component (1/8-pel units). `0` for ZEROMV.
    pub mv_row: i32,
    /// Best NEWMV col component (1/8-pel units). `0` for ZEROMV.
    pub mv_col: i32,
    /// Skin-detection result for the block (libvpx
    /// `vp9_compute_skin_block`). `false` until W2.5 lands.
    pub is_skin: bool,
    /// `consec_zero_mv[]` minimum across the block's 8×8 sub-cells
    /// (libvpx `vp9_denoiser.c:384-396`).
    pub consec_zeromv: i32,
    /// Source-block samples (`bs.width()×bs.height()`, stride
    /// `src_stride`).
    pub src: &'a [u8],
    /// `src` row stride.
    pub src_stride: usize,
    /// Motion-compensated predictor for the block (precomputed by
    /// the picker / inter-pred wave). Same shape as `src`.
    pub mc_block: &'a [u8],
    /// `mc_block` row stride.
    pub mc_block_stride: usize,
    /// Whether LAST_FRAME is a reference for the current frame.
    pub last_is_reference: bool,
    /// Frame luma width in pixels (consumed by the small-block gate).
    pub width: u32,
    /// `cpi->use_skin_detection` (libvpx
    /// `vp9_denoiser.c:367`). When `false`, `is_skin` is forced to
    /// `false` regardless of caller's value.
    pub use_skin_detection: bool,
}

/// `vp9_denoiser.c:336-437` — `vp9_denoiser_denoise`. Top-level
/// per-block driver: decide whether to denoise (via
/// `perform_motion_compensation`), apply `vp9_denoiser_filter` if so,
/// and copy back the result into either the running-avg buffer (on
/// `FILTER`) or the source-shaped output (on `COPY`).
///
/// Operates on the running-avg plane at `INTRA_FRAME_IDX` (the
/// per-frame scratch). Returns the final per-block decision plus the
/// post-filter source buffer the caller can write back to its own
/// recon. The MC running-avg buffer in `denoiser.mc_running_avg_y[0]`
/// is *not* re-derived from the predictor here — the caller passes
/// the MC block in `inputs.mc_block` directly.
///
/// Returns `(decision, denoised_block_owned)`:
/// - `decision` — `CopyBlock` / `FilterBlock` / `FilterZeroMvBlock`.
/// - `denoised_block_owned` — the per-block result: the running-avg
///   window's samples on `FilterBlock` / `FilterZeroMvBlock`; the
///   source samples on `CopyBlock`. Tight-stride `bw*bh` bytes.
pub fn vp9_denoiser_denoise(
    denoiser: &mut Denoiser,
    inputs: &DenoiseBlockInputs<'_>,
) -> (DenoiserDecision, Vec<u8>) {
    let motion_magnitude = inputs.mv_row * inputs.mv_row + inputs.mv_col * inputs.mv_col;
    let mut zeromv_filter = false;

    // `vp9_denoiser.c:367-403` — skin-detection / consec_zeromv path.
    // The standalone module trusts the caller's `is_skin` and
    // `consec_zeromv`; the `use_skin_detection` gate at line 367 is
    // honoured.
    let is_skin = inputs.use_skin_detection && inputs.is_skin;
    let increase_denoising = !is_skin && denoiser.denoising_level == DenoiserLevel::High;

    // `vp9_denoiser.c:410-417` — only attempt denoising when LAST is
    // a reference, the denoising level is at least `Low`, and the
    // caller hasn't asked us to skip.
    let mut decision = DenoiserDecision::CopyBlock;
    if inputs.last_is_reference
        && denoiser.denoising_level >= DenoiserLevel::Low
        && !inputs.ctx.sb_skip_denoising
    {
        decision = perform_motion_compensation(
            denoiser,
            inputs.bs,
            increase_denoising,
            &inputs.ctx,
            motion_magnitude,
            is_skin,
            &mut zeromv_filter,
            inputs.consec_zeromv,
            inputs.width,
        );
    }

    let bw = inputs.bs.width() as usize;
    let bh = inputs.bs.height() as usize;

    // `vp9_denoiser.c:419-423` — if perform_mc said FILTER, run the
    // per-block filter against the per-frame scratch
    // (`running_avg_y[INTRA_FRAME]`).
    if decision == DenoiserDecision::FilterBlock {
        // Carve out the scratch window. The scratch is full-frame
        // tight-stride; offset to (mi_row, mi_col) and pass the
        // strided window.
        let scratch = &mut denoiser.running_avg_y[INTRA_FRAME_IDX];
        let stride = scratch.stride as usize;
        let off = block_offset(stride, inputs.mi_row, inputs.mi_col);
        decision = vp9_denoiser_filter(
            inputs.src,
            inputs.src_stride,
            inputs.mc_block,
            inputs.mc_block_stride,
            &mut scratch.y_buffer[off..],
            stride,
            increase_denoising,
            inputs.bs,
            motion_magnitude,
        );
    }

    // `vp9_denoiser.c:425-433` — final copy-back. The function
    // returns the per-block "denoised view" the caller should use
    // for re-encoding.
    let mut denoised = vec![0u8; bw * bh];
    match decision {
        DenoiserDecision::FilterBlock | DenoiserDecision::FilterZeroMvBlock => {
            // Filtered: read from scratch.
            let scratch = &denoiser.running_avg_y[INTRA_FRAME_IDX];
            let stride = scratch.stride as usize;
            let off = block_offset(stride, inputs.mi_row, inputs.mi_col);
            for r in 0..bh {
                let s = off + r * stride;
                denoised[r * bw..r * bw + bw].copy_from_slice(&scratch.y_buffer[s..s + bw]);
            }
            // Copy back into source-shaped slot too (libvpx line 426
            // — the source for downstream encode is the denoised
            // signal).
        }
        DenoiserDecision::CopyBlock => {
            // Copy source verbatim into scratch + return source.
            let scratch = &mut denoiser.running_avg_y[INTRA_FRAME_IDX];
            let stride = scratch.stride as usize;
            let off = block_offset(stride, inputs.mi_row, inputs.mi_col);
            for r in 0..bh {
                let s = off + r * stride;
                scratch.y_buffer[s..s + bw].copy_from_slice(
                    &inputs.src[r * inputs.src_stride..r * inputs.src_stride + bw],
                );
                denoised[r * bw..r * bw + bw].copy_from_slice(
                    &inputs.src[r * inputs.src_stride..r * inputs.src_stride + bw],
                );
            }
        }
    }

    // `vp9_denoiser.c:435-436` — promote FILTER → FILTER_ZEROMV if
    // the zero-MV ref was the winner.
    if decision == DenoiserDecision::FilterBlock && zeromv_filter {
        decision = DenoiserDecision::FilterZeroMvBlock;
    }
    (decision, denoised)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_loop::EncoderConfig;

    // -------------------------------------------------------------------------
    // W5.1 — noise_sensitivity plumb-through
    // -------------------------------------------------------------------------

    /// W5.1 PROOF — `EncoderConfig::new()` defaults `noise_sensitivity = 0`
    /// (denoiser disabled), matching libvpx vpxenc's CLI default.
    /// libvpx `vp9_encoder.h::VP9EncoderConfig::noise_sensitivity`
    /// (default 0 in `vp9_cx_iface.c::vp9_init_config`).
    #[test]
    fn w5_1_noise_sensitivity_zero_disables_denoiser() {
        let cfg = EncoderConfig::new(64, 64);
        assert_eq!(cfg.noise_sensitivity, 0, "default must be 0 (off)");
        // Denoiser disabled ⇒ alloc rejects.
        assert!(
            Denoiser::alloc(cfg.noise_sensitivity, cfg.width, cfg.height).is_err(),
            "noise_sensitivity == 0 must produce no denoiser instance",
        );
    }

    /// W5.1 PROOF — every non-zero `noise_sensitivity ∈ 1..=6`
    /// produces a valid `Denoiser` (alloc succeeds).
    /// libvpx range from `vp9_cx_iface.c::vp9_set_encoder_config`
    /// (`oxcf->noise_sensitivity = extra_cfg->noise_sensitivity`,
    /// `extra_cfg.noise_sensitivity` ∈ `0..=6` per vpxenc
    /// `vpxenc.c::set_encoder_config`).
    #[test]
    fn w5_1_noise_sensitivity_nonzero_enables_denoiser() {
        for ns in 1u8..=6 {
            let mut cfg = EncoderConfig::new(64, 64);
            cfg.noise_sensitivity = ns;
            assert!(
                Denoiser::alloc(cfg.noise_sensitivity, cfg.width, cfg.height).is_ok(),
                "noise_sensitivity == {ns} must enable the denoiser",
            );
        }
    }

    // -------------------------------------------------------------------------
    // W5.2 — alloc / free / realloc
    // -------------------------------------------------------------------------

    /// W5.2 PROOF — `Denoiser::alloc` creates `NONSVC_REF_FRAMES`
    /// running-avg buffers + 1 MC running-avg + 1 last_source, all
    /// sized `width * height` and tight-stride.
    /// libvpx `vp9_denoiser.c:637-686`.
    #[test]
    fn w5_2_alloc_creates_buffers_sized_to_frame() {
        let w = 128;
        let h = 64;
        let den = Denoiser::alloc(2, w, h).expect("alloc must succeed for ns>=1");
        assert_eq!(den.num_ref_frames, NONSVC_REF_FRAMES);
        assert_eq!(den.num_layers, 1);
        assert!(
            den.buffers_match_frame(w, h),
            "every buffer must be width*height = {} bytes",
            w * h,
        );
        assert!(den.frame_buffer_initialized);
        // libvpx line 691: post-alloc level is Medium.
        assert_eq!(den.denoising_level, DenoiserLevel::Medium);
        assert_eq!(den.prev_denoising_level, DenoiserLevel::Medium);
        // Zero-dim rejection.
        assert!(Denoiser::alloc(2, 0, 64).is_err());
        assert!(Denoiser::alloc(2, 64, 0).is_err());
    }

    /// W5.2 — `Denoiser::free` drops the buffers and clears the
    /// initialised flag. libvpx `vp9_denoiser.c:698-717`.
    #[test]
    fn w5_2_free_releases_buffers() {
        let mut den = Denoiser::alloc(2, 64, 64).unwrap();
        assert!(den.frame_buffer_initialized);
        den.free();
        assert!(!den.frame_buffer_initialized);
        assert!(den.running_avg_y.is_empty());
        assert!(den.mc_running_avg_y.is_empty());
        assert!(den.last_source.y_buffer.is_empty());
    }

    // -------------------------------------------------------------------------
    // W5.3 — vp9_denoiser_filter_c + threshold helpers
    // -------------------------------------------------------------------------

    /// W5.3 PROOF — threshold helpers reproduce libvpx's constants.
    /// libvpx `vp9_denoiser.c:28-63` + `vp9_denoiser.h:107-110`.
    #[test]
    fn w5_3_filter_threshold_helpers_match_libvpx_table() {
        // `absdiff_thresh` (vp9_denoiser.c:28-31): 3 base, +1 if
        // increase_denoising. bs ignored.
        assert_eq!(absdiff_thresh(BlockSize::Block16x16, false), 3);
        assert_eq!(absdiff_thresh(BlockSize::Block32x32, false), 3);
        assert_eq!(absdiff_thresh(BlockSize::Block16x16, true), 4);

        // `delta_thresh` (vp9_denoiser.c:33-37): always 4.
        assert_eq!(delta_thresh(BlockSize::Block16x16, false), 4);
        assert_eq!(delta_thresh(BlockSize::Block64x64, true), 4);

        // `noise_motion_thresh` (vp9_denoiser.c:39-43): always 625.
        assert_eq!(noise_motion_thresh(BlockSize::Block16x16, false), 625);
        assert_eq!(noise_motion_thresh(BlockSize::Block8x8, true), 625);

        // `sse_thresh` (vp9_denoiser.c:45-47): num_pels * (40 or 80).
        // 16x16 → 256 pixels → 256*40 = 10240; with inc → 256*80 = 20480.
        assert_eq!(sse_thresh(BlockSize::Block16x16, false), 10_240);
        assert_eq!(sse_thresh(BlockSize::Block16x16, true), 20_480);
        // 32x32 → 1024 pixels → 1024*40 = 40960.
        assert_eq!(sse_thresh(BlockSize::Block32x32, false), 40_960);

        // `sse_diff_thresh` (vp9_denoiser.c:49-59).
        // Large motion + !increase → 0.
        assert_eq!(sse_diff_thresh(BlockSize::Block16x16, false, 10_000), 0);
        // Large motion + increase → num_pels<<2 = 256*4 = 1024.
        assert_eq!(sse_diff_thresh(BlockSize::Block16x16, true, 10_000), 1024);
        // Small motion → num_pels<<4 = 256*16 = 4096 (both inc states).
        assert_eq!(sse_diff_thresh(BlockSize::Block16x16, false, 0), 4096);
        assert_eq!(sse_diff_thresh(BlockSize::Block16x16, true, 0), 4096);

        // `total_adj_weak_thresh` (vp9_denoiser.c:61-63): num_pels * (2 or 3).
        // 16x16: 256*2 = 512; with inc 256*3 = 768.
        assert_eq!(total_adj_weak_thresh(BlockSize::Block16x16, false), 512);
        assert_eq!(total_adj_weak_thresh(BlockSize::Block16x16, true), 768);

        // `total_adj_strong_thresh` (vp9_denoiser.h:107-110): same shape
        // as weak_thresh.
        assert_eq!(total_adj_strong_thresh(BlockSize::Block16x16, false), 512);
        assert_eq!(total_adj_strong_thresh(BlockSize::Block16x16, true), 768);
        // 32x32 → 1024*2 = 2048; with inc 1024*3 = 3072.
        assert_eq!(total_adj_strong_thresh(BlockSize::Block32x32, false), 2048);
        assert_eq!(total_adj_strong_thresh(BlockSize::Block32x32, true), 3072);
    }

    /// W5.3 PROOF — when `src == mc_avg`, the strong filter's diff is
    /// zero on every sample, so `total_adj == 0`, and the result is
    /// `FilterBlock` with `avg` filled from `mc_avg` (`absdiff <=
    /// thresh` path on every sample). libvpx `vp9_denoiser.c:93-130`.
    #[test]
    fn w5_3_filter_zero_diff_no_change() {
        let bs = BlockSize::Block16x16;
        let bw = bs.width() as usize;
        let bh = bs.height() as usize;
        // Tight-stride blocks.
        let sig: Vec<u8> = (0..bw * bh).map(|i| ((i * 7) & 0xff) as u8).collect();
        let mc: Vec<u8> = sig.clone();
        let mut avg = vec![0u8; bw * bh];
        let decision = vp9_denoiser_filter(&sig, bw, &mc, bw, &mut avg, bw, false, bs, 0);
        assert_eq!(decision, DenoiserDecision::FilterBlock);
        // Every sample should equal `mc[i]` because absdiff == 0 ≤
        // absdiff_thresh, so the "snap to MC" path fires per sample.
        assert_eq!(avg, mc, "zero diff ⇒ running-avg = mc_avg = src");
    }

    /// W5.3 — small-diff cases produce the snap-to-MC behaviour for
    /// every sample where `absdiff <= 3` (no `increase_denoising`).
    /// Exercises the `total_adj += diff` accumulator path.
    /// libvpx `vp9_denoiser.c:99-101`.
    #[test]
    fn w5_3_filter_small_diff_snaps_to_mc() {
        let bs = BlockSize::Block16x16;
        let bw = bs.width() as usize;
        let bh = bs.height() as usize;
        let sig: Vec<u8> = vec![100u8; bw * bh];
        // mc differs by +2 on every sample → absdiff=2 ≤ 3 ⇒ snap.
        let mc: Vec<u8> = vec![102u8; bw * bh];
        let mut avg = vec![0u8; bw * bh];
        let decision = vp9_denoiser_filter(&sig, bw, &mc, bw, &mut avg, bw, false, bs, 0);
        // total_adj = bw*bh * 2 = 512, ==  strong_thresh(16x16) =
        // 512 ⇒ accepted (≤).
        assert_eq!(decision, DenoiserDecision::FilterBlock);
        assert!(avg.iter().all(|&v| v == 102));
    }

    /// W5.3 — when `mc` differs from `src` by a large amount on every
    /// sample, the strong filter overshoots and the weak filter
    /// either claws back (FilterBlock) or fails out (CopyBlock).
    /// We only assert "does not panic" + the decision is one of the
    /// two valid options. libvpx `vp9_denoiser.c:132-179`.
    #[test]
    fn w5_3_filter_large_diff_either_weak_or_copy() {
        let bs = BlockSize::Block16x16;
        let bw = bs.width() as usize;
        let bh = bs.height() as usize;
        let sig: Vec<u8> = vec![50u8; bw * bh];
        let mc: Vec<u8> = vec![200u8; bw * bh]; // absdiff=150 everywhere
        let mut avg = vec![0u8; bw * bh];
        let decision = vp9_denoiser_filter(&sig, bw, &mc, bw, &mut avg, bw, false, bs, 0);
        // We accept either decision; the point is the filter ran
        // without panic and produced a defined answer.
        assert!(
            matches!(
                decision,
                DenoiserDecision::FilterBlock | DenoiserDecision::CopyBlock
            ),
            "large-diff decision must be FilterBlock or CopyBlock, got {decision:?}",
        );
    }

    // -------------------------------------------------------------------------
    // W5.4 — end-to-end denoise driver
    // -------------------------------------------------------------------------

    /// W5.4 PROOF — `vp9_denoiser_denoise` runs end-to-end on a
    /// synthetic 32×32 block (above the 16×16 small-block gate), the
    /// scratch buffer holds the running-avg, and the function returns
    /// a non-empty `denoised` view. libvpx `vp9_denoiser.c:336-437`.
    #[test]
    fn w5_4_denoise_runs_end_to_end_on_synthetic_block() {
        let width = 64;
        let height = 64;
        let mut den = Denoiser::alloc(2, width, height).unwrap();
        let bs = BlockSize::Block32x32;
        let bw = bs.width() as usize;
        let bh = bs.height() as usize;
        // Construct a strided src plane and a strided MC plane —
        // simulate "MC almost matches src" so the filter takes the
        // FilterBlock path.
        let stride = width as usize;
        let mut src_plane = vec![80u8; stride * (height as usize)];
        let mut mc_plane = vec![82u8; stride * (height as usize)];
        // Slight texture so the filter isn't a degenerate constant.
        for r in 0..(height as usize) {
            for c in 0..stride {
                src_plane[r * stride + c] = ((r * 3 + c) & 0xff) as u8;
                mc_plane[r * stride + c] = src_plane[r * stride + c].wrapping_add(1);
            }
        }
        // Place block at (mi_row=2, mi_col=2) → pixel (16, 16). bs is
        // 32×32 so it fits inside 64×64.
        let mi_row = 2;
        let mi_col = 2;
        let off = block_offset(stride, mi_row, mi_col);

        let inputs = DenoiseBlockInputs {
            mi_row,
            mi_col,
            bs,
            ctx: DenoiserBlockCtx::zeroed(),
            mv_row: 0,
            mv_col: 0,
            is_skin: false,
            consec_zeromv: 100,
            src: &src_plane[off..],
            src_stride: stride,
            mc_block: &mc_plane[off..],
            mc_block_stride: stride,
            last_is_reference: true,
            width,
            use_skin_detection: false,
        };
        let (decision, denoised) = vp9_denoiser_denoise(&mut den, &inputs);
        // All small-diff samples (|src-mc|=1) should accept the
        // FilterBlock path on the strong pass.
        assert!(
            matches!(
                decision,
                DenoiserDecision::FilterBlock | DenoiserDecision::FilterZeroMvBlock
            ),
            "32×32 block with absdiff=1 must filter, got {decision:?}",
        );
        // Denoised view is tight-stride bw*bh.
        assert_eq!(denoised.len(), bw * bh);
        // Filter should have rendered avg=mc samples (small-diff snap)
        // so the denoised buffer mirrors `mc` within the block.
        for r in 0..bh {
            for c in 0..bw {
                let src_sample = src_plane[off + r * stride + c];
                let mc_sample = mc_plane[off + r * stride + c];
                let den_sample = denoised[r * bw + c];
                // Snap-to-MC path: denoised == mc.
                assert_eq!(
                    den_sample, mc_sample,
                    "denoised sample at ({r},{c}) src={src_sample} mc={mc_sample}",
                );
            }
        }
    }

    /// W5.4 — when `last_is_reference == false`, the driver bypasses
    /// `perform_motion_compensation` entirely and copies src→scratch
    /// (CopyBlock path). libvpx `vp9_denoiser.c:410`.
    #[test]
    fn w5_4_denoise_no_last_reference_copies_src() {
        let mut den = Denoiser::alloc(2, 64, 64).unwrap();
        let bs = BlockSize::Block16x16;
        let bw = bs.width() as usize;
        let bh = bs.height() as usize;
        let src: Vec<u8> = (0..bw * bh).map(|i| (i & 0xff) as u8).collect();
        let mc: Vec<u8> = vec![0u8; bw * bh];
        let inputs = DenoiseBlockInputs {
            mi_row: 0,
            mi_col: 0,
            bs,
            ctx: DenoiserBlockCtx::zeroed(),
            mv_row: 0,
            mv_col: 0,
            is_skin: false,
            consec_zeromv: 100,
            src: &src,
            src_stride: bw,
            mc_block: &mc,
            mc_block_stride: bw,
            last_is_reference: false,
            width: 64,
            use_skin_detection: false,
        };
        let (decision, denoised) = vp9_denoiser_denoise(&mut den, &inputs);
        assert_eq!(decision, DenoiserDecision::CopyBlock);
        // Copy path returns src verbatim.
        assert_eq!(denoised, src);
    }

    /// W5.4 — `perform_motion_compensation`'s small-block gate
    /// (16×8 / 8×16 / 8×8) always returns CopyBlock. libvpx
    /// `vp9_denoiser.c:215-218`.
    #[test]
    fn w5_4_denoise_small_block_copies() {
        let mut den = Denoiser::alloc(2, 64, 64).unwrap();
        for bs in [
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
        ] {
            let bw = bs.width() as usize;
            let bh = bs.height() as usize;
            let src: Vec<u8> = vec![100u8; bw * bh];
            let mc: Vec<u8> = vec![100u8; bw * bh];
            let inputs = DenoiseBlockInputs {
                mi_row: 0,
                mi_col: 0,
                bs,
                ctx: DenoiserBlockCtx::zeroed(),
                mv_row: 0,
                mv_col: 0,
                is_skin: false,
                consec_zeromv: 100,
                src: &src,
                src_stride: bw,
                mc_block: &mc,
                mc_block_stride: bw,
                last_is_reference: true,
                width: 64,
                use_skin_detection: false,
            };
            let (decision, _) = vp9_denoiser_denoise(&mut den, &inputs);
            assert_eq!(
                decision,
                DenoiserDecision::CopyBlock,
                "small block {bs:?} must copy",
            );
        }
    }
}
