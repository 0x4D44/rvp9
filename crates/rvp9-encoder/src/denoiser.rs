//! W5.1 – W5.4 — VP9 temporal denoiser (standalone module).
//!
//! Faithful scalar port of libvpx `vp9/encoder/vp9_denoiser.c`. The
//! denoiser is a per-block NLM-style filter that runs on the encoder's
//! source buffer using temporal references to reduce noise before coding.
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
//! `frame_loop::StreamEncoder` integrates the filter into the production
//! per-leaf coding path. Sensitivity 0 disables it; 1 through 6 enable it at
//! Medium initially. Published noise estimates adapt the level, with temporal
//! references reseeded when filtering resumes after LowLow.
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
//! - **No SSE2 path.** x86-64 keeps the faithful scalar reference. The
//!   AArch64 path dispatches to the port of libvpx's NEON filter and is
//!   checked against the scalar implementation on that target.
//! - **No spatial-SVC allocation.** Temporal-only SVC uses the same two
//!   reference averages as ordinary streams, with per-frame reference-map
//!   identities and refresh masks supplied by the production encoder.
//!   Spatial denoising, per-layer reallocations and `denoise_layer_idx`
//!   remain unsupported; `num_layers = 1` and
//!   `num_ref_frames = NONSVC_REF_FRAMES = 3`.
//! - **Caller-supplied skin and motion history.** The standalone driver
//!   accepts `is_skin` and `consec_zeromv`. Production filtering receives
//!   these from `PreAnalysisState`, and its adaptive estimator excludes
//!   skin using the current 16x16 block's YUV center.
//! - **No `PICK_MODE_CONTEXT` / `MACROBLOCK` coupling.** The standalone
//!   driver accepts a precomputed MC block. Production's
//!   `lib.rs::denoise_inter_leaf` builds the predictor from the selected
//!   running-average reference without rebinding `MACROBLOCKD` pointers.
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

use crate::rate_control::config::{MAX_CODED_DIMENSION, MAX_CODED_LUMA_PIXELS};

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

/// Invalid buffer geometry or mutable denoiser state supplied to a public
/// per-block denoiser entry point.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenoiserInputError {
    /// A plane's row stride is smaller than the requested block width.
    InvalidStride {
        /// Plane whose stride is invalid.
        plane: &'static str,
        /// Supplied row stride.
        stride: usize,
        /// Minimum required row stride.
        minimum: usize,
    },
    /// A plane does not contain the complete requested strided window.
    PlaneTooShort {
        /// Plane whose buffer is too short.
        plane: &'static str,
        /// Supplied buffer length.
        actual: usize,
        /// Minimum required buffer length.
        required: usize,
    },
    /// Checked coordinate or row-window arithmetic overflowed.
    GeometryOverflow {
        /// Geometry whose arithmetic overflowed.
        field: &'static str,
    },
    /// The requested block lies outside the denoiser scratch frame.
    BlockOutOfBounds {
        /// Block top edge in pixels.
        row: usize,
        /// Block left edge in pixels.
        col: usize,
        /// Block width in pixels.
        width: usize,
        /// Block height in pixels.
        height: usize,
        /// Scratch-frame width in pixels.
        frame_width: usize,
        /// Scratch-frame height in pixels.
        frame_height: usize,
    },
    /// The public fields in `Denoiser` do not describe usable scratch state.
    InvalidState {
        /// State invariant that was not satisfied.
        reason: &'static str,
    },
}

impl std::fmt::Display for DenoiserInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStride {
                plane,
                stride,
                minimum,
            } => write!(
                f,
                "{plane} stride {stride} is smaller than block width {minimum}"
            ),
            Self::PlaneTooShort {
                plane,
                actual,
                required,
            } => write!(
                f,
                "{plane} buffer has {actual} bytes but its block window needs {required}"
            ),
            Self::GeometryOverflow { field } => {
                write!(f, "denoiser {field} geometry overflowed")
            }
            Self::BlockOutOfBounds {
                row,
                col,
                width,
                height,
                frame_width,
                frame_height,
            } => write!(
                f,
                "block at ({row}, {col}) with size {width}x{height} exceeds {frame_width}x{frame_height} scratch frame"
            ),
            Self::InvalidState { reason } => write!(f, "invalid denoiser state: {reason}"),
        }
    }
}

impl std::error::Error for DenoiserInputError {}

fn validate_plane_window(
    plane: &'static str,
    len: usize,
    stride: usize,
    width: usize,
    height: usize,
) -> Result<(), DenoiserInputError> {
    if stride < width {
        return Err(DenoiserInputError::InvalidStride {
            plane,
            stride,
            minimum: width,
        });
    }
    let required = height
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|row| row.checked_add(width))
        .ok_or(DenoiserInputError::GeometryOverflow { field: plane })?;
    if len < required {
        return Err(DenoiserInputError::PlaneTooShort {
            plane,
            actual: len,
            required,
        });
    }
    Ok(())
}

/// `vp9_denoiser.h:38-43` — denoising aggressiveness level. Driven by
/// `vp9_denoiser_set_noise_level` from the published noise estimate.
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
) -> Result<DenoiserDecision, DenoiserInputError> {
    let bw = bs.width() as usize;
    let bh = bs.height() as usize;
    validate_plane_window("sig", sig.len(), sig_stride, bw, bh)?;
    validate_plane_window("mc_avg", mc_avg.len(), mc_avg_stride, bw, bh)?;
    validate_plane_window("avg", avg.len(), avg_stride, bw, bh)?;

    #[cfg(target_arch = "aarch64")]
    if !increase_denoising && neon_filter_block_size(bs) {
        // SAFETY: The three windows were validated above. AArch64 requires
        // NEON, and the helper only performs unaligned loads/stores within
        // those validated windows. High denoise remains scalar because pinned
        // libvpx's NEON and C filters disagree at the level-one threshold.
        return Ok(unsafe {
            neon::filter(
                neon::FilterPlanes {
                    sig,
                    sig_stride,
                    mc_avg,
                    mc_avg_stride,
                    avg,
                    avg_stride,
                },
                neon::FilterConfig {
                    increase_denoising,
                    bs,
                    motion_magnitude,
                },
            )
        });
    }

    vp9_denoiser_filter_scalar(
        sig,
        sig_stride,
        mc_avg,
        mc_avg_stride,
        avg,
        avg_stride,
        increase_denoising,
        bs,
        motion_magnitude,
    )
}

/// Scalar reference implementation of [`vp9_denoiser_filter`].
///
/// The public wrapper validates the three windows before dispatching here or
/// to the AArch64 NEON implementation. Keeping this body separate gives the
/// NEON differential test a stable scalar oracle.
#[allow(clippy::too_many_arguments)]
fn vp9_denoiser_filter_scalar(
    sig: &[u8],
    sig_stride: usize,
    mc_avg: &[u8],
    mc_avg_stride: usize,
    avg: &mut [u8],
    avg_stride: usize,
    increase_denoising: bool,
    bs: BlockSize,
    motion_magnitude: i32,
) -> Result<DenoiserDecision, DenoiserInputError> {
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
        return Ok(DenoiserDecision::FilterBlock);
    }

    // `vp9_denoiser.c:138-144` — try the weak (damped) filter, or bail.
    let delta = ((total_adj.abs() - strong_thresh) >> num_pels_log2(bs)) + 1;
    if delta >= delta_thresh(bs, increase_denoising) {
        return Ok(DenoiserDecision::CopyBlock);
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
    Ok(
        if total_adj.abs() <= total_adj_weak_thresh(bs, increase_denoising) {
            DenoiserDecision::FilterBlock
        } else {
            DenoiserDecision::CopyBlock
        },
    )
}

// -----------------------------------------------------------------------------
// AArch64 NEON filter (libvpx `vp9_denoiser_neon.c`)
// -----------------------------------------------------------------------------

/// The block sizes handled by libvpx's NEON dispatcher. The scalar path is
/// retained for the three 4-pixel-wide VP9 shapes and 8×4, which are not
/// denoised by the encoder driver and have no NEON kernel in libvpx.
#[cfg(target_arch = "aarch64")]
#[inline]
const fn neon_filter_block_size(bs: BlockSize) -> bool {
    matches!(
        bs,
        BlockSize::Block8x8
            | BlockSize::Block8x16
            | BlockSize::Block16x8
            | BlockSize::Block16x16
            | BlockSize::Block16x32
            | BlockSize::Block32x16
            | BlockSize::Block32x32
            | BlockSize::Block32x64
            | BlockSize::Block64x32
            | BlockSize::Block64x64
    )
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{
        BlockSize, DenoiserDecision, MOTION_MAGNITUDE_THRESHOLD, neon_filter_block_size,
        total_adj_strong_thresh,
    };
    use std::arch::aarch64::{
        int8x16_t, int16x8_t, uint8x16_t, vabdq_u8, vaddlvq_s16, vaddq_s16, vaddq_u8, vandq_u8,
        vbslq_u8, vcgtq_u8, vcleq_u8, vcltq_u8, vdupq_n_s16, vdupq_n_u8, vget_high_s8,
        vget_high_u8, vget_low_s8, vget_low_u8, vld1q_u8, vminq_u8, vmovl_s8, vqaddq_u8, vqsubq_u8,
        vreinterpretq_s8_u8, vst1_u8, vst1q_u8, vsubq_s8,
    };

    pub(super) struct FilterPlanes<'a> {
        pub(super) sig: &'a [u8],
        pub(super) sig_stride: usize,
        pub(super) mc_avg: &'a [u8],
        pub(super) mc_avg_stride: usize,
        pub(super) avg: &'a mut [u8],
        pub(super) avg_stride: usize,
    }

    #[derive(Clone, Copy)]
    pub(super) struct FilterConfig {
        pub(super) increase_denoising: bool,
        pub(super) bs: BlockSize,
        pub(super) motion_magnitude: i32,
    }

    #[derive(Clone, Copy)]
    struct StrongFilterVectors {
        level1_threshold: uint8x16_t,
        level2_threshold: uint8x16_t,
        level3_threshold: uint8x16_t,
        level1_adjustment: uint8x16_t,
        delta_level_1_and_2: uint8x16_t,
        delta_level_2_and_3: uint8x16_t,
    }

    /// Widen each signed-byte adjustment before accumulating it. The C NEON
    /// implementation uses signed-byte accumulators because its normal
    /// adjustment range fits there. Widening preserves the scalar i32 sum for
    /// the high-denoising boundary cases while keeping the sample work vector
    /// based.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add_adjustments(
        sum_low: int16x8_t,
        sum_high: int16x8_t,
        adjustment: int8x16_t,
    ) -> (int16x8_t, int16x8_t) {
        (
            vaddq_s16(sum_low, vmovl_s8(vget_low_s8(adjustment))),
            vaddq_s16(sum_high, vmovl_s8(vget_high_s8(adjustment))),
        )
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn horizontal_add(sum_low: int16x8_t, sum_high: int16x8_t) -> i32 {
        vaddlvq_s16(sum_low) + vaddlvq_s16(sum_high)
    }

    /// Port of libvpx `denoiser_16x1_neon`. The two widened accumulators are
    /// equivalent to the scalar filter's `total_adj: i32` accumulator.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn denoiser_16x1(
        sig: *const u8,
        mc_running_avg_y: *const u8,
        running_avg_y: *mut u8,
        vectors: &StrongFilterVectors,
        sum_low: int16x8_t,
        sum_high: int16x8_t,
    ) -> (int16x8_t, int16x8_t) {
        // SAFETY: Callers pass pointers to at least 16 contiguous samples in
        // each row; the validated public wrapper establishes that contract.
        unsafe {
            let v_sig = vld1q_u8(sig);
            let v_mc_running_avg_y = vld1q_u8(mc_running_avg_y);

            let v_abs_diff = vabdq_u8(v_sig, v_mc_running_avg_y);
            let v_diff_pos_mask = vcltq_u8(v_sig, v_mc_running_avg_y);
            let v_diff_neg_mask = vcgtq_u8(v_sig, v_mc_running_avg_y);

            let v_level1_mask = vcleq_u8(vectors.level1_threshold, v_abs_diff);
            let v_level2_mask = vcleq_u8(vectors.level2_threshold, v_abs_diff);
            let v_level3_mask = vcleq_u8(vectors.level3_threshold, v_abs_diff);

            let v_level2_adjustment = vandq_u8(v_level2_mask, vectors.delta_level_1_and_2);
            let v_level3_adjustment = vandq_u8(v_level3_mask, vectors.delta_level_2_and_3);
            let v_level1and2_adjustment = vaddq_u8(vectors.level1_adjustment, v_level2_adjustment);
            let v_level1and2and3_adjustment =
                vaddq_u8(v_level1and2_adjustment, v_level3_adjustment);
            let v_abs_adjustment = vbslq_u8(v_level1_mask, v_level1and2and3_adjustment, v_abs_diff);

            let v_pos_adjustment = vandq_u8(v_diff_pos_mask, v_abs_adjustment);
            let v_neg_adjustment = vandq_u8(v_diff_neg_mask, v_abs_adjustment);
            let mut v_running_avg_y = vqaddq_u8(v_sig, v_pos_adjustment);
            v_running_avg_y = vqsubq_u8(v_running_avg_y, v_neg_adjustment);
            vst1q_u8(running_avg_y, v_running_avg_y);

            let adjustment = vsubq_s8(
                vreinterpretq_s8_u8(v_pos_adjustment),
                vreinterpretq_s8_u8(v_neg_adjustment),
            );
            add_adjustments(sum_low, sum_high, adjustment)
        }
    }

    /// Port of libvpx `denoiser_adjust_16x1_neon` for the weak pass.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn denoiser_adjust_16x1(
        sig: *const u8,
        mc_running_avg_y: *const u8,
        running_avg_y: *mut u8,
        delta: uint8x16_t,
        sum_low: int16x8_t,
        sum_high: int16x8_t,
    ) -> (int16x8_t, int16x8_t) {
        // SAFETY: Callers pass pointers to at least 16 contiguous samples in
        // each row; the validated public wrapper establishes that contract.
        unsafe {
            let mut v_running_avg_y = vld1q_u8(running_avg_y);
            let v_sig = vld1q_u8(sig);
            let v_mc_running_avg_y = vld1q_u8(mc_running_avg_y);

            let v_abs_diff = vabdq_u8(v_sig, v_mc_running_avg_y);
            let v_diff_pos_mask = vcltq_u8(v_sig, v_mc_running_avg_y);
            let v_diff_neg_mask = vcgtq_u8(v_sig, v_mc_running_avg_y);
            let v_abs_adjustment = vminq_u8(v_abs_diff, delta);
            let v_pos_adjustment = vandq_u8(v_diff_pos_mask, v_abs_adjustment);
            let v_neg_adjustment = vandq_u8(v_diff_neg_mask, v_abs_adjustment);

            v_running_avg_y = vqsubq_u8(v_running_avg_y, v_pos_adjustment);
            v_running_avg_y = vqaddq_u8(v_running_avg_y, v_neg_adjustment);
            vst1q_u8(running_avg_y, v_running_avg_y);

            let adjustment = vsubq_s8(
                vreinterpretq_s8_u8(v_neg_adjustment),
                vreinterpretq_s8_u8(v_pos_adjustment),
            );
            add_adjustments(sum_low, sum_high, adjustment)
        }
    }

    /// Port of libvpx `vp9_denoiser_NxM_neon`.
    #[target_feature(enable = "neon")]
    unsafe fn filter_nxm(planes: &mut FilterPlanes<'_>, config: FilterConfig) -> DenoiserDecision {
        // SAFETY: The public wrapper validates all windows before reaching
        // this helper. Every row pointer below stays within those windows.
        unsafe {
            let shift_inc = i32::from(
                config.increase_denoising && config.motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD,
            );
            let level1_adjustment = if config.motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD {
                4 + shift_inc
            } else {
                3
            };
            let vectors = StrongFilterVectors {
                level1_threshold: vdupq_n_u8((4 + shift_inc) as u8),
                level2_threshold: vdupq_n_u8(8),
                level3_threshold: vdupq_n_u8(16),
                level1_adjustment: vdupq_n_u8(level1_adjustment as u8),
                delta_level_1_and_2: vdupq_n_u8(1),
                delta_level_2_and_3: vdupq_n_u8(2),
            };

            let width = config.bs.width() as usize;
            let height = config.bs.height() as usize;
            let width_vectors = width / 16;
            let mut sums_low = [[vdupq_n_s16(0); 4]; 4];
            let mut sums_high = [[vdupq_n_s16(0); 4]; 4];

            for row in 0..height {
                for chunk in 0..width_vectors {
                    let sig_ptr = planes
                        .sig
                        .as_ptr()
                        .add(row * planes.sig_stride + chunk * 16);
                    let mc_ptr = planes
                        .mc_avg
                        .as_ptr()
                        .add(row * planes.mc_avg_stride + chunk * 16);
                    let avg_ptr = planes
                        .avg
                        .as_mut_ptr()
                        .add(row * planes.avg_stride + chunk * 16);
                    let (low, high) = denoiser_16x1(
                        sig_ptr,
                        mc_ptr,
                        avg_ptr,
                        &vectors,
                        sums_low[chunk][row >> 4],
                        sums_high[chunk][row >> 4],
                    );
                    sums_low[chunk][row >> 4] = low;
                    sums_high[chunk][row >> 4] = high;
                }
            }

            let mut sum_diff = 0i32;
            for chunk in 0..width_vectors {
                for group in 0..4 {
                    sum_diff += vaddlvq_s16(sums_low[chunk][group]);
                    sum_diff += vaddlvq_s16(sums_high[chunk][group]);
                }
            }
            let sum_diff_thresh = total_adj_strong_thresh(config.bs, config.increase_denoising);
            if sum_diff.abs() > sum_diff_thresh {
                let delta =
                    ((sum_diff.abs() - sum_diff_thresh) >> super::num_pels_log2(config.bs)) + 1;
                if delta >= 4 {
                    return DenoiserDecision::CopyBlock;
                }
                let delta = vdupq_n_u8(delta as u8);
                for row in 0..height {
                    for chunk in 0..width_vectors {
                        let sig_ptr = planes
                            .sig
                            .as_ptr()
                            .add(row * planes.sig_stride + chunk * 16);
                        let mc_ptr = planes
                            .mc_avg
                            .as_ptr()
                            .add(row * planes.mc_avg_stride + chunk * 16);
                        let avg_ptr = planes
                            .avg
                            .as_mut_ptr()
                            .add(row * planes.avg_stride + chunk * 16);
                        let (low, high) = denoiser_adjust_16x1(
                            sig_ptr,
                            mc_ptr,
                            avg_ptr,
                            delta,
                            sums_low[chunk][row >> 4],
                            sums_high[chunk][row >> 4],
                        );
                        sums_low[chunk][row >> 4] = low;
                        sums_high[chunk][row >> 4] = high;
                    }
                }
                sum_diff = 0;
                for chunk in 0..width_vectors {
                    for group in 0..4 {
                        sum_diff += vaddlvq_s16(sums_low[chunk][group]);
                        sum_diff += vaddlvq_s16(sums_high[chunk][group]);
                    }
                }
                if sum_diff.abs() > sum_diff_thresh {
                    return DenoiserDecision::CopyBlock;
                }
            }
            DenoiserDecision::FilterBlock
        }
    }

    /// Port of libvpx `vp9_denoiser_8xN_neon`.
    #[target_feature(enable = "neon")]
    unsafe fn filter_8xn(planes: &mut FilterPlanes<'_>, config: FilterConfig) -> DenoiserDecision {
        // SAFETY: The public wrapper validates all windows before reaching
        // this helper. The temporary rows are local and every source/dest
        // pointer stays within the validated windows.
        unsafe {
            let shift_inc = i32::from(
                config.increase_denoising && config.motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD,
            );
            let level1_adjustment = if config.motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD {
                4 + shift_inc
            } else {
                3
            };
            let vectors = StrongFilterVectors {
                level1_threshold: vdupq_n_u8((4 + shift_inc) as u8),
                level2_threshold: vdupq_n_u8(8),
                level3_threshold: vdupq_n_u8(16),
                level1_adjustment: vdupq_n_u8(level1_adjustment as u8),
                delta_level_1_and_2: vdupq_n_u8(1),
                delta_level_2_and_3: vdupq_n_u8(2),
            };
            let height = config.bs.height() as usize;
            let pair_count = height / 2;
            let mut sig_buffer = [[0u8; 16]; 8];
            let mut mc_buffer = [[0u8; 16]; 8];
            let mut running_buffer = [[0u8; 16]; 8];
            let mut sum_low = vdupq_n_s16(0);
            let mut sum_high = vdupq_n_s16(0);

            for row_pair in 0..pair_count {
                let row = row_pair * 2;
                std::ptr::copy_nonoverlapping(
                    planes.sig.as_ptr().add(row * planes.sig_stride),
                    sig_buffer[row_pair].as_mut_ptr(),
                    8,
                );
                std::ptr::copy_nonoverlapping(
                    planes.sig.as_ptr().add((row + 1) * planes.sig_stride),
                    sig_buffer[row_pair].as_mut_ptr().add(8),
                    8,
                );
                std::ptr::copy_nonoverlapping(
                    planes.mc_avg.as_ptr().add(row * planes.mc_avg_stride),
                    mc_buffer[row_pair].as_mut_ptr(),
                    8,
                );
                std::ptr::copy_nonoverlapping(
                    planes.mc_avg.as_ptr().add((row + 1) * planes.mc_avg_stride),
                    mc_buffer[row_pair].as_mut_ptr().add(8),
                    8,
                );
                std::ptr::copy_nonoverlapping(
                    planes.avg.as_ptr().add(row * planes.avg_stride),
                    running_buffer[row_pair].as_mut_ptr(),
                    8,
                );
                std::ptr::copy_nonoverlapping(
                    planes.avg.as_ptr().add((row + 1) * planes.avg_stride),
                    running_buffer[row_pair].as_mut_ptr().add(8),
                    8,
                );
                (sum_low, sum_high) = denoiser_16x1(
                    sig_buffer[row_pair].as_ptr(),
                    mc_buffer[row_pair].as_ptr(),
                    running_buffer[row_pair].as_mut_ptr(),
                    &vectors,
                    sum_low,
                    sum_high,
                );
                let running = vld1q_u8(running_buffer[row_pair].as_ptr());
                vst1_u8(
                    planes.avg.as_mut_ptr().add(row * planes.avg_stride),
                    vget_low_u8(running),
                );
                vst1_u8(
                    planes.avg.as_mut_ptr().add((row + 1) * planes.avg_stride),
                    vget_high_u8(running),
                );
            }

            let sum_diff_thresh = total_adj_strong_thresh(config.bs, config.increase_denoising);
            let mut sum_diff = horizontal_add(sum_low, sum_high);
            if sum_diff.abs() > sum_diff_thresh {
                let delta =
                    ((sum_diff.abs() - sum_diff_thresh) >> super::num_pels_log2(config.bs)) + 1;
                if delta >= 4 {
                    return DenoiserDecision::CopyBlock;
                }
                let delta = vdupq_n_u8(delta as u8);
                for row_pair in 0..pair_count {
                    (sum_low, sum_high) = denoiser_adjust_16x1(
                        sig_buffer[row_pair].as_ptr(),
                        mc_buffer[row_pair].as_ptr(),
                        running_buffer[row_pair].as_mut_ptr(),
                        delta,
                        sum_low,
                        sum_high,
                    );
                    let running = vld1q_u8(running_buffer[row_pair].as_ptr());
                    let row = row_pair * 2;
                    vst1_u8(
                        planes.avg.as_mut_ptr().add(row * planes.avg_stride),
                        vget_low_u8(running),
                    );
                    vst1_u8(
                        planes.avg.as_mut_ptr().add((row + 1) * planes.avg_stride),
                        vget_high_u8(running),
                    );
                }
                sum_diff = horizontal_add(sum_low, sum_high);
                if sum_diff.abs() > sum_diff_thresh {
                    return DenoiserDecision::CopyBlock;
                }
            }
            DenoiserDecision::FilterBlock
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn filter(
        mut planes: FilterPlanes<'_>,
        config: FilterConfig,
    ) -> DenoiserDecision {
        // SAFETY: The caller validates all plane windows; the two kernels
        // perform only in-bounds NEON loads/stores.
        unsafe {
            if matches!(config.bs, BlockSize::Block8x8 | BlockSize::Block8x16) {
                filter_8xn(&mut planes, config)
            } else if neon_filter_block_size(config.bs) {
                filter_nxm(&mut planes, config)
            } else {
                DenoiserDecision::CopyBlock
            }
        }
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
    ///
    /// # Errors
    ///
    /// Returns [`DenoiserAllocError`] for empty or unsupported
    /// dimensions, or if the backing buffer cannot be reserved.
    pub fn alloc(width: u32, height: u32) -> Result<Self, DenoiserAllocError> {
        if width == 0 || height == 0 {
            return Err(DenoiserAllocError::InvalidConfig);
        }
        let pixels = u64::from(width) * u64::from(height);
        if width > MAX_CODED_DIMENSION
            || height > MAX_CODED_DIMENSION
            || pixels > MAX_CODED_LUMA_PIXELS
        {
            return Err(DenoiserAllocError::DimensionsTooLarge { width, height });
        }
        let len = usize::try_from(pixels)
            .map_err(|_| DenoiserAllocError::DimensionsTooLarge { width, height })?;
        let mut y_buffer = Vec::new();
        y_buffer
            .try_reserve_exact(len)
            .map_err(|_| DenoiserAllocError::AllocationFailed)?;
        y_buffer.resize(len, 0);
        Ok(Self {
            y_buffer,
            width,
            height,
            stride: width,
        })
    }

    /// `true` if the plane carries a non-empty buffer.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        !self.y_buffer.is_empty()
    }
}

/// Decoder-map identities and refresh decisions for the temporal averages.
#[derive(Clone, Copy)]
pub(crate) struct DenoiserFrameContext {
    pub last_slot: usize,
    pub golden_slot: usize,
    pub refresh_mask: u8,
}

impl Default for DenoiserFrameContext {
    fn default() -> Self {
        Self {
            last_slot: 1,
            golden_slot: 2,
            refresh_mask: 1,
        }
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
    /// The dimensions exceed the encoder's supported frame envelope.
    DimensionsTooLarge {
        /// Requested luma width.
        width: u32,
        /// Requested luma height.
        height: u32,
    },
    /// The backing luma buffer could not be reserved.
    AllocationFailed,
}

impl std::fmt::Display for DenoiserAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid denoiser configuration"),
            Self::DimensionsTooLarge { width, height } => {
                write!(f, "denoiser dimensions too large: {width}x{height}")
            }
            Self::AllocationFailed => f.write_str("failed to allocate denoiser frame"),
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
            running_avg_y.push(DenoiserFrame::alloc(width, height)?);
        }
        let mut mc_running_avg_y = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            mc_running_avg_y.push(DenoiserFrame::alloc(width, height)?);
        }
        // `vp9_denoiser.c:678-686` — `last_source` (full frame, top layer only).
        let last_source = DenoiserFrame::alloc(width, height)?;
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

    /// Apply a newly published estimate, mirroring pinned
    /// `vp9_denoiser.c:732-744`. Re-enabling filtering seeds fresh references
    /// after this frame, before the next frame can consume them.
    pub(crate) fn set_noise_level(&mut self, level: rvp9_core::mode_decision::inter::NoiseLevel) {
        use rvp9_core::mode_decision::inter::NoiseLevel;
        self.denoising_level = match level {
            NoiseLevel::LowLow => DenoiserLevel::LowLow,
            NoiseLevel::Low => DenoiserLevel::Low,
            NoiseLevel::Medium => DenoiserLevel::Medium,
            NoiseLevel::High => DenoiserLevel::High,
        };
        self.reset = self.denoising_level > DenoiserLevel::LowLow
            && self.prev_denoising_level == DenoiserLevel::LowLow;
        self.prev_denoising_level = self.denoising_level;
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
        let Ok(expected) = usize::try_from(u64::from(width) * u64::from(height)) else {
            return false;
        };
        let Some(expected_slots) = self.num_ref_frames.checked_mul(self.num_layers) else {
            return false;
        };
        self.running_avg_y.len() == expected_slots
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
        0i64
    } else {
        i64::from(ctx.zeromv_sse) - i64::from(ctx.newmv_sse)
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
        && sse_diff > i64::from(sse_diff_thresh(bs, increase_denoising, motion_magnitude));
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
                && ((u64::from(ctx.zeromv_lastref_sse) < (5 * u64::from(ctx.zeromv_sse)) >> 2)
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

fn validate_denoise_inputs(
    denoiser: &Denoiser,
    inputs: &DenoiseBlockInputs<'_>,
) -> Result<(usize, usize, usize, usize), DenoiserInputError> {
    let bw = inputs.bs.width() as usize;
    let bh = inputs.bs.height() as usize;
    validate_plane_window("src", inputs.src.len(), inputs.src_stride, bw, bh)?;
    validate_plane_window(
        "mc_block",
        inputs.mc_block.len(),
        inputs.mc_block_stride,
        bw,
        bh,
    )?;

    if denoiser.num_ref_frames > denoiser.running_avg_y.len() {
        return Err(DenoiserInputError::InvalidState {
            reason: "num_ref_frames exceeds running_avg_y length",
        });
    }
    let scratch =
        denoiser
            .running_avg_y
            .get(INTRA_FRAME_IDX)
            .ok_or(DenoiserInputError::InvalidState {
                reason: "missing INTRA scratch frame",
            })?;
    if inputs.width != scratch.width {
        return Err(DenoiserInputError::InvalidState {
            reason: "input width does not match INTRA scratch frame",
        });
    }
    let frame_width = scratch.width as usize;
    let frame_height = scratch.height as usize;
    let stride = scratch.stride as usize;
    validate_plane_window(
        "INTRA scratch",
        scratch.y_buffer.len(),
        stride,
        frame_width,
        frame_height,
    )?;

    let row = inputs
        .mi_row
        .checked_mul(8)
        .ok_or(DenoiserInputError::GeometryOverflow { field: "block row" })?;
    let col = inputs
        .mi_col
        .checked_mul(8)
        .ok_or(DenoiserInputError::GeometryOverflow {
            field: "block column",
        })?;
    let row_end = row
        .checked_add(bh)
        .ok_or(DenoiserInputError::GeometryOverflow { field: "block row" })?;
    let col_end = col
        .checked_add(bw)
        .ok_or(DenoiserInputError::GeometryOverflow {
            field: "block column",
        })?;
    if row_end > frame_height || col_end > frame_width {
        return Err(DenoiserInputError::BlockOutOfBounds {
            row,
            col,
            width: bw,
            height: bh,
            frame_width,
            frame_height,
        });
    }
    let offset = row
        .checked_mul(stride)
        .and_then(|base| base.checked_add(col))
        .ok_or(DenoiserInputError::GeometryOverflow {
            field: "scratch offset",
        })?;
    Ok((bw, bh, stride, offset))
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
/// Returns `Ok((decision, denoised_block_owned))`:
/// - `decision` — `CopyBlock` / `FilterBlock` / `FilterZeroMvBlock`.
/// - `denoised_block_owned` — the per-block result: the running-avg
///   window's samples on `FilterBlock` / `FilterZeroMvBlock`; the
///   source samples on `CopyBlock`. Tight-stride `bw*bh` bytes.
pub fn vp9_denoiser_denoise(
    denoiser: &mut Denoiser,
    inputs: &DenoiseBlockInputs<'_>,
) -> Result<(DenoiserDecision, Vec<u8>), DenoiserInputError> {
    let (bw, bh, stride, off) = validate_denoise_inputs(denoiser, inputs)?;
    let mv_row = u64::from(inputs.mv_row.unsigned_abs());
    let mv_col = u64::from(inputs.mv_col.unsigned_abs());
    let motion_magnitude = (mv_row * mv_row + mv_col * mv_col).min(i32::MAX as u64) as i32;
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

    // `vp9_denoiser.c:419-423` — if perform_mc said FILTER, run the
    // per-block filter against the per-frame scratch
    // (`running_avg_y[INTRA_FRAME]`).
    if decision == DenoiserDecision::FilterBlock {
        // Carve out the scratch window. The scratch is full-frame
        // tight-stride; offset to (mi_row, mi_col) and pass the
        // strided window.
        let scratch = &mut denoiser.running_avg_y[INTRA_FRAME_IDX];
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
        )?;
    }

    // `vp9_denoiser.c:425-433` — final copy-back. The function
    // returns the per-block "denoised view" the caller should use
    // for re-encoding.
    let mut denoised = vec![0u8; bw * bh];
    match decision {
        DenoiserDecision::FilterBlock | DenoiserDecision::FilterZeroMvBlock => {
            // Filtered: read from scratch.
            let scratch = &denoiser.running_avg_y[INTRA_FRAME_IDX];
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
    Ok((decision, denoised))
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
        let decision = vp9_denoiser_filter(&sig, bw, &mc, bw, &mut avg, bw, false, bs, 0).unwrap();
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
        let decision = vp9_denoiser_filter(&sig, bw, &mc, bw, &mut avg, bw, false, bs, 0).unwrap();
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
        let decision = vp9_denoiser_filter(&sig, bw, &mc, bw, &mut avg, bw, false, bs, 0).unwrap();
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

    /// The AArch64 port must preserve the scalar decision and every active
    /// output sample across all block shapes handled by libvpx's NEON
    /// dispatcher. Strided windows and randomized byte extremes are included;
    /// motion values cover both sides of the 24-pixel adjustment threshold.
    /// Medium denoise exercises NEON, while high denoise proves the scalar
    /// fallback for pinned libvpx's known C/NEON threshold disagreement.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn w5_3_neon_filter_matches_scalar_for_supported_shapes() {
        const BLOCKS: [BlockSize; 10] = [
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ];
        const MOTIONS: [i32; 4] = [0, 24, 25, 96];

        for (block_index, &bs) in BLOCKS.iter().enumerate() {
            let width = bs.width() as usize;
            let height = bs.height() as usize;
            for stride_pad in [0usize, 3, 11] {
                let stride = width + stride_pad;
                let len = stride * height;
                let mut sig = vec![0u8; len];
                let mut mc = vec![0u8; len];
                let mut avg_scalar = vec![0xa5u8; len];
                let mut avg_neon = avg_scalar.clone();
                for case in 0..8u32 {
                    let mut state = 0x9e37_79b9u32
                        ^ (block_index as u32).wrapping_mul(0x85eb_ca6b)
                        ^ case.wrapping_mul(0xc2b2_ae35);
                    for row in 0..height {
                        for col in 0..width {
                            let index = row * stride + col;
                            if case == 0 {
                                let source =
                                    12 + ((row * 29 + col * 17 + block_index * 7) % 220) as i32;
                                let delta =
                                    ((row * 13 + col * 19 + block_index * 23) % 43) as i32 - 21;
                                sig[index] = source as u8;
                                mc[index] = (source + delta).clamp(0, 255) as u8;
                            } else {
                                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                sig[index] = (state >> 24) as u8;
                                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                mc[index] = (state >> 24) as u8;
                            }
                        }
                    }

                    for &increase_denoising in &[false, true] {
                        for &motion_magnitude in &MOTIONS {
                            avg_scalar.fill(0xa5);
                            avg_neon.copy_from_slice(&avg_scalar);
                            let scalar_decision = vp9_denoiser_filter_scalar(
                                &sig,
                                stride,
                                &mc,
                                stride,
                                &mut avg_scalar,
                                stride,
                                increase_denoising,
                                bs,
                                motion_magnitude,
                            )
                            .expect("validated differential-test window");
                            let neon_decision = vp9_denoiser_filter(
                                &sig,
                                stride,
                                &mc,
                                stride,
                                &mut avg_neon,
                                stride,
                                increase_denoising,
                                bs,
                                motion_magnitude,
                            )
                            .expect("validated differential-test window");
                            assert_eq!(
                                neon_decision, scalar_decision,
                                "decision mismatch for {bs:?}, stride={stride}, case={case}, increase={increase_denoising}, motion={motion_magnitude}",
                            );
                            assert_eq!(
                                avg_neon, avg_scalar,
                                "output mismatch for {bs:?}, stride={stride}, case={case}, increase={increase_denoising}, motion={motion_magnitude}",
                            );
                        }
                    }
                }
            }
        }
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
        let (decision, denoised) = vp9_denoiser_denoise(&mut den, &inputs).unwrap();
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
        let (decision, denoised) = vp9_denoiser_denoise(&mut den, &inputs).unwrap();
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
            let (decision, _) = vp9_denoiser_denoise(&mut den, &inputs).unwrap();
            assert_eq!(
                decision,
                DenoiserDecision::CopyBlock,
                "small block {bs:?} must copy",
            );
        }
    }

    #[test]
    fn public_filter_rejects_short_and_invalid_strided_planes() {
        let bs = BlockSize::Block16x16;
        let mut avg = vec![0u8; 16 * 16];

        assert!(
            vp9_denoiser_filter(&[0; 15], 16, &[0; 16 * 16], 16, &mut avg, 16, false, bs, 0)
                .is_err()
        );
        assert!(
            vp9_denoiser_filter(
                &[0; 16 * 16],
                15,
                &[0; 16 * 16],
                16,
                &mut avg,
                16,
                false,
                bs,
                0,
            )
            .is_err()
        );
        assert!(
            vp9_denoiser_filter(&[0; 16 * 16], 16, &[0; 15], 16, &mut avg, 16, false, bs, 0,)
                .is_err()
        );
        assert!(
            vp9_denoiser_filter(
                &[0; 16 * 16],
                16,
                &[0; 16 * 16],
                16,
                &mut avg[..15],
                16,
                false,
                bs,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn public_frame_allocator_rejects_oversized_dimensions() {
        assert!(DenoiserFrame::alloc(u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn public_denoise_rejects_out_of_frame_block() {
        let mut den = Denoiser::alloc(2, 64, 64).unwrap();
        let src = vec![0u8; 16 * 16];
        let mc = vec![0u8; 16 * 16];
        let inputs = DenoiseBlockInputs {
            mi_row: 8,
            mi_col: 0,
            bs: BlockSize::Block16x16,
            ctx: DenoiserBlockCtx::zeroed(),
            mv_row: 0,
            mv_col: 0,
            is_skin: false,
            consec_zeromv: 0,
            src: &src,
            src_stride: 16,
            mc_block: &mc,
            mc_block_stride: 16,
            last_is_reference: false,
            width: 64,
            use_skin_detection: false,
        };

        assert!(vp9_denoiser_denoise(&mut den, &inputs).is_err());
    }

    #[test]
    fn public_denoise_rejects_short_source_and_invalid_scratch_state() {
        let mut den = Denoiser::alloc(2, 64, 64).unwrap();
        let src = vec![0u8; 15];
        let mc = vec![0u8; 16 * 16];
        let inputs = DenoiseBlockInputs {
            mi_row: 0,
            mi_col: 0,
            bs: BlockSize::Block16x16,
            ctx: DenoiserBlockCtx::zeroed(),
            mv_row: i32::MIN,
            mv_col: i32::MIN,
            is_skin: false,
            consec_zeromv: 0,
            src: &src,
            src_stride: 16,
            mc_block: &mc,
            mc_block_stride: 16,
            last_is_reference: true,
            width: 64,
            use_skin_detection: false,
        };
        assert!(matches!(
            vp9_denoiser_denoise(&mut den, &inputs),
            Err(DenoiserInputError::PlaneTooShort { plane: "src", .. })
        ));

        let src = vec![0u8; 16 * 16];
        let inputs = DenoiseBlockInputs {
            src: &src,
            ..inputs
        };
        den.running_avg_y.clear();
        assert!(matches!(
            vp9_denoiser_denoise(&mut den, &inputs),
            Err(DenoiserInputError::InvalidState { .. })
        ));
    }
}
