//! W3.1 – W3.5 — ALTREF temporal filter construction pipeline.
//!
//! Faithful port of libvpx `vp9/encoder/vp9_temporal_filter.c`. The
//! temporal filter takes N consecutive source frames plus a designated
//! ARF-source frame and produces a non-local-means-filtered output
//! frame (the alt-reference frame, ALTREF) that the encoder will later
//! emit as a no-show super-frame.
//!
//! ## Scope
//!
//! This module ships the **construction pipeline as a standalone
//! subsystem**: feed in `N` frames + the ARF index + a strength /
//! frame-count budget, get back a filtered YUV plane triplet. The
//! StreamEncoder integration (bitstream emission of the filtered
//! frame, ARF lifetime in the reference list, GF-group scheduling)
//! is a Wave 3.6+ item.
//!
//! ## libvpx file:line citation block
//!
//! - `vp9_temporal_filter.c:48-186`   — `temporal_filter_predictors_mb_c`
//!   (32×32 luma + 2× 16×16 chroma sub-pel inter-pred build).
//! - `vp9_temporal_filter.c:195-213`  — `mod_index` NLM modifier table lookup.
//! - `vp9_temporal_filter.c:236-258`  — `get_filter_weight` blk_fw dispatch.
//! - `vp9_temporal_filter.c:260-399`  — `vp9_apply_temporal_filter_c` NLM core.
//! - `vp9_temporal_filter.c:552-643`  — `temporal_filter_find_matching_mb_c`
//!   (block-match MV search).
//! - `vp9_temporal_filter.c:645-996`  — `temporal_filter_iterate_row_c`
//!   (per-row driver, accumulator → output frame normalisation).
//! - `vp9_temporal_filter.c:1031-1102` — `adjust_arnr_filter` (frame
//!   count / strength budget).
//! - `vp9_temporal_filter.c:1104-1205` — `vp9_temporal_filter` (top-level
//!   dispatcher).
//! - `vp9_temporal_filter.h:18`       — `ARNR_FILT_QINDEX` constant.
//! - `vp9_temporal_filter.h:22-33`    — `BW`, `BH`, `BLK_PELS`, `TF_SHIFT`,
//!   `TF_ROUND`, `SUB_BW`, `SUB_BH`.
//! - `vp9_encoder.h:499-505`          — `ARNRFilterData` struct.
//!
//! ## Deviations from libvpx (documented)
//!
//! - **No HBD path.** libvpx ships `vp9_highbd_apply_temporal_filter_c`
//!   (lines 401-549) and `highbd_mod_index` (lines 215-233) for 10/12-bit
//!   sources. rvp9 is 8-bit-only across the board (the rvp9-core `Pixel`
//!   policy); HBD is out of v1 scope.
//! - **No SIMD.** This is a faithful scalar reference. SIMD parity
//!   lands in Wave 6.
//! - **No FILM-content variance suppression.** libvpx
//!   `vp9_temporal_filter.c:713-734` drops strength by 2 on
//!   ultra-flat blocks when `oxcf.content == VP9E_CONTENT_FILM`. rvp9
//!   doesn't expose a `content` knob yet, so we always take the
//!   default path. Adding the variance branch when the knob lands is
//!   a one-liner.
//! - **No multi-arf gating, no SVC scaling, no row-MT.** The libvpx
//!   `vp9_temporal_filter` top-level handles those at lines 1140-1193,
//!   1201-1204. rvp9 ships single-tile single-pass single-resolution
//!   construction.
//! - **No `find_fractional_mv_step` (sub-pel ME).** libvpx refines the
//!   diamond hit with a sub-pel search; rvp9's reference uses
//!   integer-pel only. Sub-pel ME lands when the rvp9 mv_search
//!   sub-pixel module gets its temporal-filter-shaped wrapper. The
//!   predictor still supports Q3 sub-pel MVs end-to-end, so this
//!   doesn't lose any precision in the downstream NLM math — it just
//!   means the MV we feed in always has zero sub-pel bits.
//! - **No second-pass `adjust_arnr_filter` adjustments.** libvpx's
//!   `oxcf->pass == 2` branch (lines 1042-1048) reads
//!   `twopass.arnr_strength_adjustment`; rvp9 doesn't have first-pass
//!   stats yet, so we collapse to `base_strength = arnr_strength`.
//! - **`adjust_arnr_filter` Q-table inlined.** libvpx calls
//!   `vp9_convert_qindex_to_q(qindex, bit_depth)` to convert an int
//!   qindex to a real Q value (lines 1064-1069). For 8-bit, the
//!   tabulated values come from `vp9_quant_common.c::ac_qlookup`.
//!   Since the rvp9 `quant` tables aren't directly cross-importable
//!   without pulling in extra context, we use the same 8-bit lookup
//!   inline here (it's just `ac_qlookup[qindex] / 4.0` per libvpx
//!   `vp9_quant_common.c::vp9_dc_quant`/`vp9_ac_quant`, and we only
//!   need the integer truncation that libvpx uses at line 1064).

use crate::lookahead::LookaheadEntry;
use rvp9_core::inter::filter::{
    EIGHTTAP_SHARP_KERNEL, FILTER_BITS, SUBPEL_BITS, SUBPEL_MASK, SUBPEL_SHIFTS, SUBPEL_TAPS,
};

// =====================================================================
// Constants (libvpx `vp9_temporal_filter.h:18-33`).
// =====================================================================

/// `ARNR_FILT_QINDEX` — the fixed qindex used for ARF construction's
/// motion-search rate-distortion math. libvpx
/// `vp9_temporal_filter.h:18`.
pub const ARNR_FILT_QINDEX: u8 = 128;

/// `BW` / `BH` — temporal-filter block size in luma pixels. libvpx
/// `vp9_temporal_filter.h:23,25`.
pub const BW: usize = 32;
/// See [`BW`].
pub const BH: usize = 32;

/// `BLK_PELS` — luma pixels per block. libvpx `vp9_temporal_filter.h:27`.
pub const BLK_PELS: usize = BH * BW;

/// `TF_SHIFT` / `TF_ROUND` — mi-coord → mb-coord conversion. libvpx
/// `vp9_temporal_filter.h:28-29`.
pub const TF_SHIFT: usize = 2;
/// See [`TF_SHIFT`].
pub const TF_ROUND: usize = 3;

/// `SUB_BW` / `SUB_BH` — sub-block (16×16) size for the 4-way split
/// inside a 32×32 block. libvpx `vp9_temporal_filter.h:32-33`.
pub const SUB_BW: usize = 16;
/// See [`SUB_BW`].
pub const SUB_BH: usize = 16;

/// Number of 16×16 sub-blocks tiling a 32×32 block.
pub const NUM_SUB_BLOCKS: usize = 4;

/// `index_mult` modifier table for `mod_index`. libvpx
/// `vp9_temporal_filter.c:37-39`.
const INDEX_MULT: [u32; 14] = [
    0, 0, 0, 0, 49152, 39322, 32768, 28087, 24576, 21846, 19661, 17874, 0, 15124,
];

/// `MAX_LAG_BUFFERS` — upper bound on the temporal-filter frame
/// budget. Matches the lookahead's [`crate::lookahead::MAX_LAG_BUFFERS`].
const MAX_LAG_BUFFERS: usize = crate::lookahead::MAX_LAG_BUFFERS;

// =====================================================================
// Motion-vector type (libvpx `vp9_common/vp9_mv.h::MV`).
// =====================================================================

/// 1/8-pel MV (Q3 units). libvpx `MV` (`vp9_mv.h::MV`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Mv {
    /// Row component in 1/8-pel units.
    pub row: i16,
    /// Column component in 1/8-pel units.
    pub col: i16,
}

// =====================================================================
// W3.1 — ArnrFilterData
// =====================================================================

/// Per-frame reference into the temporal-filter input set.
///
/// Borrows from the lookahead-owned [`LookaheadEntry`] so the filter
/// pipeline doesn't double-buffer the source frames. The lifetime
/// `'a` ties this back to the lookahead context.
///
/// The frame at index `alt_ref_index` of [`ArnrFilterData::frames`] is
/// the **ARF source** — the centre frame the filter is anchored to.
/// All other frames are temporal neighbours whose contributions are
/// blended into the output.
#[derive(Clone, Copy, Debug)]
pub struct ArnrFrameRef<'a> {
    /// Luma plane.
    pub y: &'a [u8],
    /// Cb plane.
    pub u: &'a [u8],
    /// Cr plane.
    pub v: &'a [u8],
    /// Luma stride (== width since lookahead packs contiguous rows).
    pub y_stride: usize,
    /// Chroma stride.
    pub uv_stride: usize,
    /// Coded luma width.
    pub width: u32,
    /// Coded luma height.
    pub height: u32,
}

impl<'a> ArnrFrameRef<'a> {
    /// Borrow a [`LookaheadEntry`] as an [`ArnrFrameRef`].
    #[must_use]
    pub fn from_lookahead(entry: &'a LookaheadEntry) -> Self {
        let w = entry.width as usize;
        let uvw = w / 2;
        Self {
            y: &entry.y,
            u: &entry.u,
            v: &entry.v,
            y_stride: w,
            uv_stride: uvw,
            width: entry.width,
            height: entry.height,
        }
    }
}

/// Filter input + parameters for one ARF construction call.
///
/// Mirrors libvpx `ARNRFilterData` (`vp9_encoder.h:499-505`):
///
/// ```c
/// typedef struct ARNRFilterData {
///   YV12_BUFFER_CONFIG *frames[MAX_LAG_BUFFERS];
///   int strength;
///   int frame_count;
///   int alt_ref_index;
///   struct scale_factors sf;
/// } ARNRFilterData;
/// ```
///
/// `sf` (scale factors) is omitted — rvp9 doesn't support inter-frame
/// resize at the ARF builder level (per the documented scope above).
#[derive(Clone, Debug)]
pub struct ArnrFilterData<'a> {
    /// Source frames (capacity matches libvpx `MAX_LAG_BUFFERS`).
    pub frames: Vec<ArnrFrameRef<'a>>,
    /// Index of the ARF source within `frames`. libvpx
    /// `arnr_filter_data->alt_ref_index`.
    pub alt_ref_index: usize,
    /// Number of frames participating in the blend (= `frames.len()`).
    /// Carried explicitly to mirror libvpx's separate field.
    pub frame_count: usize,
    /// NLM strength `0..=6` (libvpx `arnr_filter_data->strength`,
    /// asserted `[0,6]` at `vp9_temporal_filter.c:279-280`).
    pub strength: u8,
    /// Q-index used for the temporal-filter rate-distortion math.
    /// Always [`ARNR_FILT_QINDEX`] in libvpx
    /// (`vp9_temporal_filter.c:1197-1199`); carried as a field so a
    /// future caller can override for experiments.
    pub filter_qindex: u8,
}

impl<'a> ArnrFilterData<'a> {
    /// Construct an [`ArnrFilterData`] from a frame list.
    ///
    /// # Panics
    ///
    /// Panics if `alt_ref_index >= frames.len()`, if `frames` is
    /// empty or exceeds [`MAX_LAG_BUFFERS`], or if `strength > 6`.
    #[must_use]
    pub fn new(frames: Vec<ArnrFrameRef<'a>>, alt_ref_index: usize, strength: u8) -> Self {
        assert!(!frames.is_empty(), "ArnrFilterData needs ≥1 frame");
        assert!(
            frames.len() <= MAX_LAG_BUFFERS,
            "ArnrFilterData frame_count exceeds MAX_LAG_BUFFERS"
        );
        assert!(alt_ref_index < frames.len(), "alt_ref_index out of bounds");
        assert!(strength <= 6, "strength must be in [0, 6]");
        let frame_count = frames.len();
        Self {
            frames,
            alt_ref_index,
            frame_count,
            strength,
            filter_qindex: ARNR_FILT_QINDEX,
        }
    }
}

// =====================================================================
// W3.2 — adjust_arnr_filter
// =====================================================================

/// libvpx 8-bit `ac_qlookup` partial table — only the entries reachable
/// from the `q > 16` branch matter for our purposes. The full table
/// lives in `vp9_quant_common.c` and the conversion is
/// `vp9_convert_qindex_to_q(qi, bd) = ac_qlookup[qi] / 4.0` (libvpx
/// `vp9_quant_common.c::vp9_convert_qindex_to_q`).
///
/// We embed the exact 256-entry table so [`adjust_arnr_filter`] is a
/// pure function. `q > 16` (libvpx `vp9_temporal_filter.c:1070`)
/// corresponds to `qindex > 60` (since `ac_qlookup[60] = 64`, q=16),
/// matching libvpx's threshold for "high enough Q that we don't
/// dampen strength".
const AC_QLOOKUP_8BIT: [i16; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
    140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188,
    191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 260,
    265, 270, 275, 280, 285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366,
    373, 380, 387, 394, 401, 408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520,
    530, 540, 550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729, 743,
    757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988, 1007, 1026, 1046,
    1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423,
    1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
];

/// Convert qindex → q value (truncated to int) for the 8-bit profile.
/// Matches the result of `(int)vp9_convert_qindex_to_q(qi, 8)` in
/// libvpx (it returns a `double = ac_qlookup[qi] / 4.0`; truncation
/// to `int` happens at the call site, `vp9_temporal_filter.c:1064`).
#[inline]
fn qindex_to_q_int(qindex: u8) -> i32 {
    i32::from(AC_QLOOKUP_8BIT[qindex as usize]) / 4
}

/// Result of [`adjust_arnr_filter`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdjustedArnrParams {
    /// Number of frames to include in the filter.
    pub frame_count: u8,
    /// NLM strength `0..=6`.
    pub strength: u8,
}

/// Apply buffer-limits + context-specific adjustments to the ARF
/// filter parameters. Pure function form of libvpx
/// `vp9_temporal_filter.c:1031-1102`.
///
/// libvpx's signature is `adjust_arnr_filter(VP9_COMP *cpi, int
/// distance, int group_boost, int *arnr_frames, int *arnr_strength)`.
/// We unpack the `cpi` fields the function actually reads and pass
/// them explicitly:
///
/// - `arnr_max_frames` (libvpx `oxcf->arnr_max_frames`, line 1037).
/// - `arnr_strength` (libvpx `oxcf->arnr_strength`, line 1047).
/// - `distance` (frames the ARF reaches forward into the GF group).
/// - `frames_after_arf` (libvpx line 1035; equals
///   `vp9_lookahead_depth(lookahead) - distance - 1`).
/// - `group_boost` (libvpx `rc->gfu_boost`, passed as the third
///   formal arg).
/// - `inter_qindex` (libvpx `rc->avg_frame_qindex[INTER_FRAME]`).
/// - `key_qindex` (libvpx `rc->avg_frame_qindex[KEY_FRAME]`).
/// - `current_video_frame_gt_1` (libvpx
///   `cm->current_video_frame > 1`, line 1064).
/// - `arf_src_offset_lt_baseline_minus1` (libvpx lines 1096-1097;
///   forces `frames = 1` when the ARF will eventually be shown as
///   `show_existing_frame`).
///
/// libvpx's pass-2 strength adjustment (lines 1042-1048) is dropped
/// per the deviation note at the top of the module.
#[must_use]
#[allow(clippy::too_many_arguments)] // mirrors libvpx's parameter set
pub fn adjust_arnr_filter(
    arnr_max_frames: u8,
    arnr_strength: u8,
    distance: u32,
    frames_after_arf: u32,
    group_boost: u32,
    inter_qindex: u8,
    key_qindex: u8,
    current_video_frame_gt_1: bool,
    arf_src_offset_lt_baseline_minus1: bool,
) -> AdjustedArnrParams {
    // libvpx `vp9_temporal_filter.c:1037` — `(arnr_max_frames - 1) >> 1`
    let mut frames_fwd: i32 = (i32::from(arnr_max_frames) - 1) >> 1;

    // libvpx `vp9_temporal_filter.c:1046-1048` — base_strength from
    // `oxcf->arnr_strength`, clipped to [0,6] (libvpx clips only on
    // the pass-2 path at line 1045; pass-1 line 1047 trusts the
    // input. We clip unconditionally because rvp9 hasn't validated
    // the config knob upstream yet).
    let base_strength: i32 = i32::from(arnr_strength).clamp(0, 6);

    // libvpx `vp9_temporal_filter.c:1051-1052` — forward/backward
    // limits.
    let frames_after_arf_i32 = frames_after_arf as i32;
    let distance_i32 = distance as i32;
    if frames_fwd > frames_after_arf_i32 {
        frames_fwd = frames_after_arf_i32;
    }
    if frames_fwd > distance_i32 {
        frames_fwd = distance_i32;
    }

    let mut frames_bwd: i32 = frames_fwd;

    // libvpx `vp9_temporal_filter.c:1057-1058` — even-length filter
    // bonus backward frame.
    if frames_bwd < distance_i32 {
        frames_bwd += (i32::from(arnr_max_frames) + 1) & 0x1;
    }

    // libvpx `vp9_temporal_filter.c:1061`.
    let mut frames: i32 = frames_bwd + 1 + frames_fwd;

    // libvpx `vp9_temporal_filter.c:1063-1069` — strength dampening
    // by active max Q.
    let q: i32 = if current_video_frame_gt_1 {
        qindex_to_q_int(inter_qindex)
    } else {
        qindex_to_q_int(key_qindex)
    };
    let mut strength: i32 = if q > 16 {
        base_strength
    } else {
        let mut s = base_strength - ((16 - q) / 2);
        if s < 0 {
            s = 0;
        }
        s
    };

    // libvpx `vp9_temporal_filter.c:1078-1085` — gf-boost gating.
    let group_boost_i32 = group_boost as i32;
    if frames > group_boost_i32 / 150 {
        frames = group_boost_i32 / 150;
        frames += if (frames & 1) == 0 { 1 } else { 0 };
    }
    if strength > group_boost_i32 / 300 {
        strength = group_boost_i32 / 300;
    }

    // libvpx `vp9_temporal_filter.c:1096-1098` — show_existing_frame
    // gate.
    if arf_src_offset_lt_baseline_minus1 {
        frames = 1;
    }

    // libvpx asserts `strength ∈ [0,6]` at apply_temporal_filter
    // (line 279-280). Saturate here so the caller always passes a
    // valid value; libvpx itself can underflow into negative values
    // for very low Q only when `base_strength` is also small, in
    // which case the clamp to 0 at line 1074 already fires. The
    // upper clamp matches libvpx's pass-2 line 1045.
    strength = strength.clamp(0, 6);
    if frames < 1 {
        frames = 1;
    }

    AdjustedArnrParams {
        frame_count: frames as u8,
        strength: strength as u8,
    }
}

// =====================================================================
// W3.3 — temporal_filter_predictors_mb (scalar, EIGHTTAP_SHARP).
// =====================================================================

/// Build a 32×32 luma + 2× 16×16 chroma sub-pel prediction for one
/// temporal-filter block.
///
/// Scalar reference for libvpx
/// `temporal_filter_predictors_mb_c`
/// (`vp9_temporal_filter.c:48-186`). Output layout matches libvpx's
/// `pred[BLK_PELS * 3]` packing:
///
/// - `pred[0 .. BLK_PELS]`               — luma, stride `BW`.
/// - `pred[BLK_PELS .. 2*BLK_PELS]`      — Cb, stride `BW / 2 = 16`.
/// - `pred[2*BLK_PELS .. 3*BLK_PELS]`    — Cr, stride `BW / 2 = 16`.
///
/// `use_32x32 == true` paths the predictor through the single 32×32
/// MV (`mv_row` / `mv_col` for luma in Q3 units, derived from the
/// libvpx `MV mv = {mv_row, mv_col}` at line 71). `use_32x32 ==
/// false` paths it through four independent 16×16 MVs from
/// `blk_mvs[0..4]`.
///
/// `x` / `y` are the **block origin within the frame**, in luma
/// pixels — libvpx passes `mb_col * BW`, `mb_row * BH`
/// (`vp9_temporal_filter.c:809`).
///
/// # libvpx call sites
///
/// Luma path:    `vp9_temporal_filter.c:93-94` (use_32x32) /
/// `vp9_temporal_filter.c:124-127` (4×16).
///
/// Chroma path:  `vp9_temporal_filter.c:96-104` (use_32x32) /
/// `vp9_temporal_filter.c:162-171` (4×16).
///
/// # Subsampling
///
/// rvp9 ships 4:2:0 only — `ss_x = ss_y = 1`. The libvpx
/// `mv_precision_uv = MV_PRECISION_Q4` branch (lines 59-61) handles
/// the doubling of the chroma MV; the `MV_PRECISION_Q3` else-branch
/// at lines 62-64 is the 4:4:4 path that rvp9 doesn't exercise.
#[allow(clippy::too_many_arguments)] // mirrors libvpx
pub fn temporal_filter_predictors_mb(
    y_frame: &[u8],
    u_frame: &[u8],
    v_frame: &[u8],
    y_stride: usize,
    uv_stride: usize,
    y_width: usize,
    y_height: usize,
    block_x: usize,
    block_y: usize,
    mv_row_q3: i16,
    mv_col_q3: i16,
    blk_mvs: &[Mv; NUM_SUB_BLOCKS],
    use_32x32: bool,
    pred: &mut [u8; BLK_PELS * 3],
) {
    // 4:2:0 — `ss_x = ss_y = 1`. UV block is BW/2 × BH/2.
    let uv_block_width = BW / 2;
    let uv_block_height = BH / 2;

    // Luma plane.
    if use_32x32 {
        // libvpx `vp9_temporal_filter.c:93-94`
        build_inter_predictor(
            y_frame,
            y_stride,
            y_width,
            y_height,
            block_x,
            block_y,
            BW,
            BH,
            mv_row_q3,
            mv_col_q3,
            /*is_chroma=*/ false,
            &mut pred[0..BLK_PELS],
            BW,
        );
        // Cb / Cr (libvpx lines 96-104). The chroma plane is at
        // half resolution; libvpx's `MV_PRECISION_Q4` doubles the
        // MV. We mirror that by feeding the chroma origin and
        // calling build_inter_predictor with is_chroma=true.
        let uv_x = block_x / 2;
        let uv_y = block_y / 2;
        let uv_w = y_width / 2;
        let uv_h = y_height / 2;
        build_inter_predictor(
            u_frame,
            uv_stride,
            uv_w,
            uv_h,
            uv_x,
            uv_y,
            uv_block_width,
            uv_block_height,
            mv_row_q3,
            mv_col_q3,
            /*is_chroma=*/ true,
            &mut pred[BLK_PELS..2 * BLK_PELS],
            uv_block_width,
        );
        build_inter_predictor(
            v_frame,
            uv_stride,
            uv_w,
            uv_h,
            uv_x,
            uv_y,
            uv_block_width,
            uv_block_height,
            mv_row_q3,
            mv_col_q3,
            /*is_chroma=*/ true,
            &mut pred[2 * BLK_PELS..3 * BLK_PELS],
            uv_block_width,
        );
        return;
    }

    // use_32x32 == false: 4× 16×16 sub-blocks. libvpx loops i in
    // 0..BH step BH/2 (=16), j in 0..BW step BW/2 (=16),
    // raster-order producing blk_mvs[0..4]. Lines 111-135.
    let ys = BH / 2;
    let xs = BW / 2;
    let mut k = 0usize;
    for i in (0..BH).step_by(ys) {
        for j in (0..BW).step_by(xs) {
            let mv = blk_mvs[k];
            let dst_off = i * BW + j;
            // Sub-block luma. Note libvpx's per-sub-block origin
            // adjustment: the destination origin is `(block_x + j,
            // block_y + i)`; the predictor builder applies the MV
            // relative to that.
            build_inter_predictor(
                y_frame,
                y_stride,
                y_width,
                y_height,
                block_x + j,
                block_y + i,
                xs,
                ys,
                mv.row,
                mv.col,
                false,
                &mut pred[dst_off..dst_off + (ys - 1) * BW + xs],
                BW,
            );
            k += 1;
        }
    }

    // U / V — libvpx lines 138-185. Sub-block grid is half size in
    // chroma; same blk_mvs reused (`mv_precision_uv = MV_PRECISION_Q4`).
    let uv_xs = uv_block_width / 2;
    let uv_ys = uv_block_height / 2;
    let uv_w = y_width / 2;
    let uv_h = y_height / 2;
    let uv_x_base = block_x / 2;
    let uv_y_base = block_y / 2;
    let mut k = 0usize;
    for i in (0..uv_block_height).step_by(uv_ys) {
        for j in (0..uv_block_width).step_by(uv_xs) {
            let mv = blk_mvs[k];
            let dst_off = i * uv_block_width + j;
            let u_dst = &mut pred
                [BLK_PELS + dst_off..BLK_PELS + dst_off + (uv_ys - 1) * uv_block_width + uv_xs];
            build_inter_predictor(
                u_frame,
                uv_stride,
                uv_w,
                uv_h,
                uv_x_base + j,
                uv_y_base + i,
                uv_xs,
                uv_ys,
                mv.row,
                mv.col,
                true,
                u_dst,
                uv_block_width,
            );
            let v_dst = &mut pred[2 * BLK_PELS + dst_off
                ..2 * BLK_PELS + dst_off + (uv_ys - 1) * uv_block_width + uv_xs];
            build_inter_predictor(
                v_frame,
                uv_stride,
                uv_w,
                uv_h,
                uv_x_base + j,
                uv_y_base + i,
                uv_xs,
                uv_ys,
                mv.row,
                mv.col,
                true,
                v_dst,
                uv_block_width,
            );
            k += 1;
        }
    }
}

/// Build one inter-predicted plane block.
///
/// Faithful scalar port of libvpx `vp9_build_inter_predictor`
/// (`vp9_reconinter.c:43-59`) + `inter_predictor` dispatch
/// (`vp9_reconinter.h:23-32`). EIGHTTAP_SHARP kernel; no scaling.
///
/// `mv_row_q3` / `mv_col_q3` are the **luma-domain Q3** MV components.
/// For chroma (`is_chroma == true`), libvpx promotes them to Q4 by
/// `mv * 2` (line 49-50 of `vp9_reconinter.c`).
///
/// The reference is **edge-clamped**: samples outside the frame
/// repeat the boundary pixel. libvpx does this by extending the
/// reference buffer with a border (`VP9ENCBORDERINPIXELS = 32`); rvp9
/// emulates the same effect by clamping coordinates in the inner
/// loop. This is correct for sub-pel MVs whose footprint partially
/// straddles the frame edge.
#[allow(clippy::too_many_arguments)]
fn build_inter_predictor(
    src: &[u8],
    src_stride: usize,
    src_width: usize,
    src_height: usize,
    dst_x: usize,
    dst_y: usize,
    w: usize,
    h: usize,
    mv_row_q3: i16,
    mv_col_q3: i16,
    is_chroma: bool,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // libvpx `vp9_reconinter.c:48-50`: Q3 → Q4 (chroma uses Q4 MV
    // because the chroma plane is half-size).
    let (mv_row_q4, mv_col_q4): (i32, i32) = if is_chroma {
        // Q3 luma MV → Q4 chroma MV: `mv * 2` for both axes (4:2:0).
        (i32::from(mv_row_q3) * 2, i32::from(mv_col_q3) * 2)
    } else {
        // libvpx: `is_q4 ? row : row * 2` — for the luma call we have
        // `precision = MV_PRECISION_Q3`, so `mv_q4 = mv * 2`.
        (i32::from(mv_row_q3) * 2, i32::from(mv_col_q3) * 2)
    };

    // libvpx `vp9_reconinter.c:55`: `src += (mv.row >> SUBPEL_BITS) *
    // stride + (mv.col >> SUBPEL_BITS)`. Sub-pel offsets are the
    // bottom SUBPEL_BITS.
    let int_row = mv_row_q4 >> SUBPEL_BITS;
    let int_col = mv_col_q4 >> SUBPEL_BITS;
    let subpel_x = (mv_col_q4 & SUBPEL_MASK) as usize;
    let subpel_y = (mv_row_q4 & SUBPEL_MASK) as usize;
    debug_assert!(subpel_x < SUBPEL_SHIFTS && subpel_y < SUBPEL_SHIFTS);

    // The footprint we need from `src` is, in source coordinates:
    //   columns: dst_x + int_col - 3            .. dst_x + int_col + w + 3
    //   rows:    dst_y + int_row - 3            .. dst_y + int_row + h + 7
    //                                              (8-tap vertical = h + 7)
    // libvpx relies on the source border; we materialise a padded
    // patch with edge clamping. This costs O((w + 7)*(h + 7)) — fine
    // for 32×32 / 16×16 blocks.

    let pad = SUBPEL_TAPS / 2 - 1; // 3
    let patch_w = w + (SUBPEL_TAPS - 1); // w + 7
    let patch_h = h + (SUBPEL_TAPS - 1); // h + 7

    // Source-coordinate top-left of the patch:
    let src_x0 = dst_x as i32 + int_col - pad as i32;
    let src_y0 = dst_y as i32 + int_row - pad as i32;

    let mut patch = vec![0u8; patch_w * patch_h];
    for py in 0..patch_h {
        let sy = (src_y0 + py as i32).clamp(0, src_height as i32 - 1) as usize;
        for px in 0..patch_w {
            let sx = (src_x0 + px as i32).clamp(0, src_width as i32 - 1) as usize;
            patch[py * patch_w + px] = src[sy * src_stride + sx];
        }
    }

    // Dispatch matches libvpx's `predict[subpel_x != 0][subpel_y != 0]`
    // table:
    //   [0][0] -> copy
    //   [1][0] -> horiz-only
    //   [0][1] -> vert-only
    //   [1][1] -> 2-D full convolve
    // libvpx `vp9_reconinter.h:29`.
    let kernel = &EIGHTTAP_SHARP_KERNEL;
    let filter_x = &kernel[subpel_x];
    let filter_y = &kernel[subpel_y];
    match (subpel_x != 0, subpel_y != 0) {
        (false, false) => {
            // Integer-pel copy. Patch[pad..pad+h, pad..pad+w] is the
            // exact w×h slab.
            for y in 0..h {
                for x in 0..w {
                    dst[y * dst_stride + x] = patch[(pad + y) * patch_w + (pad + x)];
                }
            }
        }
        (true, false) => {
            // Horizontal-only. convolve8_horiz wants src pre-shifted
            // left by 3 cols; our patch already starts at src_x0 = -3.
            // We use a single-row indexing loop because the patch
            // has shape (h + 7) × (w + 7) and we want only the
            // first h rows aligned vertically with the output.
            for y in 0..h {
                let row = &patch[(pad + y) * patch_w..(pad + y) * patch_w + patch_w];
                for x in 0..w {
                    let mut sum: i32 = 0;
                    for k in 0..SUBPEL_TAPS {
                        sum += i32::from(row[x + k]) * i32::from(filter_x[k]);
                    }
                    dst[y * dst_stride + x] = round_clip_filter(sum);
                }
            }
        }
        (false, true) => {
            // Vertical-only. Patch starts at src_y0 = -3 so row y in
            // the output sums patch[y..y+8] columns at pad+x.
            for y in 0..h {
                for x in 0..w {
                    let mut sum: i32 = 0;
                    for k in 0..SUBPEL_TAPS {
                        sum += i32::from(patch[(y + k) * patch_w + (pad + x)])
                            * i32::from(filter_y[k]);
                    }
                    dst[y * dst_stride + x] = round_clip_filter(sum);
                }
            }
        }
        (true, true) => {
            // 2-D full convolve. Horizontal pass into a `patch_h`-row
            // intermediate of shape patch_h × w, then vertical pass.
            // Mirrors libvpx `vpx_convolve8_c` (`vpx_convolve.c:158-190`).
            let mut intermediate = vec![0u8; patch_h * w];
            for y in 0..patch_h {
                let row = &patch[y * patch_w..y * patch_w + patch_w];
                for x in 0..w {
                    let mut sum: i32 = 0;
                    for k in 0..SUBPEL_TAPS {
                        sum += i32::from(row[x + k]) * i32::from(filter_x[k]);
                    }
                    intermediate[y * w + x] = round_clip_filter(sum);
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mut sum: i32 = 0;
                    for k in 0..SUBPEL_TAPS {
                        sum += i32::from(intermediate[(y + k) * w + x]) * i32::from(filter_y[k]);
                    }
                    dst[y * dst_stride + x] = round_clip_filter(sum);
                }
            }
        }
    }
}

#[inline]
fn round_clip_filter(sum: i32) -> u8 {
    let rounded = (sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS;
    rounded.clamp(0, 255) as u8
}

// =====================================================================
// W3.4 — apply_temporal_filter (scalar NLM core).
// =====================================================================

/// `mod_index` — NLM modifier index lookup. libvpx
/// `vp9_temporal_filter.c:195-213`.
#[inline]
#[must_use]
pub fn mod_index(
    sum_dist: i32,
    index: i32,
    rounding: i32,
    strength: i32,
    filter_weight: i32,
) -> i32 {
    debug_assert!((0..=13).contains(&index));
    debug_assert!(INDEX_MULT[index as usize] != 0);

    // libvpx: `((unsigned)clamp(sum_dist, 0, UINT16_MAX) * index_mult[index]) >> 16`
    let clamped = sum_dist.clamp(0, u16::MAX as i32) as u32;
    let mut modi = ((clamped.wrapping_mul(INDEX_MULT[index as usize])) >> 16) as i32;
    modi += rounding;
    modi >>= strength;

    // libvpx: `mod = VPXMIN(16, mod);`
    if modi > 16 {
        modi = 16;
    }
    // libvpx: `mod = 16 - mod; mod *= filter_weight;`
    modi = 16 - modi;
    modi *= filter_weight;
    modi
}

/// `get_filter_weight` — per-pixel weight from blk_fw quad. libvpx
/// `vp9_temporal_filter.c:236-258`.
#[inline]
fn get_filter_weight(
    i: usize,
    j: usize,
    block_height: usize,
    block_width: usize,
    blk_fw: &[i32; NUM_SUB_BLOCKS],
    use_32x32: bool,
) -> i32 {
    if use_32x32 {
        return blk_fw[0];
    }
    if i < block_height / 2 {
        if j < block_width / 2 {
            return blk_fw[0];
        }
        return blk_fw[1];
    }
    if j < block_width / 2 {
        return blk_fw[2];
    }
    blk_fw[3]
}

/// Non-local-means temporal filter — scalar reference. libvpx
/// `vp9_apply_temporal_filter_c` (`vp9_temporal_filter.c:260-399`).
///
/// Inputs:
/// - `y_frame1`, `u_frame1`, `v_frame1` — the ARF source for this block.
/// - `y_pred`, `u_pred`, `v_pred` — the motion-compensated prediction
///   from the neighbour frame.
/// - `accumulator` / `count` — the 3-plane accumulator triple
///   (`y_acc[BLK_PELS]`, `u_acc[BLK_PELS/4]`, `v_acc[BLK_PELS/4]`),
///   packed identically to libvpx
///   (`vp9_temporal_filter.c:706-707`).
/// - `strength`, `blk_fw`, `use_32x32`, `ss_x`, `ss_y` — NLM tuning.
///
/// libvpx asserts `strength ∈ [0, 6]` (lines 279-280); we do the
/// same. `strength == 0` ⇒ `rounding = 0`, the modifier saturates at
/// 16 with no shifting → `modifier = (16 - 16) * filter_weight = 0`
/// ⇒ accumulator is untouched. (Tested by `w3_4` PROOF below.)
#[allow(clippy::too_many_arguments)] // mirrors libvpx
pub fn apply_temporal_filter(
    y_frame1: &[u8],
    y_stride: usize,
    y_pred: &[u8],
    y_buf_stride: usize,
    u_frame1: &[u8],
    v_frame1: &[u8],
    uv_stride: usize,
    u_pred: &[u8],
    v_pred: &[u8],
    uv_buf_stride: usize,
    block_width: usize,
    block_height: usize,
    ss_x: u32,
    ss_y: u32,
    strength: i32,
    blk_fw: &[i32; NUM_SUB_BLOCKS],
    use_32x32: bool,
    y_accumulator: &mut [u32],
    y_count: &mut [u16],
    u_accumulator: &mut [u32],
    u_count: &mut [u16],
    v_accumulator: &mut [u32],
    v_count: &mut [u16],
) {
    assert!((0..=6).contains(&strength), "strength must be in [0, 6]");

    let rounding: i32 = (1 << strength) >> 1;
    let uv_block_width = block_width >> ss_x;
    let uv_block_height = block_height >> ss_y;

    // libvpx uses fixed BLK_PELS staging; we allocate exactly the
    // sizes we need for the requested block shape.
    let y_buf_pels = block_width * block_height;
    let uv_buf_pels = uv_block_width * uv_block_height;
    let mut y_diff_sse = vec![0u16; y_buf_pels];
    let mut u_diff_sse = vec![0u16; uv_buf_pels];
    let mut v_diff_sse = vec![0u16; uv_buf_pels];

    // libvpx `vp9_temporal_filter.c:288-294` — Y diff².
    for i in 0..block_height {
        for j in 0..block_width {
            let diff =
                i32::from(y_frame1[i * y_stride + j]) - i32::from(y_pred[i * y_buf_stride + j]);
            // libvpx stores `diff * diff` in u16; max value is
            // 255² = 65025, fits.
            y_diff_sse[i * block_width + j] = (diff * diff) as u16;
        }
    }
    // libvpx `vp9_temporal_filter.c:295-306` — UV diff².
    {
        let mut idx = 0;
        for i in 0..uv_block_height {
            for j in 0..uv_block_width {
                let du = i32::from(u_frame1[i * uv_stride + j])
                    - i32::from(u_pred[i * uv_buf_stride + j]);
                let dv = i32::from(v_frame1[i * uv_stride + j])
                    - i32::from(v_pred[i * uv_buf_stride + j]);
                u_diff_sse[idx] = (du * du) as u16;
                v_diff_sse[idx] = (dv * dv) as u16;
                idx += 1;
            }
        }
    }

    // libvpx `vp9_temporal_filter.c:308-398` — main NLM loop.
    let mut k = 0usize; // luma acc index
    let mut m = 0usize; // chroma acc index
    for i in 0..block_height {
        for j in 0..block_width {
            let pixel_value = i32::from(y_pred[i * y_buf_stride + j]);
            let filter_weight =
                get_filter_weight(i, j, block_height, block_width, blk_fw, use_32x32);

            // libvpx `vp9_temporal_filter.c:315-339` — Y 3×3 NLM
            // window + chroma cross-term.
            let mut y_index: i32 = 0;
            let uv_r = i >> ss_y;
            let uv_c = j >> ss_x;
            let mut modifier: i32 = 0;
            for idy in -1i32..=1 {
                for idx in -1i32..=1 {
                    let row = i as i32 + idy;
                    let col = j as i32 + idx;
                    if row >= 0 && row < block_height as i32 && col >= 0 && col < block_width as i32
                    {
                        modifier +=
                            i32::from(y_diff_sse[row as usize * block_width + col as usize]);
                        y_index += 1;
                    }
                }
            }
            debug_assert!(y_index > 0);

            modifier += i32::from(u_diff_sse[uv_r * uv_block_width + uv_c]);
            modifier += i32::from(v_diff_sse[uv_r * uv_block_width + uv_c]);
            y_index += 2;

            let modifier = mod_index(modifier, y_index, rounding, strength, filter_weight);
            y_count[k] = y_count[k].wrapping_add(modifier as u16);
            y_accumulator[k] =
                y_accumulator[k].wrapping_add((modifier as u32) * (pixel_value as u32));
            k += 1;

            // libvpx `vp9_temporal_filter.c:350-396` — chroma update,
            // gated by `!(i & ss_y) && !(j & ss_x)` (= one chroma
            // sample per 2×2 luma quad).
            if (i & ss_y as usize) == 0 && (j & ss_x as usize) == 0 {
                let u_pixel_value = i32::from(u_pred[uv_r * uv_buf_stride + uv_c]);
                let v_pixel_value = i32::from(v_pred[uv_r * uv_buf_stride + uv_c]);
                let mut cr_index: i32 = 0;
                let mut u_mod: i32 = 0;
                let mut v_mod: i32 = 0;

                for idy in -1i32..=1 {
                    for idx in -1i32..=1 {
                        let row = uv_r as i32 + idy;
                        let col = uv_c as i32 + idx;
                        if row >= 0
                            && row < uv_block_height as i32
                            && col >= 0
                            && col < uv_block_width as i32
                        {
                            u_mod +=
                                i32::from(u_diff_sse[row as usize * uv_block_width + col as usize]);
                            v_mod +=
                                i32::from(v_diff_sse[row as usize * uv_block_width + col as usize]);
                            cr_index += 1;
                        }
                    }
                }
                debug_assert!(cr_index > 0);

                // libvpx `vp9_temporal_filter.c:375-382` — luma
                // cross-term for the chroma modifier.
                let mut y_diff_local: i32 = 0;
                for idy in 0..=ss_y as i32 {
                    for idx in 0..=ss_x as i32 {
                        let row = (uv_r << ss_y) as i32 + idy;
                        let col = (uv_c << ss_x) as i32 + idx;
                        y_diff_local +=
                            i32::from(y_diff_sse[row as usize * block_width + col as usize]);
                        cr_index += 1;
                    }
                }
                u_mod += y_diff_local;
                v_mod += y_diff_local;

                let u_mod = mod_index(u_mod, cr_index, rounding, strength, filter_weight);
                let v_mod = mod_index(v_mod, cr_index, rounding, strength, filter_weight);

                u_count[m] = u_count[m].wrapping_add(u_mod as u16);
                u_accumulator[m] =
                    u_accumulator[m].wrapping_add((u_mod as u32) * (u_pixel_value as u32));
                v_count[m] = v_count[m].wrapping_add(v_mod as u16);
                v_accumulator[m] =
                    v_accumulator[m].wrapping_add((v_mod as u32) * (v_pixel_value as u32));
                m += 1;
            }
        }
    }
}

// =====================================================================
// W3.5 — find_matching_mb + iterate_row + vp9_temporal_filter.
// =====================================================================

/// `fixed_divide[i]` = `0x80000 / i` for `i ∈ 1..512`. Used to
/// normalise `accumulator / count` into 8-bit output without a real
/// division. libvpx `vp9_temporal_filter.c:36, 191-192`.
fn fixed_divide_table() -> [u32; 512] {
    let mut t = [0u32; 512];
    for (i, slot) in t.iter_mut().enumerate().skip(1) {
        *slot = 0x80000 / (i as u32);
    }
    t
}

/// Output of a temporal-filter construction call.
#[derive(Clone, Debug)]
pub struct FilteredArnrFrame {
    /// Luma plane, row-major, stride == `width`.
    pub y: Vec<u8>,
    /// Cb plane, row-major, stride == `width / 2`.
    pub u: Vec<u8>,
    /// Cr plane, row-major, stride == `width / 2`.
    pub v: Vec<u8>,
    /// Coded luma width.
    pub width: u32,
    /// Coded luma height.
    pub height: u32,
}

/// Block-match motion search. Diamond-style integer-pel SAD search
/// over a small window around (0, 0). Scalar reference for libvpx
/// `temporal_filter_find_matching_mb_c`
/// (`vp9_temporal_filter.c:552-643`).
///
/// libvpx's path is `vp9_full_pixel_search(... MESH ...)` followed by
/// `find_fractional_mv_step`. rvp9's scalar reference uses a simple
/// 3-step diamond (8 neighbours per step, halving step size) and
/// returns integer-pel MVs only. This is enough to produce a
/// representative ARF in test fixtures; production parity needs the
/// rvp9 `mv_search` module to grow a temporal-filter-shaped wrapper
/// (deferred to Wave 4+).
///
/// Returns `(best_mv_q3, blk_mvs_q3, blk_bestsme)` where
/// `blk_bestsme[k]` is the per-sub-block SAD (in libvpx that's an
/// SSE; we use SAD because the rvp9 mv_search reference is SAD-based).
#[allow(clippy::too_many_arguments)]
fn find_matching_mb(
    arf_y: &[u8],
    arf_off: usize,
    cand_y: &[u8],
    cand_off: usize,
    stride: usize,
    frame_width: usize,
    frame_height: usize,
    block_x: usize,
    block_y: usize,
) -> (Mv, [Mv; NUM_SUB_BLOCKS], [u32; NUM_SUB_BLOCKS]) {
    // Full-block 32x32 search. Step sequence (4, 2, 1) at integer
    // pel — small enough to keep the reference fast, large enough
    // to cover the typical libvpx MESH range. libvpx uses MESH
    // (which can reach quite far); rvp9 truncates by design.
    let (best_dy, best_dx) = diamond_search_int(
        arf_y,
        arf_off,
        cand_y,
        cand_off,
        stride,
        frame_width,
        frame_height,
        block_x,
        block_y,
        BW,
        BH,
        0,
        0,
        &[4, 2, 1],
    );
    let best_mv = Mv {
        row: (best_dy * 8) as i16,
        col: (best_dx * 8) as i16,
    };

    // 4 × 16×16 sub-blocks, seeded from the parent MV. libvpx
    // `vp9_temporal_filter.c:613-636`.
    let mut blk_mvs = [Mv::default(); NUM_SUB_BLOCKS];
    let mut blk_sme = [0u32; NUM_SUB_BLOCKS];
    let mut k = 0;
    for i in (0..BH).step_by(SUB_BH) {
        for j in (0..BW).step_by(SUB_BW) {
            let (dy, dx) = diamond_search_int(
                arf_y,
                arf_off + i * stride + j,
                cand_y,
                cand_off + i * stride + j,
                stride,
                frame_width,
                frame_height,
                block_x + j,
                block_y + i,
                SUB_BW,
                SUB_BH,
                best_dy,
                best_dx,
                &[2, 1],
            );
            blk_mvs[k] = Mv {
                row: (dy * 8) as i16,
                col: (dx * 8) as i16,
            };
            // SAD at the chosen MV — clamped to frame bounds.
            blk_sme[k] = sad_at_offset_clamped(
                arf_y,
                arf_off + i * stride + j,
                cand_y,
                cand_off + i * stride + j,
                stride,
                frame_width,
                frame_height,
                block_x + j,
                block_y + i,
                SUB_BW,
                SUB_BH,
                dy,
                dx,
            );
            k += 1;
        }
    }
    (best_mv, blk_mvs, blk_sme)
}

#[allow(clippy::too_many_arguments)]
fn diamond_search_int(
    arf: &[u8],
    arf_off: usize,
    cand: &[u8],
    cand_off: usize,
    stride: usize,
    fw: usize,
    fh: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    start_dy: i32,
    start_dx: i32,
    steps: &[i32],
) -> (i32, i32) {
    let mut best_dy = start_dy;
    let mut best_dx = start_dx;
    let mut best_sad = sad_at_offset_clamped(
        arf, arf_off, cand, cand_off, stride, fw, fh, bx, by, bw, bh, best_dy, best_dx,
    );
    for &step in steps {
        // 8-neighbour diamond.
        let candidates = [
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (-step, step),
            (step, -step),
            (step, step),
        ];
        for (rdy, rdx) in candidates {
            let dy = best_dy + rdy;
            let dx = best_dx + rdx;
            let s = sad_at_offset_clamped(
                arf, arf_off, cand, cand_off, stride, fw, fh, bx, by, bw, bh, dy, dx,
            );
            if s < best_sad {
                best_sad = s;
                best_dy = dy;
                best_dx = dx;
            }
        }
    }
    (best_dy, best_dx)
}

#[allow(clippy::too_many_arguments)]
fn sad_at_offset_clamped(
    arf: &[u8],
    arf_off: usize,
    cand: &[u8],
    _cand_off: usize,
    stride: usize,
    fw: usize,
    fh: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    dy: i32,
    dx: i32,
) -> u32 {
    let mut total: u32 = 0;
    for y in 0..bh {
        let sy = (by as i32 + dy + y as i32).clamp(0, fh as i32 - 1) as usize;
        let ay = (by + y).min(fh - 1);
        for x in 0..bw {
            let sx = (bx as i32 + dx + x as i32).clamp(0, fw as i32 - 1) as usize;
            let ax = (bx + x).min(fw - 1);
            // ARF read at (ax, ay), candidate read at (sx, sy).
            // Both offsets are in source coords; we ignore the
            // pre-supplied offsets because we compute fresh
            // coordinates from `bx, by, dx, dy` to handle edge
            // clamping uniformly.
            let _ = (arf_off, _cand_off);
            let a = arf[ay * stride + ax];
            let c = cand[sy * stride + sx];
            total += (i32::from(a) - i32::from(c)).unsigned_abs();
        }
    }
    total
}

/// Run the temporal filter over one row of 32×32 blocks. Faithful
/// port of libvpx `vp9_temporal_filter_iterate_row_c`
/// (`vp9_temporal_filter.c:645-996`).
///
/// `mb_row` is the row of 32×32 blocks (in libvpx units of `BH`).
/// `mb_col_start` / `mb_col_end` define the column range — libvpx
/// uses these for tile slicing; rvp9's reference covers the full
/// frame width.
///
/// Writes to `out_y` / `out_u` / `out_v` at the same offsets the
/// ARF source occupies. Uses [`fixed_divide_table`] to normalise
/// `accumulator / count` per pixel (libvpx lines 905-947).
#[allow(clippy::too_many_arguments)]
pub fn temporal_filter_iterate_row(
    data: &ArnrFilterData<'_>,
    mb_row: usize,
    mb_col_start: usize,
    mb_col_end: usize,
    fixed_divide: &[u32; 512],
    out_y: &mut [u8],
    out_u: &mut [u8],
    out_v: &mut [u8],
) {
    let frames = &data.frames;
    let frame_count = data.frame_count;
    let alt_ref_index = data.alt_ref_index;
    let strength = i32::from(data.strength);

    let arf = frames[alt_ref_index];
    let y_stride = arf.y_stride;
    let uv_stride = arf.uv_stride;
    let frame_width = arf.width as usize;
    let frame_height = arf.height as usize;
    let mb_uv_height = BH / 2;
    let mb_uv_width = BW / 2;

    let mut mb_y_offset = mb_row * BH * y_stride + BW * mb_col_start;
    let mut mb_uv_offset = mb_row * mb_uv_height * uv_stride + mb_uv_width * mb_col_start;

    for mb_col in mb_col_start..mb_col_end {
        let block_x = mb_col * BW;
        let block_y = mb_row * BH;
        let visible_width = BW.min(frame_width - block_x);
        let visible_height = BH.min(frame_height - block_y);

        // libvpx filters full macroblocks against its replicated source
        // border (vp9_temporal_filter.c:802-846). Materialise that border
        // only for a partial block; motion search/prediction already clamp
        // their reads to the visible source dimensions.
        let mut edge_source = [0u8; BLK_PELS * 3];
        let partial = visible_width != BW || visible_height != BH;
        if partial {
            temporal_filter_predictors_mb(
                arf.y,
                arf.u,
                arf.v,
                y_stride,
                uv_stride,
                frame_width,
                frame_height,
                block_x,
                block_y,
                0,
                0,
                &[Mv::default(); NUM_SUB_BLOCKS],
                true,
                &mut edge_source,
            );
        }

        let mut accumulator = vec![0u32; BLK_PELS * 3];
        let mut count = vec![0u16; BLK_PELS * 3];

        for (frame_index, cand) in frames.iter().enumerate().take(frame_count) {
            // libvpx `vp9_temporal_filter.c:738-750` — per-frame
            // init.
            let mut ref_mv = Mv::default();
            let mut blk_mvs = [Mv::default(); NUM_SUB_BLOCKS];
            let mut blk_fw = [0i32; NUM_SUB_BLOCKS];
            let use_32x32;

            if frame_index == alt_ref_index {
                // libvpx `vp9_temporal_filter.c:752-754` — self-blend
                // path: zero MV, weight=2, use_32x32=true.
                blk_fw = [2, 2, 2, 2];
                use_32x32 = true;
            } else {
                let thresh_low = 10000i32;
                let thresh_high = 20000i32;
                let (best, sub_mvs, sub_sme) = find_matching_mb(
                    arf.y,
                    mb_y_offset,
                    cand.y,
                    mb_y_offset,
                    y_stride,
                    frame_width,
                    frame_height,
                    block_x,
                    block_y,
                );
                ref_mv = best;
                blk_mvs = sub_mvs;

                let err = sad_at_offset_clamped(
                    arf.y,
                    mb_y_offset,
                    cand.y,
                    mb_y_offset,
                    y_stride,
                    frame_width,
                    frame_height,
                    block_x,
                    block_y,
                    BW,
                    BH,
                    i32::from(best.row) / 8,
                    i32::from(best.col) / 8,
                ) as i32;
                let err16: i32 = sub_sme.iter().map(|&s| s as i32).sum();
                let mut max_err = i32::MIN;
                let mut min_err = i32::MAX;
                for &s in &sub_sme {
                    let s = s as i32;
                    if s < min_err {
                        min_err = s;
                    }
                    if s > max_err {
                        max_err = s;
                    }
                }
                // libvpx `vp9_temporal_filter.c:774-790` — 32x32 vs
                // 4×16x16 decision.
                const THR_SHIFT: i32 = 2;
                if (err.saturating_mul(15) < (err16 << 4) && max_err - min_err < 10000)
                    || (err.saturating_mul(14) < (err16 << 4) && max_err - min_err < 5000)
                {
                    use_32x32 = true;
                    let w = if err < (thresh_low << THR_SHIFT) {
                        2
                    } else if err < (thresh_high << THR_SHIFT) {
                        1
                    } else {
                        0
                    };
                    blk_fw = [w, w, w, w];
                } else {
                    use_32x32 = false;
                    for k in 0..NUM_SUB_BLOCKS {
                        let s = sub_sme[k] as i32;
                        blk_fw[k] = if s < thresh_low {
                            2
                        } else if s < thresh_high {
                            1
                        } else {
                            0
                        };
                    }
                }

                // libvpx `vp9_temporal_filter.c:792-799` — distance
                // attenuation.
                let dist = (frame_index as i32 - alt_ref_index as i32).unsigned_abs() as usize;
                for fw in blk_fw.iter_mut().take(NUM_SUB_BLOCKS) {
                    match dist {
                        1 => *fw = (*fw).min(2),
                        2 | 3 => *fw = (*fw).min(1),
                        _ => {}
                    }
                }
            }

            if blk_fw[0] | blk_fw[1] | blk_fw[2] | blk_fw[3] != 0 {
                // libvpx `vp9_temporal_filter.c:802-810` — build
                // predictor.
                let mut predictor = [0u8; BLK_PELS * 3];
                temporal_filter_predictors_mb(
                    cand.y,
                    cand.u,
                    cand.v,
                    y_stride,
                    uv_stride,
                    frame_width,
                    frame_height,
                    block_x,
                    block_y,
                    ref_mv.row,
                    ref_mv.col,
                    &blk_mvs,
                    use_32x32,
                    &mut predictor,
                );

                let (arf_y_block, arf_u_block, arf_v_block, source_y_stride, source_uv_stride) =
                    if partial {
                        (
                            &edge_source[..BLK_PELS],
                            &edge_source[BLK_PELS..2 * BLK_PELS],
                            &edge_source[2 * BLK_PELS..],
                            BW,
                            mb_uv_width,
                        )
                    } else {
                        (
                            &arf.y[mb_y_offset..],
                            &arf.u[mb_uv_offset..],
                            &arf.v[mb_uv_offset..],
                            y_stride,
                            uv_stride,
                        )
                    };

                let (pred_y, pred_uv) = predictor.split_at(BLK_PELS);
                let (pred_u, pred_v) = pred_uv.split_at(BLK_PELS);

                let (acc_y, acc_uv) = accumulator.split_at_mut(BLK_PELS);
                let (acc_u, acc_v) = acc_uv.split_at_mut(BLK_PELS);
                let (cnt_y, cnt_uv) = count.split_at_mut(BLK_PELS);
                let (cnt_u, cnt_v) = cnt_uv.split_at_mut(BLK_PELS);

                apply_temporal_filter(
                    arf_y_block,
                    source_y_stride,
                    pred_y,
                    BW,
                    arf_u_block,
                    arf_v_block,
                    source_uv_stride,
                    pred_u,
                    pred_v,
                    mb_uv_width,
                    BW,
                    BH,
                    1,
                    1, // 4:2:0
                    strength,
                    &blk_fw,
                    use_32x32,
                    acc_y,
                    cnt_y,
                    acc_u,
                    cnt_u,
                    acc_v,
                    cnt_v,
                );
            }
        }

        // libvpx `vp9_temporal_filter.c:905-991` — normalise
        // accumulator to 8-bit output.
        let mut byte = mb_y_offset;
        let mut k = 0usize;
        for i in 0..BH {
            for j in 0..BW {
                let c = count[k] as usize;
                let pval = if c > 0 {
                    let v = (accumulator[k] + (count[k] as u32 >> 1)) * fixed_divide[c];
                    (v >> 19) as u8
                } else {
                    0
                };
                if i < visible_height && j < visible_width {
                    out_y[byte] = pval;
                }
                byte += 1;
                k += 1;
            }
            byte = mb_y_offset + (i + 1) * y_stride;
        }

        let mut byte = mb_uv_offset;
        let mut k = BLK_PELS;
        for i in 0..mb_uv_height {
            for j in 0..mb_uv_width {
                let m = k + BLK_PELS;
                let cu = count[k] as usize;
                let cv = count[m] as usize;
                let pu = if cu > 0 {
                    let v = (accumulator[k] + (count[k] as u32 >> 1)) * fixed_divide[cu];
                    (v >> 19) as u8
                } else {
                    0
                };
                let pv = if cv > 0 {
                    let v = (accumulator[m] + (count[m] as u32 >> 1)) * fixed_divide[cv];
                    (v >> 19) as u8
                } else {
                    0
                };
                if i < visible_height / 2 && j < visible_width / 2 {
                    out_u[byte] = pu;
                    out_v[byte] = pv;
                }
                byte += 1;
                k += 1;
            }
            byte = mb_uv_offset + (i + 1) * uv_stride;
        }

        mb_y_offset += BW;
        mb_uv_offset += mb_uv_width;
    }
}

/// Top-level dispatcher — build the filtered ARF from `data`. Mirrors
/// libvpx `vp9_temporal_filter` (`vp9_temporal_filter.c:1104-1205`)
/// with the deviations noted at the top of the module.
///
/// The caller is expected to have already chosen the ARF source
/// (centre frame) and the participating neighbours: pack them into
/// `data.frames` with `data.alt_ref_index` pointing at the centre.
/// `data.strength` should come from [`adjust_arnr_filter`].
///
/// # Panics
///
/// Panics if `data.frames` is empty, dimensions differ across
/// frames, or dimensions are zero or odd. Partial macroblocks use
/// replicated source edges, matching libvpx's source frame border.
#[must_use]
pub fn vp9_temporal_filter(data: &ArnrFilterData<'_>) -> FilteredArnrFrame {
    assert!(!data.frames.is_empty(), "temporal filter needs ≥1 frame");
    let arf = data.frames[data.alt_ref_index];
    let width = arf.width as usize;
    let height = arf.height as usize;
    assert!(
        width > 0 && height > 0 && width.is_multiple_of(2) && height.is_multiple_of(2),
        "rvp9 temporal filter requires positive even 4:2:0 dimensions"
    );
    for f in &data.frames {
        assert_eq!(f.width, arf.width, "all frames must share width");
        assert_eq!(f.height, arf.height, "all frames must share height");
    }

    let uv_w = width / 2;
    let uv_h = height / 2;
    // The row API writes using the source strides. Pack those rows back
    // to the tight-stride output promised by FilteredArnrFrame below.
    let mut out_y = vec![0u8; arf.y_stride * height];
    let mut out_u = vec![0u8; arf.uv_stride * uv_h];
    let mut out_v = vec![0u8; arf.uv_stride * uv_h];

    let mb_rows = height.div_ceil(BH);
    let mb_cols = width.div_ceil(BW);
    let fixed_divide = fixed_divide_table();

    for mb_row in 0..mb_rows {
        temporal_filter_iterate_row(
            data,
            mb_row,
            0,
            mb_cols,
            &fixed_divide,
            &mut out_y,
            &mut out_u,
            &mut out_v,
        );
    }

    fn pack_rows(plane: Vec<u8>, stride: usize, width: usize) -> Vec<u8> {
        if stride == width {
            plane
        } else {
            plane
                .chunks_exact(stride)
                .flat_map(|row| row[..width].iter().copied())
                .collect()
        }
    }

    FilteredArnrFrame {
        y: pack_rows(out_y, arf.y_stride, width),
        u: pack_rows(out_u, arf.uv_stride, uv_w),
        v: pack_rows(out_v, arf.uv_stride, uv_w),
        width: arf.width,
        height: arf.height,
    }
}

// =====================================================================
// PROOF tests.
// =====================================================================

#[cfg(test)]
mod tests {
    //! PROOF tests for W3.1 – W3.5 — the temporal-filter cluster.
    use super::*;

    /// W3.1 PROOF — [`ArnrFilterData::new`] holds N frames and stamps
    /// the right shape. Mirrors libvpx `ARNRFilterData` field set
    /// (`vp9_encoder.h:499-505`).
    #[test]
    fn w3_1_arnr_filter_data_alloc_holds_frames() {
        // 64×64 frames (2 MB rows × 2 MB cols).
        let w = 64u32;
        let h = 64u32;
        let make = |seed: u8| {
            let y = vec![seed; (w * h) as usize];
            let u = vec![128u8; (w / 2 * h / 2) as usize];
            let v = vec![128u8; (w / 2 * h / 2) as usize];
            (y, u, v)
        };
        let (y0, u0, v0) = make(50);
        let (y1, u1, v1) = make(60);
        let (y2, u2, v2) = make(70);
        let f0 = ArnrFrameRef {
            y: &y0,
            u: &u0,
            v: &v0,
            y_stride: w as usize,
            uv_stride: (w / 2) as usize,
            width: w,
            height: h,
        };
        let f1 = ArnrFrameRef {
            y: &y1,
            u: &u1,
            v: &v1,
            y_stride: w as usize,
            uv_stride: (w / 2) as usize,
            width: w,
            height: h,
        };
        let f2 = ArnrFrameRef {
            y: &y2,
            u: &u2,
            v: &v2,
            y_stride: w as usize,
            uv_stride: (w / 2) as usize,
            width: w,
            height: h,
        };
        let data = ArnrFilterData::new(vec![f0, f1, f2], 1, 3);
        assert_eq!(data.frame_count, 3);
        assert_eq!(data.frames.len(), 3);
        assert_eq!(data.alt_ref_index, 1);
        assert_eq!(data.strength, 3);
        assert_eq!(data.filter_qindex, ARNR_FILT_QINDEX);
    }

    /// W3.2 PROOF — `adjust_arnr_filter` clamps strength to `[0, 6]`
    /// for representative inputs. libvpx clamps base_strength to
    /// `[0,6]` at `vp9_temporal_filter.c:1045` (pass-2 path) and
    /// derived `strength` is rate-limited by `group_boost / 300` at
    /// line 1083-1085. We test the clamp explicitly.
    #[test]
    fn w3_2_adjust_arnr_filter_clamps_strength() {
        // Case 1: caller passes oversized arnr_strength=10 — should
        // be clipped to 6 by our defensive clamp (matches libvpx's
        // pass-2 clamp at line 1045).
        let p = adjust_arnr_filter(
            /*arnr_max_frames=*/ 7, /*arnr_strength=*/ 10, /*distance=*/ 4,
            /*frames_after_arf=*/ 10, /*group_boost=*/ 10_000,
            /*inter_qindex=*/ 100, /*key_qindex=*/ 100,
            /*current_video_frame_gt_1=*/ true,
            /*arf_src_offset_lt_baseline_minus1=*/ false,
        );
        assert!(p.strength <= 6, "strength must be ≤ 6 ({})", p.strength);

        // Case 2: low Q (qindex=0 → q=1) with `current_video_frame_gt_1`
        // forces the line 1073 branch `strength = base_strength -
        // ((16 - q) / 2)`, which can underflow. We clamp to 0
        // (matches the `if (strength < 0) strength = 0` at line
        // 1074).
        let p = adjust_arnr_filter(7, 3, 1, 10, 10_000, 0, 0, true, false);
        assert!(p.strength <= 6);
        // base=3, q=1, strength = 3 - (15/2) = 3 - 7 = -4 → 0.
        assert_eq!(p.strength, 0);

        // Case 3: group_boost is small so the line 1083-1085 cap
        // fires (strength = group_boost / 300).
        let p = adjust_arnr_filter(7, 6, 1, 10, 600, 200, 200, true, false);
        // q = ac_qlookup[200]/4 = 432/4 = 108 > 16 ⇒ strength=6.
        // Then 6 > 600/300 = 2 ⇒ strength clamped to 2.
        assert_eq!(p.strength, 2);
    }

    /// W3.3 PROOF — predictor with `mv=(0,0)` is bit-exact integer
    /// copy. libvpx: `inter_predictor` dispatches to the [0][0]
    /// (copy) entry when `subpel_x == 0 && subpel_y == 0`
    /// (`vp9_reconinter.h:29`).
    #[test]
    fn w3_3_predictors_mb_round_trip_at_zero_mv() {
        // 64×64 frame; pick the 32×32 block at (16, 16).
        let w = 64usize;
        let h = 64usize;
        let mut y = vec![0u8; w * h];
        for i in 0..h {
            for j in 0..w {
                // Distinctive pattern.
                y[i * w + j] = ((i * 7 + j * 13) & 0xff) as u8;
            }
        }
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![64u8; (w / 2) * (h / 2)];
        let mut pred = [0u8; BLK_PELS * 3];
        temporal_filter_predictors_mb(
            &y,
            &u,
            &v,
            w,
            w / 2,
            w,
            h,
            /*block_x=*/ 16,
            /*block_y=*/ 16,
            /*mv_row_q3=*/ 0,
            /*mv_col_q3=*/ 0,
            &[Mv::default(); NUM_SUB_BLOCKS],
            /*use_32x32=*/ true,
            &mut pred,
        );
        // Verify luma round-trip.
        for i in 0..BH {
            for j in 0..BW {
                let expected = y[(16 + i) * w + (16 + j)];
                let got = pred[i * BW + j];
                assert_eq!(
                    got, expected,
                    "luma mismatch at ({},{}): expected {expected}, got {got}",
                    i, j
                );
            }
        }
        // Verify chroma round-trip — chroma origin is (8, 8) in
        // 4:2:0 (= luma origin / 2). Chroma block is 16×16.
        for i in 0..BH / 2 {
            for j in 0..BW / 2 {
                let expected_u = u[(8 + i) * (w / 2) + (8 + j)];
                let expected_v = v[(8 + i) * (w / 2) + (8 + j)];
                let got_u = pred[BLK_PELS + i * (BW / 2) + j];
                let got_v = pred[2 * BLK_PELS + i * (BW / 2) + j];
                assert_eq!(got_u, expected_u, "U mismatch at ({i},{j})");
                assert_eq!(got_v, expected_v, "V mismatch at ({i},{j})");
            }
        }
    }

    /// W3.4 PROOF — `strength == 0` makes the NLM filter a no-op
    /// (accumulator untouched). libvpx `vp9_apply_temporal_filter_c`
    /// at `vp9_temporal_filter.c:270`: `rounding = (1 << 0) >> 1 =
    /// 0`. Then in `mod_index` (line 202-207): `mod = clamp(sum_dist,
    /// 0, U16) * index_mult[index] >> 16 + 0 >> 0`. Result is
    /// bounded by 16 (line 207), capped at exactly 16 when the
    /// strength shift is 0, giving `mod = 16 - 16 = 0` after line
    /// 209. Then `filter_weight * 0 = 0`. So the accumulator is
    /// untouched, regardless of source vs pred.
    ///
    /// NB: This is libvpx-faithful — the [0,6] strength range is
    /// asserted (line 279-280). `strength=0` is the lower bound and
    /// represents "filter as gently as possible"; the math collapses
    /// to a no-op in that edge case.
    #[test]
    fn w3_4_apply_temporal_filter_zero_strength_passthrough() {
        let bw = BW;
        let bh = BH;
        let y_src = vec![100u8; bw * bh];
        let y_pred = vec![200u8; bw * bh]; // big difference!
        let u_src = vec![50u8; (bw / 2) * (bh / 2)];
        let v_src = vec![60u8; (bw / 2) * (bh / 2)];
        let u_pred = vec![150u8; (bw / 2) * (bh / 2)];
        let v_pred = vec![160u8; (bw / 2) * (bh / 2)];

        let mut y_acc = vec![0u32; bw * bh];
        let mut y_cnt = vec![0u16; bw * bh];
        let mut u_acc = vec![0u32; (bw / 2) * (bh / 2)];
        let mut u_cnt = vec![0u16; (bw / 2) * (bh / 2)];
        let mut v_acc = vec![0u32; (bw / 2) * (bh / 2)];
        let mut v_cnt = vec![0u16; (bw / 2) * (bh / 2)];

        apply_temporal_filter(
            &y_src,
            bw,
            &y_pred,
            bw,
            &u_src,
            &v_src,
            bw / 2,
            &u_pred,
            &v_pred,
            bw / 2,
            bw,
            bh,
            1,
            1,
            /*strength=*/ 0,
            &[2, 2, 2, 2],
            /*use_32x32=*/ true,
            &mut y_acc,
            &mut y_cnt,
            &mut u_acc,
            &mut u_cnt,
            &mut v_acc,
            &mut v_cnt,
        );

        // strength=0 + large diff: modifier = 0, so accumulator
        // stays zero, count stays zero.
        assert!(
            y_acc.iter().all(|&v| v == 0),
            "y_acc should be untouched at strength=0"
        );
        assert!(
            y_cnt.iter().all(|&v| v == 0),
            "y_cnt should be untouched at strength=0"
        );
        assert!(u_acc.iter().all(|&v| v == 0));
        assert!(v_acc.iter().all(|&v| v == 0));
        assert!(u_cnt.iter().all(|&v| v == 0));
        assert!(v_cnt.iter().all(|&v| v == 0));
    }

    /// W3.5 PROOF — pipeline smoke: 3 identical frames in ⇒ output
    /// ≈ input (modulo `accumulator/count` rounding from the
    /// fixed_divide table). libvpx normalises via
    /// `vp9_temporal_filter.c:910-914`:
    /// `pval = ((accumulator + count/2) * fixed_divide[count]) >> 19`,
    /// which approximates `accumulator / count`. With identical
    /// frames the modifier per-pixel-per-frame is `16 * 2 = 32`
    /// (max weight from `blk_fw[k] = 2`), so total count =
    /// `3 frames × 32 = 96`; the rounded division equals the source
    /// pixel within ±1 due to integer fixed_divide.
    #[test]
    fn w3_5_vp9_temporal_filter_pipeline_passes_smoke() {
        let w = 64u32;
        let h = 64u32;
        let mut y = vec![0u8; (w * h) as usize];
        for i in 0..h as usize {
            for j in 0..w as usize {
                y[i * (w as usize) + j] = ((i * 5 + j * 7) & 0xff) as u8;
            }
        }
        let u = vec![100u8; (w / 2 * h / 2) as usize];
        let v = vec![150u8; (w / 2 * h / 2) as usize];

        let f = ArnrFrameRef {
            y: &y,
            u: &u,
            v: &v,
            y_stride: w as usize,
            uv_stride: (w / 2) as usize,
            width: w,
            height: h,
        };
        let frames = vec![f, f, f];
        let data = ArnrFilterData::new(frames, 1, 3);
        let out = vp9_temporal_filter(&data);

        assert_eq!(out.width, w);
        assert_eq!(out.height, h);
        assert_eq!(out.y.len(), (w * h) as usize);

        // Every pixel within ±1 of the source.
        let mut max_err = 0i32;
        for i in 0..h as usize {
            for j in 0..w as usize {
                let src = i32::from(y[i * (w as usize) + j]);
                let dst = i32::from(out.y[i * (w as usize) + j]);
                max_err = max_err.max((src - dst).abs());
            }
        }
        assert!(
            max_err <= 1,
            "identical-frame pipeline must round-trip within ±1; got max_err = {max_err}"
        );
        // Chroma planes too.
        for &uv_pixel in &out.u {
            assert!(
                (i32::from(uv_pixel) - 100).abs() <= 1,
                "U mismatch: got {uv_pixel} expected ~100"
            );
        }
        for &uv_pixel in &out.v {
            assert!(
                (i32::from(uv_pixel) - 150).abs() <= 1,
                "V mismatch: got {uv_pixel} expected ~150"
            );
        }
    }

    /// W3.4 PROOF (bonus) — `mod_index` table sanity. libvpx
    /// `vp9_temporal_filter.c:195-213`: `index_mult[4..=11, 13]`
    /// are non-zero. We assert the libvpx-canonical entries match.
    #[test]
    fn w3_4_mod_index_table_matches_libvpx() {
        assert_eq!(INDEX_MULT[4], 49152);
        assert_eq!(INDEX_MULT[5], 39322);
        assert_eq!(INDEX_MULT[6], 32768);
        assert_eq!(INDEX_MULT[7], 28087);
        assert_eq!(INDEX_MULT[8], 24576);
        assert_eq!(INDEX_MULT[9], 21846);
        assert_eq!(INDEX_MULT[10], 19661);
        assert_eq!(INDEX_MULT[11], 17874);
        assert_eq!(INDEX_MULT[13], 15124);
    }
}
