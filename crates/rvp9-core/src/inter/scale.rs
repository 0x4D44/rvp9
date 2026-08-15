//! Reference-frame scale factors and the per-block `inter_predictor`
//! driver.
//!
//! When the reference frame has a different resolution to the current
//! frame (the resize feature, or spatial SVC's lower-layer ref), the
//! decoder maps current-frame coordinates and motion vectors into the
//! reference using a Q14 fixed-point scaling factor. This module ports
//! libvpx's small `vp9_setup_scale_factors_for_frame` /
//! `vp9_scale_mv` machinery (`vp9/common/vp9_scale.{h,c}`) plus the
//! `inter_predictor` dispatcher (`vp9/common/vp9_reconinter.h:23-32`)
//! that picks the right convolve kernel given subpel offsets, scaling,
//! and compound-average flag.
//!
//! ## Scope (Task C.2)
//!
//! Pure data + dispatch. Phase D will:
//!
//! 1. Build a [`ScaleFactors`] per reference frame at frame-header
//!    decode time (`vp9_decodeframe.c:1559-1606`).
//! 2. Per inter block, compute `(src, src_stride, subpel_x_q4,
//!    subpel_y_q4)` from `(ref_buf, mv_q4, block_origin)` — see the
//!    snippet in [`inter_predictor`]'s doc.
//! 3. Hand it all to [`inter_predictor`].
//!
//! Border extension (the next sub-task, C.3) reshapes how Phase D
//! computes the integer-pel base offset when the MV reaches off the
//! reference edge; the scale-factor maths here is unaffected — it
//! generates a sample position in the ref's coordinate system, and
//! C.3 will handle reads beyond `[0, ref_w) × [0, ref_h)`.

use super::filter::{
    SUBPEL_BITS, SUBPEL_MASK, SUBPEL_SHIFTS, convolve8_scaled_avg_horiz, convolve8_scaled_avg_vert,
    convolve8_scaled_horiz, convolve8_scaled_vert, copy_edge_replicated_window, kernel_for,
};
use crate::bitstream::frame::InterpFilter;

/// libvpx `REF_SCALE_SHIFT` (`vp9_scale.h:21`). Q14 fixed-point shift
/// used for [`ScaleFactors`].
pub const REF_SCALE_SHIFT: u32 = 14;

/// libvpx `REF_NO_SCALE` (`vp9_scale.h:22`). `1 << REF_SCALE_SHIFT`,
/// the fixed-point value for a unit scale factor (ref dimension ==
/// cur dimension).
pub const REF_NO_SCALE: i32 = 1 << REF_SCALE_SHIFT;

/// libvpx `REF_INVALID_SCALE` (`vp9_scale.h:23`). Sentinel value for
/// reference frames whose dimensions fail
/// [`valid_ref_frame_size`].
pub const REF_INVALID_SCALE: i32 = -1;

/// Per-reference-frame Q14 fixed-point scale factor.
///
/// libvpx `struct scale_factors` (`vp9_scale.h:25-38`), minus the
/// function-pointer dispatch tables — Rust handles that via the match
/// in [`inter_predictor`].
///
/// Each axis carries:
///
/// - `*_scale_fp`: `(ref_dim << REF_SCALE_SHIFT) / cur_dim`, the Q14
///   ratio mapping current-frame coordinates to reference-frame
///   coordinates. Equals [`REF_NO_SCALE`] for a unit scale,
///   [`REF_INVALID_SCALE`] when the (ref, cur) pair fails the
///   `valid_ref_frame_size` envelope (libvpx `valid_ref_frame_size`,
///   `vp9_scale.h:61-65`).
/// - `*_step_q4`: `scaled_value(SUBPEL_SHIFTS) = scale_fp >>
///   (REF_SCALE_SHIFT - SUBPEL_BITS)`. The per-sample Q4 increment
///   threaded into the scaled convolve kernels. Equals 16 for a unit
///   scale.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ScaleFactors {
    /// Q14 horizontal scale factor: `(ref_w << 14) / cur_w`.
    /// [`REF_INVALID_SCALE`] if the ref/cur pair is out of range.
    pub x_scale_fp: i32,
    /// Q14 vertical scale factor.
    pub y_scale_fp: i32,
    /// Q4 horizontal per-sample step (libvpx `sf->x_step_q4`).
    pub x_step_q4: i32,
    /// Q4 vertical per-sample step (libvpx `sf->y_step_q4`).
    pub y_step_q4: i32,
}

impl ScaleFactors {
    /// Build a [`ScaleFactors`] from a (ref, cur) dimension pair.
    ///
    /// Mirrors libvpx `vp9_setup_scale_factors_for_frame`
    /// (`vp9_scale.c:46-71`). If the pair fails
    /// [`valid_ref_frame_size`], both `x_scale_fp` and `y_scale_fp`
    /// are set to [`REF_INVALID_SCALE`] and the `_step_q4` fields are
    /// left at 0 — exactly mirroring libvpx's early-return path.
    /// Callers should test [`Self::is_valid`] before consuming the
    /// step fields.
    #[must_use]
    pub fn new(ref_w: u32, ref_h: u32, cur_w: u32, cur_h: u32) -> Self {
        if !valid_ref_frame_size(ref_w, ref_h, cur_w, cur_h) {
            return Self {
                x_scale_fp: REF_INVALID_SCALE,
                y_scale_fp: REF_INVALID_SCALE,
                x_step_q4: 0,
                y_step_q4: 0,
            };
        }
        let x_scale_fp = fixed_point_scale_factor(ref_w, cur_w);
        let y_scale_fp = fixed_point_scale_factor(ref_h, cur_h);
        // libvpx: `sf->x_step_q4 = scaled_x(16, sf)` =
        // `(16 * x_scale_fp) >> 14` = `x_scale_fp >> 10`. Using the
        // generic `scaled_value` here keeps the code paths uniform.
        let x_step_q4 = scaled_value(SUBPEL_SHIFTS as i32, x_scale_fp);
        let y_step_q4 = scaled_value(SUBPEL_SHIFTS as i32, y_scale_fp);
        Self {
            x_scale_fp,
            y_scale_fp,
            x_step_q4,
            y_step_q4,
        }
    }

    /// libvpx `vp9_is_valid_scale` (`vp9_scale.h:51-54`). True iff
    /// both axes hold a real (non-[`REF_INVALID_SCALE`]) scale.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.x_scale_fp != REF_INVALID_SCALE && self.y_scale_fp != REF_INVALID_SCALE
    }

    /// libvpx `vp9_is_scaled` (`vp9_scale.h:56-59`). True iff the
    /// scale is valid and at least one axis is non-unit.
    #[must_use]
    pub fn is_scaled(&self) -> bool {
        self.is_valid() && (self.x_scale_fp != REF_NO_SCALE || self.y_scale_fp != REF_NO_SCALE)
    }

    /// Convert a current-frame integer-pel x coordinate plus a Q4 MV
    /// component into the corresponding reference-frame Q4 sub-pel
    /// sample position.
    ///
    /// libvpx splits this across `scaled_buffer_offset` (the integer
    /// part) and `vp9_scale_mv` (the masked fractional + scaled-MV
    /// part). For C.2 we expose the *full* sum — Phase D either uses
    /// it directly (when stepping through scaled convolves) or
    /// decomposes it into `(int_offset, subpel_q4)` for the unscaled
    /// fast path.
    #[must_use]
    pub fn scaled_x_q4(&self, x: i32, mv_q4: i16) -> i32 {
        scaled_value(x << SUBPEL_BITS, self.x_scale_fp)
            + scaled_value(i32::from(mv_q4), self.x_scale_fp)
    }

    /// Vertical counterpart of [`Self::scaled_x_q4`].
    #[must_use]
    pub fn scaled_y_q4(&self, y: i32, mv_q4: i16) -> i32 {
        scaled_value(y << SUBPEL_BITS, self.y_scale_fp)
            + scaled_value(i32::from(mv_q4), self.y_scale_fp)
    }
}

/// libvpx `valid_ref_frame_size` (`vp9_scale.h:61-65`). The spec'd
/// envelope: ref dim must be in `[cur / 16, 2 * cur]` per axis.
/// References outside this range fail `vp9_setup_scale_factors_for_frame`
/// and produce a [`REF_INVALID_SCALE`] [`ScaleFactors`].
#[must_use]
pub fn valid_ref_frame_size(ref_w: u32, ref_h: u32, cur_w: u32, cur_h: u32) -> bool {
    // Promote to u64 so `16 * ref_w` never overflows for the
    // multi-megapixel frame sizes VP9 permits.
    let ref_w = u64::from(ref_w);
    let ref_h = u64::from(ref_h);
    let cur_w = u64::from(cur_w);
    let cur_h = u64::from(cur_h);
    2 * cur_w >= ref_w && 2 * cur_h >= ref_h && cur_w <= 16 * ref_w && cur_h <= 16 * ref_h
}

/// libvpx `get_fixed_point_scale_factor` (`vp9_scale.c:29-35`).
/// `(other << 14) / this`. Caller must ensure `this != 0` — the
/// `valid_ref_frame_size` gate suffices for valid VP9 streams (frame
/// dimensions are encoded with at least 1 in libvpx's `read_frame_size`).
#[inline]
fn fixed_point_scale_factor(other: u32, this: u32) -> i32 {
    debug_assert!(this != 0, "fixed_point_scale_factor: this == 0");
    // libvpx uses `(other_size << 14) / this_size` in signed int
    // arithmetic. Cur/ref widths fit comfortably in i32 even after
    // the 14-bit shift (VP9 caps frame size at 65536, so the
    // shifted value is ≤ 65536 << 14 = 2^30).
    ((other as i64) << REF_SCALE_SHIFT) as i32 / this as i32
}

/// libvpx static helper `scaled_x` / `scaled_y` (`vp9_scale.c:16-22`).
/// `(val * scale_fp) >> REF_SCALE_SHIFT`, with the multiply widened to
/// 64-bit to match libvpx's `(int64_t)val * sf->x_scale_fp`.
#[inline]
fn scaled_value(val: i32, scale_fp: i32) -> i32 {
    ((i64::from(val) * i64::from(scale_fp)) >> REF_SCALE_SHIFT) as i32
}

/// Per-block inter-prediction kernel dispatcher.
///
/// Mirrors libvpx `inter_predictor` (`vp9_reconinter.h:23-32`). Given
/// a reference pointer pre-shifted to the integer-pel base of the MV
/// (i.e. the caller has applied `src += (mv_q4_total >> SUBPEL_BITS) *
/// stride` already), the per-axis sub-pel offset `subpel_*_q4`, and a
/// [`ScaleFactors`] for the ref, picks one of the eight convolve
/// kernels:
///
/// | x filtering | y filtering | selected path              |
/// |-------------|-------------|----------------------------|
/// | no          | no          | [`convolve_copy`]          |
/// | no          | yes         | [`convolve8_scaled_vert`]  |
/// | yes         | no          | [`convolve8_scaled_horiz`] |
/// | yes         | yes         | [`convolve8_scaled`]       |
///
/// The `_avg` variant (`is_compound_avg`) follows the same table with
/// `convolve_avg` / `convolve8_avg_*` / `convolve8_scaled_avg`.
///
/// An axis needs filtering when either its starting sub-pel offset is
/// non-zero or its step differs from [`SUBPEL_SHIFTS`]. This matches
/// libvpx's `vp9_setup_scale_factors_for_frame` table and, critically,
/// the caller's per-axis three-pixel source pre-shift.
///
/// ## Phase D call-site recipe
///
/// ```text
/// // mv_q4 is already in Q4 (1/16-pel) units, in the reference's coord system.
/// let total_x_q4 = sf.scaled_x_q4(block_origin_x, mv_q4.col);
/// let total_y_q4 = sf.scaled_y_q4(block_origin_y, mv_q4.row);
/// let int_x = total_x_q4 >> SUBPEL_BITS;
/// let int_y = total_y_q4 >> SUBPEL_BITS;
/// let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
/// let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;
/// let src = &ref_buf[(int_y as usize) * ref_stride + (int_x as usize)..];
/// inter_predictor(src, ref_stride, dst, dst_stride,
///                 subpel_x_q4, subpel_y_q4, &sf, interp, w, h, false);
/// ```
///
/// Border-extend coordination (C.3): the `int_x`/`int_y` integer-pel
/// base may land outside the reference frame, especially with large
/// MVs. C.3 will widen each ref buffer with an extended border so
/// the `convolve*` reads never fault. C.2 itself does not clamp.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
pub fn inter_predictor(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    subpel_x_q4: i32,
    subpel_y_q4: i32,
    sf: &ScaleFactors,
    interp_filter: InterpFilter,
    w: usize,
    h: usize,
    is_compound_avg: bool,
) {
    let dispatch = crate::dsp::get();
    inter_predictor_with_dispatch(
        src,
        src_stride,
        dst,
        dst_stride,
        subpel_x_q4,
        subpel_y_q4,
        sf,
        interp_filter,
        w,
        h,
        is_compound_avg,
        &dispatch,
    );
}

/// R.16: caller-passes-dispatch variant of [`inter_predictor`]. Hot
/// callers (decoder per-MC path) can use this to skip the per-call
/// `Arc::clone` from `crate::dsp::get()`. The public
/// [`inter_predictor`] just forwards here after acquiring its own
/// dispatch.
///
/// R.13 tried switching `inter_predictor` itself to `get_ref()` and
/// regressed; passing the existing `&SimdDispatch` (which the caller
/// already has in scope) avoids both the Arc atomic AND any
/// codegen-changes that R.13 induced.
#[allow(clippy::too_many_arguments)]
pub fn inter_predictor_with_dispatch(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    subpel_x_q4: i32,
    subpel_y_q4: i32,
    sf: &ScaleFactors,
    interp_filter: InterpFilter,
    w: usize,
    h: usize,
    is_compound_avg: bool,
    dispatch: &crate::dsp::SimdDispatch,
) {
    debug_assert!(
        sf.is_valid(),
        "inter_predictor: scale factor is REF_INVALID_SCALE"
    );
    debug_assert!(
        (0..SUBPEL_SHIFTS as i32).contains(&subpel_x_q4),
        "inter_predictor: subpel_x_q4 = {subpel_x_q4} out of [0, {SUBPEL_SHIFTS})",
    );
    debug_assert!(
        (0..SUBPEL_SHIFTS as i32).contains(&subpel_y_q4),
        "inter_predictor: subpel_y_q4 = {subpel_y_q4} out of [0, {SUBPEL_SHIFTS})",
    );

    let filters = kernel_for(interp_filter);

    let x_step_scaled = sf.x_step_q4 != SUBPEL_SHIFTS as i32;
    let y_step_scaled = sf.y_step_q4 != SUBPEL_SHIFTS as i32;
    if x_step_scaled || y_step_scaled {
        let x_filtering = subpel_x_q4 != 0 || x_step_scaled;
        let y_filtering = subpel_y_q4 != 0 || y_step_scaled;

        match (x_filtering, y_filtering, is_compound_avg) {
            (false, true, false) => convolve8_scaled_vert(
                src,
                src_stride,
                dst,
                dst_stride,
                filters,
                subpel_y_q4,
                sf.y_step_q4,
                w,
                h,
            ),
            (false, true, true) => convolve8_scaled_avg_vert(
                src,
                src_stride,
                dst,
                dst_stride,
                filters,
                subpel_y_q4,
                sf.y_step_q4,
                w,
                h,
            ),
            (true, false, false) => convolve8_scaled_horiz(
                src,
                src_stride,
                dst,
                dst_stride,
                filters,
                subpel_x_q4,
                sf.x_step_q4,
                w,
                h,
            ),
            (true, false, true) => convolve8_scaled_avg_horiz(
                src,
                src_stride,
                dst,
                dst_stride,
                filters,
                subpel_x_q4,
                sf.x_step_q4,
                w,
                h,
            ),
            (true, true, false) => (dispatch.convolve8_scaled())(
                src,
                src_stride,
                dst,
                dst_stride,
                filters,
                subpel_x_q4,
                sf.x_step_q4,
                subpel_y_q4,
                sf.y_step_q4,
                w,
                h,
            ),
            (true, true, true) => (dispatch.convolve8_scaled_avg())(
                src,
                src_stride,
                dst,
                dst_stride,
                filters,
                subpel_x_q4,
                sf.x_step_q4,
                subpel_y_q4,
                sf.y_step_q4,
                w,
                h,
            ),
            (false, false, _) => unreachable!("a scaled step must filter at least one axis"),
        }
        return;
    }

    // Unscaled path: step == SUBPEL_SHIFTS on both axes.
    let x_nonzero = subpel_x_q4 != 0;
    let y_nonzero = subpel_y_q4 != 0;
    let filter_x = &filters[subpel_x_q4 as usize];
    let filter_y = &filters[subpel_y_q4 as usize];

    match (x_nonzero, y_nonzero, is_compound_avg) {
        (false, false, false) => (dispatch.convolve_copy())(src, src_stride, dst, dst_stride, w, h),
        (false, false, true) => (dispatch.convolve_avg())(src, src_stride, dst, dst_stride, w, h),
        (false, true, false) => {
            (dispatch.convolve8_vert())(src, src_stride, dst, dst_stride, filter_y, w, h)
        }
        (false, true, true) => {
            (dispatch.convolve8_avg_vert())(src, src_stride, dst, dst_stride, filter_y, w, h)
        }
        (true, false, false) => {
            (dispatch.convolve8_horiz())(src, src_stride, dst, dst_stride, filter_x, w, h)
        }
        (true, false, true) => {
            (dispatch.convolve8_avg_horiz())(src, src_stride, dst, dst_stride, filter_x, w, h)
        }
        (true, true, false) => {
            (dispatch.convolve8())(src, src_stride, dst, dst_stride, filter_x, filter_y, w, h)
        }
        (true, true, true) => {
            (dispatch.convolve8_avg())(src, src_stride, dst, dst_stride, filter_x, filter_y, w, h)
        }
    }
}

/// Re-export of [`SUBPEL_MASK`] for callers that build sub-pel
/// positions from a [`ScaleFactors`] result. Same value as in
/// [`super::filter`].
pub const SUBPEL_MASK_Q4: i32 = SUBPEL_MASK;

/// Inputs for decoder-exact inter prediction from a tight reference plane.
///
/// `plane_origin` is expressed in the destination plane's coordinates, while
/// `phase_origin` is expressed in luma coordinates. They differ for subsampled
/// chroma, matching libvpx's scaled-reference mapping.
#[derive(Clone, Copy, Debug)]
pub struct TightInterPredictionParams<'a> {
    /// Visible, unbordered reference plane.
    pub reference: &'a [u8],
    /// Reference-plane row stride.
    pub reference_stride: usize,
    /// Visible reference-plane dimensions.
    pub reference_dimensions: (usize, usize),
    /// Current-plane dimensions.
    pub current_dimensions: (usize, usize),
    /// Block origin in current-plane coordinates.
    pub plane_origin: (usize, usize),
    /// Block origin used for fractional scaled phase calculation.
    pub phase_origin: (usize, usize),
    /// Prediction block dimensions.
    pub block_dimensions: (usize, usize),
    /// Motion vector in Q4 units, `(col, row)`.
    pub mv_q4: (i32, i32),
    /// Interpolation filter.
    pub interp_filter: InterpFilter,
    /// Whether to average the prediction into the destination.
    pub is_compound_avg: bool,
}

/// Build one decoder-exact inter prediction from a tight reference plane.
///
/// The function owns scaled-coordinate mapping and visible-edge replication,
/// so encoder mode decision and reconstruction can share the same pixel
/// oracle without materializing a separately filtered analysis reference.
pub fn build_inter_prediction_from_tight_plane(
    params: TightInterPredictionParams<'_>,
    dst: &mut [u8],
    dst_stride: usize,
    border_scratch: &mut [u8],
) {
    const VP9_INTERP_EXTEND: i32 = 4;

    let (ref_w, ref_h) = params.reference_dimensions;
    let (cur_w, cur_h) = params.current_dimensions;
    let (plane_x, plane_y) = params.plane_origin;
    let (phase_x, phase_y) = params.phase_origin;
    let (bw, bh) = params.block_dimensions;
    let (mv_q4_col, mv_q4_row) = params.mv_q4;
    let sf = ScaleFactors::new(ref_w as u32, ref_h as u32, cur_w as u32, cur_h as u32);
    assert!(
        sf.is_valid(),
        "inter reference dimensions must be VP9-valid"
    );
    assert!(
        sf.x_step_q4 <= SUBPEL_SHIFTS as i32 && sf.y_step_q4 <= SUBPEL_SHIFTS as i32,
        "tight-plane prediction only accepts equal-size or lower-layer references",
    );
    assert!(
        params.reference.len()
            >= params
                .reference_stride
                .saturating_mul(ref_h.saturating_sub(1))
                .saturating_add(ref_w),
        "tight reference plane is smaller than its declared dimensions",
    );
    assert!(
        dst.len()
            >= dst_stride
                .saturating_mul(bh.saturating_sub(1))
                .saturating_add(bw),
        "prediction destination is smaller than the requested block",
    );

    let mul_q14 = |value: i32, scale_fp: i32| {
        ((i64::from(value) * i64::from(scale_fp)) >> REF_SCALE_SHIFT) as i32
    };
    let (total_x_q4, total_y_q4) = if sf.is_scaled() {
        let scaled_mv_col = mul_q14(mv_q4_col, sf.x_scale_fp)
            + (mul_q14((phase_x as i32) << SUBPEL_BITS, sf.x_scale_fp) & SUBPEL_MASK);
        let scaled_mv_row = mul_q14(mv_q4_row, sf.y_scale_fp)
            + (mul_q14((phase_y as i32) << SUBPEL_BITS, sf.y_scale_fp) & SUBPEL_MASK);
        let base_int_x = mul_q14(plane_x as i32, sf.x_scale_fp);
        let base_int_y = mul_q14(plane_y as i32, sf.y_scale_fp);
        (
            (base_int_x << SUBPEL_BITS) + scaled_mv_col,
            (base_int_y << SUBPEL_BITS) + scaled_mv_row,
        )
    } else {
        (
            sf.scaled_x_q4(plane_x as i32, mv_q4_col as i16),
            sf.scaled_y_q4(plane_y as i32, mv_q4_row as i16),
        )
    };
    let int_x = total_x_q4 >> SUBPEL_BITS;
    let int_y = total_y_q4 >> SUBPEL_BITS;
    let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
    let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;
    let x_pad = subpel_x_q4 != 0 || sf.x_step_q4 != SUBPEL_SHIFTS as i32;
    let y_pad = subpel_y_q4 != 0 || sf.y_step_q4 != SUBPEL_SHIFTS as i32;
    let last_x_q4 = total_x_q4 + (bw as i32 - 1) * sf.x_step_q4;
    let last_y_q4 = total_y_q4 + (bh as i32 - 1) * sf.y_step_q4;
    let mut x0 = int_x;
    let mut y0 = int_y;
    let mut x1 = (last_x_q4 >> SUBPEL_BITS) + 1;
    let mut y1 = (last_y_q4 >> SUBPEL_BITS) + 1;
    if x_pad {
        x0 -= VP9_INTERP_EXTEND - 1;
        x1 += VP9_INTERP_EXTEND;
    }
    if y_pad {
        y0 -= VP9_INTERP_EXTEND - 1;
        y1 += VP9_INTERP_EXTEND;
    }

    let predict = |src: &[u8], src_stride: usize, dst: &mut [u8]| {
        inter_predictor_with_dispatch(
            src,
            src_stride,
            dst,
            dst_stride,
            subpel_x_q4,
            subpel_y_q4,
            &sf,
            params.interp_filter,
            bw,
            bh,
            params.is_compound_avg,
            crate::dsp::get_ref(),
        );
    };
    let needs_scratch = x0 < 0 || y0 < 0 || x1 > ref_w as i32 || y1 > ref_h as i32;
    if !needs_scratch {
        let row_pre = if y_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
        let col_pre = if x_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
        let src_base =
            (int_y + row_pre) as usize * params.reference_stride + (int_x + col_pre) as usize;
        predict(&params.reference[src_base..], params.reference_stride, dst);
        return;
    }

    let scratch_w = (x1 - x0 + 1) as usize;
    let scratch_h = (y1 - y0 + 1) as usize;
    assert!(
        scratch_w.saturating_mul(scratch_h) <= border_scratch.len(),
        "inter prediction border scratch is too small for {scratch_w}x{scratch_h}",
    );
    copy_edge_replicated_window(
        params.reference,
        params.reference_stride,
        ref_w,
        ref_h,
        x0 as isize,
        y0 as isize,
        scratch_w,
        scratch_h,
        border_scratch,
        scratch_w,
    );
    predict(&border_scratch[..scratch_w * scratch_h], scratch_w, dst);
}

#[cfg(test)]
mod tests {
    use super::super::filter::{
        convolve8, convolve8_horiz, convolve8_scaled, convolve8_scaled_horiz,
    };
    use super::*;

    // --- Constants and validity gate -----------------------------------

    #[test]
    fn constants_match_libvpx() {
        // libvpx `vp9_scale.h:21-23`.
        assert_eq!(REF_SCALE_SHIFT, 14);
        assert_eq!(REF_NO_SCALE, 1 << 14);
        assert_eq!(REF_INVALID_SCALE, -1);
    }

    #[test]
    fn tight_scaled_prediction_matches_normative_fractional_pixels() {
        const REF_W: usize = 64;
        const REF_H: usize = 32;
        let reference: Vec<u8> = (0..REF_W * REF_H)
            .map(|index| ((index * 37 + index / REF_W * 19) & 0xff) as u8)
            .collect();
        let mut prediction = [0u8; 16 * 16];
        let mut border = [0u8; 72 * 72];
        build_inter_prediction_from_tight_plane(
            TightInterPredictionParams {
                reference: &reference,
                reference_stride: REF_W,
                reference_dimensions: (REF_W, REF_H),
                current_dimensions: (REF_W * 2, REF_H * 2),
                plane_origin: (32, 16),
                phase_origin: (32, 16),
                block_dimensions: (16, 16),
                mv_q4: (-8, -8),
                interp_filter: InterpFilter::EightTap,
                is_compound_avg: false,
            },
            &mut prediction,
            16,
            &mut border,
        );

        assert_eq!(
            &prediction[..16],
            &[
                255, 197, 88, 43, 69, 85, 63, 55, 67, 89, 106, 133, 180, 238, 255, 200
            ]
        );
    }

    #[test]
    fn valid_ref_frame_size_envelope() {
        // libvpx `valid_ref_frame_size` (`vp9_scale.h:61-65`):
        // `2 * cur >= ref && cur <= 16 * ref` per axis.
        assert!(valid_ref_frame_size(640, 480, 640, 480));
        // 2x upscale ref: 2 * cur >= ref → ref <= 1280. 1280 is OK.
        assert!(valid_ref_frame_size(1280, 960, 640, 480));
        assert!(!valid_ref_frame_size(1281, 960, 640, 480));
        // 16x downscale ref: cur <= 16 * ref → cur=640 ⇒ ref >= 40.
        assert!(valid_ref_frame_size(40, 30, 640, 480));
        assert!(!valid_ref_frame_size(39, 30, 640, 480));
        // Asymmetric: only y fails.
        assert!(!valid_ref_frame_size(640, 1000, 640, 480));
    }

    #[test]
    fn invalid_scale_is_propagated() {
        let sf = ScaleFactors::new(1281, 960, 640, 480);
        assert_eq!(sf.x_scale_fp, REF_INVALID_SCALE);
        assert_eq!(sf.y_scale_fp, REF_INVALID_SCALE);
        assert!(!sf.is_valid());
        assert!(!sf.is_scaled());
    }

    // --- Identity scale ------------------------------------------------

    #[test]
    fn identity_scale_is_not_scaled() {
        let sf = ScaleFactors::new(640, 480, 640, 480);
        assert_eq!(sf.x_scale_fp, REF_NO_SCALE);
        assert_eq!(sf.y_scale_fp, REF_NO_SCALE);
        assert_eq!(sf.x_step_q4, SUBPEL_SHIFTS as i32);
        assert_eq!(sf.y_step_q4, SUBPEL_SHIFTS as i32);
        assert!(sf.is_valid());
        assert!(!sf.is_scaled());
    }

    #[test]
    fn identity_scaled_x_y_q4() {
        // Identity: scaled_x_q4(x, mv) = (x << SUBPEL_BITS) + mv.
        let sf = ScaleFactors::new(640, 480, 640, 480);
        assert_eq!(sf.scaled_x_q4(0, 0), 0);
        assert_eq!(sf.scaled_x_q4(10, 0), 10 << SUBPEL_BITS);
        assert_eq!(sf.scaled_x_q4(10, 5), (10 << SUBPEL_BITS) + 5);
        assert_eq!(sf.scaled_x_q4(10, -5), (10 << SUBPEL_BITS) - 5);
        assert_eq!(sf.scaled_y_q4(7, 13), (7 << SUBPEL_BITS) + 13);
    }

    // --- 2× and 0.5× -------------------------------------------------

    #[test]
    fn two_times_upscale() {
        // ref = 2 * cur: x_scale_fp = (2*cur << 14)/cur = 2 << 14 = 1 << 15.
        let sf = ScaleFactors::new(1280, 720, 640, 360);
        assert_eq!(sf.x_scale_fp, 1 << 15);
        assert_eq!(sf.y_scale_fp, 1 << 15);
        // x_step_q4 = scaled_value(16, 1<<15) = (16 * (1<<15)) >> 14 = 32.
        assert_eq!(sf.x_step_q4, 32);
        assert_eq!(sf.y_step_q4, 32);
        assert!(sf.is_scaled());
        // scaled_x_q4(x, 0) = scaled_value(x << 4, 1<<15) = (x << 4) * 2 = 2x in Q4.
        // Concretely: cur-pel x=5 ⇒ ref-pel 10 ⇒ Q4 = 160.
        assert_eq!(sf.scaled_x_q4(5, 0), 5 << (SUBPEL_BITS + 1));
        assert_eq!(sf.scaled_x_q4(5, 0), 10 << SUBPEL_BITS);
    }

    #[test]
    fn half_times_downscale() {
        // ref = cur / 2: x_scale_fp = (cur/2 << 14)/cur = 1 << 13.
        let sf = ScaleFactors::new(640, 360, 1280, 720);
        assert_eq!(sf.x_scale_fp, 1 << 13);
        assert_eq!(sf.y_scale_fp, 1 << 13);
        // x_step_q4 = (16 * (1<<13)) >> 14 = 8.
        assert_eq!(sf.x_step_q4, 8);
        assert_eq!(sf.y_step_q4, 8);
        assert!(sf.is_scaled());
        // scaled_x_q4(x, 0) = (x << 4) * 0.5 in Q4 ⇒ x << 3.
        assert_eq!(sf.scaled_x_q4(10, 0), 10 << (SUBPEL_BITS - 1));
        // Odd x rounds via truncation (libvpx's `(val * scale) >> 14`
        // is an arithmetic right-shift; for positive values this is
        // floor toward -inf which is same as truncation toward zero).
        assert_eq!(sf.scaled_x_q4(11, 0), (11 << 4) / 2);
    }

    #[test]
    fn asymmetric_scale() {
        // 2x in x, 0.5x in y.
        let sf = ScaleFactors::new(1280, 360, 640, 720);
        assert_eq!(sf.x_scale_fp, 1 << 15);
        assert_eq!(sf.y_scale_fp, 1 << 13);
        assert_eq!(sf.x_step_q4, 32);
        assert_eq!(sf.y_step_q4, 8);
        assert!(sf.is_scaled());
    }

    #[test]
    fn rounding_boundary_off_by_one() {
        // 3:7 ratio — neither power of two, so the divide isn't
        // exact. Hand-traced: x_scale_fp = (300 << 14) / 700.
        let sf = ScaleFactors::new(300, 100, 700, 100);
        // (300 << 14) = 4_915_200; / 700 = 7021 (truncated).
        assert_eq!(sf.x_scale_fp, 4_915_200 / 700);
        // y axis is identity (100/100).
        assert_eq!(sf.y_scale_fp, REF_NO_SCALE);
        assert!(sf.is_scaled());
        // scaled_x_q4(5, 0) = (80 * 7021) >> 14 = 561680 >> 14 = 34.
        assert_eq!(sf.scaled_x_q4(5, 0), ((80i64 * 7021) >> 14) as i32);
    }

    #[test]
    fn resize_vector_scale_factors() {
        // The conformance corpus `vp90-2-05-resize.ivf` carries
        // 352×288 ⇆ 176×144 transitions (the typical CIF/QCIF pair
        // libvpx uses for resize coverage). Both ratios are 2:1, so
        // we expect the same outputs as `two_times_upscale` /
        // `half_times_downscale` but at the canonical dimensions.
        let upscale = ScaleFactors::new(352, 288, 176, 144);
        assert_eq!(upscale.x_scale_fp, 1 << 15);
        assert_eq!(upscale.y_scale_fp, 1 << 15);
        assert!(upscale.is_scaled());

        let downscale = ScaleFactors::new(176, 144, 352, 288);
        assert_eq!(downscale.x_scale_fp, 1 << 13);
        assert_eq!(downscale.y_scale_fp, 1 << 13);
        assert!(downscale.is_scaled());

        // And an unscaled ref (same dims) sits next to them.
        let unscaled = ScaleFactors::new(352, 288, 352, 288);
        assert!(!unscaled.is_scaled());
    }

    // --- inter_predictor dispatch -------------------------------------

    /// Pad a `w x h` payload with a halo so the 8-tap kernel can read
    /// its full footprint, replicating edge pixels into the halo.
    /// Returns `(buf, stride, x_off, y_off)`.
    fn pad_block(payload: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize, usize) {
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut buf = vec![0u8; stride * padded_h];
        for y in 0..padded_h {
            for x in 0..stride {
                let cy = y.saturating_sub(pad).min(h - 1);
                let cx = x.saturating_sub(pad).min(w - 1);
                buf[y * stride + x] = payload[cy * w + cx];
            }
        }
        (buf, stride, pad, pad)
    }

    #[test]
    fn inter_predictor_identity_passthrough() {
        // is_scaled == false, subpel == 0 → convolve_copy.
        let sf = ScaleFactors::new(8, 8, 8, 8);
        let payload: Vec<u8> = (0..64u8).collect();
        let (src, stride, xo, yo) = pad_block(&payload, 8, 8);
        let mut dst = vec![0u8; 64];
        // No 8-tap halo needed for the copy path, but pre-shift the
        // src to the destination origin anyway so we're testing the
        // exact same call shape Phase D will use.
        inter_predictor(
            &src[yo * stride + xo..],
            stride,
            &mut dst,
            8,
            0,
            0,
            &sf,
            InterpFilter::EightTap,
            8,
            8,
            false,
        );
        assert_eq!(dst, payload);
    }

    #[test]
    fn inter_predictor_identity_with_subpel_x_uses_horiz() {
        // is_scaled == false, subpel_x != 0, subpel_y == 0 →
        // convolve8_horiz. Differential: compare with a direct call.
        let sf = ScaleFactors::new(8, 8, 8, 8);
        let payload: Vec<u8> = (0..64u8).map(|v| v.wrapping_mul(3)).collect();
        let (src, stride, xo, yo) = pad_block(&payload, 8, 8);
        let mut dst_via_predictor = vec![0u8; 64];
        let mut dst_direct = vec![0u8; 64];

        // src must be pre-shifted left by 3 (the 8-tap horiz halo).
        let src_base = yo * stride + xo - 3;
        inter_predictor(
            &src[src_base..],
            stride,
            &mut dst_via_predictor,
            8,
            5,
            0,
            &sf,
            InterpFilter::EightTap,
            8,
            8,
            false,
        );
        convolve8_horiz(
            &src[src_base..],
            stride,
            &mut dst_direct,
            8,
            &kernel_for(InterpFilter::EightTap)[5],
            8,
            8,
        );
        assert_eq!(dst_via_predictor, dst_direct);
    }

    #[test]
    fn inter_predictor_identity_with_subpel_xy_uses_full_2d() {
        let sf = ScaleFactors::new(8, 8, 8, 8);
        let payload: Vec<u8> = (0..64u8)
            .map(|v| v.wrapping_mul(5).wrapping_add(7))
            .collect();
        let (src, stride, xo, yo) = pad_block(&payload, 8, 8);
        let mut dst_via_predictor = vec![0u8; 64];
        let mut dst_direct = vec![0u8; 64];

        // src must be pre-shifted left 3 cols AND up 3 rows.
        let src_base = (yo - 3) * stride + (xo - 3);
        inter_predictor(
            &src[src_base..],
            stride,
            &mut dst_via_predictor,
            8,
            7,
            11,
            &sf,
            InterpFilter::EightTap,
            8,
            8,
            false,
        );
        convolve8(
            &src[src_base..],
            stride,
            &mut dst_direct,
            8,
            &kernel_for(InterpFilter::EightTap)[7],
            &kernel_for(InterpFilter::EightTap)[11],
            8,
            8,
        );
        assert_eq!(dst_via_predictor, dst_direct);
    }

    #[test]
    fn inter_predictor_scaled_uses_scaled_2d() {
        // is_scaled == true → convolve8_scaled regardless of subpel.
        // Differential vs a direct convolve8_scaled call.
        let sf = ScaleFactors::new(640, 480, 320, 240);
        assert!(sf.is_scaled());
        assert_eq!(sf.x_step_q4, 32);
        assert_eq!(sf.y_step_q4, 32);

        let payload: Vec<u8> = (0..256u16).map(|v| v as u8).collect();
        let (src, stride, xo, yo) = pad_block(&payload, 16, 16);
        let mut dst_via_predictor = vec![0u8; 64];
        let mut dst_direct = vec![0u8; 64];

        let src_base = (yo - 3) * stride + (xo - 3);
        inter_predictor(
            &src[src_base..],
            stride,
            &mut dst_via_predictor,
            8,
            3,
            9,
            &sf,
            InterpFilter::EightTap,
            8,
            8,
            false,
        );
        convolve8_scaled(
            &src[src_base..],
            stride,
            &mut dst_direct,
            8,
            kernel_for(InterpFilter::EightTap),
            3,
            sf.x_step_q4,
            9,
            sf.y_step_q4,
            8,
            8,
        );
        assert_eq!(dst_via_predictor, dst_direct);
    }

    #[test]
    fn inter_predictor_compound_avg_passthrough() {
        // is_scaled == false, subpel == 0, is_compound_avg == true →
        // convolve_avg.
        let sf = ScaleFactors::new(8, 8, 8, 8);
        let payload: Vec<u8> = (0..64u8).map(|v| v.wrapping_mul(2)).collect();
        let (src, stride, xo, yo) = pad_block(&payload, 8, 8);
        let mut dst = vec![100u8; 64];

        inter_predictor(
            &src[yo * stride + xo..],
            stride,
            &mut dst,
            8,
            0,
            0,
            &sf,
            InterpFilter::EightTap,
            8,
            8,
            true,
        );
        // Each output pixel = (100 + payload + 1) >> 1.
        for i in 0..64 {
            let expected = ((100u32 + u32::from(payload[i]) + 1) >> 1) as u8;
            assert_eq!(dst[i], expected, "mismatch at idx {i}");
        }
    }

    // --- Scaled-convolve differential vs libvpx scalar ref ------------

    /// Verbatim port of libvpx `convolve_horiz` with the step
    /// argument honoured. Lives in test scope only.
    #[allow(clippy::too_many_arguments)]
    fn libvpx_convolve_horiz_ref(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filters: &[[i16; 8]; SUBPEL_SHIFTS],
        x0_q4: i32,
        x_step_q4: i32,
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            let mut x_q4 = x0_q4;
            for x in 0..w {
                let src_x_base = (x_q4 >> SUBPEL_BITS) as usize;
                let filter = &filters[(x_q4 & SUBPEL_MASK) as usize];
                let mut sum: i32 = 0;
                let row = y * src_stride;
                for k in 0..8 {
                    sum += i32::from(src[row + src_x_base + k]) * i32::from(filter[k]);
                }
                let rounded = (sum + (1 << 6)) >> 7;
                dst[y * dst_stride + x] = rounded.clamp(0, 255) as u8;
                x_q4 += x_step_q4;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn libvpx_convolve_vert_ref(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filters: &[[i16; 8]; SUBPEL_SHIFTS],
        y0_q4: i32,
        y_step_q4: i32,
        w: usize,
        h: usize,
    ) {
        for x in 0..w {
            let mut y_q4 = y0_q4;
            for y in 0..h {
                let src_y_base = (y_q4 >> SUBPEL_BITS) as usize;
                let filter = &filters[(y_q4 & SUBPEL_MASK) as usize];
                let mut sum: i32 = 0;
                for k in 0..8 {
                    sum += i32::from(src[(src_y_base + k) * src_stride + x]) * i32::from(filter[k]);
                }
                let rounded = (sum + (1 << 6)) >> 7;
                dst[y * dst_stride + x] = rounded.clamp(0, 255) as u8;
                y_q4 += y_step_q4;
            }
        }
    }

    #[test]
    fn inter_predictor_width_only_scale_matches_libvpx_horiz() {
        let sf = ScaleFactors::new(32, 16, 16, 16);
        assert_eq!(sf.x_step_q4, 32);
        assert_eq!(sf.y_step_q4, SUBPEL_SHIFTS as i32);

        let payload: Vec<u8> = (0..16)
            .flat_map(|y| (0..32).map(move |x| (y * 13 + x * 7) as u8))
            .collect();
        let (src, stride, xo, yo) = pad_block(&payload, 32, 16);
        let src_base = yo * stride + xo - 3;
        let src = &src[src_base..];
        let filters = kernel_for(InterpFilter::EightTapSharp);
        let mut actual = vec![0u8; 8 * 8];
        let mut expected = vec![0u8; 8 * 8];

        inter_predictor(
            src,
            stride,
            &mut actual,
            8,
            5,
            0,
            &sf,
            InterpFilter::EightTapSharp,
            8,
            8,
            false,
        );
        libvpx_convolve_horiz_ref(
            src,
            stride,
            &mut expected,
            8,
            filters,
            5,
            sf.x_step_q4,
            8,
            8,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn inter_predictor_height_only_scale_matches_libvpx_vert_avg() {
        let sf = ScaleFactors::new(16, 32, 16, 16);
        assert_eq!(sf.x_step_q4, SUBPEL_SHIFTS as i32);
        assert_eq!(sf.y_step_q4, 32);

        let payload: Vec<u8> = (0..32)
            .flat_map(|y| (0..16).map(move |x| (y * 5 + x * 17) as u8))
            .collect();
        let (src, stride, xo, yo) = pad_block(&payload, 16, 32);
        let src_base = (yo - 3) * stride + xo;
        let src = &src[src_base..];
        let filters = kernel_for(InterpFilter::EightTapSmooth);
        let initial: Vec<u8> = (0..64).map(|i| (31 + i * 3) as u8).collect();
        let mut actual = initial.clone();
        let mut pred = vec![0u8; 8 * 8];

        inter_predictor(
            src,
            stride,
            &mut actual,
            8,
            0,
            11,
            &sf,
            InterpFilter::EightTapSmooth,
            8,
            8,
            true,
        );
        libvpx_convolve_vert_ref(src, stride, &mut pred, 8, filters, 11, sf.y_step_q4, 8, 8);
        let expected: Vec<u8> = initial
            .iter()
            .zip(pred)
            .map(|(&old, new)| ((u16::from(old) + u16::from(new) + 1) >> 1) as u8)
            .collect();

        assert_eq!(actual, expected);
    }

    /// Verbatim port of libvpx `vpx_convolve8_c` with full step args.
    #[allow(clippy::too_many_arguments)]
    fn libvpx_convolve_2d_ref(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filters: &[[i16; 8]; SUBPEL_SHIFTS],
        x0_q4: i32,
        x_step_q4: i32,
        y0_q4: i32,
        y_step_q4: i32,
        w: usize,
        h: usize,
    ) {
        let intermediate_height =
            (((h as i32 - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) as usize + 8;
        let mut temp = vec![0u8; 64 * intermediate_height];
        libvpx_convolve_horiz_ref(
            src,
            src_stride,
            &mut temp,
            64,
            filters,
            x0_q4,
            x_step_q4,
            w,
            intermediate_height,
        );
        libvpx_convolve_vert_ref(&temp, 64, dst, dst_stride, filters, y0_q4, y_step_q4, w, h);
    }

    #[test]
    fn differential_random_scaled_2d() {
        // 100 random inputs across all four kernels and various scale
        // factors / sub-pel offsets — confirms convolve8_scaled
        // matches the libvpx reference byte-for-byte.
        let w = 8;
        let h = 8;
        let pad = 16usize; // generous halo for stepping
        let stride = w * 4 + 2 * pad;
        let padded_h = h * 4 + 2 * pad;
        let mut state = 0x5CA1_E5C2u32;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let kernel_choices = [
            InterpFilter::EightTap,
            InterpFilter::EightTapSmooth,
            InterpFilter::EightTapSharp,
            InterpFilter::Bilinear,
        ];
        // Mix of step values: 16 (unit), 32 (2x ref), 24 (1.5x),
        // 12 (0.75x), 8 (0.5x).
        let step_choices = [16, 32, 24, 12, 8];

        for iter in 0..100 {
            let mut src = vec![0u8; stride * padded_h];
            for cell in src.iter_mut() {
                *cell = (rand() & 0xff) as u8;
            }
            let kernel = kernel_choices[(iter as usize) % 4];
            let filters = kernel_for(kernel);
            let x0 = rand() as i32 & SUBPEL_MASK;
            let y0 = rand() as i32 & SUBPEL_MASK;
            let x_step = step_choices[(rand() as usize) % step_choices.len()];
            let y_step = step_choices[(rand() as usize) % step_choices.len()];
            let mut dst_kernel = vec![0u8; w * h];
            let mut dst_ref = vec![0u8; w * h];
            let src_off = (pad - 3) * stride + (pad - 3);

            convolve8_scaled(
                &src[src_off..],
                stride,
                &mut dst_kernel,
                w,
                filters,
                x0,
                x_step,
                y0,
                y_step,
                w,
                h,
            );
            libvpx_convolve_2d_ref(
                &src[src_off..],
                stride,
                &mut dst_ref,
                w,
                filters,
                x0,
                x_step,
                y0,
                y_step,
                w,
                h,
            );
            assert_eq!(
                dst_kernel, dst_ref,
                "scaled 2d mismatch at iter={iter} x0={x0} xs={x_step} y0={y0} ys={y_step}",
            );
        }
    }

    #[test]
    fn scaled_horiz_with_step_16_matches_unscaled() {
        // Bit-equivalence check: convolve8_scaled_horiz with
        // x_step_q4 = 16 must equal convolve8_horiz with the same
        // single sub-pel offset.
        let w = 8;
        let h = 8;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let mut src = vec![0u8; stride * (h + 2 * pad)];
        let mut state = 0x900D_F00Du32;
        for cell in src.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *cell = (state & 0xff) as u8;
        }
        let filters = kernel_for(InterpFilter::EightTapSharp);
        let x0 = 11;
        let src_off = pad * stride + pad - 3;
        let mut dst_scaled = vec![0u8; w * h];
        let mut dst_unscaled = vec![0u8; w * h];
        convolve8_scaled_horiz(
            &src[src_off..],
            stride,
            &mut dst_scaled,
            w,
            filters,
            x0,
            16,
            w,
            h,
        );
        convolve8_horiz(
            &src[src_off..],
            stride,
            &mut dst_unscaled,
            w,
            &filters[x0 as usize],
            w,
            h,
        );
        assert_eq!(dst_scaled, dst_unscaled);
    }
}
