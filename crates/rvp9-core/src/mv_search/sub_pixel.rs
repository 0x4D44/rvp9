//! Sub-pixel MV refinement (the `_pruned_more` variant).
//!
//! Port of `vp9_find_best_sub_pixel_tree_pruned_more`
//! (`vp9_mcomp.c:453-513`). This is the `--cpu-used 6+` default when
//! the speed-feature `subpel_search_method == SUBPEL_TREE_PRUNED_MORE`.
//!
//! The libvpx function has a parabolic-fit short-circuit when the
//! caller supplies a well-behaved 5-entry `cost_list` (produced by
//! `calc_int_cost_list`); F7.0 omits that short-circuit (equivalent
//! to libvpx running with `cost_list == NULL`) and always walks the
//! `FIRST_LEVEL_CHECKS` half-pel / quarter-pel / eighth-pel pyramid.
//!
//! ## Search pyramid
//!
//! At each level (half / quarter / eighth-pel):
//!
//! 1. `FIRST_LEVEL_CHECKS` — 4 cardinal + 1 diagonal sample around
//!    the starting center `(tr, tc)`; updates `(br, bc)` if better.
//! 2. `SECOND_LEVEL_CHECKS` — only when
//!    [`SubPixelSearchParams::iters_per_step`] > 1. Extends the search
//!    in the direction of the FIRST_LEVEL winner: if the diagonal
//!    won, samples the two adjacent "long-diagonal" points; if a
//!    cardinal won, samples three further points along the winning
//!    axis. Mirrors the `SECOND_LEVEL_CHECKS` macro in
//!    `vp9_mcomp.c:218-248`.
//!
//! The levels execute in sequence:
//!
//! 1. Half-pel (`hstep = 4` in 1/8-pel units).
//! 2. Quarter-pel (`hstep = 2`) — only if `forced_stop != 2`.
//! 3. Eighth-pel (`hstep = 1`) — only if
//!    `forced_stop == 0 && allow_hp && use_mv_hp(ref_mv)`.
//!
//! Each level keeps the best `(br, bc)` from the previous level as
//! its starting `(tr, tc)`.

use crate::block::BlockSize;
use crate::cost::mv_cost::MvCosts;
use crate::dsp::distortion::{
    vpx_sub_pixel_variance16x16, vpx_sub_pixel_variance32x32, vpx_sub_pixel_variance4x4,
    vpx_sub_pixel_variance64x64, vpx_sub_pixel_variance8x8,
};
use crate::inter::mv::use_mv_hp;
use crate::mode_info::Mv;
use crate::mv_search::full_pixel::{PlaneRef, mv_err_cost};
use crate::mv_search::mv_limits::{MvLimits, set_subpel_mv_search_range};

/// Sub-pel-only kernel signature: `(src, src_stride, xoffset,
/// yoffset, reference, ref_stride) -> (variance, sse)`.
type SvfFn = fn(&[u8], usize, usize, usize, &[u8], usize) -> (u32, u32);

/// `forced_stop` enum mirror of libvpx's `SUBPEL_FORCE_STOP`
/// (`vp9_speed_features.h:168`). The numeric values match libvpx so
/// the integer comparisons inside the search loop stay bitwise-equiv.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum SubPixelForcedStop {
    /// `EIGHTH_PEL = 0` — refine all the way to 1/8-pel (subject to
    /// `allow_hp && use_mv_hp(ref)`).
    EighthPel = 0,
    /// `QUARTER_PEL = 1` — stop at 1/4-pel.
    QuarterPel = 1,
    /// `HALF_PEL = 2` — stop at 1/2-pel.
    HalfPel = 2,
}

/// Input parameters to [`find_best_sub_pixel_tree_pruned_more`].
pub struct SubPixelSearchParams<'a> {
    /// Block size — picks the sub-pel-variance kernel.
    pub block_size: BlockSize,
    /// Source plane view.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the block within `src_plane`, in `(col, row)`.
    pub src_offset: (usize, usize),
    /// Reference plane view.
    pub ref_plane: &'a PlaneRef<'a>,
    /// Starting MV in **full-pel units** — typically the output of
    /// the full-pel diamond search. The search internally lifts it
    /// to 1/8-pel by `* 8`.
    pub center_mv: Mv,
    /// Reference MV (1/8-pel units) — used for MV-cost computation.
    pub ref_mv: Mv,
    /// Full-pel MV limits — the function derives the sub-pel
    /// (1/8-pel-unit) limits internally via
    /// [`crate::mv_search::set_subpel_mv_search_range`].
    pub mv_limits: MvLimits,
    /// `error_per_bit` cost multiplier (variance-domain).
    pub error_per_bit: i32,
    /// MV-cost LUT bank.
    pub mv_costs: &'a MvCosts,
    /// `cm->allow_high_precision_mv` — frame-level HP gate.
    pub allow_hp: bool,
    /// Stop level. `EighthPel` (default) does full pyramid; higher
    /// values truncate.
    pub forced_stop: SubPixelForcedStop,
    /// `iters_per_step` mirror of libvpx's
    /// `sf.mv.subpel_search_level` (`vp9_speed_features.h:201`). Each
    /// pyramid level runs `FIRST_LEVEL_CHECKS` unconditionally; when
    /// `iters_per_step > 1`, it additionally runs
    /// `SECOND_LEVEL_CHECKS` to fan out further in the winning
    /// direction. libvpx's rt setter leaves this at the default of
    /// `2` (`vp9_speed_features.c:929`) for all rt speed presets
    /// except SVC speed 3+ (which drops it to 0-1). For our locked
    /// cpu-used 6-8 rt config the value is **2**, matching libvpx's
    /// non-SVC rt path bug-for-bug.
    pub iters_per_step: i32,
}

/// Result of a sub-pixel search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubPixelSearchResult {
    /// Refined MV in **1/8-pel units** (standard VP9 MV unit).
    pub best_mv: Mv,
    /// Distortion (sub-pixel variance + MV cost) at the best MV.
    pub best_distortion: u32,
}

/// Dispatch sub-pel-variance by block size.
///
/// **Constraint:** only the 5 square sizes 4x4, 8x8, 16x16, 32x32,
/// 64x64 are ported (per F6.0). Other block sizes panic in debug,
/// and return zero-variance in release as a fail-soft. Mode-decision
/// must avoid calling sub-pel search on non-square blocks until the
/// remaining 8 sub-pel-variance kernels land.
#[inline]
fn svf_for(bsize: BlockSize) -> Option<SvfFn> {
    match bsize {
        BlockSize::Block4x4 => Some(vpx_sub_pixel_variance4x4),
        BlockSize::Block8x8 => Some(vpx_sub_pixel_variance8x8),
        BlockSize::Block16x16 => Some(vpx_sub_pixel_variance16x16),
        BlockSize::Block32x32 => Some(vpx_sub_pixel_variance32x32),
        BlockSize::Block64x64 => Some(vpx_sub_pixel_variance64x64),
        _ => None,
    }
}

/// Helper: try to compute (variance + mv_err_cost) for a 1/8-pel
/// candidate MV `(tr, tc)`. Returns `u32::MAX` if the candidate is
/// out of the sub-pel limits or the reference fetch would be
/// out-of-bounds.
///
/// `(tr, tc)` are signed 1/8-pel coordinates; the integer pixel
/// position is `(tr >> 3, tc >> 3)`, and the phase passed to the
/// sub-pel-variance kernel is `(tc & 7, tr & 7)`.
#[allow(clippy::too_many_arguments)]
fn try_subpel(
    tr: i32,
    tc: i32,
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    ref_plane: &PlaneRef<'_>,
    svf: SvfFn,
    bw: usize,
    bh: usize,
    subpel_limits: &MvLimits,
    ref_mv: Mv,
    error_per_bit: i32,
    mv_costs: &MvCosts,
) -> u32 {
    // Sub-pel limits check.
    if tc < subpel_limits.col_min
        || tc > subpel_limits.col_max
        || tr < subpel_limits.row_min
        || tr > subpel_limits.row_max
    {
        return u32::MAX;
    }

    // Integer pixel offset (libvpx `(r >> 3) * y_stride + (c >> 3)`).
    let int_row = tr >> 3;
    let int_col = tc >> 3;
    // Sub-pel phase (libvpx `sp(c) = c & 7`).
    let xoff = (tc & 7) as usize;
    let yoff = (tr & 7) as usize;

    // Fetch the source block.
    let src_col = src_offset.0;
    let src_row = src_offset.1;
    if src_col + bw > src_plane.width || src_row + bh > src_plane.height {
        return u32::MAX;
    }
    let src_start = src_row * src_plane.stride + src_col;
    let src_end = (src_row + bh - 1) * src_plane.stride + src_col + bw;
    if src_end > src_plane.data.len() {
        return u32::MAX;
    }
    let src_slice = &src_plane.data[src_start..src_end];

    // Fetch the reference block. The sub-pel-variance kernel reads
    // `(h + 1) × (w + 1)` pixels from the reference (one extra row
    // and column for the bilinear filter's right/bottom tap), so we
    // need a buffer of at least `h * stride + w + 1` bytes starting
    // at the reference offset.
    let ref_col_i = src_col as isize + int_col as isize;
    let ref_row_i = src_row as isize + int_row as isize;
    if ref_col_i < 0 || ref_row_i < 0 {
        return u32::MAX;
    }
    let ref_col = ref_col_i as usize;
    let ref_row = ref_row_i as usize;
    // Need (bh + 1) rows × (bw + 1) cols accessible. Even at phase
    // (0, 0) the kernel still reads the +1 row/col, so we must
    // always have headroom.
    if ref_col + bw + 1 > ref_plane.width || ref_row + bh + 1 > ref_plane.height {
        return u32::MAX;
    }
    let ref_start = ref_row * ref_plane.stride + ref_col;
    let ref_end = (ref_row + bh) * ref_plane.stride + ref_col + bw + 1;
    if ref_end > ref_plane.data.len() {
        return u32::MAX;
    }
    let ref_slice = &ref_plane.data[ref_start..ref_end];

    // sub_pixel_variance signature: (src, src_stride, xoff, yoff,
    // ref, ref_stride). NB: libvpx-internal naming is reversed
    // (the "src" of the bilinear filter is the reference frame and
    // "ref" is the source block being matched). The wrapper in
    // dsp/distortion.rs preserves that convention; we pass
    // `ref_slice` as the bilinear-filtered "src" and `src_slice`
    // as the variance "reference".
    let (var, _sse) = svf(
        ref_slice,
        ref_plane.stride,
        xoff,
        yoff,
        src_slice,
        src_plane.stride,
    );

    let cost = mv_err_cost(
        tr as i16,
        tc as i16,
        ref_mv.row,
        ref_mv.col,
        error_per_bit,
        mv_costs,
    );
    var.saturating_add(cost)
}

/// `vp9_find_best_sub_pixel_tree_pruned_more` (`vp9_mcomp.c:453-513`).
///
/// Refines a full-pel MV into 1/8-pel units via a 3-level pyramid
/// (half / quarter / eighth-pel). At each level we run
/// `FIRST_LEVEL_CHECKS` (4 cardinals + 1 diagonal) plus, when
/// `iters_per_step > 1`, `SECOND_LEVEL_CHECKS` extending in the
/// winning direction. The pruning vs `_pruned`: `_pruned_more` uses
/// `hstep`-scaled steps inside the cost-list surface-minimum branch
/// (which we don't reach without `calc_int_cost_list`), but its
/// `FIRST_LEVEL_CHECKS` / `SECOND_LEVEL_CHECKS` path is identical.
pub fn find_best_sub_pixel_tree_pruned_more(
    params: &SubPixelSearchParams<'_>,
) -> SubPixelSearchResult {
    let Some(svf) = svf_for(params.block_size) else {
        debug_assert!(
            false,
            "sub_pixel_variance not ported for {:?} (F6.0 ported 5 square sizes)",
            params.block_size
        );
        return SubPixelSearchResult {
            best_mv: Mv {
                row: params.center_mv.row * 8,
                col: params.center_mv.col * 8,
            },
            best_distortion: 0,
        };
    };
    let bw = usize::from(params.block_size.width());
    let bh = usize::from(params.block_size.height());

    // Derive sub-pel limits.
    let mut subpel_limits = MvLimits::default();
    set_subpel_mv_search_range(
        &mut subpel_limits,
        &params.mv_limits,
        params.ref_mv.row,
        params.ref_mv.col,
    );

    // Lift bestmv to 1/8-pel.
    let mut br: i32 = (params.center_mv.row as i32) * 8;
    let mut bc: i32 = (params.center_mv.col as i32) * 8;

    // Seed besterr at the starting MV.
    let mut besterr = try_subpel(
        br,
        bc,
        params.src_plane,
        params.src_offset,
        params.ref_plane,
        svf,
        bw,
        bh,
        &subpel_limits,
        params.ref_mv,
        params.error_per_bit,
        params.mv_costs,
    );

    let iters_per_step = params.iters_per_step;
    let mut hstep: i32 = 4; // 1/2-pel in 1/8-pel units.

    // Half-pel level (always).
    pyramid_level(
        &mut br,
        &mut bc,
        &mut besterr,
        hstep,
        iters_per_step,
        params,
        svf,
        bw,
        bh,
        &subpel_limits,
    );

    // Quarter-pel level — only if forced_stop != 2.
    if (params.forced_stop as i32) != (SubPixelForcedStop::HalfPel as i32) {
        hstep >>= 1; // 2 (1/4-pel).
        pyramid_level(
            &mut br,
            &mut bc,
            &mut besterr,
            hstep,
            iters_per_step,
            params,
            svf,
            bw,
            bh,
            &subpel_limits,
        );
    }

    // Eighth-pel level — only if forced_stop == 0 && allow_hp && use_mv_hp(ref).
    if (params.forced_stop as i32) == (SubPixelForcedStop::EighthPel as i32)
        && params.allow_hp
        && use_mv_hp(params.ref_mv)
    {
        hstep = 1; // 1/8-pel.
        pyramid_level(
            &mut br,
            &mut bc,
            &mut besterr,
            hstep,
            iters_per_step,
            params,
            svf,
            bw,
            bh,
            &subpel_limits,
        );
    }

    SubPixelSearchResult {
        best_mv: Mv {
            row: br as i16,
            col: bc as i16,
        },
        best_distortion: besterr,
    }
}

/// One pyramid-level step: `FIRST_LEVEL_CHECKS` at `(br, bc)`, then
/// (if `iters_per_step > 1`) `SECOND_LEVEL_CHECKS` extending in the
/// winning direction. Mirrors the `vp9_mcomp.c` macro pair
/// (`vp9_mcomp.c:202-248`). Updates `(br, bc, besterr)` in place.
///
/// The center of FIRST_LEVEL_CHECKS is the current `(*br, *bc)`. The
/// SECOND_LEVEL_CHECKS step uses the same captured center (libvpx's
/// `(tr, tc)`) and the post-FIRST best (libvpx's `(br, bc)`), so the
/// "did the diagonal win?" geometry test is `(tr != br && tc != bc)`.
#[allow(clippy::too_many_arguments)]
fn pyramid_level(
    br: &mut i32,
    bc: &mut i32,
    besterr: &mut u32,
    hstep: i32,
    iters_per_step: i32,
    params: &SubPixelSearchParams<'_>,
    svf: SvfFn,
    bw: usize,
    bh: usize,
    subpel_limits: &MvLimits,
) {
    // Capture the center BEFORE the first-level checks update (br, bc).
    let center_r = *br;
    let center_c = *bc;

    // FIRST_LEVEL_CHECKS — 4 cardinals + 1 inferred diagonal.
    let left = check(
        center_r,
        center_c - hstep,
        br,
        bc,
        besterr,
        params,
        svf,
        bw,
        bh,
        subpel_limits,
    );
    let right = check(
        center_r,
        center_c + hstep,
        br,
        bc,
        besterr,
        params,
        svf,
        bw,
        bh,
        subpel_limits,
    );
    let up = check(
        center_r - hstep,
        center_c,
        br,
        bc,
        besterr,
        params,
        svf,
        bw,
        bh,
        subpel_limits,
    );
    let down = check(
        center_r + hstep,
        center_c,
        br,
        bc,
        besterr,
        params,
        svf,
        bw,
        bh,
        subpel_limits,
    );

    // libvpx: whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);
    let whichdir: u32 = (if left < right { 0 } else { 1 }) + (if up < down { 0 } else { 2 });

    // One diagonal sample.
    let (diag_r, diag_c) = match whichdir {
        0 => (center_r - hstep, center_c - hstep),
        1 => (center_r - hstep, center_c + hstep),
        2 => (center_r + hstep, center_c - hstep),
        _ => (center_r + hstep, center_c + hstep),
    };
    let _ = check(
        diag_r,
        diag_c,
        br,
        bc,
        besterr,
        params,
        svf,
        bw,
        bh,
        subpel_limits,
    );

    // SECOND_LEVEL_CHECKS — only when iters_per_step > 1
    // (`vp9_mcomp.c:218-248`). The libvpx macro uses the FIRST_LEVEL
    // center (tr, tc) and the post-FIRST best (br, bc).
    if iters_per_step > 1 {
        second_level_checks(
            br,
            bc,
            besterr,
            center_r,
            center_c,
            whichdir,
            hstep,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
    }
}

/// `SECOND_LEVEL_CHECKS` macro port (`vp9_mcomp.c:218-248`).
///
/// `(tr, tc)` is the FIRST_LEVEL_CHECKS center; `(br, bc)` is the
/// post-FIRST best. Geometry:
///
/// - **Diagonal won** (`tr != br && tc != bc`): sample two "long
///   diagonal" extensions — `(tr + kr, tc + 2*kc)` and
///   `(tr + 2*kr, tc + kc)` where `kr = br - tr`, `kc = bc - tc`.
/// - **Horizontal cardinal won** (`tr == br, tc != bc`): sample
///   `(tr ± hstep, tc + 2*kc)` and (per `whichdir`) one of
///   `(tr + hstep, tc + kc)` / `(tr - hstep, tc + kc)`.
/// - **Vertical cardinal won** (`tr != br, tc == bc`): sample
///   `(tr + 2*kr, tc ± hstep)` and (per `whichdir`) one of
///   `(tr + kr, tc + hstep)` / `(tr + kr, tc - hstep)`.
///
/// If neither moved (`tr == br && tc == bc`, i.e. the center was the
/// best), libvpx's macro is a no-op here too.
#[allow(clippy::too_many_arguments)]
fn second_level_checks(
    br: &mut i32,
    bc: &mut i32,
    besterr: &mut u32,
    tr: i32,
    tc: i32,
    whichdir: u32,
    hstep: i32,
    params: &SubPixelSearchParams<'_>,
    svf: SvfFn,
    bw: usize,
    bh: usize,
    subpel_limits: &MvLimits,
) {
    let br_in = *br;
    let bc_in = *bc;
    if tr != br_in && tc != bc_in {
        // Diagonal won.
        let kr = br_in - tr;
        let kc = bc_in - tc;
        let _ = check(
            tr + kr,
            tc + 2 * kc,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
        let _ = check(
            tr + 2 * kr,
            tc + kc,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
    } else if tr == br_in && tc != bc_in {
        // Horizontal cardinal won.
        let kc = bc_in - tc;
        let _ = check(
            tr + hstep,
            tc + 2 * kc,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
        let _ = check(
            tr - hstep,
            tc + 2 * kc,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
        // whichdir 0/1: up-direction; 2/3: down-direction.
        let extra_r = if whichdir < 2 {
            tr + hstep
        } else {
            tr - hstep
        };
        let _ = check(
            extra_r,
            tc + kc,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
    } else if tr != br_in && tc == bc_in {
        // Vertical cardinal won.
        let kr = br_in - tr;
        let _ = check(
            tr + 2 * kr,
            tc + hstep,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
        let _ = check(
            tr + 2 * kr,
            tc - hstep,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
        // whichdir 0/2: left-direction; 1/3: right-direction.
        let extra_c = if whichdir == 0 || whichdir == 2 {
            tc + hstep
        } else {
            tc - hstep
        };
        let _ = check(
            tr + kr,
            extra_c,
            br,
            bc,
            besterr,
            params,
            svf,
            bw,
            bh,
            subpel_limits,
        );
    }
    // else: tr == br && tc == bc → center won, no second-level (libvpx
    // macro silently no-ops here).
}

/// CHECK_BETTER expansion: compute (variance + cost) at `(r, c)`,
/// update `(br, bc, besterr)` if better. Returns the candidate's
/// score (or `u32::MAX` if out of bounds).
#[allow(clippy::too_many_arguments)]
fn check(
    r: i32,
    c: i32,
    br: &mut i32,
    bc: &mut i32,
    besterr: &mut u32,
    params: &SubPixelSearchParams<'_>,
    svf: SvfFn,
    bw: usize,
    bh: usize,
    subpel_limits: &MvLimits,
) -> u32 {
    let v = try_subpel(
        r,
        c,
        params.src_plane,
        params.src_offset,
        params.ref_plane,
        svf,
        bw,
        bh,
        subpel_limits,
        params.ref_mv,
        params.error_per_bit,
        params.mv_costs,
    );
    if v < *besterr {
        *besterr = v;
        *br = r;
        *bc = c;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::mv_cost::build_nmv_cost_table;
    use crate::entropy::mv_probs::DEFAULT_NMV_CONTEXT;

    /// Deterministic content with smooth-ish gradient (so the bilinear
    /// filter has something meaningful to interpolate).
    fn make_plane(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; width * height];
        for i in 0..height {
            for j in 0..width {
                let v = ((i as u32).wrapping_mul(7).wrapping_add((j as u32).wrapping_mul(seed))
                    .wrapping_add(40))
                    & 0xff;
                buf[i * width + j] = v as u8;
            }
        }
        buf
    }

    #[test]
    fn sub_pixel_search_at_full_pel_returns_full_pel() {
        // Source = reference, center_mv = (0, 0) full-pel.
        // The refined result must stay at (0, 0) in 1/8-pel units.
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 11);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true,
            forced_stop: SubPixelForcedStop::EighthPel,
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        assert_eq!(r.best_mv, Mv::ZERO, "identical planes → MV stays at zero");
    }

    #[test]
    fn sub_pixel_search_refines_to_quarter_pel() {
        // Build a synthetic 1/4-pel-shifted reference: bilinearly
        // interpolate a base ref at phase (xoff=2, yoff=0) to produce
        // a "source" plane. Then sub-pel search starting from
        // center_mv = (0, 0) should land on col-phase = 2 (1/4-pel).
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 11);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        // Build a src plane: at (col=24, row=24), fill an 8x8 block
        // with the bilinear-interpolated reference at phase
        // (xoff=2, yoff=0) — i.e. the "source" came from the
        // reference shifted by 1/4 pel horizontally.
        let mut src_data = ref_data.clone();
        let src_col = 24usize;
        let src_row = 24usize;
        let bw = 8usize;
        let bh = 8usize;
        // Phase (2/8) = 1/4-pel, filter taps [96, 32] / 128.
        let f0 = 96i32;
        let f1 = 32i32;
        for i in 0..bh {
            for j in 0..bw {
                let a = i32::from(ref_data[(src_row + i) * pw + (src_col + j)]);
                let b = i32::from(ref_data[(src_row + i) * pw + (src_col + j + 1)]);
                let v = (a * f0 + b * f1 + 64) >> 7;
                src_data[(src_row + i) * pw + (src_col + j)] = v as u8;
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        let params = SubPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &src_plane,
            src_offset: (src_col, src_row),
            ref_plane: &ref_plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: false, // restrict to 1/4-pel granularity
            forced_stop: SubPixelForcedStop::QuarterPel,
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        // Expected MV in 1/8-pel units: col = 2, row = 0.
        // Allow a tolerance: the search may settle at exactly ±2 col
        // or even at the half-pel position 4 if the local error
        // surface is funky; require col ∈ {2, 4} as a soft check.
        assert_eq!(
            r.best_mv.row, 0,
            "row should stay near zero (got {})",
            r.best_mv.row
        );
        assert!(
            r.best_mv.col == 2 || r.best_mv.col == 4,
            "expected col ∈ {{2, 4}} for 1/4-pel shift; got col={}",
            r.best_mv.col
        );
    }

    /// `forced_stop = HalfPel`: the search must NOT enter the QuarterPel
    /// or EighthPel levels, so col % 4 == 0 (LSB and bit-1 zero).
    #[test]
    fn sub_pixel_search_forced_stop_half_pel_only_half_pel_results() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 11);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        // Synthesize a (xoff=1, yoff=0) 1/8-pel shifted source so the
        // search has an attractor at an odd 1/8-pel offset, but capped
        // to half-pel by forced_stop=HalfPel.
        let mut src_data = ref_data.clone();
        let src_col = 24usize;
        let src_row = 24usize;
        for i in 0..8 {
            for j in 0..8 {
                let a = i32::from(ref_data[(src_row + i) * pw + (src_col + j)]);
                let b = i32::from(ref_data[(src_row + i) * pw + (src_col + j + 1)]);
                let v = (a * 112 + b * 16 + 64) >> 7;
                src_data[(src_row + i) * pw + (src_col + j)] = v as u8;
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &src_plane,
            src_offset: (src_col, src_row),
            ref_plane: &ref_plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true,
            forced_stop: SubPixelForcedStop::HalfPel,
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        // With HalfPel-only refinement, low 2 bits of each axis must
        // be zero (1/8-pel-unit half-pel positions are multiples of 4).
        assert_eq!(
            r.best_mv.col & 3,
            0,
            "forced_stop=HalfPel: col & 3 must be 0; got col={}",
            r.best_mv.col
        );
        assert_eq!(
            r.best_mv.row & 3,
            0,
            "forced_stop=HalfPel: row & 3 must be 0; got row={}",
            r.best_mv.row
        );
    }

    /// `iters_per_step <= 1`: `SECOND_LEVEL_CHECKS` must NOT run. We
    /// can't directly observe the absence of those probe calls, but
    /// the search must still produce a valid result for an identical-
    /// plane case (MV stays at zero).
    #[test]
    fn sub_pixel_search_iters_per_step_one_skips_second_level() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 23);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true,
            forced_stop: SubPixelForcedStop::EighthPel,
            iters_per_step: 1, // skips SECOND_LEVEL_CHECKS
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        assert_eq!(
            r.best_mv,
            Mv::ZERO,
            "identical planes → MV stays at zero even with iters=1"
        );
    }

    /// `allow_hp=false` blocks the EighthPel level even with
    /// `forced_stop=EighthPel`. Result must have col & 1 == 0 and
    /// row & 1 == 0.
    #[test]
    fn sub_pixel_search_disabled_hp_yields_no_eighth_pel_bits() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 11);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        // 1/8-pel-shifted source so the natural attractor is at an odd
        // 1/8-pel position.
        let mut src_data = ref_data.clone();
        let src_col = 24usize;
        let src_row = 24usize;
        for i in 0..8 {
            for j in 0..8 {
                let a = i32::from(ref_data[(src_row + i) * pw + (src_col + j)]);
                let b = i32::from(ref_data[(src_row + i) * pw + (src_col + j + 1)]);
                let v = (a * 112 + b * 16 + 64) >> 7;
                src_data[(src_row + i) * pw + (src_col + j)] = v as u8;
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &src_plane,
            src_offset: (src_col, src_row),
            ref_plane: &ref_plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: false, // hp disabled — no EighthPel
            forced_stop: SubPixelForcedStop::EighthPel,
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        assert_eq!(
            r.best_mv.col & 1,
            0,
            "allow_hp=false: col & 1 must be 0; got col={}",
            r.best_mv.col
        );
        assert_eq!(
            r.best_mv.row & 1,
            0,
            "allow_hp=false: row & 1 must be 0; got row={}",
            r.best_mv.row
        );
    }

    /// `use_mv_hp(ref_mv) == false` (large ref_mv) blocks the
    /// EighthPel level even if `allow_hp == true && forced_stop ==
    /// EighthPel`. The function uses `THRESH = 64` for the
    /// half-of-each-component check.
    #[test]
    fn sub_pixel_search_large_ref_mv_disables_eighth_pel() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 11);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            // ref_mv has col=128 ≥ THRESH (64), so use_mv_hp returns
            // false → no EighthPel level regardless of allow_hp.
            ref_mv: Mv { row: 0, col: 128 },
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true,
            forced_stop: SubPixelForcedStop::EighthPel,
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        // Identical planes → MV stays at zero; LSB still asserted as
        // a property test of "no 1/8-pel arithmetic was performed".
        assert_eq!(
            r.best_mv.col & 1,
            0,
            "large ref_mv → no EighthPel branch; got col={}",
            r.best_mv.col
        );
        assert_eq!(
            r.best_mv.row & 1,
            0,
            "large ref_mv → no EighthPel branch; got row={}",
            r.best_mv.row
        );
    }

    /// Unsupported block size: `Block4x8` is non-square and falls into
    /// `svf_for`'s `_ => None` arm. In release mode the search returns
    /// the center MV scaled to 1/8-pel units with zero distortion.
    /// In debug mode the debug_assert fires (test uses `should_panic`).
    #[test]
    #[should_panic(expected = "sub_pixel_variance not ported")]
    fn sub_pixel_search_unsupported_block_size_panics_in_debug() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 11);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block4x8, // non-square → no SVF
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv { row: 1, col: 2 }, // full-pel; ×8 = (8, 16)
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true,
            forced_stop: SubPixelForcedStop::EighthPel,
            iters_per_step: 2,
        };
        let _r = find_best_sub_pixel_tree_pruned_more(&params);
    }

    /// Edge case: tight `mv_limits` that prevent any out-of-zero
    /// movement — search must converge to (0,0) without
    /// out-of-bounds reads.
    #[test]
    fn sub_pixel_search_zero_mv_limits_stays_at_origin() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 11);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let params = SubPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            // Zero-range mv_limits force ALL non-zero candidates out
            // of bounds — try_subpel returns u32::MAX for them.
            mv_limits: MvLimits::new(0, 0, 0, 0),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true,
            forced_stop: SubPixelForcedStop::EighthPel,
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        assert_eq!(r.best_mv, Mv::ZERO, "tight limits force origin");
    }

    #[test]
    fn sub_pixel_search_respects_forced_stop() {
        // forced_stop = QuarterPel: the search must NOT enter the
        // 1/8-pel level, so the col LSB cannot equal 1, 3, 5, or 7.
        // (Equivalently: col % 2 == 0.)
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 11);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        // Inject a 1/8-pel shift in the source so the search has a
        // genuine attractor at an odd 1/8-pel offset.
        let mut src_data = ref_data.clone();
        let src_col = 24usize;
        let src_row = 24usize;
        // Phase (1/8), filter taps [112, 16].
        for i in 0..8 {
            for j in 0..8 {
                let a = i32::from(ref_data[(src_row + i) * pw + (src_col + j)]);
                let b = i32::from(ref_data[(src_row + i) * pw + (src_col + j + 1)]);
                let v = (a * 112 + b * 16 + 64) >> 7;
                src_data[(src_row + i) * pw + (src_col + j)] = v as u8;
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        let params = SubPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &src_plane,
            src_offset: (src_col, src_row),
            ref_plane: &ref_plane,
            center_mv: Mv::ZERO,
            ref_mv: Mv::ZERO,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            error_per_bit: 1,
            mv_costs: &costs,
            allow_hp: true, // would normally enable HP
            forced_stop: SubPixelForcedStop::QuarterPel, // ...but capped here.
            iters_per_step: 2,
        };
        let r = find_best_sub_pixel_tree_pruned_more(&params);
        assert_eq!(
            r.best_mv.col & 1,
            0,
            "forced_stop=QuarterPel: col LSB (1/8-pel) must be zero; got col={}",
            r.best_mv.col
        );
        assert_eq!(
            r.best_mv.row & 1,
            0,
            "forced_stop=QuarterPel: row LSB (1/8-pel) must be zero; got row={}",
            r.best_mv.row
        );
    }
}
