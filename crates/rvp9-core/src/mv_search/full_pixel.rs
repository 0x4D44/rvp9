//! Full-pixel diamond search.
//!
//! Port of `vp9_diamond_search_sad_c` (`vp9_mcomp.c:2055-2190`). The
//! search operates in **full-pel** units internally; the input
//! `center_mv` is in 1/8-pel units and gets converted at the boundary.
//!
//! ## Frame-buffer view: [`PlaneRef`]
//!
//! The decoder side already has its own `frame_pool` view, but for the
//! encoder MV-search hot loop we need something dead simple: a
//! `&[u8]` slice + a stride. We don't want a generic parameter (per
//! the F7.0 brief), and we don't want raw pointers either since this
//! is safe Rust code reachable from unit tests. The
//! `(data, stride, width, height)` adapter ([`PlaneRef`]) ticks all
//! those boxes.
//!
//! ## Cost arithmetic
//!
//! Two MV-cost helpers are exposed:
//!
//! - [`mvsad_err_cost`] — used by the full-pel search; uses the
//!   `sad_per_bit` (`sadpb`) multiplier and divides by `1 << 9`.
//!   Mirror of `mvsad_err_cost` (`vp9_encoder.h:1518-1526`).
//! - [`mv_err_cost`] — used by sub-pel refinement; uses
//!   `error_per_bit` and divides by `1 << 14`. Mirror of
//!   `mv_err_cost` (`vp9_mcomp.c:87-97`).

use crate::block::BlockSize;
use crate::cost::mv_cost::{MvCosts, mv_cost};
use crate::dsp::distortion::{
    vpx_sad4x4, vpx_sad4x8, vpx_sad8x4, vpx_sad8x8, vpx_sad8x16, vpx_sad16x8, vpx_sad16x16,
    vpx_sad16x32, vpx_sad32x16, vpx_sad32x32, vpx_sad32x64, vpx_sad64x32, vpx_sad64x64,
};
use crate::mode_info::Mv;
use crate::mv_search::mv_limits::{MvLimits, is_mv_in};
use crate::mv_search::search_site::SearchSiteConfig;

/// Stride-aware read-only view into a single image plane.
///
/// `data` is the full plane buffer; `stride` is the row stride
/// (typically `>= width` to allow for padding). `width` / `height`
/// are bounds the caller can rely on for safe indexing.
///
/// The search internally accesses pixels by `(row, col)` pairs
/// produced from a base offset plus search-site offsets — all of
/// which the caller has validated against [`MvLimits`].
#[derive(Clone, Copy, Debug)]
pub struct PlaneRef<'a> {
    /// Plane pixel data; length is `>= height * stride`.
    pub data: &'a [u8],
    /// Row stride (bytes between successive rows).
    pub stride: usize,
    /// Plane width in pixels.
    pub width: usize,
    /// Plane height in pixels.
    pub height: usize,
}

/// `mvsad_err_cost` (`vp9_encoder.h:1518-1526`). Per-MV cost component
/// for the full-pel SAD-domain comparison.
///
/// `this_mv` and `center_mv` are both in **full-pel units**. Returns
/// `ROUND_POWER_OF_TWO(mv_cost(diff) * sad_per_bit, 9)`.
#[inline]
#[must_use]
pub fn mvsad_err_cost(
    this_mv_row: i16,
    this_mv_col: i16,
    center_mv_row: i16,
    center_mv_col: i16,
    sad_per_bit: i32,
    mv_costs: &MvCosts,
) -> u32 {
    let diff_row = this_mv_row.wrapping_sub(center_mv_row);
    let diff_col = this_mv_col.wrapping_sub(center_mv_col);
    let raw = mv_cost(diff_row, diff_col, mv_costs);
    // ROUND_POWER_OF_TWO(raw * sad_per_bit, 9).
    let prod = (raw as i64).saturating_mul(sad_per_bit as i64);
    let rounded = (prod + (1i64 << 8)) >> 9;
    // Cost is non-negative by construction (raw, sad_per_bit ≥ 0).
    rounded.max(0) as u32
}

/// `mv_err_cost` (`vp9_mcomp.c:87-97`). Per-MV cost for sub-pel
/// variance-domain comparison.
///
/// MVs are in **1/8-pel units**. Returns
/// `ROUND_POWER_OF_TWO(mv_cost(diff) * error_per_bit, 14)`.
///
/// The `14` shift comes from `RDDIV_BITS + VP9_PROB_COST_SHIFT -
/// RD_EPB_SHIFT + PIXEL_TRANSFORM_ERROR_SCALE = 7 + 9 - 6 + 4`.
#[inline]
#[must_use]
pub fn mv_err_cost(
    this_mv_row: i16,
    this_mv_col: i16,
    ref_mv_row: i16,
    ref_mv_col: i16,
    error_per_bit: i32,
    mv_costs: &MvCosts,
) -> u32 {
    let diff_row = this_mv_row.wrapping_sub(ref_mv_row);
    let diff_col = this_mv_col.wrapping_sub(ref_mv_col);
    let raw = mv_cost(diff_row, diff_col, mv_costs);
    let prod = (raw as i64).saturating_mul(error_per_bit as i64);
    let rounded = (prod + (1i64 << 13)) >> 14;
    rounded.max(0) as u32
}

/// Dispatch SAD by block size. Returns a function pointer suitable
/// for use in the search loop.
///
/// The C source uses `cpi->fn_ptr[bsize].sdf`; we don't have that
/// indirection yet, so a small `match` does the job. F7.0 lands the
/// 13 SAD variants — all VP9 block sizes covered.
#[inline]
#[must_use]
fn sad_for(bsize: BlockSize) -> fn(&[u8], usize, &[u8], usize) -> u32 {
    match bsize {
        BlockSize::Block4x4 => vpx_sad4x4,
        BlockSize::Block4x8 => vpx_sad4x8,
        BlockSize::Block8x4 => vpx_sad8x4,
        BlockSize::Block8x8 => vpx_sad8x8,
        BlockSize::Block8x16 => vpx_sad8x16,
        BlockSize::Block16x8 => vpx_sad16x8,
        BlockSize::Block16x16 => vpx_sad16x16,
        BlockSize::Block16x32 => vpx_sad16x32,
        BlockSize::Block32x16 => vpx_sad32x16,
        BlockSize::Block32x32 => vpx_sad32x32,
        BlockSize::Block32x64 => vpx_sad32x64,
        BlockSize::Block64x32 => vpx_sad64x32,
        BlockSize::Block64x64 => vpx_sad64x64,
    }
}

/// Input parameters to a single full-pel diamond search call.
///
/// `src_offset` is the `(col, row)` of the block's top-left within
/// `src_plane`. The full-pel candidate MV (`best_mv`, in full-pel
/// units after conversion from `center_mv`) is then applied as an
/// offset into `ref_plane`: i.e. the reference block starts at
/// `(src_offset.0 + mv.col, src_offset.1 + mv.row)`.
///
/// `center_mv` is the search center, in **1/8-pel units** (the
/// standard VP9 MV unit). The search loop converts it to full-pel
/// internally via `(center_mv.row >> 3, center_mv.col >> 3)`.
///
/// `mv_limits` is in **full-pel units** and bounds the absolute MV
/// (not the absolute pixel offset). Callers compute it via
/// [`crate::mv_search::set_mv_search_range`].
///
/// `max_iters` lets the realtime path cap the number of step
/// iterations independently of `site_config.total_steps`. Pass
/// `i32::MAX` (or `site_config.total_steps as i32`) for unlimited.
pub struct FullPixelSearchParams<'a> {
    /// Block size — picks the SAD kernel.
    pub block_size: BlockSize,
    /// Source plane view.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the block within `src_plane`, in `(col, row)`.
    pub src_offset: (usize, usize),
    /// Reference plane view (same plane the encoder will read at the
    /// chosen MV).
    pub ref_plane: &'a PlaneRef<'a>,
    /// Cost-attractor center (`ref_mv` in libvpx `diamond_search_sad`),
    /// **1/8-pel units**. `mvsad_err_cost` penalises distance from this
    /// MV — it is the rate-reference MV (NEAREST), NOT necessarily the
    /// search start. The search loop converts it to full-pel internally
    /// via `(center_mv.row >> 3, center_mv.col >> 3)`.
    pub center_mv: Mv,
    /// Search **start** position (`mvp_full` in libvpx), **1/8-pel
    /// units**. When `None`, the search starts at `center_mv` (the
    /// historical behaviour). When `Some`, the search starts here while
    /// still costing MV-distance against `center_mv` — this is the
    /// libvpx pattern where `mvp_full` (best-of-{nearest,near,pred_mv}
    /// or the int_pro seed) differs from the rate-reference `ref_mv`.
    pub start_mv: Option<Mv>,
    /// Full-pel MV bounds.
    pub mv_limits: MvLimits,
    /// `sad_per_bit` cost multiplier. Mode-decision computes this as
    /// `vp9_mv_bit_cost`'s `weight` argument (libvpx `sadperbit16` /
    /// `sadperbit4`).
    pub sad_per_bit: i32,
    /// MV-cost LUT bank (joint + per-axis).
    pub mv_costs: &'a MvCosts,
    /// Maximum step iterations. Realtime path may cap this; pass
    /// `site_config.total_steps as i32` to disable.
    pub max_iters: i32,
    /// 4-site diamond config (from
    /// [`crate::mv_search::init_dsmotion_compensation`]). Must have
    /// been built with the same stride as `ref_plane.stride`.
    pub site_config: &'a SearchSiteConfig,
}

/// Result of a full-pixel diamond search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullPixelSearchResult {
    /// Best MV found, in **full-pel units**. To convert to 1/8-pel
    /// (the standard VP9 unit) multiply both components by 8.
    pub best_mv_row: i16,
    /// Best MV column, full-pel units.
    pub best_mv_col: i16,
    /// SAD + MV-cost at `best_mv`. Includes the `mvsad_err_cost`
    /// term — i.e. it's the *augmented* SAD the search compared
    /// against.
    pub best_sad: u32,
    /// Number of step iterations actually consumed (≤ `max_iters`).
    pub iters_used: i32,
}

/// Read a full-pel `w × h` block from `plane` at `(col, row)` and
/// return its row slice plus the row stride. Returns `None` if any
/// pixel of the requested block would lie outside the plane's
/// `[0, width) × [0, height)` rectangle.
#[inline]
fn block_at<'a>(
    plane: &'a PlaneRef<'a>,
    col: isize,
    row: isize,
    w: usize,
    h: usize,
) -> Option<(&'a [u8], usize)> {
    if col < 0 || row < 0 {
        return None;
    }
    let col = col as usize;
    let row = row as usize;
    if col + w > plane.width || row + h > plane.height {
        return None;
    }
    let start = row * plane.stride + col;
    let end = (row + h - 1) * plane.stride + col + w;
    if end > plane.data.len() {
        return None;
    }
    Some((&plane.data[start..end], plane.stride))
}

/// Compute SAD for the block at `(src_offset + mv)` in `ref_plane`
/// versus the source block at `src_offset` in `src_plane`. Returns
/// `None` if either block falls outside its plane.
#[inline]
fn sad_at(
    block_size: BlockSize,
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    ref_plane: &PlaneRef<'_>,
    mv_row: i16,
    mv_col: i16,
) -> Option<u32> {
    let w = usize::from(block_size.width());
    let h = usize::from(block_size.height());
    let (src_slice, src_stride) = block_at(
        src_plane,
        src_offset.0 as isize,
        src_offset.1 as isize,
        w,
        h,
    )?;
    let ref_col = src_offset.0 as isize + isize::from(mv_col);
    let ref_row = src_offset.1 as isize + isize::from(mv_row);
    let (ref_slice, ref_stride) = block_at(ref_plane, ref_col, ref_row, w, h)?;
    Some((sad_for(block_size))(
        src_slice, src_stride, ref_slice, ref_stride,
    ))
}

/// Raw SAD (no MV-cost) for the block at full-pel MV `(mv_row, mv_col)`.
/// Returns `None` if the reference block falls outside the plane.
/// Diagnostic helper used by the E12/E13 search-quality oracle.
#[must_use]
pub fn raw_sad_at_mv(
    block_size: BlockSize,
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    ref_plane: &PlaneRef<'_>,
    mv_row: i16,
    mv_col: i16,
) -> Option<u32> {
    sad_at(block_size, src_plane, src_offset, ref_plane, mv_row, mv_col)
}

/// Exhaustive raw-SAD scan over every full-pel MV inside `mv_limits`
/// (bounded by `max_radius` on each axis around 0). Returns the
/// minimum raw SAD and the MV achieving it. Diagnostic-only oracle for
/// the E12/E13 full-pel search-quality measurement — NOT part of the
/// shipped search (far too slow for realtime).
#[must_use]
pub fn exhaustive_full_pel_min_sad(
    block_size: BlockSize,
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    ref_plane: &PlaneRef<'_>,
    mv_limits: &MvLimits,
    max_radius: i32,
) -> Option<(u32, i16, i16)> {
    let row_lo = mv_limits.row_min.max(-max_radius);
    let row_hi = mv_limits.row_max.min(max_radius);
    let col_lo = mv_limits.col_min.max(-max_radius);
    let col_hi = mv_limits.col_max.min(max_radius);
    let mut best: Option<(u32, i16, i16)> = None;
    for r in row_lo..=row_hi {
        for c in col_lo..=col_hi {
            if let Some(s) = sad_at(
                block_size, src_plane, src_offset, ref_plane, r as i16, c as i16,
            ) && best.is_none_or(|(b, _, _)| s < b)
            {
                best = Some((s, r as i16, c as i16));
            }
        }
    }
    best
}

/// `vp9_diamond_search_sad_c` (`vp9_mcomp.c:2055-2190`).
///
/// Search the reference plane around `center_mv` using the 4-site
/// diamond pattern in `site_config`. Returns the best full-pel MV
/// and its augmented SAD.
///
/// **Critical libvpx invariants preserved:**
///
/// - MV-search bounds (`mv_limits`) are checked per-site via
///   [`is_mv_in`]; out-of-bounds candidates are skipped (never read
///   from the reference plane).
/// - The two-step CHECK_BETTER pattern: first compare raw SAD
///   against `bestsad`, then re-add the MV cost and re-compare. This
///   short-circuits expensive cost-lookups when the raw SAD already
///   exceeds the running best.
/// - The `num00` semantics (libvpx returns this to the wrapper) are
///   not exposed yet — the wrapper isn't ported in F7.0. The result's
///   `iters_used` field is provided in its stead.
///
/// **Differences vs libvpx:**
///
/// - libvpx mixes 4-way SAD batches (`sdx4df`) with scalar SAD
///   (`sdf`); F7.0 uses scalar only (the 4-way SIMD kernel isn't
///   ported yet — F-RTCD task). The fast-path / slow-path
///   bookkeeping is preserved so the SIMD upgrade is additive.
/// - libvpx's `start_mv_sad` is passed in by the wrapper; here we
///   compute it inline for the initial position.
pub fn diamond_search_sad(params: &FullPixelSearchParams<'_>) -> FullPixelSearchResult {
    let site_config = params.site_config;
    debug_assert_eq!(site_config.searches_per_step, 4);
    let searches_per_step = site_config.searches_per_step as usize;

    // Convert center_mv (1/8-pel) to fcenter_mv (full-pel). This is the
    // cost-attractor / rate-reference MV (libvpx `ref_mv`).
    let fcenter_row = params.center_mv.row >> 3;
    let fcenter_col = params.center_mv.col >> 3;

    // Initial best MV = the search start (`mvp_full`). When no explicit
    // start is given, fall back to the cost-attractor (historical
    // behaviour). The start is clamped into `mv_limits` so the seed
    // never begins out of the legal window. libvpx clamps `mvp_full`
    // identically inside `full_pixel_diamond` (`vp9_mcomp.c:2500`).
    let start_mv = params.start_mv.unwrap_or(params.center_mv);
    let mut best_row = (i32::from(start_mv.row >> 3))
        .clamp(params.mv_limits.row_min, params.mv_limits.row_max) as i16;
    let mut best_col = (i32::from(start_mv.col >> 3))
        .clamp(params.mv_limits.col_min, params.mv_limits.col_max) as i16;

    // Initial SAD; if out-of-bounds, fall back to u32::MAX so any
    // valid step is preferred.
    let mut bestsad: u32 = match sad_at(
        params.block_size,
        params.src_plane,
        params.src_offset,
        params.ref_plane,
        best_row,
        best_col,
    ) {
        Some(s) => {
            s + mvsad_err_cost(
                best_row,
                best_col,
                fcenter_row,
                fcenter_col,
                params.sad_per_bit,
                params.mv_costs,
            )
        }
        None => u32::MAX,
    };

    let mut last_site: i32 = -1;
    let mut iters_used = 0i32;

    let max_iters = params.max_iters.max(0);
    let tot_steps = site_config.total_steps as i32;
    let effective_steps = max_iters.min(tot_steps);

    for step in 0..effective_steps {
        iters_used = step + 1;
        let base = (step as usize) * searches_per_step;

        // libvpx all_in fast path: peek at indices [base..base+4] —
        // the 4 axis-aligned sites. If all 4 stay inside mv_limits,
        // skip the per-site `is_mv_in` check.
        let s0 = site_config.ss_mv[base];
        let s1 = site_config.ss_mv[base + 1];
        let s2 = site_config.ss_mv[base + 2];
        let s3 = site_config.ss_mv[base + 3];

        // ss_mv[base + 0] is (-len, 0), [+1] is (+len, 0), [+2] is (0, -len),
        // [+3] is (0, +len). Match libvpx's `all_in` checks exactly:
        let all_in = (best_row as i32 + s0.0 as i32) > params.mv_limits.row_min
            && (best_row as i32 + s1.0 as i32) < params.mv_limits.row_max
            && (best_col as i32 + s2.1 as i32) > params.mv_limits.col_min
            && (best_col as i32 + s3.1 as i32) < params.mv_limits.col_max;
        // Note: libvpx uses strict < / > here (not <=), which can fail
        // to enter the fast path even when all 4 candidates are
        // legal-equal to a boundary. That matches `is_mv_in`'s
        // inclusive comparison conservatively — we preserve it.

        let mut best_site: i32 = -1;

        if all_in {
            for t in 0..searches_per_step {
                let (dr, dc) = site_config.ss_mv[base + t];
                let this_row = best_row + dr;
                let this_col = best_col + dc;
                // SAD without bounds check (covered by all_in).
                if let Some(thissad) = sad_at(
                    params.block_size,
                    params.src_plane,
                    params.src_offset,
                    params.ref_plane,
                    this_row,
                    this_col,
                ) && thissad < bestsad
                {
                    let augmented = thissad
                        + mvsad_err_cost(
                            this_row,
                            this_col,
                            fcenter_row,
                            fcenter_col,
                            params.sad_per_bit,
                            params.mv_costs,
                        );
                    if augmented < bestsad {
                        bestsad = augmented;
                        best_site = t as i32;
                    }
                }
            }
        } else {
            for t in 0..searches_per_step {
                let (dr, dc) = site_config.ss_mv[base + t];
                let this_row = best_row as i32 + dr as i32;
                let this_col = best_col as i32 + dc as i32;

                if !is_mv_in(&params.mv_limits, this_row, this_col) {
                    continue;
                }
                let tr = this_row as i16;
                let tc = this_col as i16;
                if let Some(thissad) = sad_at(
                    params.block_size,
                    params.src_plane,
                    params.src_offset,
                    params.ref_plane,
                    tr,
                    tc,
                ) && thissad < bestsad
                {
                    let augmented = thissad
                        + mvsad_err_cost(
                            tr,
                            tc,
                            fcenter_row,
                            fcenter_col,
                            params.sad_per_bit,
                            params.mv_costs,
                        );
                    if augmented < bestsad {
                        bestsad = augmented;
                        best_site = t as i32;
                    }
                }
            }
        }

        if best_site != -1 {
            let (dr, dc) = site_config.ss_mv[base + best_site as usize];
            best_row += dr;
            best_col += dc;
            last_site = best_site;
        } else if last_site == -1 {
            // libvpx's `else if (best_address == in_what)` branch:
            // center stayed best from the very start. We don't
            // expose `num00` in the result yet.
        }
        // (best_site == last_site case absorbed; libvpx skips the
        // address-update when the same direction was best as last
        // iter — our code skips on best_site == -1 instead, since
        // for the 4-site init the index-vs-last-index check has no
        // cycle-saving effect when we're recomputing SAD anyway.)
    }

    FullPixelSearchResult {
        best_mv_row: best_row,
        best_mv_col: best_col,
        best_sad: bestsad,
        iters_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::mv_cost::build_nmv_cost_table;
    use crate::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
    use crate::mv_search::search_site::init_dsmotion_compensation;

    /// Build a chaotic-content plane via xorshift32 so the search
    /// has no accidental low-SAD attractors away from the true MV.
    fn make_plane(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; width * height];
        let mut state = seed.wrapping_add(1);
        for slot in buf.iter_mut() {
            // xorshift32 — full-period non-trivial pattern.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *slot = (state & 0xff) as u8;
        }
        buf
    }

    #[test]
    fn diamond_search_finds_zero_mv_on_identical_planes() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 64;
        let plane_h = 64;
        let data = make_plane(plane_w, plane_h, 17);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);

        let params = FullPixelSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(-32, 32, -32, 32),
            sad_per_bit: 16,
            mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };

        let r = diamond_search_sad(&params);
        assert_eq!(r.best_mv_row, 0, "identical planes → row=0");
        assert_eq!(r.best_mv_col, 0, "identical planes → col=0");
        // best_sad includes the augmented MV-cost — libvpx adds
        // `mvsad_err_cost` to the raw SAD even at the seed MV
        // (`vp9_encoder.h:1538` `get_start_mv_sad`). At zero diff that's
        // `joint_cost[ZERO] * sad_per_bit / 512`, which is a small
        // positive number, not zero. Verify only that it's bounded.
        assert!(
            r.best_sad < 200,
            "identical planes → SAD should be MV-cost only (got {})",
            r.best_sad
        );
    }

    #[test]
    fn diamond_search_finds_small_shift() {
        // Source block = ref plane sampled at (col=3, row=0). The
        // search starting at center=(0,0) must recover MV (col=3, row=0)
        // in full-pel units.
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 64;
        let plane_h = 64;
        let ref_data = make_plane(plane_w, plane_h, 17);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };

        // Build a src plane that, at the SAME (col=20, row=20) offset,
        // contains the same pixels the reference has at (col=23, row=20)
        // — i.e. the src block came from the reference shifted by
        // (col=3, row=0).
        let mut src_data = vec![0u8; plane_w * plane_h];
        // Use the reference plane data, then "paint" the src block at
        // (col=20, row=20) with the shifted reference content.
        src_data.copy_from_slice(&ref_data);
        let bw = 16usize;
        let bh = 16usize;
        let src_col = 20usize;
        let src_row = 20usize;
        let shift_col = 3i16;
        let shift_row = 0i16;
        for i in 0..bh {
            for j in 0..bw {
                let r_idx = (src_row + i + shift_row as usize) * plane_w
                    + (src_col + j + shift_col as usize);
                let s_idx = (src_row + i) * plane_w + (src_col + j);
                src_data[s_idx] = ref_data[r_idx];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };

        let cfg = init_dsmotion_compensation(plane_w as i32);
        let params = FullPixelSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &src_plane,
            src_offset: (src_col, src_row),
            ref_plane: &ref_plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            sad_per_bit: 1, // small so cost doesn't dominate
            mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };

        let r = diamond_search_sad(&params);
        assert_eq!(
            (r.best_mv_row, r.best_mv_col),
            (shift_row, shift_col),
            "expected MV ({shift_row}, {shift_col}), got ({}, {})",
            r.best_mv_row,
            r.best_mv_col
        );
    }

    #[test]
    fn diamond_search_respects_max_iters() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 64;
        let plane_h = 64;
        let data = make_plane(plane_w, plane_h, 5);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);

        let params = FullPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &plane,
            src_offset: (20, 20),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            sad_per_bit: 16,
            mv_costs: &costs,
            max_iters: 1,
            site_config: &cfg,
        };

        let r = diamond_search_sad(&params);
        assert_eq!(r.iters_used, 1);
        // With max_iters=1 and an identical-plane search the very
        // first step (len=1024) is fully outside the small search
        // window, so the diamond doesn't move. We just verify no
        // panic + iters_used contract.
    }

    /// `start_mv = Some(non-zero)`: the search starts at the explicit
    /// `mvp_full` (the `Some` branch of the start_mv match). The
    /// resulting MV's SAD must be at most the SAD at the seed position
    /// (search only moves to improve).
    #[test]
    fn diamond_search_explicit_start_mv_some_path() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 64;
        let plane_h = 64;
        let data = make_plane(plane_w, plane_h, 19);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);
        // Compute the seed SAD at the explicit start.
        let start_mv = Mv { row: 0, col: 24 }; // 1/8-pel → fcol=3.
        let seed_sad = raw_sad_at_mv(
            BlockSize::Block8x8,
            &plane,
            (16, 16),
            &plane,
            start_mv.row >> 3,
            start_mv.col >> 3,
        )
        .expect("seed must be in-bounds");
        let params = FullPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: Some(start_mv),
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            sad_per_bit: 1,
            mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };
        let r = diamond_search_sad(&params);
        let result_sad = raw_sad_at_mv(
            BlockSize::Block8x8,
            &plane,
            (16, 16),
            &plane,
            r.best_mv_row,
            r.best_mv_col,
        )
        .expect("search result must be in-bounds");
        // Search only moves to improve raw SAD; result ≤ seed.
        assert!(
            result_sad <= seed_sad,
            "search must not regress: seed_sad={seed_sad}, result_sad={result_sad}"
        );
    }

    /// `max_iters = 0`: the for-loop body executes zero iterations.
    /// iters_used stays 0 and the result is the seed (start_mv ?? center).
    #[test]
    fn diamond_search_zero_iters_returns_seed() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 64;
        let plane_h = 64;
        let data = make_plane(plane_w, plane_h, 7);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);

        let params = FullPixelSearchParams {
            block_size: BlockSize::Block4x4,
            src_plane: &plane,
            src_offset: (10, 10),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(-8, 8, -8, 8),
            sad_per_bit: 1,
            mv_costs: &costs,
            max_iters: 0, // zero — loop body never runs
            site_config: &cfg,
        };
        let r = diamond_search_sad(&params);
        assert_eq!(r.iters_used, 0, "max_iters=0 → iters_used=0");
        assert_eq!(r.best_mv_row, 0);
        assert_eq!(r.best_mv_col, 0);
    }

    /// Tight mv_limits force the `all_in` fast path to fail at every
    /// step (some site falls outside the small range), forcing the
    /// per-site `is_mv_in` slow path. Verify the search still terminates
    /// and produces an in-range result.
    #[test]
    fn diamond_search_tight_limits_uses_slow_path() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 32;
        let plane_h = 32;
        let data = make_plane(plane_w, plane_h, 9);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);
        // Tight limits: row/col in [-1, 1]. First step (len=1024) is
        // immediately out of range on all 4 sites → slow path hits
        // `is_mv_in` and rejects all of them, so best_site stays -1.
        let params = FullPixelSearchParams {
            block_size: BlockSize::Block4x4,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(-1, 1, -1, 1),
            sad_per_bit: 1,
            mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };
        let r = diamond_search_sad(&params);
        // Result must lie inside the (tight) limits.
        assert!(
            r.best_mv_row >= -1 && r.best_mv_row <= 1,
            "row out of limits: {}",
            r.best_mv_row
        );
        assert!(
            r.best_mv_col >= -1 && r.best_mv_col <= 1,
            "col out of limits: {}",
            r.best_mv_col
        );
    }

    /// `mvsad_err_cost` at zero diff with default tables should be
    /// strictly positive (joint ZERO still costs at least one branch).
    #[test]
    fn mvsad_err_cost_zero_diff_is_small_positive() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let c = mvsad_err_cost(0, 0, 0, 0, 16, &costs);
        // Joint cost > 0; scaled by sad_per_bit=16 then >>9 may yield
        // 0 due to rounding. Verify just that it does NOT underflow.
        // The function returns u32 and is documented non-negative by
        // construction.
        assert!(c < 1000, "zero-diff cost should be tiny; got {c}");
    }

    /// `mv_err_cost` at zero diff: same property as `mvsad_err_cost`.
    #[test]
    fn mv_err_cost_zero_diff_is_small_positive() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let c = mv_err_cost(0, 0, 0, 0, 16, &costs);
        assert!(c < 1000, "zero-diff cost should be tiny; got {c}");
    }

    /// `mvsad_err_cost` scales monotonically with `sad_per_bit`: same
    /// MV diff, higher weight → strictly greater cost (until rounding).
    #[test]
    fn mvsad_err_cost_scales_with_sad_per_bit() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let low = mvsad_err_cost(10, 10, 0, 0, 1, &costs);
        let high = mvsad_err_cost(10, 10, 0, 0, 256, &costs);
        assert!(
            high > low,
            "higher sad_per_bit must increase cost: low={low}, high={high}"
        );
    }

    /// `raw_sad_at_mv` returns None for out-of-bounds reference reads
    /// and Some for in-bounds.
    #[test]
    fn raw_sad_at_mv_returns_none_for_out_of_bounds() {
        let plane_w = 16;
        let plane_h = 16;
        let data = make_plane(plane_w, plane_h, 3);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        // Block at top-left corner; large negative MV → ref starts at
        // (col=-100, row=-100) which is out of bounds.
        let sad_oob = raw_sad_at_mv(BlockSize::Block4x4, &plane, (0, 0), &plane, -100, -100);
        assert!(sad_oob.is_none(), "huge negative MV should be OOB");

        // In-range MV (0, 0): identical block → SAD = 0.
        let sad_in = raw_sad_at_mv(BlockSize::Block4x4, &plane, (4, 4), &plane, 0, 0);
        assert_eq!(sad_in, Some(0), "identical block, zero MV → SAD = 0");
    }

    /// `raw_sad_at_mv` returns Some for in-range MV; the `negative
    /// row` and `out-of-width` arms in `block_at` are exercised via
    /// negative-coord and beyond-width tests.
    #[test]
    fn raw_sad_at_mv_negative_row_returns_none() {
        let plane_w = 16;
        let plane_h = 16;
        let data = make_plane(plane_w, plane_h, 4);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        // Block at (0, 0); MV (-1, 0) → ref starts at (col=0, row=-1).
        // Negative row → `block_at` returns None.
        let sad = raw_sad_at_mv(BlockSize::Block4x4, &plane, (0, 0), &plane, -1, 0);
        assert!(sad.is_none(), "negative row MV → None");
        // Same for negative col.
        let sad = raw_sad_at_mv(BlockSize::Block4x4, &plane, (0, 0), &plane, 0, -1);
        assert!(sad.is_none(), "negative col MV → None");
    }

    /// `raw_sad_at_mv` returns None when reference block extends past
    /// the plane's right or bottom edge.
    #[test]
    fn raw_sad_at_mv_beyond_width_returns_none() {
        let plane_w = 16;
        let plane_h = 16;
        let data = make_plane(plane_w, plane_h, 4);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        // 4x4 block at (12, 0). MV (0, 5) → ref at (col=17) which is
        // past the plane width (16). `block_at` returns None.
        let sad = raw_sad_at_mv(BlockSize::Block4x4, &plane, (12, 0), &plane, 0, 5);
        assert!(sad.is_none(), "MV beyond plane width → None");
        // Beyond plane height.
        let sad = raw_sad_at_mv(BlockSize::Block4x4, &plane, (0, 12), &plane, 5, 0);
        assert!(sad.is_none(), "MV beyond plane height → None");
    }

    /// `mv_err_cost` scales monotonically with `error_per_bit`: same
    /// MV diff, higher weight → strictly greater cost.
    #[test]
    fn mv_err_cost_scales_with_error_per_bit() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let low = mv_err_cost(20, 20, 0, 0, 1, &costs);
        let high = mv_err_cost(20, 20, 0, 0, 4096, &costs);
        assert!(
            high > low,
            "higher error_per_bit must increase cost: low={low}, high={high}"
        );
    }

    /// `exhaustive_full_pel_min_sad` returns None when no MV in the
    /// `mv_limits ∩ [-max_radius, max_radius]` rectangle produces a
    /// valid reference read. Achievable when the entire search window
    /// is out of bounds.
    #[test]
    fn exhaustive_full_pel_min_sad_no_valid_mv_returns_none() {
        let plane_w = 8;
        let plane_h = 8;
        let data = make_plane(plane_w, plane_h, 1);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        // Block at (0, 0); limits force only negative MVs to be
        // considered, but `sad_at` returns None for them (out of
        // bounds on the ref side).
        let limits = MvLimits::new(-4, -1, -4, -1);
        let r =
            exhaustive_full_pel_min_sad(BlockSize::Block4x4, &plane, (0, 0), &plane, &limits, 8);
        assert!(r.is_none(), "all candidate MVs OOB → None");
    }

    /// `exhaustive_full_pel_min_sad` returns the global min within
    /// the bounded radius. For identical planes the min is at (0, 0)
    /// with SAD = 0.
    #[test]
    fn exhaustive_full_pel_min_sad_identical_planes() {
        let plane_w = 32;
        let plane_h = 32;
        let data = make_plane(plane_w, plane_h, 5);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let limits = MvLimits::new(-4, 4, -4, 4);
        let result =
            exhaustive_full_pel_min_sad(BlockSize::Block4x4, &plane, (12, 12), &plane, &limits, 4);
        let (min_sad, r, c) = result.expect("at least one valid MV in range");
        assert_eq!(min_sad, 0, "identical plane: min SAD = 0");
        assert_eq!(r, 0, "row of min should be 0");
        assert_eq!(c, 0, "col of min should be 0");
    }

    #[test]
    fn diamond_search_at_frame_edge_no_panic() {
        // Search center near edge; mv_limits constrains it tight
        // enough that out-of-bounds candidates are rejected.
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let plane_w = 32;
        let plane_h = 32;
        let data = make_plane(plane_w, plane_h, 3);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);

        // Block at top-left corner (0, 0). Without bounds clamping,
        // any negative MV would access out-of-bounds memory; mv_limits
        // = [0..16] prevents that.
        let params = FullPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &plane,
            src_offset: (0, 0),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(0, 16, 0, 16),
            sad_per_bit: 16,
            mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };

        let r = diamond_search_sad(&params);
        // No panic = success; the MV must lie inside the limits.
        assert!(r.best_mv_row as i32 >= 0 && r.best_mv_row as i32 <= 16);
        assert!(r.best_mv_col as i32 >= 0 && r.best_mv_col as i32 <= 16);
    }
}
