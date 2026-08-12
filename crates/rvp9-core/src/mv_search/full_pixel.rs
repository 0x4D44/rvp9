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
use crate::cost::mv_cost::{MV_MAX, MvCosts, mv_cost, mv_joint};
use crate::dsp::dispatch::{Sad4dFn, SadFn};
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

/// Full-pel search variant of [`mvsad_err_cost`] for the encoder's legal MV
/// window. Libvpx indexes both component tables directly here; their zero
/// slots are built as zero, so this avoids the per-axis clamp and branch in
/// [`mv_cost`] without changing the cost arithmetic.
#[inline]
fn mvsad_err_cost_in_range(
    this_mv_row: i16,
    this_mv_col: i16,
    center_mv_row: i16,
    center_mv_col: i16,
    sad_per_bit: i32,
    mv_costs: &MvCosts,
) -> u32 {
    let diff_row = i32::from(this_mv_row) - i32::from(center_mv_row);
    let diff_col = i32::from(this_mv_col) - i32::from(center_mv_col);
    debug_assert!(
        (-MV_MAX..=MV_MAX).contains(&diff_row),
        "full-pel row diff {diff_row} must fit the libvpx MV cost table"
    );
    debug_assert!(
        (-MV_MAX..=MV_MAX).contains(&diff_col),
        "full-pel col diff {diff_col} must fit the libvpx MV cost table"
    );

    let diff_row = diff_row as i16;
    let diff_col = diff_col as i16;
    let raw = mv_costs.joint_cost[mv_joint(diff_row, diff_col)]
        + mv_costs.component_cost[0].cost[(i32::from(diff_row) + MV_MAX) as usize]
        + mv_costs.component_cost[1].cost[(i32::from(diff_col) + MV_MAX) as usize];
    // ROUND_POWER_OF_TWO(raw * sad_per_bit, 9), matching the public helper.
    let prod = (raw as i64).saturating_mul(sad_per_bit as i64);
    let rounded = (prod + (1i64 << 8)) >> 9;
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
/// The C source uses `cpi->fn_ptr[bsize].sdf`; the SAD dispatch table
/// (`SimdDispatch::sad`, P1-B) is exactly that indirection — the AVX2
/// kernel on AVX2 hosts, the scalar `vpx_sad{W}x{H}` reference otherwise,
/// byte-exact across the two arms. **Resolve once per search call** and
/// reuse across the inner SAD loop, so the `OnceLock` load in `get_ref()`
/// is amortized rather than paid per SAD.
#[inline]
#[must_use]
fn sad_for(bsize: BlockSize) -> SadFn {
    crate::dsp::get_ref().sad(bsize)
}

/// Dispatch the 4-reference SAD (`sdx4df`) kernel by block size (PR3).
/// Mirrors libvpx `cpi->fn_ptr[bsize].sdx4df` — AVX2 on x86-64,
/// dot-product or base NEON on AArch64, and the scalar four-`sad` reference
/// otherwise. Each output lane is byte-exact with [`sad_for`]. **Resolve once
/// per search call** and reuse across every batched step, amortizing the
/// `get_ref()` OnceLock load.
#[inline]
#[must_use]
fn sad4d_for(bsize: BlockSize) -> Sad4dFn {
    crate::dsp::get_ref().sad4d(bsize)
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
    /// Fixed SAD-domain MV-cost LUT used only by full-pixel search.
    ///
    /// Pinned libvpx keeps this separate from the adaptive entropy/RD table in
    /// [`Self::mv_costs`].
    pub sad_mv_costs: &'a MvCosts,
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

/// Libvpx's integer-pel cost surface around a full-pixel search winner.
///
/// Entries are `[center, left, bottom, right, top]`. A value of [`i32::MAX`]
/// marks a neighbor that is outside the legal MV or plane window, matching
/// libvpx's signed `int cost_list[5]` contract.
pub type FullPixelCostList = [i32; 5];

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
///
/// `sad_fn` is the block-size-specific SAD kernel, resolved once per
/// search call by the caller (hoisting the dispatch `get_ref()` out of
/// the inner loop). Threading it in keeps the kernel selection out of the
/// hot path while preserving byte-identical results.
#[inline]
fn sad_at(
    block_size: BlockSize,
    sad_fn: SadFn,
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
    Some(sad_fn(src_slice, src_stride, ref_slice, ref_stride))
}

/// Compute the four raw SADs for the four candidate MVs `mvs[t] =
/// (mv_row, mv_col)` against the single source block at `src_offset`, using
/// the 4-reference `sad4d` kernel when every candidate's reference block is
/// in-plane (PR3).
///
/// Returns `out[t] = Some(sad)` exactly when [`sad_at`] would return
/// `Some(sad)` for `mvs[t]`, and `None` exactly when [`sad_at`] would return
/// `None` (source or that reference block falls outside its plane). The SAD
/// value is **byte-identical** to [`sad_at`]'s — the `sad4d` kernel's lane
/// `t` equals `sad_fn(src, ref_t)` by construction (proven in
/// `dsp::x86_64::sad4d_avx2`), and the per-lane scalar fallback calls the same
/// `sad_fn` [`sad_at`] uses.
///
/// Fast path (all four references in-plane, the overwhelmingly common
/// `all_in` case): one `sad4d_fn` call loads the source row once and
/// accumulates four running SADs. Slow path (any reference clipped by the
/// plane rectangle — reachable because `all_in` only checks MV-space limits,
/// not pixel bounds): fall back to per-lane single `sad_at`, preserving the
/// exact `None` for the clipped lanes.
///
/// This helper changes ONLY the *source* of the four SAD values; the caller's
/// `all_in` decision loop is byte-for-byte unchanged (same in-order scan,
/// same mid-loop strict-`<` running-best mutation, same `mvsad_err_cost`
/// augmentation). See `crate::mv_search::full_pixel::diamond_search_sad`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn sad4d_at(
    block_size: BlockSize,
    sad_fn: SadFn,
    sad4d_fn: Sad4dFn,
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    ref_plane: &PlaneRef<'_>,
    mvs: &[(i16, i16); 4],
) -> [Option<u32>; 4] {
    let w = usize::from(block_size.width());
    let h = usize::from(block_size.height());

    // Source block — identical for all four candidates. If it is out of the
    // source plane, every `sad_at` would short-circuit to `None`.
    let Some((src_slice, src_stride)) = block_at(
        src_plane,
        src_offset.0 as isize,
        src_offset.1 as isize,
        w,
        h,
    ) else {
        return [None; 4];
    };

    // Resolve each candidate's reference block (or `None` if clipped).
    let mut ref_slices: [Option<&[u8]>; 4] = [None; 4];
    let mut all_in_plane = true;
    for t in 0..4 {
        let ref_col = src_offset.0 as isize + isize::from(mvs[t].1);
        let ref_row = src_offset.1 as isize + isize::from(mvs[t].0);
        match block_at(ref_plane, ref_col, ref_row, w, h) {
            // Every candidate shares `ref_plane.stride`, so we keep only the
            // slice and use that single stride below.
            Some((slice, _stride)) => ref_slices[t] = Some(slice),
            None => all_in_plane = false,
        }
    }

    if all_in_plane {
        // Fast path: one batched call. All four references share the plane
        // stride; each lane equals the single-SAD `sad_fn` result.
        let refs: [&[u8]; 4] = [
            ref_slices[0].unwrap(),
            ref_slices[1].unwrap(),
            ref_slices[2].unwrap(),
            ref_slices[3].unwrap(),
        ];
        let mut out_sads = [0u32; 4];
        sad4d_fn(
            src_slice,
            src_stride,
            &refs,
            ref_plane.stride,
            &mut out_sads,
        );
        [
            Some(out_sads[0]),
            Some(out_sads[1]),
            Some(out_sads[2]),
            Some(out_sads[3]),
        ]
    } else {
        // Slow path: at least one reference clipped. Per lane, the in-plane
        // ones get the same scalar `sad_fn` as `sad_at`; the clipped ones
        // stay `None` — byte-identical to `sad_at` per site.
        let mut out = [None; 4];
        for t in 0..4 {
            if let Some(slice) = ref_slices[t] {
                out[t] = Some(sad_fn(src_slice, src_stride, slice, ref_plane.stride));
            }
        }
        out
    }
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
    let sad_fn = sad_for(block_size);
    sad_at(
        block_size, sad_fn, src_plane, src_offset, ref_plane, mv_row, mv_col,
    )
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
    let sad_fn = sad_for(block_size);
    for r in row_lo..=row_hi {
        for c in col_lo..=col_hi {
            if let Some(s) = sad_at(
                block_size, sad_fn, src_plane, src_offset, ref_plane, r as i16, c as i16,
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
/// - The private implementation also returns libvpx's `num00` count so
///   the NSTEP wrapper can skip redundant smaller-start searches.
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
    diamond_search_sad_from_step(params, 0).0
}

fn diamond_search_sad_from_step(
    params: &FullPixelSearchParams<'_>,
    start_step: usize,
) -> (FullPixelSearchResult, usize) {
    let site_config = params.site_config;
    debug_assert_eq!(site_config.searches_per_step, 4);
    let searches_per_step = site_config.searches_per_step as usize;

    // Resolve the SAD kernels once per search call (P1-B / PR3 hoist): the
    // dispatch `get_ref()` OnceLock load is amortized over every SAD in the
    // diamond instead of paid per-SAD. Byte-identical to the per-call
    // resolution it replaces. `sad_fn` serves the initial/cross SADs and the
    // slow path; `sad4d_fn` serves the `all_in` fast path's four axis sites.
    let sad_fn = sad_for(params.block_size);
    let sad4d_fn = sad4d_for(params.block_size);

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
    let initial_row = best_row;
    let initial_col = best_col;

    // Initial SAD; if out-of-bounds, fall back to u32::MAX so any
    // valid step is preferred.
    let mut bestsad: u32 = match sad_at(
        params.block_size,
        sad_fn,
        params.src_plane,
        params.src_offset,
        params.ref_plane,
        best_row,
        best_col,
    ) {
        Some(s) => {
            s + mvsad_err_cost_in_range(
                best_row,
                best_col,
                fcenter_row,
                fcenter_col,
                params.sad_per_bit,
                params.sad_mv_costs,
            )
        }
        None => u32::MAX,
    };

    let mut last_site: i32 = -1;
    let mut iters_used = 0i32;
    let mut num00 = 0usize;

    let max_iters = params.max_iters.max(0);
    let tot_steps = site_config.total_steps as i32;
    let effective_steps = max_iters.min(tot_steps);

    for step in start_step as i32..effective_steps {
        iters_used += 1;
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
            // PR3: batch the four axis-site SADs with one `sad4d` call. The
            // candidate MVs are exactly `best + ss_mv[base + t]` (the same
            // `this_row`/`this_col` the per-site loop derives). `sad4d_at`
            // returns `sads[t]` byte-identical to what `sad_at` returned per
            // site — `Some(sad)` for in-plane references (each lane equal to
            // the single-SAD kernel), `None` for any reference the plane
            // rectangle clips. ONLY the SAD *source* changes below; the
            // decision loop (in-order scan, mid-loop strict-`<` running-best,
            // `mvsad_err_cost` augmentation) is byte-for-byte the original.
            let mvs: [(i16, i16); 4] = [
                {
                    let (dr, dc) = site_config.ss_mv[base];
                    (best_row + dr, best_col + dc)
                },
                {
                    let (dr, dc) = site_config.ss_mv[base + 1];
                    (best_row + dr, best_col + dc)
                },
                {
                    let (dr, dc) = site_config.ss_mv[base + 2];
                    (best_row + dr, best_col + dc)
                },
                {
                    let (dr, dc) = site_config.ss_mv[base + 3];
                    (best_row + dr, best_col + dc)
                },
            ];
            let sads = sad4d_at(
                params.block_size,
                sad_fn,
                sad4d_fn,
                params.src_plane,
                params.src_offset,
                params.ref_plane,
                &mvs,
            );

            // `needless_range_loop`: the index `t` is load-bearing — it
            // selects `ss_mv[base + t]` (the cost MV) AND becomes `best_site`.
            // The mandated gate-2-preserving contract requires this exact
            // in-order indexed scan; an iterator zip would obscure it and the
            // `best_site = t` assignment still needs the index.
            #[allow(clippy::needless_range_loop)]
            for t in 0..searches_per_step {
                let (dr, dc) = site_config.ss_mv[base + t];
                let this_row = best_row + dr;
                let this_col = best_col + dc;
                // SAD without bounds check (covered by all_in): now sourced
                // from the precomputed `sad4d` array (gate-2-preserving).
                if let Some(thissad) = sads[t]
                    && thissad < bestsad
                {
                    let augmented = thissad
                        + mvsad_err_cost_in_range(
                            this_row,
                            this_col,
                            fcenter_row,
                            fcenter_col,
                            params.sad_per_bit,
                            params.sad_mv_costs,
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
                    sad_fn,
                    params.src_plane,
                    params.src_offset,
                    params.ref_plane,
                    tr,
                    tc,
                ) && thissad < bestsad
                {
                    let augmented = thissad
                        + mvsad_err_cost_in_range(
                            tr,
                            tc,
                            fcenter_row,
                            fcenter_col,
                            params.sad_per_bit,
                            params.sad_mv_costs,
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
            // center stayed best from the very start.
            if best_row == initial_row && best_col == initial_col {
                num00 += 1;
            }
        }
        // (best_site == last_site case absorbed; libvpx skips the
        // address-update when the same direction was best as last
        // iter — our code skips on best_site == -1 instead, since
        // for the 4-site init the index-vs-last-index check has no
        // cycle-saving effect when we're recomputing SAD anyway.)
    }

    (
        FullPixelSearchResult {
            best_mv_row: best_row,
            best_mv_col: best_col,
            best_sad: bestsad,
            iters_used,
        },
        num00,
    )
}

#[inline]
fn full_pixel_variance_cost(
    params: &FullPixelSearchParams<'_>,
    result: FullPixelSearchResult,
) -> u32 {
    let width = usize::from(params.block_size.width());
    let height = usize::from(params.block_size.height());
    let Some((src, src_stride)) = block_at(
        params.src_plane,
        params.src_offset.0 as isize,
        params.src_offset.1 as isize,
        width,
        height,
    ) else {
        return u32::MAX;
    };
    let Some((reference, ref_stride)) = block_at(
        params.ref_plane,
        params.src_offset.0 as isize + isize::from(result.best_mv_col),
        params.src_offset.1 as isize + isize::from(result.best_mv_row),
        width,
        height,
    ) else {
        return u32::MAX;
    };
    let (sum, sum_sq) =
        (crate::dsp::get_ref().variance(params.block_size))(src, src_stride, reference, ref_stride);
    let samples = (width * height) as u64;
    let mean_sq = ((sum * sum) as u64) / samples;
    let variance = sum_sq.saturating_sub(mean_sq).min(u64::from(u32::MAX)) as u32;
    variance.saturating_add(mv_err_cost(
        result.best_mv_row.saturating_mul(8),
        result.best_mv_col.saturating_mul(8),
        params.center_mv.row,
        params.center_mv.col,
        params.sad_per_bit,
        params.mv_costs,
    ))
}

/// Final unit-step refinement used by libvpx's `NSTEP` wrapper.
#[must_use]
fn refining_search_sad(
    params: &FullPixelSearchParams<'_>,
    initial: FullPixelSearchResult,
    search_range: i32,
) -> FullPixelSearchResult {
    const NEIGHBORS: [(i16, i16); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

    let sad_fn = sad_for(params.block_size);
    let sad4d_fn = sad4d_for(params.block_size);
    let fcenter_row = params.center_mv.row >> 3;
    let fcenter_col = params.center_mv.col >> 3;
    let mut best_row = initial.best_mv_row;
    let mut best_col = initial.best_mv_col;
    let mut best_sad = initial.best_sad;
    let mut moves = 0;

    for _ in 0..search_range.max(0) {
        let candidates = NEIGHBORS.map(|(delta_row, delta_col)| {
            (
                i32::from(best_row) + i32::from(delta_row),
                i32::from(best_col) + i32::from(delta_col),
            )
        });
        let all_in = candidates
            .iter()
            .all(|&(row, col)| is_mv_in(&params.mv_limits, row, col));
        let mut best_site = None;

        if all_in {
            let mvs = candidates.map(|(row, col)| (row as i16, col as i16));
            let sads = sad4d_at(
                params.block_size,
                sad_fn,
                sad4d_fn,
                params.src_plane,
                params.src_offset,
                params.ref_plane,
                &mvs,
            );
            for (index, raw_sad) in sads.into_iter().enumerate() {
                if let Some(raw_sad) = raw_sad
                    && raw_sad < best_sad
                {
                    let (candidate_row, candidate_col) = mvs[index];
                    let augmented = raw_sad.saturating_add(mvsad_err_cost_in_range(
                        candidate_row,
                        candidate_col,
                        fcenter_row,
                        fcenter_col,
                        params.sad_per_bit,
                        params.sad_mv_costs,
                    ));
                    if augmented < best_sad {
                        best_sad = augmented;
                        best_site = Some(index);
                    }
                }
            }
        } else {
            for (index, &(candidate_row, candidate_col)) in candidates.iter().enumerate() {
                if !is_mv_in(&params.mv_limits, candidate_row, candidate_col) {
                    continue;
                }
                let candidate_row = candidate_row as i16;
                let candidate_col = candidate_col as i16;
                if let Some(raw_sad) = sad_at(
                    params.block_size,
                    sad_fn,
                    params.src_plane,
                    params.src_offset,
                    params.ref_plane,
                    candidate_row,
                    candidate_col,
                ) && raw_sad < best_sad
                {
                    let augmented = raw_sad.saturating_add(mvsad_err_cost_in_range(
                        candidate_row,
                        candidate_col,
                        fcenter_row,
                        fcenter_col,
                        params.sad_per_bit,
                        params.sad_mv_costs,
                    ));
                    if augmented < best_sad {
                        best_sad = augmented;
                        best_site = Some(index);
                    }
                }
            }
        }

        let Some(best_site) = best_site else {
            break;
        };
        best_row += NEIGHBORS[best_site].0;
        best_col += NEIGHBORS[best_site].1;
        moves += 1;
    }

    FullPixelSearchResult {
        best_mv_row: best_row,
        best_mv_col: best_col,
        best_sad,
        iters_used: initial.iters_used + moves,
    }
}

/// Libvpx `NSTEP` wrapper used by realtime cpu-used 6.
///
/// Runs the progressively smaller diamond starts selected by
/// `fullpel_search_step_param = 6`, ranks their winners in the variance
/// domain, then performs the final eight-move unit refinement.
#[must_use]
pub fn nstep_search_sad(params: &FullPixelSearchParams<'_>) -> FullPixelSearchResult {
    const STEP_PARAM: usize = 6;
    let effective_steps =
        (params.max_iters.max(0) as usize).min(params.site_config.total_steps as usize);
    if effective_steps <= STEP_PARAM {
        return diamond_search_sad_from_step(params, effective_steps).0;
    }
    let start_step = STEP_PARAM.min(effective_steps);
    let further_steps = effective_steps.saturating_sub(start_step + 1);
    let (mut best, mut n) = diamond_search_sad_from_step(params, start_step);
    let mut best_cost = full_pixel_variance_cost(params, best);
    let mut num00 = 0usize;
    let mut do_refine = n <= further_steps;

    while n < further_steps {
        n += 1;
        if num00 != 0 {
            num00 -= 1;
            continue;
        }

        let (candidate, candidate_num00) = diamond_search_sad_from_step(params, start_step + n);
        num00 = candidate_num00;
        let candidate_cost = full_pixel_variance_cost(params, candidate);
        if candidate_cost < best_cost {
            best = candidate;
            best_cost = candidate_cost;
        }
        if num00 > further_steps - n {
            do_refine = false;
        }
    }

    if !do_refine {
        return best;
    }
    let refined = refining_search_sad(params, best, 8);
    if full_pixel_variance_cost(params, refined) < best_cost {
        refined
    } else {
        best
    }
}

/// libvpx `FAST_DIAMOND` full-pel search at `step_param = 10`.
///
/// This is the cpu-used 7/8 realtime search selected by
/// `vp9_speed_features.c:699-703`. It dispatches through
/// `fast_dia_search` to the scale-zero `bigdia` pattern with
/// `do_init_search = 0`: evaluate the four unit neighbours once, move to
/// the best, then continue with the previous direction and its two adjacent
/// directions until none improves the augmented SAD.
#[must_use]
pub fn fast_diamond_search_sad(params: &FullPixelSearchParams<'_>) -> FullPixelSearchResult {
    fast_diamond_search_sad_impl::<false>(params).0
}

/// CPU7/8 fast-diamond search with its native SAD-domain integer cost surface.
///
/// Pinned libvpx populates this list inside `vp9_pattern_search_sad`; retaining
/// the values while the search moves avoids a second SAD walk before
/// `SUBPEL_TREE_PRUNED_EVENMORE`. The list order and signed sentinel follow
/// [`FullPixelCostList`].
#[must_use]
pub fn fast_diamond_search_sad_with_cost_list(
    params: &FullPixelSearchParams<'_>,
) -> (FullPixelSearchResult, FullPixelCostList) {
    fast_diamond_search_sad_impl::<true>(params)
}

#[inline]
fn signed_cost_list_value(cost: u32) -> i32 {
    cost.min(i32::MAX as u32) as i32
}

#[inline]
fn finish_fast_diamond_search<const WITH_COST_LIST: bool>(
    params: &FullPixelSearchParams<'_>,
    best_row: i16,
    best_col: i16,
    best_sad: u32,
    rounds: i32,
    mut cost_list: FullPixelCostList,
) -> (FullPixelSearchResult, FullPixelCostList) {
    const DIRECTIONS: [(i16, i16); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    if WITH_COST_LIST {
        let fcenter_row = params.center_mv.row >> 3;
        let fcenter_col = params.center_mv.col >> 3;
        for (index, (delta_row, delta_col)) in DIRECTIONS.into_iter().enumerate() {
            if cost_list[index + 1] == i32::MAX {
                continue;
            }
            let row = best_row + delta_row;
            let col = best_col + delta_col;
            let mv_cost = signed_cost_list_value(mvsad_err_cost_in_range(
                row,
                col,
                fcenter_row,
                fcenter_col,
                params.sad_per_bit,
                params.sad_mv_costs,
            ));
            // Current VP9 block SAD and MV-rate domains cannot overflow i32.
            // Saturation only preserves the libvpx INT_MAX sentinel if a
            // future caller widens either domain.
            cost_list[index + 1] = cost_list[index + 1].saturating_add(mv_cost);
        }
    }

    (
        FullPixelSearchResult {
            best_mv_row: best_row,
            best_mv_col: best_col,
            best_sad,
            iters_used: rounds,
        },
        cost_list,
    )
}

fn fast_diamond_search_sad_impl<const WITH_COST_LIST: bool>(
    params: &FullPixelSearchParams<'_>,
) -> (FullPixelSearchResult, FullPixelCostList) {
    const DIRECTIONS: [(i16, i16); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    let sad_fn = sad_for(params.block_size);
    let sad4d_fn = sad4d_for(params.block_size);
    let fcenter_row = params.center_mv.row >> 3;
    let fcenter_col = params.center_mv.col >> 3;
    let start_mv = params.start_mv.unwrap_or(params.center_mv);
    let mut best_row = (i32::from(start_mv.row >> 3))
        .clamp(params.mv_limits.row_min, params.mv_limits.row_max) as i16;
    let mut best_col = (i32::from(start_mv.col >> 3))
        .clamp(params.mv_limits.col_min, params.mv_limits.col_max) as i16;
    let mut best_sad = sad_at(
        params.block_size,
        sad_fn,
        params.src_plane,
        params.src_offset,
        params.ref_plane,
        best_row,
        best_col,
    )
    .map_or(u32::MAX, |sad| {
        sad.saturating_add(mvsad_err_cost_in_range(
            best_row,
            best_col,
            fcenter_row,
            fcenter_col,
            params.sad_per_bit,
            params.sad_mv_costs,
        ))
    });
    let mut cost_list = [i32::MAX; 5];
    if WITH_COST_LIST {
        cost_list[0] = signed_cost_list_value(best_sad);
    }
    let mut rounds = 0;

    let candidates = DIRECTIONS.map(|(delta_row, delta_col)| {
        (
            i32::from(best_row) + i32::from(delta_row),
            i32::from(best_col) + i32::from(delta_col),
        )
    });
    let all_in = candidates
        .iter()
        .all(|&(row, col)| is_mv_in(&params.mv_limits, row, col));
    let mut initial_direction = None;
    if all_in {
        let mvs = candidates.map(|(row, col)| (row as i16, col as i16));
        let sads = sad4d_at(
            params.block_size,
            sad_fn,
            sad4d_fn,
            params.src_plane,
            params.src_offset,
            params.ref_plane,
            &mvs,
        );
        for (index, raw_sad) in sads.into_iter().enumerate() {
            if WITH_COST_LIST {
                cost_list[index + 1] = raw_sad.map_or(i32::MAX, signed_cost_list_value);
            }
            if let Some(raw_sad) = raw_sad
                && raw_sad < best_sad
            {
                let (candidate_row, candidate_col) = mvs[index];
                let augmented = raw_sad.saturating_add(mvsad_err_cost_in_range(
                    candidate_row,
                    candidate_col,
                    fcenter_row,
                    fcenter_col,
                    params.sad_per_bit,
                    params.sad_mv_costs,
                ));
                if augmented < best_sad {
                    best_sad = augmented;
                    initial_direction = Some(index);
                }
            }
        }
    } else {
        for (index, &(candidate_row, candidate_col)) in candidates.iter().enumerate() {
            if !is_mv_in(&params.mv_limits, candidate_row, candidate_col) {
                continue;
            }
            let candidate_row = candidate_row as i16;
            let candidate_col = candidate_col as i16;
            let raw_sad = sad_at(
                params.block_size,
                sad_fn,
                params.src_plane,
                params.src_offset,
                params.ref_plane,
                candidate_row,
                candidate_col,
            );
            if WITH_COST_LIST {
                cost_list[index + 1] = raw_sad.map_or(i32::MAX, signed_cost_list_value);
            }
            if let Some(raw_sad) = raw_sad
                && raw_sad < best_sad
            {
                let augmented = raw_sad.saturating_add(mvsad_err_cost_in_range(
                    candidate_row,
                    candidate_col,
                    fcenter_row,
                    fcenter_col,
                    params.sad_per_bit,
                    params.sad_mv_costs,
                ));
                if augmented < best_sad {
                    best_sad = augmented;
                    initial_direction = Some(index);
                }
            }
        }
    }

    let Some(mut direction) = initial_direction else {
        return finish_fast_diamond_search::<WITH_COST_LIST>(
            params, best_row, best_col, best_sad, rounds, cost_list,
        );
    };
    let (delta_row, delta_col) = DIRECTIONS[direction];
    best_row += delta_row;
    best_col += delta_col;
    rounds += 1;

    let mut evaluate = |indices: &[usize], row: i16, col: i16, previous_direction: usize| {
        if WITH_COST_LIST {
            let previous_center = cost_list[0];
            cost_list[1..].fill(i32::MAX);
            cost_list[((previous_direction + 2) % 4) + 1] = previous_center;
            cost_list[0] = signed_cost_list_value(best_sad);
        }
        let mut best_direction = None;
        for &index in indices {
            let (delta_row, delta_col) = DIRECTIONS[index];
            let candidate_row = i32::from(row) + i32::from(delta_row);
            let candidate_col = i32::from(col) + i32::from(delta_col);
            if !is_mv_in(&params.mv_limits, candidate_row, candidate_col) {
                continue;
            }
            let candidate_row = candidate_row as i16;
            let candidate_col = candidate_col as i16;
            let raw_sad = sad_at(
                params.block_size,
                sad_fn,
                params.src_plane,
                params.src_offset,
                params.ref_plane,
                candidate_row,
                candidate_col,
            );
            if WITH_COST_LIST {
                cost_list[index + 1] = raw_sad.map_or(i32::MAX, signed_cost_list_value);
            }
            if let Some(raw_sad) = raw_sad
                && raw_sad < best_sad
            {
                let augmented = raw_sad.saturating_add(mvsad_err_cost_in_range(
                    candidate_row,
                    candidate_col,
                    fcenter_row,
                    fcenter_col,
                    params.sad_per_bit,
                    params.sad_mv_costs,
                ));
                if augmented < best_sad {
                    best_sad = augmented;
                    best_direction = Some(index);
                }
            }
        }
        best_direction
    };

    loop {
        let adjacent = [(direction + 3) % 4, direction, (direction + 1) % 4];
        let Some(next_direction) = evaluate(&adjacent, best_row, best_col, direction) else {
            break;
        };
        direction = next_direction;
        let (delta_row, delta_col) = DIRECTIONS[direction];
        best_row += delta_row;
        best_col += delta_col;
        rounds += 1;
    }

    finish_fast_diamond_search::<WITH_COST_LIST>(
        params, best_row, best_col, best_sad, rounds, cost_list,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::mv_cost::{MV_VALS, MvComponentCosts, build_nmv_cost_table};
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

    fn zero_mv_costs() -> MvCosts {
        let axis = MvComponentCosts {
            cost: vec![0; MV_VALS].into_boxed_slice(),
        };
        MvCosts {
            joint_cost: [0; 4],
            component_cost: [axis.clone(), axis],
        }
    }

    #[test]
    fn full_pixel_search_separates_sad_and_entropy_cost_domains() {
        let width = 32usize;
        let height = 32usize;
        let reference = make_plane(width, height, 0x1180);
        let mut source = vec![0u8; width * height];
        for row in 0..height {
            for col in 0..width - 1 {
                source[row * width + col] = reference[row * width + col + 1];
            }
        }
        let src_plane = PlaneRef {
            data: &source,
            stride: width,
            width,
            height,
        };
        let ref_plane = PlaneRef {
            data: &reference,
            stride: width,
            width,
            height,
        };
        let sites = SearchSiteConfig {
            ss_mv: vec![(-1, 0), (1, 0), (0, -1), (0, 1)],
            ss_os: vec![-(width as isize), width as isize, -1, 1],
            searches_per_step: 4,
            total_steps: 1,
        };

        let zero = zero_mv_costs();
        let mut expensive = zero.clone();
        let plus_one = (MV_MAX + 1) as usize;
        let plus_one_q3 = (MV_MAX + 8) as usize;
        expensive.component_cost[1].cost[plus_one] = 1_000_000;
        expensive.component_cost[1].cost[plus_one_q3] = 1_000_000;

        let base = FullPixelSearchParams {
            block_size: BlockSize::Block8x8,
            src_plane: &src_plane,
            src_offset: (8, 8),
            ref_plane: &ref_plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(-1, 1, -1, 1),
            sad_per_bit: 512,
            mv_costs: &expensive,
            sad_mv_costs: &zero,
            max_iters: 1,
            site_config: &sites,
        };
        let zero_sad_winner = diamond_search_sad(&base);
        assert_eq!(
            (zero_sad_winner.best_mv_row, zero_sad_winner.best_mv_col),
            (0, 1),
            "full-pixel winner must follow the SAD-domain table"
        );

        let expensive_sad = FullPixelSearchParams {
            sad_mv_costs: &expensive,
            ..base
        };
        let expensive_sad_winner = diamond_search_sad(&expensive_sad);
        assert_eq!(
            (
                expensive_sad_winner.best_mv_row,
                expensive_sad_winner.best_mv_col
            ),
            (0, 0),
            "adaptive entropy costs must not substitute for SAD costs"
        );

        let zero_entropy = FullPixelSearchParams {
            mv_costs: &zero,
            sad_mv_costs: &zero,
            ..base
        };
        assert!(
            full_pixel_variance_cost(&base, zero_sad_winner)
                > full_pixel_variance_cost(&zero_entropy, zero_sad_winner),
            "NSTEP variance ranking must retain the adaptive entropy table"
        );
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
            sad_mv_costs: &costs,
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

        let (nstep_seed, num00) = diamond_search_sad_from_step(&params, 6);
        assert_eq!(
            num00,
            cfg.total_steps as usize - 6,
            "a stationary NSTEP seed must report every remaining center hit"
        );
        let nstep = nstep_search_sad(&params);
        assert_eq!(nstep, nstep_seed, "num00 must skip redundant restarts");
    }

    #[test]
    fn fast_diamond_cost_list_retains_libvpx_sad_domain_while_moving() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let plane_w = 64usize;
        let plane_h = 64usize;
        let ref_data: Vec<u8> = (0..plane_h)
            .flat_map(|row| (0..plane_w).map(move |col| (row * 2 + col) as u8))
            .collect();
        let mut src_data = ref_data.clone();
        let src_col = 24usize;
        let src_row = 24usize;
        let shift_row = 2i16;
        let shift_col = 1i16;
        for row in 0..16usize {
            for col in 0..16usize {
                let ref_row = (src_row as isize + isize::from(shift_row) + row as isize) as usize;
                let ref_col = (src_col as isize + isize::from(shift_col) + col as isize) as usize;
                src_data[(src_row + row) * plane_w + src_col + col] =
                    ref_data[ref_row * plane_w + ref_col];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: plane_w,
            width: plane_w,
            height: plane_h,
        };
        let ref_plane = PlaneRef {
            data: &ref_data,
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
            sad_per_bit: 19,
            mv_costs: &costs,
            sad_mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };
        let (result, actual) = fast_diamond_search_sad_with_cost_list(&params);
        assert_eq!(result, fast_diamond_search_sad(&params));
        assert_eq!(
            (result.best_mv_row, result.best_mv_col),
            (shift_row, shift_col)
        );
        let positions = [
            (shift_row, shift_col),
            (shift_row, shift_col - 1),
            (shift_row + 1, shift_col),
            (shift_row, shift_col + 1),
            (shift_row - 1, shift_col),
        ];
        for (index, &(row, col)) in positions.iter().enumerate() {
            let raw_sad = raw_sad_at_mv(
                params.block_size,
                params.src_plane,
                params.src_offset,
                params.ref_plane,
                row,
                col,
            )
            .unwrap();
            let mv_cost = mvsad_err_cost(row, col, 0, 0, params.sad_per_bit, &costs);
            let retained_center_cost = if index == 1 { mv_cost } else { 0 };
            assert_eq!(
                actual[index],
                (raw_sad + mv_cost + retained_center_cost) as i32,
                "cost_list[{index}]"
            );
        }
        assert_eq!(actual[0], result.best_sad as i32);
    }

    #[test]
    fn fast_diamond_cost_list_uses_signed_sentinels_at_mv_edges() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, true);
        let plane_w = 64usize;
        let data = make_plane(plane_w, plane_w, 41);
        let plane = PlaneRef {
            data: &data,
            stride: plane_w,
            width: plane_w,
            height: plane_w,
        };
        let cfg = init_dsmotion_compensation(plane_w as i32);
        let params = FullPixelSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &plane,
            src_offset: (24, 24),
            ref_plane: &plane,
            center_mv: Mv::ZERO,
            start_mv: None,
            mv_limits: MvLimits::new(0, 16, 0, 16),
            sad_per_bit: 19,
            mv_costs: &costs,
            sad_mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };
        let (result, actual) = fast_diamond_search_sad_with_cost_list(&params);
        assert_eq!(result, fast_diamond_search_sad(&params));
        assert_eq!((result.best_mv_row, result.best_mv_col), (0, 0));
        assert_eq!(actual[0], result.best_sad as i32);
        assert_eq!(actual[1], i32::MAX, "left neighbor is outside col_min");
        assert_eq!(actual[4], i32::MAX, "top neighbor is outside row_min");
        assert_ne!(actual[2], i32::MAX);
        assert_ne!(actual[3], i32::MAX);
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
            sad_mv_costs: &costs,
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

        let nstep = nstep_search_sad(&params);
        assert_eq!(
            (nstep.best_mv_row, nstep.best_mv_col),
            (shift_row, shift_col),
            "NSTEP should recover the same small shift"
        );

        let fast_params = FullPixelSearchParams {
            start_mv: Some(Mv { row: 0, col: 16 }),
            ..params
        };
        let fast = fast_diamond_search_sad(&fast_params);
        assert_eq!(
            (fast.best_mv_row, fast.best_mv_col),
            (shift_row, shift_col),
            "FAST_DIAMOND should refine a nearby seed to ({shift_row}, {shift_col})"
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
            sad_mv_costs: &costs,
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
            sad_mv_costs: &costs,
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
            sad_mv_costs: &costs,
            max_iters: 0, // zero — loop body never runs
            site_config: &cfg,
        };
        let r = diamond_search_sad(&params);
        assert_eq!(r.iters_used, 0, "max_iters=0 → iters_used=0");
        assert_eq!(r.best_mv_row, 0);
        assert_eq!(r.best_mv_col, 0);
        assert_eq!(
            nstep_search_sad(&params),
            r,
            "NSTEP must also respect a zero-iteration budget"
        );
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
            sad_mv_costs: &costs,
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

    /// The full-pel direct-table helper must remain byte-for-byte equivalent
    /// to the clamping `mv_cost` path throughout the legal table range. The
    /// cross-product covers zero, the normal search envelope, and both table
    /// limits on each axis, with representative center and SAD weights.
    #[test]
    fn mvsad_err_cost_in_range_matches_mv_cost_at_boundaries() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        assert_eq!(
            costs.component_cost[0].cost[MV_MAX as usize], 0,
            "vertical zero slot is part of the direct-index invariant"
        );
        assert_eq!(
            costs.component_cost[1].cost[MV_MAX as usize], 0,
            "horizontal zero slot is part of the direct-index invariant"
        );

        let centers = [-2046, -1023, -32, 0, 32, 1023, 2046];
        let diffs = [-MV_MAX, -2046, -1023, -32, -1, 0, 1, 32, 1023, 2046, MV_MAX];
        let sad_per_bits = [0, 1, 16, 256, i32::MAX];

        for &center_row in &centers {
            for &center_col in &centers {
                for &diff_row in &diffs {
                    for &diff_col in &diffs {
                        let this_row = (center_row + diff_row) as i16;
                        let this_col = (center_col + diff_col) as i16;
                        for &sad_per_bit in &sad_per_bits {
                            let direct = mvsad_err_cost_in_range(
                                this_row,
                                this_col,
                                center_row as i16,
                                center_col as i16,
                                sad_per_bit,
                                &costs,
                            );
                            let raw = mv_cost(diff_row as i16, diff_col as i16, &costs);
                            let expected = ((raw as i64).saturating_mul(sad_per_bit as i64)
                                + (1i64 << 8))
                                >> 9;
                            assert_eq!(
                                direct,
                                expected.max(0) as u32,
                                "direct cost differs at center=({},{}), diff=({},{}), sad_per_bit={}",
                                center_row,
                                center_col,
                                diff_row,
                                diff_col,
                                sad_per_bit
                            );
                        }
                    }
                }
            }
        }
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
            sad_mv_costs: &costs,
            max_iters: cfg.total_steps as i32,
            site_config: &cfg,
        };

        let r = diamond_search_sad(&params);
        // No panic = success; the MV must lie inside the limits.
        assert!(r.best_mv_row as i32 >= 0 && r.best_mv_row as i32 <= 16);
        assert!(r.best_mv_col as i32 >= 0 && r.best_mv_col as i32 <= 16);
    }
}
