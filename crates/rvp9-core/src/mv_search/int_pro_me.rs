//! SB-level integer-projection motion estimation
//! (`vp9_int_pro_motion_estimation`).
//!
//! Port of `vp9_int_pro_motion_estimation` (`vp9_mcomp.c:2264-2400`)
//! and its supporting helpers `vector_match`, `vpx_int_pro_row_c`,
//! `vpx_int_pro_col_c`, `vpx_vector_var_c`
//! (`vp9_mcomp.c:2192-2254`, `vpx_dsp/avg.c:345-387`).
//!
//! The algorithm produces a cheap SB-level integer-pel MV by
//! comparing 1D row/column projections of source and reference
//! windows, rather than running a full 2D SAD search. The result
//! seeds the per-leaf picker's NEWMV diamond-search center via
//! `x->pred_mv[LAST_FRAME]` in libvpx.
//!
//! ## Why this matters
//!
//! Without an SB-level coarse-ME seed, the per-leaf NEWMV diamond
//! search starts from `(0, 0)` (or `nearest`, which is also `(0, 0)`
//! at frame boundaries / startup). At low Q, the MV-bit-cost of a
//! sub-pel NEWMV exceeds the SSE gain over NEARESTMV-at-(0, 0), so
//! the picker never selects NEWMV. With no NEWMV picks, the spatial
//! MV-ref propagation chain never fires either (every neighbour is
//! also NEARESTMV-at-(0, 0)). On continuous-motion content
//! (foreman_cif) this collapses inter prediction to "always copy LAST
//! frame" → PSNR decays linearly every frame.
//!
//! The libvpx fix: do a cheap 1D projection search at the 64×64 SB
//! level, get a starter MV, hand that to the per-leaf picker as the
//! NEWMV diamond seed. The seed has to actually beat NEARESTMV in
//! RD, which it can if the SB-level motion is large enough that a
//! single MV explains most of the SB.
//!
//! ## Algorithm
//!
//! For a 64×64 SB at `(sb_x, sb_y)`:
//!
//! 1. Build `hbuf[search_width]` = column-sums of the reference,
//!    starting at column `sb_x - bw/2`. Length is `search_width =
//!    2*bw`. Each entry sums `bh` rows. Each entry is normalised by
//!    `bh >> 1` (divisor matching libvpx `vpx_int_pro_row_c`).
//! 2. Build `vbuf[search_height]` = row-sums of the reference,
//!    starting at row `sb_y - bh/2`. Length is `search_height =
//!    2*bh`. Each entry sums `bw` columns. Each entry is normalised
//!    by `3 + (bw >> 5)` (libvpx's `norm_factor`).
//! 3. Build `src_hbuf[bw]` (column sums of source SB) and
//!    `src_vbuf[bh]` (row sums of source SB) with the same
//!    normalisations.
//! 4. `vector_match(hbuf, src_hbuf, bwl)` returns the best column
//!    shift in `[-bw/2, bw/2]` (full-pel units). Same for rows.
//! 5. Refine the (row, col) candidate with a 4-point diamond
//!    (`{-1, 0}`, `{0, -1}`, `{0, 1}`, `{1, 0}`) using the 2D SAD
//!    kernel for the SB size.
//! 6. Refine with a 2-point cross derived from the diamond results.
//! 7. Multiply by 8 to convert full-pel → Q3 (1/8-pel) units.
//! 8. Clamp to the subpel MV search range (caller's responsibility
//!    via [`MvLimits`]).
//!
//! ## Limitations / divergences
//!
//! - **8-bit only.** libvpx falls back to a single full-SB SAD for
//!   HBD; we don't ship that fallback because rvp9 only supports
//!   Profile 0 (8-bit).
//! - **No scaled-ref handling.** rvp9's single-LASTFRAME setup
//!   doesn't use scale factors; if/when GOLDEN / ALTREF land with
//!   resolution-scaled refs, this needs the
//!   `vp9_get_scaled_ref_frame` swap-out logic.

use crate::block::BlockSize;
use crate::dsp::dispatch::SadFn;
use crate::mode_info::Mv;
use crate::mv_search::full_pixel::PlaneRef;

/// Edge-clamped reference pixel read.
///
/// libvpx runs `vp9_int_pro_motion_estimation` against a reference
/// frame buffer whose borders are physically extended by edge
/// replication (`vpx_extend_frame_borders`,
/// `vp9/common/vp9_reconinter.c` → `vpx_dsp/vpx_convolve.c`), so the
/// projection window and the 2D-SAD halo near a frame edge read the
/// replicated border rather than going out of bounds. rvp9's
/// `PlaneRef` is the bare visible plane (no padding), so we replicate
/// that border semantics on the fly: any `(y, x)` outside the visible
/// plane clamps to the nearest edge pixel. This is the faithful
/// equivalent of an extended-border frame and lets every SB — including
/// the frame-edge SBs — get a valid motion seed instead of bailing.
#[inline]
fn clamp_px(plane: &PlaneRef<'_>, y: i32, x: i32) -> u8 {
    let cy = y.clamp(0, plane.height as i32 - 1) as usize;
    let cx = x.clamp(0, plane.width as i32 - 1) as usize;
    plane.data[cy * plane.stride + cx]
}

/// Output of [`int_pro_motion_estimation`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IntProMeResult {
    /// Resulting integer-pel MV (in Q3 / 1/8-pel units — i.e. the
    /// full-pel offset times 8). `Mv::ZERO` when the search settled on
    /// zero motion — caller treats `Mv::ZERO` as "no useful seed".
    pub mv: Mv,
    /// Best SAD over the SB at the returned MV. `u32::MAX` only for the
    /// unsupported-block-size early return (the [`ZERO`] sentinel);
    /// every supported SB — including frame-edge SBs — now produces a
    /// real SAD via the edge-clamped border reads.
    ///
    /// [`ZERO`]: IntProMeResult::ZERO
    pub best_sad: u32,
}

impl IntProMeResult {
    /// Zero MV / no-search-performed sentinel.
    pub const ZERO: Self = Self {
        mv: Mv::ZERO,
        best_sad: u32::MAX,
    };
}

/// `vpx_int_pro_row_c` (`vpx_dsp/avg.c:345-359`). Sum the next
/// `16` columns of `ref` along `height` rows, normalise by
/// `height >> 1`, write into `hbuf[0..16]`.
///
/// Caller must ensure `ref[0 .. 16 + (height-1) * ref_stride]` is
/// in bounds.
#[inline]
fn vpx_int_pro_row(hbuf: &mut [i16], ref_plane: &[u8], ref_stride: usize, height: usize) {
    debug_assert_eq!(hbuf.len(), 16);
    debug_assert!(height >= 2);
    #[cfg(target_arch = "aarch64")]
    if matches!(height, 16 | 32 | 64) {
        crate::dsp::aarch64::vpx_int_pro_row_neon(hbuf, ref_plane, ref_stride, height);
        return;
    }

    let norm_factor = (height >> 1) as i32;
    for idx in 0..16 {
        let mut sum: i32 = 0;
        for i in 0..height {
            sum += i32::from(ref_plane[i * ref_stride + idx]);
        }
        hbuf[idx] = (sum / norm_factor) as i16;
    }
}

/// `vpx_int_pro_col_c` (`vpx_dsp/avg.c:362-368`). Return the sum
/// of the next `width` columns of `ref`. Caller divides by the
/// matching `norm_factor`.
#[inline]
fn vpx_int_pro_col(ref_row: &[u8], width: usize) -> i32 {
    debug_assert!(ref_row.len() >= width);
    let mut sum: i32 = 0;
    for &b in ref_row.iter().take(width) {
        sum += i32::from(b);
    }
    sum
}

/// Edge-clamped variant of [`vpx_int_pro_row`]: same as
/// `vpx_int_pro_row_c` but the reference window base is an absolute
/// (possibly out-of-plane) `(base_y, base_x)` and every pixel read
/// clamps to the visible-plane border (see [`clamp_px`]). This is what
/// libvpx's extended-border frame supplies for free at frame edges.
#[inline]
fn int_pro_row_clamped(
    hbuf: &mut [i16],
    plane: &PlaneRef<'_>,
    base_y: i32,
    base_x: i32,
    height: usize,
) {
    debug_assert_eq!(hbuf.len(), 16);
    debug_assert!(height >= 2);
    let norm_factor = (height >> 1) as i32;
    for (idx, slot) in hbuf.iter_mut().enumerate() {
        let mut sum: i32 = 0;
        for i in 0..height {
            sum += i32::from(clamp_px(plane, base_y + i as i32, base_x + idx as i32));
        }
        *slot = (sum / norm_factor) as i16;
    }
}

/// Edge-clamped variant of [`vpx_int_pro_col`]: sum `width` columns of
/// the reference starting at absolute `(base_y, base_x)`, clamping each
/// read to the visible-plane border.
#[inline]
fn int_pro_col_clamped(plane: &PlaneRef<'_>, base_y: i32, base_x: i32, width: usize) -> i32 {
    let mut sum: i32 = 0;
    for c in 0..width {
        sum += i32::from(clamp_px(plane, base_y, base_x + c as i32));
    }
    sum
}

fn fill_int_pro_rows(
    output: &mut [i16],
    plane: &PlaneRef<'_>,
    base_y: i32,
    base_x: i32,
    height: usize,
) {
    debug_assert_eq!(output.len() % 16, 0);
    let direct = base_y >= 0
        && base_x >= 0
        && base_y as usize + height <= plane.height
        && base_x as usize + output.len() <= plane.width;

    for idx in (0..output.len()).step_by(16) {
        if direct {
            let start = base_y as usize * plane.stride + base_x as usize + idx;
            vpx_int_pro_row(
                &mut output[idx..idx + 16],
                &plane.data[start..],
                plane.stride,
                height,
            );
        } else {
            int_pro_row_clamped(
                &mut output[idx..idx + 16],
                plane,
                base_y,
                base_x + idx as i32,
                height,
            );
        }
    }
}

fn fill_int_pro_cols(
    output: &mut [i16],
    plane: &PlaneRef<'_>,
    base_y: i32,
    base_x: i32,
    width: usize,
    norm_shift: usize,
) {
    let direct = base_y >= 0
        && base_x >= 0
        && base_y as usize + output.len() <= plane.height
        && base_x as usize + width <= plane.width;

    for (idx, slot) in output.iter_mut().enumerate() {
        let sum = if direct {
            let start = (base_y as usize + idx) * plane.stride + base_x as usize;
            vpx_int_pro_col(&plane.data[start..], width)
        } else {
            int_pro_col_clamped(plane, base_y + idx as i32, base_x, width)
        };
        *slot = (sum >> norm_shift) as i16;
    }
}

/// `vpx_vector_var_c` (`vpx_dsp/avg.c:373-387`). Compute the
/// "variance" between two `4 << bwl`-length int16 vectors:
///
/// ```text
/// width = 4 << bwl
/// for i in 0..width: diff = ref[i] - src[i]; mean += diff; sse += diff*diff
/// var = sse - ((mean * mean) >> (bwl + 2))
/// ```
///
/// `bwl` ranges over `{1, 2, 3, 4}` (16, 32, 64 vs the 16/32/64
/// SB widths we care about). The result fits in i32 with margin
/// (sse: 26 bits dynamic range per libvpx comment).
#[inline]
fn vpx_vector_var(ref_vec: &[i16], src_vec: &[i16], bwl: u32) -> i32 {
    let width = 4usize << bwl;
    debug_assert!(ref_vec.len() >= width);
    debug_assert!(src_vec.len() >= width);
    let mut sse: i32 = 0;
    let mut mean: i32 = 0;
    for i in 0..width {
        let diff = i32::from(ref_vec[i]) - i32::from(src_vec[i]);
        mean += diff;
        sse += diff * diff;
    }
    sse - ((mean * mean) >> (bwl + 2))
}

/// `vector_match` (`vp9_mcomp.c:2192-2255`). Find the best 1D
/// translation in `[0, bw]` of a length-`bw` window of `src`
/// against `ref`, then return `center - (bw >> 1)` (i.e. the shift
/// centred around the SB's natural location).
///
/// `bwl` is `b_width_log2_lookup[bsize]` (= log2(bw / 4)). The
/// `ref` slice must be `>= 2*bw` long (libvpx's `search_width =
/// 2*bw`).
fn vector_match(ref_vec: &[i16], src_vec: &[i16], bwl: u32) -> i32 {
    let bw = 4i32 << bwl;
    let bw_usz = bw as usize;
    debug_assert!(ref_vec.len() >= 2 * bw_usz);
    debug_assert!(src_vec.len() >= bw_usz);

    let mut best_sad = i32::MAX;
    let mut offset = 0i32;

    // Coarse step: 16.
    let mut d = 0i32;
    while d <= bw {
        let this_sad = vpx_vector_var(&ref_vec[d as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            offset = d;
        }
        d += 16;
    }
    let mut center = offset;

    // Step: ±8.
    for d in [-8i32, 8i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }
    offset = center;

    // Step: ±4.
    for d in [-4i32, 4i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }
    offset = center;

    // Step: ±2.
    for d in [-2i32, 2i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }
    offset = center;

    // Step: ±1.
    for d in [-1i32, 1i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }

    center - (bw >> 1)
}

/// Run the 2D SAD for an SB-supported block size via the pre-resolved
/// kernel `sad_fn`.
///
/// `sad_fn` is the dispatch-resolved SAD kernel for `bsize`, hoisted out
/// of the search loop once per `int_pro_motion_estimation` call (P1-B).
/// The `match` is **retained as a fail-soft gate**: only the four
/// SB-supported sizes (8×8/16×16/32×32/64×64) invoke the kernel; any other
/// `bsize` returns the `u32::MAX` sentinel verbatim (the same behaviour as
/// before — never indexing a slot table out of bounds nor panicking).
fn sad_block(
    bsize: BlockSize,
    sad_fn: SadFn,
    src: &[u8],
    src_stride: usize,
    ref_block: &[u8],
    ref_stride: usize,
) -> u32 {
    match bsize {
        BlockSize::Block8x8
        | BlockSize::Block16x16
        | BlockSize::Block32x32
        | BlockSize::Block64x64 => sad_fn(src, src_stride, ref_block, ref_stride),
        _ => u32::MAX,
    }
}

/// Gather one visible-or-replicated block a row at a time.
///
/// Unlike [`clamp_px`], this computes the horizontal edge spans once per
/// block and the vertical clamp once per row. The middle of every row remains
/// a contiguous copy from the visible plane.
fn gather_clamped_block(
    plane: &PlaneRef<'_>,
    abs_y: i32,
    abs_x: i32,
    width: usize,
    height: usize,
    dst: &mut [u8],
) {
    debug_assert_eq!(dst.len(), width * height);
    let plane_width = plane.width as i32;
    let plane_height = plane.height as i32;
    let block_width = width as i32;
    let left = (-abs_x).clamp(0, block_width) as usize;
    let right = (abs_x + block_width - plane_width).clamp(0, block_width) as usize;
    let middle = width - left - right;
    let src_x = abs_x.max(0) as usize;

    for (row_index, dst_row) in dst.chunks_exact_mut(width).enumerate() {
        let src_y = (abs_y + row_index as i32).clamp(0, plane_height - 1) as usize;
        let row_start = src_y * plane.stride;
        let src_row = &plane.data[row_start..row_start + plane.width];

        if left != 0 {
            dst_row[..left].fill(src_row[0]);
        }
        if middle != 0 {
            dst_row[left..left + middle].copy_from_slice(&src_row[src_x..src_x + middle]);
        }
        if right != 0 {
            dst_row[width - right..].fill(src_row[plane.width - 1]);
        }
    }
}

/// Edge-clamped 2D SAD: gather a `bw×bh` reference block whose top-left
/// is at absolute `(abs_y, abs_x)` into a contiguous scratch buffer
/// (clamping every read to the visible-plane border, replicating
/// libvpx's extended-border behaviour), then run the matching SAD
/// kernel against the source slice. Used by the post-projection 2D
/// refinement so frame-edge SBs refine against the replicated border
/// instead of bailing.
fn sad_block_clamped(
    bsize: BlockSize,
    sad_fn: SadFn,
    src: &[u8],
    src_stride: usize,
    ref_plane: &PlaneRef<'_>,
    abs_pos: (i32, i32),
    edge_scratch: &mut Option<[u8; 64 * 64]>,
) -> u32 {
    let (abs_y, abs_x) = abs_pos;
    let bw = usize::from(bsize.width());
    let bh = usize::from(bsize.height());

    // The common case is wholly inside the visible plane. Its bytes are
    // already laid out exactly as the SAD kernel expects, so call it directly
    // with the plane stride. Only probes crossing a visible edge need the
    // replicated-border gather below. This mirrors libvpx's direct reads from
    // its physically extended reference frame.
    if abs_y >= 0 && abs_x >= 0 {
        let y = abs_y as usize;
        let x = abs_x as usize;
        if y + bh <= ref_plane.height && x + bw <= ref_plane.width {
            let start = y * ref_plane.stride + x;
            let end = (y + bh - 1) * ref_plane.stride + x + bw;
            if end <= ref_plane.data.len() {
                return sad_block(
                    bsize,
                    sad_fn,
                    src,
                    src_stride,
                    &ref_plane.data[start..],
                    ref_plane.stride,
                );
            }
        }
    }

    // The inactive Option payload needs no initialization. The first edge
    // probe initializes it once; later probes in this search overwrite and
    // reuse the same active block instead of zeroing another 4 KiB array.
    let scratch = edge_scratch.get_or_insert([0u8; 64 * 64]);
    gather_clamped_block(ref_plane, abs_y, abs_x, bw, bh, &mut scratch[..bh * bw]);
    sad_block(bsize, sad_fn, src, src_stride, &scratch[..bh * bw], bw)
}

/// `b_width_log2_lookup[bsize]` for the square SB block sizes we
/// support. Mirrors libvpx `vp9_common_data.c:15`.
#[inline]
fn b_log2(bsize: BlockSize) -> u32 {
    match bsize {
        BlockSize::Block8x8 => 1,
        BlockSize::Block16x16 => 2,
        BlockSize::Block32x32 => 3,
        BlockSize::Block64x64 => 4,
        _ => 0,
    }
}

/// Port of `vp9_int_pro_motion_estimation` (`vp9_mcomp.c:2264-2400`).
///
/// Computes an integer-pel MV (returned in Q3 / 1/8-pel units) for
/// the source SB at `(sb_px_x, sb_px_y)` against the reference plane.
/// `bsize` must be one of the square sizes
/// `{Block8x8, Block16x16, Block32x32, Block64x64}`; other sizes
/// return [`IntProMeResult::ZERO`].
///
/// The returned MV is **unclamped**. Callers that need to feed it
/// into a per-leaf picker with an `MvLimits` envelope should clamp
/// it themselves (libvpx calls `clamp_mv` against the subpel-MV
/// search range derived from `vp9_set_subpel_mv_search_range`).
///
/// The reference plane must be the SAME dimensions as the source
/// plane (no scaled-ref support); the caller is responsible for
/// performing the equivalent of libvpx's `vp9_setup_pre_planes`
/// upstream.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn int_pro_motion_estimation(
    src_plane: &PlaneRef<'_>,
    ref_plane: &PlaneRef<'_>,
    sb_px_x: usize,
    sb_px_y: usize,
    bsize: BlockSize,
) -> IntProMeResult {
    let bw = usize::from(bsize.width());
    let bh = usize::from(bsize.height());
    if bw != bh
        || !matches!(
            bsize,
            BlockSize::Block8x8
                | BlockSize::Block16x16
                | BlockSize::Block32x32
                | BlockSize::Block64x64
        )
    {
        return IntProMeResult::ZERO;
    }

    // The 1D-projection search reads
    //   ref[sb_y - bh/2 .. sb_y + bh + bh/2 - 1]
    //      [sb_x - bw/2 .. sb_x + bw + bw/2 - 1]
    // and
    //   src[sb_y .. sb_y + bh - 1][sb_x .. sb_x + bw - 1].
    // The post-search 2D-SAD refinement reads an additional ±1
    // ref-pixel halo around the chosen MV.
    //
    // E10 — frame-edge handling. libvpx runs this against a reference
    // frame whose borders are physically extended by edge replication
    // (`vpx_extend_frame_borders`), so the projection window and the
    // 2D-SAD halo near a frame edge read the replicated border and the
    // search always produces a seed. rvp9's `PlaneRef` is the bare
    // visible plane, so every reference read goes through [`clamp_px`]
    // (edge-clamp == border replication). The SOURCE reads stay inside
    // the visible plane: an SB top-left is always inside the frame, and
    // libvpx never reads source pixels outside the SB extent. The only
    // source-OOB case is a partial right/bottom edge SB; we guard those
    // reads too via the source-side clamp at the build sites below.
    //
    // Previously this function bailed (ZERO sentinel) whenever the
    // window touched a border, which on a 256×256 frame disabled the
    // seed on 12 of 16 SBs (every edge SB) — the root of the no-motion-
    // foothold degeneracy (E3/E9). With clamped reads all 16 SBs get a
    // valid seed.

    let search_width = bw << 1;
    let search_height = bh << 1;
    debug_assert!(search_width <= 128);
    debug_assert!(search_height <= 128);

    let src_stride = src_plane.stride;
    let bwl = b_log2(bsize);

    // ref_buf base for the row scan: starts (bw>>1) columns LEFT
    // of the SB and (bh>>1) above is NOT applied to the row scan —
    // the row scan stays at the SB's top row (matching libvpx, which
    // offsets only the column for the horizontal projection). Each
    // 16-col chunk builds via the edge-clamped row projection so the
    // leftmost (bw>>1) columns of an edge SB replicate the border.
    let base_x_row = sb_px_x as i32 - (bw >> 1) as i32;
    let mut hbuf = [0i16; 128];
    fill_int_pro_rows(
        &mut hbuf[..search_width],
        ref_plane,
        sb_px_y as i32,
        base_x_row,
        bh,
    );

    // vbuf: row-sums of ref, starting (bh>>1) rows ABOVE the SB.
    let mut vbuf = [0i16; 128];
    let norm_factor = 3 + (bw >> 5);
    let base_y_col = sb_px_y as i32 - (bh >> 1) as i32;
    fill_int_pro_cols(
        &mut vbuf[..search_height],
        ref_plane,
        base_y_col,
        sb_px_x as i32,
        bw,
        norm_factor,
    );

    // src_hbuf: column-sums of src SB. libvpx loops idx 0..bw step
    // 16 (Block16x16 → 1 iter, Block64x64 → 4 iters).
    let mut src_hbuf = [0i16; 64];
    if bw >= 16 {
        let mut idx = 0;
        while idx < bw {
            let off = sb_px_y * src_stride + sb_px_x + idx;
            vpx_int_pro_row(
                &mut src_hbuf[idx..idx + 16],
                &src_plane.data[off..],
                src_stride,
                bh,
            );
            idx += 16;
        }
    } else {
        // Block8x8: libvpx doesn't use int_pro for 8x8 (variance
        // partitioner only descends to 16x16 for the SB seed
        // call), but we keep a defensive scalar fallback so the
        // helper is total. Compute the 8 column-sums manually.
        let norm_factor = (bh >> 1) as i32;
        for idx in 0..bw {
            let mut sum: i32 = 0;
            for i in 0..bh {
                sum += i32::from(src_plane.data[(sb_px_y + i) * src_stride + sb_px_x + idx]);
            }
            src_hbuf[idx] = (sum / norm_factor) as i16;
        }
    }

    // src_vbuf: row-sums of src SB.
    let mut src_vbuf = [0i16; 64];
    {
        let norm_factor = 3 + (bw >> 5);
        for idx in 0..bh {
            let row_off = (sb_px_y + idx) * src_stride + sb_px_x;
            let sum = vpx_int_pro_col(&src_plane.data[row_off..], bw);
            src_vbuf[idx] = (sum >> norm_factor) as i16;
        }
    }

    // 1D matches → integer-pel MV offsets in `[-bw/2, bw/2]`.
    let col = vector_match(&hbuf[..search_width], &src_hbuf[..bw], bwl);
    let row = vector_match(&vbuf[..search_height], &src_vbuf[..bh], bwl);

    // 2D refinement: 4-pt diamond + 2-pt cross, both gated by
    // staying inside the ref plane (with the ±1 halo the diamond
    // needs).
    let src_off = sb_px_y * src_stride + sb_px_x;
    let src_slice = &src_plane.data[src_off..];

    // E10 — every 2D-SAD probe reads through the edge-clamped block
    // gather, mirroring libvpx's extended-border frame. The reference
    // window can extend past any frame edge and replicates the border
    // pixel instead of bailing, so the diamond/cross refinement runs
    // for every SB including frame-edge SBs.
    // Resolve the SAD kernel once per estimation call (P1-B hoist): the
    // dispatch `get_ref()` OnceLock load is amortized over every 2D-SAD
    // probe below instead of paid per probe. `sad_block` keeps its own
    // fail-soft `match` so an unsupported `bsize` still returns `u32::MAX`.
    let sad_fn = crate::dsp::get_ref().sad(bsize);
    let mut edge_scratch = None;
    let mut try_sad = |row: i32, col: i32| -> u32 {
        let abs_x = sb_px_x as i32 + col;
        let abs_y = sb_px_y as i32 + row;
        sad_block_clamped(
            bsize,
            sad_fn,
            src_slice,
            src_stride,
            ref_plane,
            (abs_y, abs_x),
            &mut edge_scratch,
        )
    };

    let mut best_sad = try_sad(row, col);

    // 4-pt diamond: { (-1,0), (0,-1), (0,1), (1,0) } (libvpx's
    // `search_pos` order, vp9_mcomp.c:2257-2262).
    let diamond = [(-1i32, 0i32), (0i32, -1i32), (0i32, 1i32), (1i32, 0i32)];
    let mut diamond_sads = [u32::MAX; 4];
    for (i, &(dr, dc)) in diamond.iter().enumerate() {
        diamond_sads[i] = try_sad(row + dr, col + dc);
    }

    // libvpx walks the 4 diamond results in order and updates
    // `*tmp_mv` (the running best) on strict `<`. `this_mv` is the
    // PRE-diamond center used by the cross-pos refinement
    // (vp9_mcomp.c:2345 `this_mv = *tmp_mv` taken BEFORE the
    // sdx4df update). We mirror that exactly.
    let pre_diamond_row = row;
    let pre_diamond_col = col;
    let mut best_row = row;
    let mut best_col = col;
    for (i, &(dr, dc)) in diamond.iter().enumerate() {
        if diamond_sads[i] < best_sad {
            best_sad = diamond_sads[i];
            best_row = pre_diamond_row + dr;
            best_col = pre_diamond_col + dc;
        }
    }

    // 2-pt cross derived from diamond SADs (libvpx
    // `vp9_mcomp.c:2369-2377`):
    //   this_sad[0] = up; this_sad[3] = down.
    //   this_sad[1] = left; this_sad[2] = right.
    // If up < down, this_mv.row -= 1; else this_mv.row += 1.
    // If left < right, this_mv.col -= 1; else this_mv.col += 1.
    let cross_row = if diamond_sads[0] < diamond_sads[3] {
        pre_diamond_row - 1
    } else {
        pre_diamond_row + 1
    };
    let cross_col = if diamond_sads[1] < diamond_sads[2] {
        pre_diamond_col - 1
    } else {
        pre_diamond_col + 1
    };
    let cross_sad = try_sad(cross_row, cross_col);
    if cross_sad < best_sad {
        best_sad = cross_sad;
        best_row = cross_row;
        best_col = cross_col;
    }

    // Convert full-pel → Q3 (1/8-pel).
    let mv_row_q3 = (best_row * 8).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    let mv_col_q3 = (best_col * 8).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;

    IntProMeResult {
        mv: Mv {
            row: mv_row_q3,
            col: mv_col_q3,
        },
        best_sad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rowwise_clamped_block_gather_matches_pixel_oracle() {
        let width = 97usize;
        let height = 83usize;
        let stride = 104usize;
        let mut pixels = vec![0u8; stride * height];
        for row in 0..height {
            for col in 0..width {
                pixels[row * stride + col] = ((row * 37 + col * 19 + (row ^ col) * 7) & 0xff) as u8;
            }
        }
        let plane = PlaneRef {
            data: &pixels,
            stride,
            width,
            height,
        };

        for side in [8usize, 16, 32, 64] {
            let side_i32 = side as i32;
            for (abs_y, abs_x) in [
                (-side_i32 - 3, -side_i32 - 5),
                (-2, 3),
                (3, -2),
                (height as i32 - 2, width as i32 - 2),
                (height as i32 + 2, width as i32 + 3),
                (7, 11),
            ] {
                let mut actual = vec![0xA5; side * side];
                gather_clamped_block(&plane, abs_y, abs_x, side, side, &mut actual);
                let mut expected = vec![0; side * side];
                for row in 0..side {
                    for col in 0..side {
                        expected[row * side + col] =
                            clamp_px(&plane, abs_y + row as i32, abs_x + col as i32);
                    }
                }
                assert_eq!(actual, expected, "side={side} y={abs_y} x={abs_x}");
            }
        }
    }

    #[test]
    fn interior_projection_fast_paths_match_clamped_oracles() {
        let width = 256usize;
        let height = 256usize;
        let pixels: Vec<u8> = (0..width * height)
            .map(|index| {
                let row = index / width;
                let col = index % width;
                ((row * 37 + col * 19 + (row ^ col) * 7) & 0xff) as u8
            })
            .collect();
        let plane = PlaneRef {
            data: &pixels,
            stride: width,
            width,
            height,
        };

        let mut actual_rows = [0i16; 128];
        fill_int_pro_rows(&mut actual_rows, &plane, 64, 32, 64);
        let mut expected_rows = [0i16; 128];
        for idx in (0..expected_rows.len()).step_by(16) {
            int_pro_row_clamped(
                &mut expected_rows[idx..idx + 16],
                &plane,
                64,
                32 + idx as i32,
                64,
            );
        }
        assert_eq!(actual_rows, expected_rows);

        let mut actual_cols = [0i16; 128];
        fill_int_pro_cols(&mut actual_cols, &plane, 32, 64, 64, 5);
        let mut expected_cols = [0i16; 128];
        for (idx, slot) in expected_cols.iter_mut().enumerate() {
            *slot = (int_pro_col_clamped(&plane, 32 + idx as i32, 64, 64) >> 5) as i16;
        }
        assert_eq!(actual_cols, expected_cols);
    }

    fn direct_path_probe(
        _src: &[u8],
        _src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
    ) -> u32 {
        assert_eq!(ref_stride, 96, "interior path must retain plane stride");
        assert_eq!(
            reference[0], 0xA7,
            "interior path must start at the requested reference pixel"
        );
        0x51AD
    }

    #[test]
    fn interior_clamped_sad_reads_directly_from_plane() {
        let mut reference = vec![0u8; 96 * 96];
        reference[17 * 96 + 19] = 0xA7;
        let plane = PlaneRef {
            data: &reference,
            stride: 96,
            width: 96,
            height: 96,
        };
        let source = [0u8; 8 * 8];
        let mut edge_scratch = None;
        assert_eq!(
            sad_block_clamped(
                BlockSize::Block8x8,
                direct_path_probe,
                &source,
                8,
                &plane,
                (17, 19),
                &mut edge_scratch,
            ),
            0x51AD
        );
        assert!(
            edge_scratch.is_none(),
            "interior probes must not initialize edge scratch"
        );
    }

    #[test]
    fn edge_clamped_sad_reuses_one_lazily_initialized_scratch() {
        let reference: Vec<u8> = (0..16 * 16)
            .map(|index| ((index * 37 + 11) & 0xff) as u8)
            .collect();
        let plane = PlaneRef {
            data: &reference,
            stride: 16,
            width: 16,
            height: 16,
        };
        let source = [17u8; 8 * 8];
        let sad_fn = crate::dsp::get_ref().sad(BlockSize::Block8x8);
        let mut edge_scratch = None;

        let _ = sad_block_clamped(
            BlockSize::Block8x8,
            sad_fn,
            &source,
            8,
            &plane,
            (-2, -3),
            &mut edge_scratch,
        );
        edge_scratch.as_mut().expect("edge initializes scratch")[4095] = 0xA5;
        let _ = sad_block_clamped(
            BlockSize::Block8x8,
            sad_fn,
            &source,
            8,
            &plane,
            (12, 13),
            &mut edge_scratch,
        );
        assert_eq!(
            edge_scratch.as_ref().expect("scratch remains active")[4095],
            0xA5,
            "a later small edge probe must not reinitialize the whole scratch"
        );
    }

    /// Constant-content src and ref: libvpx's `vector_match`
    /// returns `0 - bw/2 = -bw/2` for both axes (the strict-`<`
    /// updates never fire when all SADs tie at 0), then the 4-pt
    /// diamond + cross can't improve so the result stays at
    /// `(-bh/2, -bw/2)` in full-pel = `(-bh*4, -bw*4)` Q3.
    /// We mirror this bug-for-bug — constant input never reaches
    /// the picker in real encoder runs (variance partitioning
    /// gates on non-flat regions).
    ///
    /// For natural content the result is non-pathological because
    /// `vector_match`'s first-encountered minimum is the real
    /// motion location, not the boundary.
    #[test]
    fn constant_content_returns_libvpx_canonical_value() {
        let w = 256;
        let h = 256;
        let buf = vec![128u8; w * h];
        let src = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let r = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let result = int_pro_motion_estimation(&src, &r, 64, 64, BlockSize::Block64x64);
        // Block64x64 → bw = bh = 64 → result = (-32, -32) FP =
        // (-256, -256) Q3.
        assert_eq!(
            result.mv,
            Mv {
                row: -256,
                col: -256
            }
        );
    }

    /// Source plane == reference plane with non-constant content
    /// (so vector_match's strict-`<` updates actually fire) → the
    /// MV should be at or very close to (0, 0). With a textured
    /// stationary scene, the SAD at offset = bw/2 (which maps to
    /// MV = 0) is the true minimum.
    #[test]
    fn stationary_textured_content_yields_zero_mv() {
        let w = 256;
        let h = 256;
        // Random-ish texture so the projections discriminate.
        let buf: Vec<u8> = (0..w * h)
            .map(|i| (((i * 73 + i / 13) ^ 0x5a) & 0xff) as u8)
            .collect();
        let src = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let r = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let result = int_pro_motion_estimation(&src, &r, 64, 64, BlockSize::Block64x64);
        // Should be zero (or very close) since src == ref.
        let row_fp = i32::from(result.mv.row) / 8;
        let col_fp = i32::from(result.mv.col) / 8;
        assert!(row_fp.abs() <= 1, "row MV should be ~0, got {} FP", row_fp);
        assert!(col_fp.abs() <= 1, "col MV should be ~0, got {} FP", col_fp);
    }

    /// Synthetic horizontal translation: ref's content (vertical
    /// step pattern) is shifted left by `shift` cols relative to
    /// src. So src matches ref at col offset = +shift.
    /// Vertical-step pattern ensures the column-projection signal
    /// is strong (rows sum to the same total, but columns vary
    /// sharply at the step boundary).
    #[test]
    fn horizontal_translation_is_recovered() {
        let w = 256;
        let h = 256;
        // Vertical stripes with sharp transitions every 32 cols.
        // The column projection vbuf will have a strong step at
        // each multiple of 32; matching src vs ref will recover
        // the shift.
        let make = |shift_x: i32| {
            let mut buf = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let sx = x as i32 - shift_x;
                    if sx >= 0 && (sx as usize) < w {
                        buf[y * w + x] = if ((sx as usize) / 32).is_multiple_of(2) {
                            50
                        } else {
                            200
                        };
                    } else {
                        buf[y * w + x] = 128;
                    }
                }
            }
            buf
        };
        let src_buf = make(0);
        let shift = 4i32;
        let ref_buf = make(-shift); // ref shifted LEFT by `shift`
        let src = PlaneRef {
            data: &src_buf,
            stride: w,
            width: w,
            height: h,
        };
        let r = PlaneRef {
            data: &ref_buf,
            stride: w,
            width: w,
            height: h,
        };
        let result = int_pro_motion_estimation(&src, &r, 96, 96, BlockSize::Block64x64);
        // src is unshifted; ref has the pattern shifted LEFT by
        // `shift`. So ref[x] = src[x + shift], i.e. to match src
        // we sample ref at col offset = -shift. Expected MV col
        // ≈ -shift.
        let col_fp = i32::from(result.mv.col) / 8;
        assert!(
            (col_fp - (-shift)).abs() <= 2,
            "expected col MV near {} (full-pel), got {} ({} Q3)",
            -shift,
            col_fp,
            result.mv.col
        );
    }

    /// E10 — SB at a frame edge no longer bails. libvpx runs the
    /// projection against an edge-replicated extended border, and we
    /// mirror that with [`clamp_px`], so even the top-left (0, 0) SB —
    /// which has no real left/top padding — produces a valid seed
    /// (real `best_sad`, not the `u32::MAX` sentinel).
    #[test]
    fn frame_edge_still_produces_a_seed() {
        let w = 256;
        let h = 256;
        // Random-ish texture so the projections discriminate.
        let buf: Vec<u8> = (0..w * h)
            .map(|i| (((i * 73 + i / 13) ^ 0x5a) & 0xff) as u8)
            .collect();
        let src = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let r = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        // Block64x64 at (0, 0): needs 32 pixels of left/top padding,
        // which the border-clamp supplies by replication.
        let result = int_pro_motion_estimation(&src, &r, 0, 0, BlockSize::Block64x64);
        assert_ne!(result.best_sad, u32::MAX, "edge SB should get a real SAD");
        // src == ref → stationary content → seed should be ~0 motion.
        let row_fp = i32::from(result.mv.row) / 8;
        let col_fp = i32::from(result.mv.col) / 8;
        assert!(
            row_fp.abs() <= 1,
            "edge-SB row MV should be ~0, got {row_fp} FP"
        );
        assert!(
            col_fp.abs() <= 1,
            "edge-SB col MV should be ~0, got {col_fp} FP"
        );
    }

    /// E10 — every SB of a 256×256 frame (4×4 grid of 64×64 SBs)
    /// now produces a seed (no `u32::MAX` bail), including the 12 edge
    /// SBs that previously bailed. This is the direct unit-level
    /// confirmation of the foothold fix.
    #[test]
    fn all_sbs_seeded_no_edge_bail() {
        let w = 256;
        let h = 256;
        let buf: Vec<u8> = (0..w * h)
            .map(|i| (((i * 73 + i / 13) ^ 0x5a) & 0xff) as u8)
            .collect();
        let src = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let r = PlaneRef {
            data: &buf,
            stride: w,
            width: w,
            height: h,
        };
        let mut seeded = 0;
        for sb_row in 0..4 {
            for sb_col in 0..4 {
                let res = int_pro_motion_estimation(
                    &src,
                    &r,
                    sb_col * 64,
                    sb_row * 64,
                    BlockSize::Block64x64,
                );
                if res.best_sad != u32::MAX {
                    seeded += 1;
                }
            }
        }
        assert_eq!(seeded, 16, "all 16 SBs should be seeded, got {seeded}");
    }

    /// vector_match: constant input → libvpx's strict-`<` keeps
    /// `offset = 0` (the first try is the first hit) → returns
    /// `0 - bw/2 = -bw/2`. Document this bug-for-bug match.
    #[test]
    fn vector_match_constant_input_returns_neg_half_bw() {
        let bwl: u32 = 2;
        let bw = 4i32 << bwl; // 16
        let ref_vec = vec![100i16; (2 * bw) as usize];
        let src_vec = vec![100i16; bw as usize];
        let mv = vector_match(&ref_vec, &src_vec, bwl);
        assert_eq!(mv, -bw / 2);
    }

    /// vector_match: src matches ref starting at offset = bw/2
    /// (i.e. zero motion). With a unique minimum at that offset,
    /// vector_match returns 0.
    #[test]
    fn vector_match_centered_minimum_returns_zero() {
        let bwl: u32 = 3;
        let bw = (4i32 << bwl) as usize; // 32
        // Build a ref where the matching window is at offset bw/2
        // (centred).
        let mut ref_vec = vec![0i16; 2 * bw];
        for (i, v) in ref_vec.iter_mut().enumerate() {
            // Position-dependent values; src matches the
            // bw-window starting at bw/2 of ref.
            *v = ((i.wrapping_mul(13) + 7) & 0xff) as i16;
        }
        let src_vec: Vec<i16> = ref_vec[bw / 2..bw / 2 + bw].to_vec();
        let mv = vector_match(&ref_vec, &src_vec, bwl);
        assert_eq!(mv, 0);
    }
}
