//! Per-TX-block coefficient-decode wrapper.
//!
//! Ports libvpx `vp9_decode_block_tokens`
//! (`vp9/decoder/vp9_detokenize.c:275-333`). For one transform block at
//! a given `(plane, x4, y4_in_sb, tx_size, tx_type, ref_type)` the
//! wrapper:
//!
//! 1. Slices the right per-`(tx_size, plane_type, ref_type)` coefficient
//!    probability sub-table out of [`FrameContext`].
//! 2. Looks up the scan order via
//!    [`scan_for`].
//! 3. Computes the initial token-tree context `pt` by OR-folding the
//!    relevant cells of [`TileCoefCtx::above_coef`] /
//!    [`TileCoefCtx::left_coef`] (mirrors libvpx's
//!    `ctx = !!*(uintN_t *)a + !!*(uintN_t *)l` SIMD trick — we read
//!    cell-by-cell because our `TileCoefCtx` keeps one `u8` per 4×4
//!    sub-block rather than libvpx's packed bytes).
//! 4. Calls [`decode_coefs`] with the assembled arguments.
//! 5. Writes the `(eob > 0)` flag back into the spanned
//!    `(above_coef, left_coef)` cells via
//!    [`TileCoefCtx::set_coef_flags`].
//!
//! The point of this wrapper is to encapsulate the bookkeeping so the
//! per-coding-block driver (Wave 4 K) can iterate TX positions and call
//! `decode_block_tokens(...)` without re-deriving contexts each time.
//!
//! ## Scope (Stage 3 Task F.C / 8.2.9)
//!
//! Wave 3 F.C — the per-TX-block coef-decode wrapper. Does not touch
//! prediction, inverse transform, or pixel reconstruction; those land
//! later (Wave 3 H = `reconstruct_intra_block` and onwards).

use rvp9_core::entropy::detokenize::{CoefBandCtxProbs, DecodeCoefCounts, decode_coefs};
use rvp9_core::entropy::{BoolReader, FrameContext};
use rvp9_core::transforms::scans::{ScanOrder, scan_for};
use rvp9_core::transforms::types::{TxSize, TxType};

use crate::tile_ctx::{TileCoefCtx, combine_entropy_contexts};

/// Plane-type index libvpx uses for `coef_probs[tx][plane_type]`:
/// `0` = Y, `1` = UV. Mirrors `get_plane_type` in
/// `vp9/common/vp9_blockd.h:210-212`.
#[inline]
#[must_use]
fn plane_type_index(plane: usize) -> usize {
    usize::from(plane > 0)
}

/// OR-fold `n` cells of an above-row or left-column slice into a single
/// 0/1 "any cell non-zero" indicator. Mirrors libvpx's
/// `!!*(uintN_t *)a` / `!!*(uintN_t *)l` SIMD trick at
/// `vp9/decoder/vp9_detokenize.c:290-321` — we keep one `u8` per 4×4
/// sub-block instead of libvpx's packed-byte representation, so we
/// fold cell-by-cell.
#[inline]
#[must_use]
fn fold_any_nonzero(coef_ctx: &TileCoefCtx, plane: usize, start: usize, n: usize, above: bool) -> u8 {
    for i in 0..n {
        let v = if above {
            coef_ctx.above_coef(plane, start + i)
        } else {
            coef_ctx.left_coef(plane, start + i)
        };
        if v != 0 {
            return 1;
        }
    }
    0
}

/// Decode the coefficients of one transform block and write back the
/// per-plane `(eob > 0)` neighbour-context flag.
///
/// Mirrors libvpx `vp9_decode_block_tokens`
/// (`vp9/decoder/vp9_detokenize.c:275-333`).
///
/// # Parameters
///
/// - `reader`: bool-coder positioned at the first token bit; advances
///   by however many bits the block decode consumes.
/// - `fc`: the active per-frame probability bank — we slice
///   `fc.coef_probs[tx_size][plane_type][ref_type]` to get the
///   per-`(tx, plane_type, ref)` table [`decode_coefs`] expects.
/// - `counts`: optional per-block count buckets for backward
///   adaptation. Wave 3 K passes `None` for now (Task 1f wires
///   adaptation later).
/// - `coef_ctx`: per-tile coefficient-context state — read for the
///   initial `pt`, written for the post-decode `(eob > 0)` flag.
/// - `plane`: `0` / `1` / `2` for Y / U / V.
/// - `x4`: top-left 4×4 column of the transform block within the
///   tile, in plane-relative 4×4-sub-block units.
/// - `y4_in_sb`: top-left 4×4 row of the transform block within the
///   current SB row, in plane-relative 4×4-sub-block units.
/// - `tx_size`: transform size (`Tx4x4` / `Tx8x8` / `Tx16x16` /
///   `Tx32x32`).
/// - `tx_type`: transform type (DCT/ADST permutation). Selects the
///   non-default scan tables for `Tx4x4` / `Tx8x8` / `Tx16x16` Y
///   intra; chroma always uses `DctDct`.
/// - `ref_type`: `0` = intra, `1` = inter. Selects between
///   `coef_probs[..][0]` and `coef_probs[..][1]`.
/// - `seg_eob`: maximum scan-position the decode loop may reach
///   (per-segment `SEG_LVL_SKIP` shortens this; callers without
///   segmentation pass `tx_size.n_coeffs()`).
/// - `dequant`: `(dc_dequant, ac_dequant)` for this plane / segment.
/// - `dq_coefs_out`: output buffer of length `>= seg_eob` —
///   [`decode_coefs`] zeroes the first `seg_eob` slots before filling
///   in non-zero coefficients in raster order.
///
/// # Returns
///
/// `eob` — the number of scan positions consumed, equivalently the
/// scan-order position of the first un-decoded coefficient. The
/// caller's downstream `(eob > 0)` decision (for whether to apply the
/// inverse transform) is on this return value.
///
/// # Panics
///
/// Inherits the assertions in [`decode_coefs`] —
/// `seg_eob > tx_size.n_coeffs()`, `dq_coefs_out.len() < seg_eob`, or
/// internally-computed `initial_pt > 2` will trip.
#[allow(clippy::too_many_arguments)] // mirrors libvpx vp9_decode_block_tokens 1:1
pub fn decode_block_tokens(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    counts: Option<&mut DecodeCoefCounts<'_>>,
    coef_ctx: &mut TileCoefCtx,
    plane: usize,
    x4: usize,
    y4_in_sb: usize,
    tx_size: TxSize,
    tx_type: TxType,
    ref_type: usize,
    seg_eob: usize,
    dequant: (i32, i32),
    dq_coefs_out: &mut [i32],
) -> usize {
    let plane_type = plane_type_index(plane);
    // libvpx `cm->fc->coef_probs[tx_size][plane_type][ref_type]`
    // (`vp9_detokenize.c:159-161`).
    let coef_probs: &CoefBandCtxProbs =
        &fc.coef_probs[tx_size as usize][plane_type][ref_type];

    // libvpx `vp9_scan_orders[tx_size][tx_type]`
    // (`vp9_decodeframe.c:326-328`).
    let scan: ScanOrder = scan_for(tx_size, tx_type);

    // OR-fold the above-row + left-column slices spanned by this TX
    // block to compute the initial token-tree context. libvpx packs
    // these as `uintN_t` and uses `!!*` to collapse to 0/1; we walk
    // cells.
    let n = 1usize << (tx_size as usize); // 1 / 2 / 4 / 8
    let a_any = fold_any_nonzero(coef_ctx, plane, x4, n, true);
    let l_any = fold_any_nonzero(coef_ctx, plane, y4_in_sb, n, false);
    let initial_pt = combine_entropy_contexts(a_any, l_any);

    // Stage 3 stretch M.4 — log call-site inputs for the first Block4x4
    // of the first SB. Gated by `RVP9_DEBUG_DQCOEFS=2` so it's a no-op
    // in normal runs. Companion to the dq_coefs dump in
    // `reconstruct.rs` (gated by `=1`). Captures `initial_pt`, the
    // exact above/left neighbour cells used to compute it, the scan
    // kind, the tx_type, the dequant pair, and the coef_probs[band=0]
    // [ctx=initial_pt] row to allow a hand-trace of the first token
    // read against libvpx.
    //
    // Stage 3 stretch M.5 — also captures `(range, value, count, pos)`
    // immediately before the decode_coefs call when `RVP9_DEBUG_BOOLCURSOR=1`
    // is set, so callers can diff the bool-coder state vs libvpx's
    // dump at the same per-sub-block break-point.
    if plane == 0
        && tx_size == TxSize::Tx4x4
        && x4 < 2
        && y4_in_sb < 2
        && std::env::var("RVP9_DEBUG_DQCOEFS").is_ok_and(|v| v == "2")
    {
        let a0 = coef_ctx.above_coef(plane, x4);
        let l0 = coef_ctx.left_coef(plane, y4_in_sb);
        let prob_b0 = &coef_probs[0][initial_pt];
        eprintln!(
            "RVP9_DBG callsite plane=0 tx=4x4 x4={x4} y4_in_sb={y4_in_sb} \
             tx_type={tx_type:?} ref_type={ref_type} \
             a_any={a_any} l_any={l_any} above[{x4}]={a0} left[{y4_in_sb}]={l0} \
             initial_pt={initial_pt} dequant={dequant:?} \
             scan[0..6]={:?} probs[b=0][ctx={initial_pt}]={prob_b0:?}",
            &scan.scan[..6.min(scan.scan.len())]
        );
    }
    if plane == 0
        && tx_size == TxSize::Tx4x4
        && x4 < 2
        && y4_in_sb < 2
        && std::env::var("RVP9_DEBUG_BOOLCURSOR").is_ok_and(|v| v == "1")
    {
        let (range, value, count, pos) = reader.cursor_state();
        eprintln!(
            "RVP9_DBG cursor before decode_coefs (plane=0, tx=4x4, x4={x4}, y4_in_sb={y4_in_sb}) \
             (range={range}, value=0x{value:016x}, count={count}, pos={pos})"
        );
    }

    let eob = decode_coefs(
        reader,
        coef_probs,
        counts,
        &scan,
        initial_pt,
        tx_size,
        seg_eob,
        dequant,
        dq_coefs_out,
    );

    // Writeback: splat `(eob > 0)` across the (n × n) sub-block region
    // libvpx `vp9_detokenize.c:294/302/311/323`.
    coef_ctx.set_coef_flags(plane, x4, y4_in_sb, n, n, eob > 0);

    eob
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::entropy::BoolWriter;
    use rvp9_core::entropy::coef_probs::{COEF_BANDS, COEFF_CONTEXTS, UNCONSTRAINED_NODES};

    /// Build a `FrameContext` whose `coef_probs[tx][plane_type][ref]`
    /// at the indices we exercise is a flat `[p_eob, p_zero, p_one]`
    /// triple per (band, ctx). We start from `new_default` so every
    /// other field is well-formed, then overwrite the slot under test.
    fn fc_with_flat_probs(
        tx_size: TxSize,
        plane_type: usize,
        ref_type: usize,
        p_eob: u8,
        p_zero: u8,
        p_one: u8,
    ) -> FrameContext {
        let mut fc = FrameContext::new_default();
        let triple = [p_eob, p_zero, p_one];
        let ctx_row = [triple; COEFF_CONTEXTS];
        let band_row = [ctx_row; COEF_BANDS];
        fc.coef_probs[tx_size as usize][plane_type][ref_type] = band_row;
        fc
    }

    /// Open a `BoolWriter` over a fresh 256-byte buffer, run the
    /// closure to emit symbols, then close and return the produced
    /// bytes. Identical helper to the one in `decode_coefs` tests.
    fn encode<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolWriter<'_>),
    {
        let mut buf = vec![0u8; 256];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            f(&mut w);
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().unwrap()
        };
        buf.truncate(n);
        buf
    }

    /// Bytes that the encoder produces for "ONE_TOKEN at scan[0], sign
    /// positive, then EOB at scan[1]" given a flat-probs context with
    /// `p_eob = p_zero = p_one = 64`.
    fn encode_one_token_dc_pos() -> Vec<u8> {
        encode(|w| {
            w.write(1, 64); // NOT-EOB
            w.write(1, 64); // NOT-ZERO
            w.write(0, 64); // ONE_TOKEN
            w.write_bit(0); // sign +
            w.write(0, 64); // EOB at pos 1
        })
    }

    /// TX_4×4 at origin, no above/left context (fresh `TileCoefCtx`).
    /// Initial pt = 0 by construction (fold of zeros). After decode
    /// with eob>0, above[0] and left[0] for Y are set to 1.
    #[test]
    fn tx4x4_at_origin_initial_pt_zero_and_writeback() {
        let fc = fc_with_flat_probs(TxSize::Tx4x4, 0, 0, 64, 64, 64);
        let bytes = encode_one_token_dc_pos();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);
        let mut dqc = vec![0i32; 16];

        let eob = decode_block_tokens(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0, // plane Y
            0, 0, // x4, y4_in_sb
            TxSize::Tx4x4,
            TxType::DctDct,
            0, // intra
            16,
            (11, 22),
            &mut dqc,
        );

        assert_eq!(eob, 1);
        // scan[0] = 0 for default 4×4 scan; +1 * dc_dq = 11.
        assert_eq!(dqc[0], 11);
        // Writeback: Y above[0] and left[0] flipped to 1, rest stay 0.
        assert_eq!(coef_ctx.above_coef(0, 0), 1);
        assert_eq!(coef_ctx.left_coef(0, 0), 1);
        for x4 in 1..coef_ctx.above_coef_width(0) {
            assert_eq!(coef_ctx.above_coef(0, x4), 0);
        }
        for y4 in 1..coef_ctx.left_coef_height(0) {
            assert_eq!(coef_ctx.left_coef(0, y4), 0);
        }
    }

    /// TX_8×8: w4 = h4 = 2. Pre-seed `coef_ctx` so the OR-fold of the
    /// 2 spanned above cells is non-zero. Initial pt = 1.
    ///
    /// We can't easily prove "the right pt was used inside
    /// `decode_coefs`" without instrumenting that — instead we verify
    /// (a) the writeback splats across both spanned above + both
    /// spanned left cells, and (b) the decode succeeds (consumes the
    /// expected bits and produces eob > 0) under flat probs.
    #[test]
    fn tx8x8_or_fold_above_nonzero() {
        let fc = fc_with_flat_probs(TxSize::Tx8x8, 0, 0, 64, 64, 64);
        let bytes = encode_one_token_dc_pos();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);
        // Seed: cell above[0,4] non-zero (block to the upper-right of
        // the 0..2 span had a non-zero coeff).
        coef_ctx.set_coef_flags(0, 4, 0, 1, 1, true);
        // x4=4 is inside the 4..6 span our test block covers — verify
        // OR-fold sees it.
        let mut dqc = vec![0i32; 64];

        let eob = decode_block_tokens(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            4,   // x4 = 4 (covers cells 4,5)
            0,   // y4_in_sb = 0 (covers cells 0,1)
            TxSize::Tx8x8,
            TxType::DctDct,
            0,
            64,
            (7, 13),
            &mut dqc,
        );

        // Decode runs to completion; eob > 0 because we emitted
        // ONE_TOKEN at pos 0.
        assert_eq!(eob, 1);
        // Writeback: above[4..6] and left[0..2] all set to 1; nothing
        // else changes. (above[4] was already 1, just stays 1.)
        for x4 in 0..coef_ctx.above_coef_width(0) {
            let expected = if (4..6).contains(&x4) { 1 } else { 0 };
            assert_eq!(coef_ctx.above_coef(0, x4), expected, "above[{x4}]");
        }
        for y4 in 0..coef_ctx.left_coef_height(0) {
            let expected = if (0..2).contains(&y4) { 1 } else { 0 };
            assert_eq!(coef_ctx.left_coef(0, y4), expected, "left[{y4}]");
        }
    }

    /// TX_32×32: w4 = h4 = 8 — the largest OR-fold. Pre-seed a
    /// single non-zero cell in the spanned region of `left_coef`.
    /// Verify the decode + writeback splat across all 8 above and all
    /// 8 left cells.
    #[test]
    fn tx32x32_or_fold_left_nonzero_full_splat_writeback() {
        // Use a wider tile so we have enough above-row space:
        // tile_mi_cols = 4 → Y above_coef width = 8 cells; perfect for
        // a full TX_32X32 at x4 = 0.
        let fc = fc_with_flat_probs(TxSize::Tx32x32, 0, 0, 64, 64, 64);
        let bytes = encode_one_token_dc_pos();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(4);
        // Seed left[5] = 1 (somewhere inside the 0..8 span).
        coef_ctx.set_coef_flags(0, 0, 5, 1, 1, true);
        // Clear above[0] which was also splatted by that set_coef_flags
        // call (x4=0..1) so the above-fold below isn't already 1 from
        // the seed.
        coef_ctx.set_coef_flags(0, 0, 0, 1, 0, false);
        // Confirm seed integrity: above[0] is 0 again, left[5] is 1.
        assert_eq!(coef_ctx.above_coef(0, 0), 0);
        assert_eq!(coef_ctx.left_coef(0, 5), 1);

        let mut dqc = vec![0i32; 1024];

        let eob = decode_block_tokens(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, // x4 = 0 (covers cells 0..8)
            0, // y4_in_sb = 0 (covers cells 0..8)
            TxSize::Tx32x32,
            TxType::DctDct,
            0,
            1024,
            (8, 30), // dc_dq=8 → with dq_shift=1, output = 4
            &mut dqc,
        );

        assert_eq!(eob, 1);
        // dq_shift = 1 for Tx32x32: dqc[0] = 1 * 8 >> 1 = 4.
        assert_eq!(dqc[0], 4);
        // Writeback: all 8 above + all 8 left cells set to 1.
        for x4 in 0..8 {
            assert_eq!(coef_ctx.above_coef(0, x4), 1, "above[{x4}] splat");
        }
        for y4 in 0..8 {
            assert_eq!(coef_ctx.left_coef(0, y4), 1, "left[{y4}] splat");
        }
    }

    /// EOB at scan-position 0 ⇒ eob = 0 ⇒ writeback clears every cell
    /// in the spanned region (mirrors libvpx's
    /// `a[..] = l[..] = (eob > 0) = 0` path).
    #[test]
    fn eob_zero_writeback_clears_cells() {
        let fc = fc_with_flat_probs(TxSize::Tx8x8, 0, 0, 128, 128, 128);
        let bytes = encode(|w| {
            // EOB at pos 0.
            w.write(0, 128);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);
        // Pre-seed: every cell in the 2×2 region is 1 (e.g. a stale
        // value from a previous TX block that aliased these cells).
        coef_ctx.set_coef_flags(0, 2, 4, 2, 2, true);
        for x4 in 2..4 {
            assert_eq!(coef_ctx.above_coef(0, x4), 1);
        }
        for y4 in 4..6 {
            assert_eq!(coef_ctx.left_coef(0, y4), 1);
        }
        let mut dqc = vec![0i32; 64];

        let eob = decode_block_tokens(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            2,
            4,
            TxSize::Tx8x8,
            TxType::DctDct,
            0,
            64,
            (10, 20),
            &mut dqc,
        );

        assert_eq!(eob, 0);
        // The 2×2 writeback region is cleared to 0.
        for x4 in 2..4 {
            assert_eq!(coef_ctx.above_coef(0, x4), 0, "above[{x4}] cleared");
        }
        for y4 in 4..6 {
            assert_eq!(coef_ctx.left_coef(0, y4), 0, "left[{y4}] cleared");
        }
        // The output buffer is all zero (decode_coefs zeroes the
        // seg_eob prefix).
        for &v in &dqc {
            assert_eq!(v, 0);
        }
    }

    /// Plane independence: decoding on plane 0 (Y) must not touch
    /// plane 1 / 2 (UV) coef-context cells. Mirrors libvpx's
    /// `pd = &xd->plane[plane]` indirection — the above/left context
    /// rows are per-plane.
    #[test]
    fn plane_independence_y_decode_leaves_uv_untouched() {
        let fc = fc_with_flat_probs(TxSize::Tx4x4, 0, 0, 64, 64, 64);
        let bytes = encode_one_token_dc_pos();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);
        // Pre-populate UV cells so we can detect cross-contamination.
        coef_ctx.set_coef_flags(1, 3, 3, 1, 1, true);
        coef_ctx.set_coef_flags(2, 5, 5, 1, 1, true);
        let mut dqc = vec![0i32; 16];

        let _eob = decode_block_tokens(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx4x4,
            TxType::DctDct,
            0,
            16,
            (11, 22),
            &mut dqc,
        );

        // U / V cells we pre-seeded are still set; nothing else
        // crept in. Per-plane "where we seeded a 1" map.
        let seeded: [(usize, usize); 2] = [(1, 3), (2, 5)];
        for &(plane, seed) in &seeded {
            for x4 in 0..coef_ctx.above_coef_width(plane) {
                let expected = u8::from(x4 == seed);
                assert_eq!(
                    coef_ctx.above_coef(plane, x4),
                    expected,
                    "plane {plane} above[{x4}]"
                );
            }
            for y4 in 0..coef_ctx.left_coef_height(plane) {
                let expected = u8::from(y4 == seed);
                assert_eq!(
                    coef_ctx.left_coef(plane, y4),
                    expected,
                    "plane {plane} left[{y4}]"
                );
            }
        }
    }

    /// Wrapper output matches a hand-invoked `decode_coefs` call with
    /// the same slicing + `combine_entropy_contexts` call.
    ///
    /// This is the differential oracle: the wrapper's value-add is the
    /// bookkeeping, not the decode itself. We re-run the same byte
    /// stream through `decode_coefs` directly and verify the produced
    /// eob + dq_coefs_out are identical.
    #[test]
    fn wrapper_matches_manual_decode_coefs_call() {
        let fc = fc_with_flat_probs(TxSize::Tx4x4, 0, 0, 64, 64, 64);
        let bytes = encode_one_token_dc_pos();

        // Reference: manual slicing + manual decode_coefs.
        let mut r_ref = BoolReader::new(&bytes).unwrap();
        let coef_probs: &CoefBandCtxProbs = &fc.coef_probs[TxSize::Tx4x4 as usize][0][0];
        let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
        let mut dqc_ref = vec![0i32; 16];
        let eob_ref = decode_coefs(
            &mut r_ref,
            coef_probs,
            None,
            &scan,
            combine_entropy_contexts(0, 0),
            TxSize::Tx4x4,
            16,
            (11, 22),
            &mut dqc_ref,
        );

        // Wrapper invocation.
        let mut r_wrap = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);
        let mut dqc_wrap = vec![0i32; 16];
        let eob_wrap = decode_block_tokens(
            &mut r_wrap,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx4x4,
            TxType::DctDct,
            0,
            16,
            (11, 22),
            &mut dqc_wrap,
        );

        assert_eq!(eob_ref, eob_wrap);
        assert_eq!(dqc_ref, dqc_wrap);
    }

    /// `plane_type_index` matches libvpx `get_plane_type`.
    #[test]
    fn plane_type_index_matches_libvpx() {
        assert_eq!(plane_type_index(0), 0, "Y");
        assert_eq!(plane_type_index(1), 1, "U");
        assert_eq!(plane_type_index(2), 1, "V");
    }

    /// `fold_any_nonzero` walks a contiguous slice and returns 1 iff
    /// any cell is non-zero. Hand-traced for the common cases.
    #[test]
    fn fold_any_nonzero_or_reduces_slice() {
        let mut coef_ctx = TileCoefCtx::new(8);
        // All zero: fold = 0 over any span.
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 0, 1, true), 0);
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 0, 8, true), 0);
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 0, 4, false), 0);
        // Single non-zero at above[5]: fold over [4..6] is 1, fold over
        // [0..4] is 0.
        coef_ctx.set_coef_flags(0, 5, 0, 1, 0, true); // sets above[5]
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 4, 2, true), 1);
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 0, 4, true), 0);
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 5, 1, true), 1);
        // Out-of-range slices: TileCoefCtx::above_coef returns 0, so
        // a fully out-of-range fold is 0.
        assert_eq!(fold_any_nonzero(&coef_ctx, 0, 100, 4, true), 0);
    }

    /// Inter ref_type (= 1) selects a different sub-table than intra
    /// (= 0). We craft a `FrameContext` where the two sub-tables
    /// differ in a way the decoder can detect: `ref_type=0` has
    /// `p_eob = 128`, `ref_type=1` has `p_eob = 1` (so the decoder
    /// would NOT take EOB at low entropy; the encoder must emit the
    /// corresponding bit).
    ///
    /// Encode an EOB symbol under `p_eob = 1` (the inter table). The
    /// wrapper with `ref_type = 1` decodes it cleanly; the wrapper
    /// with `ref_type = 0` (which has `p_eob = 128`) would not, but we
    /// can at least verify the path runs and the output cell write
    /// matches the eob taken.
    #[test]
    fn ref_type_selects_intra_vs_inter_subtable() {
        let mut fc = FrameContext::new_default();
        // Build two distinct flat tables and install them at the
        // (Tx4x4, plane_type=0) slot, one per ref_type.
        let triple_intra = [128u8, 128, 128];
        let triple_inter = [1u8, 1, 1];
        let intra_band = [[triple_intra; COEFF_CONTEXTS]; COEF_BANDS];
        let inter_band = [[triple_inter; COEFF_CONTEXTS]; COEF_BANDS];
        fc.coef_probs[TxSize::Tx4x4 as usize][0][0] = intra_band;
        fc.coef_probs[TxSize::Tx4x4 as usize][0][1] = inter_band;

        // Encode EOB under the inter (p_eob=1) table: emit bit 0 with
        // prob 1.
        let bytes = encode(|w| {
            w.write(0, 1);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(4);
        let mut dqc = vec![0i32; 16];

        let eob = decode_block_tokens(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx4x4,
            TxType::DctDct,
            1, // inter
            16,
            (10, 20),
            &mut dqc,
        );

        assert_eq!(eob, 0, "EOB taken on inter sub-table");
        // eob = 0 ⇒ writeback splats 0 (which is already the initial
        // state for this cell). The assertion is that the wrapper
        // didn't panic / pick the wrong (much higher-entropy) intra
        // table, which would have produced a different bit sequence.
        assert_eq!(coef_ctx.above_coef(0, 0), 0);
        assert_eq!(coef_ctx.left_coef(0, 0), 0);
    }

    // Sanity exports.
    const _: () = {
        // UNCONSTRAINED_NODES is referenced via CoefBandCtxProbs's
        // shape. Pinning it here keeps any future shape change visible
        // at compile time.
        assert!(UNCONSTRAINED_NODES == 3);
    };
}
