//! Coefficient-bit-aware transform-domain RD estimate for one Y plane
//! block.
//!
//! Faithful port of libvpx's `block_yrd`
//! (`vp9/encoder/vp9_pickmode.c:718-844`, v1.14.1, commit
//! `12f3a2ac6`). The realtime nonrd picker uses this to refine the
//! initial `model_rd_for_sb_y` rate estimate with a coefficient-domain
//! measure: SATD of the quantized coefficients (`vpx_satd_c`,
//! `vpx_dsp/avg.c:334-341`) for the rate term, and
//! `block_error_fp(coeff, dqcoeff)` (`vp9_rdopt.c:334-345`) for the
//! distortion term. Both are deliberately coarse models, ~10-20× cheaper
//! than the exact coef-prob-tree cost the RDO path uses
//! (`vp9_rdopt.c:cost_coeffs`).
//!
//! ## Divergences from libvpx
//!
//! - **DCT vs Hadamard (DV-F1-1).** libvpx's `block_yrd` runs
//!   `vpx_hadamard_{8x8,16x16}` to produce the SATD inputs; rvp9 reuses
//!   the DCT coefficients the picker's existing
//!   `reconstruct_via_round_trip` already computed (see
//!   `mode_decision::inter`). Halves the transform cost; may need a
//!   scalar correction at F.4 if the SATD-to-bits proportionality is
//!   materially off.
//! - **`vp9_quantize_b` vs `vp9_quantize_fp` (DV-F1-2).** rvp9's picker
//!   uses `vp9_quantize_b` (with zbin pre-test) where libvpx's
//!   `block_yrd` uses `vp9_quantize_fp` (no zbin). The two produce
//!   different qcoeff magnitudes, so the SATD scale differs slightly;
//!   we accept this for the F.1 first cut.
//! - **Simple-yrd shortcut not ported.** libvpx's `block_yrd` has a
//!   speed-feature gate (`use_simple_block_yrd`) that bypasses the
//!   full path for `speed >= 8` / SVC non-base layers. We always run
//!   the full path, which matches the cpu-used-7 baseline; cpu-used-8
//!   tuning lands as F.1.1.
//! - **`*sse` rescale + skippable early-exit not surfaced here.**
//!   libvpx's `block_yrd` consumes a caller-supplied `model_rd_for_sb_y`
//!   `sse` and re-emits it for the skip-vs-non-skip RDCOST tie-break.
//!   rvp9 computes pixel-SSE-of-recon directly in the caller
//!   (`evaluate_candidate`), so the `*sse` plumbing isn't needed here.
//!   The `skippable` flag IS surfaced (`BlockYrdResult::skippable`)
//!   because the caller's skip-flag tie-break still needs it.
//!
//! See `wrk_docs/Track_F/F1_HLD_block_yrd_port.md` for the full design.

use crate::cost::VP9_PROB_COST_SHIFT;
use crate::transforms::TxSize;

/// Output of [`block_yrd`].
///
/// `coef_rate` is in `vp9_prob_cost` units (1/512-bit), on the same
/// scale as the picker's `mode_rate` and `mv_rate`. **Held as `u64` to
/// avoid the early `u32::MAX` saturation** that masks K1 calibration
/// signal at low q (large qcoeffs): pre-fix the field saturated inside
/// `block_yrd` / `reconstruct_via_round_trip` itself; post-fix the
/// caller saturate-casts to `u32` only when adding to the rate sum,
/// preserving K1 signal across the realistic operating range. `coef_dist`
/// is the coefficient-domain squared error accumulator, kept as `i64`
/// to avoid overflow on a saturated 32×32 quant-error block (Q7
/// resolution). `skippable` mirrors libvpx's `*skippable` output: `true`
/// iff every tile had `eob == 0`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct BlockYrdResult {
    /// Coefficient-bit rate estimate in `vp9_prob_cost` units (1/512-bit).
    /// `u64` (not `u32`) to avoid early saturation — see struct doc.
    pub coef_rate: u64,
    /// Coefficient-domain squared error:
    /// `sum_i (coeff[i] - dqcoeff[i])^2 >> 2`.
    /// `i64` to avoid overflow on saturated 32×32 quant-error blocks.
    pub coef_dist: i64,
    /// `true` iff every tile had `eob == 0` (residual fully quantized to
    /// zero). Mirrors libvpx's `*skippable`.
    pub skippable: bool,
}

/// Coefficient-bit-aware transform-domain RD estimate.
///
/// `coeff_tiles`, `qcoeff_tiles`, `dqcoeff_tiles`, and `eob_tiles` are
/// the per-tile output of the caller's forward DCT + quantize pipeline.
/// All four slices must have the same length (one entry per transform
/// tile). Each inner coefficient slice must have exactly
/// `tx_size.n_coeffs()` entries.
///
/// `tx_size` is the per-tile transform size (the same value the caller
/// used to run the forward transform).
///
/// Returns the libvpx-faithful triple `(coef_rate, coef_dist,
/// skippable)`. See [`BlockYrdResult`] for the units / semantics.
///
/// # Current callers
///
/// The F.1 inter picker (`mode_decision::inter::reconstruct_via_round_trip`)
/// **inlines** the per-tile accumulator inside its existing forward-DCT
/// loop instead of allocating slice-of-slice buffers and calling this
/// function — see the `by_skippable` / `by_eob_cost` / `by_rate_satd`
/// / `by_dist` locals in `inter.rs`. This function exists as the
/// canonical reference implementation (and is unit-tested directly,
/// see `tests` below) and is the entry point intended for the future
/// intra-path port (libvpx `vp9_pickmode.c:1073` via
/// `estimate_block_intra`, HLD F.1 §5.1).
///
/// # libvpx reference
///
/// `vp9/encoder/vp9_pickmode.c:718-844`, v1.14.1, commit `12f3a2ac6`.
/// The output corresponds to libvpx's `this_rdc->rate`, `this_rdc->dist`,
/// and `*skippable` after the function returns. (We don't surface the
/// `*sse` out-param — see the module-level divergence note.)
#[must_use]
pub fn block_yrd(
    coeff_tiles: &[&[i32]],
    qcoeff_tiles: &[&[i16]],
    dqcoeff_tiles: &[&[i16]],
    eob_tiles: &[u16],
    tx_size: TxSize,
) -> BlockYrdResult {
    debug_assert_eq!(coeff_tiles.len(), qcoeff_tiles.len());
    debug_assert_eq!(coeff_tiles.len(), dqcoeff_tiles.len());
    debug_assert_eq!(coeff_tiles.len(), eob_tiles.len());

    let n = tx_size.n_coeffs();

    // libvpx initialises `*skippable = 1` at the top of `block_yrd`
    // (line 768). An empty tile list trivially satisfies the
    // all-eob-zero invariant.
    let mut skippable = true;
    let mut eob_cost: u32 = 0;
    let mut rate_satd: u32 = 0;
    let mut dist: i64 = 0;

    for ((coeff, qcoeff), (dqcoeff, &eob)) in coeff_tiles
        .iter()
        .zip(qcoeff_tiles.iter())
        .zip(dqcoeff_tiles.iter().zip(eob_tiles.iter()))
    {
        debug_assert_eq!(coeff.len(), n);
        debug_assert_eq!(qcoeff.len(), n);
        debug_assert_eq!(dqcoeff.len(), n);

        // libvpx line 804: `*skippable &= (*eob == 0)`.
        skippable &= eob == 0;
        // libvpx line 805: `eob_cost += 1` per emitted sub-block.
        eob_cost = eob_cost.saturating_add(1);

        // libvpx lines 830-833 — rate path:
        //   if (*eob == 1) this_rdc->rate += abs(qcoeff[0]);
        //   else if (*eob > 1) this_rdc->rate += vpx_satd(qcoeff, step << 4);
        if eob == 1 {
            rate_satd = rate_satd.saturating_add(i32::from(qcoeff[0]).unsigned_abs());
        } else if eob > 1 {
            rate_satd = rate_satd.saturating_add(satd_i16(qcoeff));
        }

        // libvpx line 835: `dist += vp9_block_error_fp(coeff, dqcoeff, ...) >> 2`.
        dist = dist.saturating_add(block_error_fp_i32(coeff, dqcoeff) >> 2);
    }

    // libvpx lines 842-843 — final rate scale:
    //   this_rdc->rate <<= (2 + VP9_PROB_COST_SHIFT);
    //   this_rdc->rate += (eob_cost << VP9_PROB_COST_SHIFT);
    //
    // `VP9_PROB_COST_SHIFT = 9`, so the combined shift is 11. Held as
    // `u64` to preserve K1 calibration signal — the pre-fix `.min(u32::MAX
    // as u64) as u32` truncation hit on low-q blocks (large qcoeffs ⇒
    // rate_satd > 2_097_152 ⇒ shifted_rate > u32::MAX at K1=2), making
    // K1 ∈ {0..4} produce identical-looking outputs and masking real
    // signal. Saturate-cast back to u32 happens at the caller's rate-sum
    // site in `evaluate_candidate` (where the cap is honest rather than
    // covert).
    let shifted_rate = (rate_satd as u64) << (2 + VP9_PROB_COST_SHIFT);
    let eob_term = (eob_cost as u64) << VP9_PROB_COST_SHIFT;
    let coef_rate = shifted_rate.saturating_add(eob_term);

    BlockYrdResult {
        coef_rate,
        coef_dist: dist,
        skippable,
    }
}

/// Sum of absolute values of quantized coefficients.
///
/// Mirrors libvpx `vpx_satd_c` (`vpx_dsp/avg.c:334-341`):
///
/// ```c
/// int vpx_satd_c(const tran_low_t *coeff, int length) {
///     int satd = 0;
///     for (i = 0; i < length; ++i) satd += abs(coeff[i]);
///     return satd;
/// }
/// ```
///
/// Returns `u32` because the input quantized coefficients are bounded
/// (`tran_low_t` is `int16_t` in non-highbd builds, dynamic range
/// `[-32640, 32640]`); over `length <= 1024` the SATD fits in 26 bits.
/// We use `u32` so the caller can compose with the libvpx rate
/// accumulator without sign-juggling.
#[must_use]
pub(crate) fn satd_i16(qcoeff: &[i16]) -> u32 {
    let mut satd: u32 = 0;
    for &q in qcoeff {
        // `i16::abs()` overflows on `i16::MIN`; promote to i32 first.
        satd = satd.saturating_add(i32::from(q).unsigned_abs());
    }
    satd
}

/// Coefficient-domain squared quantization error.
///
/// Mirrors libvpx `vp9_block_error_fp_c` (`vp9_rdopt.c:334-345`):
///
/// ```c
/// int64_t vp9_block_error_fp_c(const tran_low_t *coeff,
///                              const tran_low_t *dqcoeff, int block_size) {
///     int64_t error = 0;
///     for (i = 0; i < block_size; i++) {
///         const int diff = coeff[i] - dqcoeff[i];
///         error += diff * diff;
///     }
///     return error;
/// }
/// ```
///
/// rvp9's picker carries un-quantized coefficients as `i32` (the
/// forward DCT writes `i32` into its scratch buffer), so this helper
/// takes `coeff: &[i32]`. `dqcoeff` stays `i16` to match the rvp9 quant
/// pipeline's output type. libvpx's `tran_low_t` is `int16_t` for the
/// non-highbd build and the subtraction sits comfortably in `i32`.
#[must_use]
pub(crate) fn block_error_fp_i32(coeff: &[i32], dqcoeff: &[i16]) -> i64 {
    debug_assert_eq!(coeff.len(), dqcoeff.len());
    let mut error: i64 = 0;
    for (&c, &dq) in coeff.iter().zip(dqcoeff.iter()) {
        let diff = c - i32::from(dq);
        error = error.saturating_add(i64::from(diff) * i64::from(diff));
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `(coeff, qcoeff, dqcoeff)` tile triple of length `n`,
    /// filled with zero. Convenience for the small-block unit tests.
    fn zero_tile(n: usize) -> (Vec<i32>, Vec<i16>, Vec<i16>) {
        (vec![0i32; n], vec![0i16; n], vec![0i16; n])
    }

    /// All-zero residual: every tile has `eob == 0`. The libvpx code
    /// initialises `*skippable = 1` (line 768), increments `eob_cost`
    /// once per visited tile (line 805), then the rate path adds
    /// nothing (the `if (*eob == 1) ... else if (*eob > 1)` branch
    /// never fires when `eob == 0`). The final rate is
    /// `0 << 11 + eob_cost << 9`, but the post-loop `if (*skippable)`
    /// gate at libvpx line 814 short-circuits the rate to 0 and sets
    /// `dist = *sse` — we model this here by reporting the rate
    /// from the formula and letting the caller use `skippable` to
    /// gate the rate (see HLD §5.5).
    ///
    /// Per HLD §7.1: "assert skippable == true, coef_rate == 0
    /// (no SATD bits, no EOB charge: eob_cost == 0)". The HLD's
    /// test description maps to the "empty tile list" edge case
    /// (no tiles visited → eob_cost stays at zero). With one
    /// all-zero tile, `eob_cost == 1` and the rate is `1 << 9 = 512`
    /// (one tile of EOB charge). We test BOTH variants.
    #[test]
    fn block_yrd_all_zero_residual_is_skippable() {
        // Variant A: zero tiles → coef_rate == 0, skippable == true.
        let r0 = block_yrd(&[], &[], &[], &[], TxSize::Tx16x16);
        assert!(r0.skippable, "no tiles → trivially skippable");
        assert_eq!(r0.coef_rate, 0, "no tiles → no rate");
        assert_eq!(r0.coef_dist, 0);

        // Variant B: one all-zero tile → still skippable (eob == 0),
        // coef_rate is the EOB charge (`1 << VP9_PROB_COST_SHIFT`).
        let (c, q, dq) = zero_tile(256);
        let eobs = [0u16];
        let r = block_yrd(
            &[&c[..]],
            &[&q[..]],
            &[&dq[..]],
            &eobs,
            TxSize::Tx16x16,
        );
        assert!(r.skippable, "all-zero qcoeff → skippable");
        // No SATD contribution; only the per-tile EOB cost.
        assert_eq!(
            r.coef_rate,
            1u64 << VP9_PROB_COST_SHIFT,
            "all-zero tile → rate is EOB-cost only",
        );
        assert_eq!(r.coef_dist, 0, "coef == dqcoeff → zero block-error");
    }

    /// Single non-zero qcoeff at position 0, `eob == 1`. The libvpx
    /// rate path takes the `if (*eob == 1)` branch (line 830) and adds
    /// `abs(qcoeff[0])` directly. Final rate is
    /// `(|qcoeff[0]| << 2 + eob_cost) << VP9_PROB_COST_SHIFT`.
    #[test]
    fn block_yrd_all_zero_except_first_uses_dc_path() {
        let n = 64;
        let mut q = vec![0i16; n];
        q[0] = 7;
        let dq = vec![0i16; n];
        let c = vec![0i32; n];
        let eobs = [1u16];

        let r = block_yrd(
            &[&c[..]],
            &[&q[..]],
            &[&dq[..]],
            &eobs,
            TxSize::Tx8x8,
        );
        assert!(!r.skippable, "eob == 1 → not skippable");

        // Hand-compute the expected rate.
        let satd: u64 = 7;
        let expected = (satd << 2) << VP9_PROB_COST_SHIFT;
        let eob_term: u64 = 1u64 << VP9_PROB_COST_SHIFT;
        assert_eq!(r.coef_rate, expected + eob_term);
    }

    /// Multiple non-zero qcoeffs with `eob > 1`. The libvpx rate path
    /// takes the `else if (*eob > 1)` branch (line 832) and adds
    /// `vpx_satd(qcoeff, ...)` — the SUM of absolute values across the
    /// whole tile.
    #[test]
    fn block_yrd_satd_path_for_eob_gt_1() {
        let n = 64;
        let mut q = vec![0i16; n];
        q[0] = 5;
        q[1] = -3;
        q[2] = 4;
        let dq = vec![0i16; n];
        let c = vec![0i32; n];
        let eobs = [3u16];

        let r = block_yrd(
            &[&c[..]],
            &[&q[..]],
            &[&dq[..]],
            &eobs,
            TxSize::Tx8x8,
        );
        // satd = 5 + 3 + 4 = 12.
        let satd: u64 = 12;
        let expected: u64 = ((satd << 2) << VP9_PROB_COST_SHIFT)
            + (1u64 << VP9_PROB_COST_SHIFT);
        assert_eq!(r.coef_rate, expected);
        assert!(!r.skippable);
    }

    /// `coef = [10, 20, 30, ...]`, `dqcoeff = [8, 18, 28, ...]` — per-coef
    /// errors are `[2, 2, 2, ...]`, squared `[4, 4, 4, ...]`. The libvpx
    /// `>> 2` shift at line 835 divides the total by 4.
    ///
    /// We pad the tile to a full TX_8x8 (64 coeffs) with zero-error
    /// entries so the shape matches the libvpx scan, then check the
    /// HLD §7.1 hand-computed value: 3 non-zero diffs of magnitude 2
    /// each → `(4 + 4 + 4) >> 2 = 3`.
    #[test]
    fn block_yrd_dist_is_block_error_fp_over_4() {
        let n = 64;
        let mut c = vec![0i32; n];
        let mut dq = vec![0i16; n];
        c[0] = 10;
        dq[0] = 8;
        c[1] = 20;
        dq[1] = 18;
        c[2] = 30;
        dq[2] = 28;
        // qcoeff doesn't enter the dist computation, just keep it
        // matched to dq so eob > 0 makes sense for the test.
        let q = dq.clone();
        let eobs = [3u16];

        let r = block_yrd(
            &[&c[..]],
            &[&q[..]],
            &[&dq[..]],
            &eobs,
            TxSize::Tx8x8,
        );
        // (4 + 4 + 4) >> 2 = 3.
        assert_eq!(r.coef_dist, 3);
    }

    /// Scale a tile's qcoeff by 2× and the rate must strictly increase
    /// — the single most important sanity check, exposes any off-by-one
    /// in the SATD path.
    #[test]
    fn block_yrd_monotone_in_residual_energy() {
        let n = 256;
        // Block A: small qcoeffs.
        let mut q_a = vec![0i16; n];
        q_a[0] = 3;
        q_a[1] = -2;
        q_a[5] = 4;
        q_a[20] = 1;
        // Block B: 2× the magnitudes of A.
        let q_b: Vec<i16> = q_a.iter().map(|&v| v * 2).collect();

        let dq = vec![0i16; n];
        let c = vec![0i32; n];
        let eob_a = [4u16];
        let eob_b = [4u16];

        let r_a = block_yrd(
            &[&c[..]],
            &[&q_a[..]],
            &[&dq[..]],
            &eob_a,
            TxSize::Tx16x16,
        );
        let r_b = block_yrd(
            &[&c[..]],
            &[&q_b[..]],
            &[&dq[..]],
            &eob_b,
            TxSize::Tx16x16,
        );
        assert!(
            r_b.coef_rate > r_a.coef_rate,
            "B's rate ({}) must exceed A's ({})",
            r_b.coef_rate, r_a.coef_rate,
        );
    }

    /// Multi-tile sum: a Block64x64 → 4× Tx32x32 tiles should produce
    /// the same `coef_rate` and `coef_dist` whether the four tiles are
    /// passed in one call or summed across four separate calls.
    #[test]
    fn block_yrd_multi_tile_64x64_sums_correctly() {
        let n = 1024;
        // Build four distinct tiles so the sum is non-degenerate.
        let mk_tile = |seed: i16| -> (Vec<i32>, Vec<i16>, Vec<i16>) {
            let mut q = vec![0i16; n];
            let mut dq = vec![0i16; n];
            let mut c = vec![0i32; n];
            q[0] = seed;
            q[10] = seed * 2;
            dq[0] = seed - 1;
            dq[10] = seed * 2;
            c[0] = i32::from(seed) + 5;
            c[10] = i32::from(seed) * 2 + 3;
            (c, q, dq)
        };
        let t0 = mk_tile(3);
        let t1 = mk_tile(5);
        let t2 = mk_tile(7);
        let t3 = mk_tile(11);

        let coeffs: Vec<&[i32]> = vec![&t0.0, &t1.0, &t2.0, &t3.0];
        let qs: Vec<&[i16]> = vec![&t0.1, &t1.1, &t2.1, &t3.1];
        let dqs: Vec<&[i16]> = vec![&t0.2, &t1.2, &t2.2, &t3.2];
        let eobs = [11u16, 11u16, 11u16, 11u16];

        let r_all = block_yrd(&coeffs, &qs, &dqs, &eobs, TxSize::Tx32x32);

        // Per-tile partial sums of the pre-scale rate_satd and dist.
        // We can't directly read the internal accumulators, so we
        // exercise the equivalent invariant by calling block_yrd
        // four times with one tile each, summing the rates and dist,
        // and undoing the libvpx EOB charge (which is per-call).
        let mut sum_rate: u64 = 0;
        let mut sum_dist: i64 = 0;
        for i in 0..4 {
            let r_i = block_yrd(
                &[coeffs[i]],
                &[qs[i]],
                &[dqs[i]],
                &[eobs[i]],
                TxSize::Tx32x32,
            );
            // Strip the per-call eob_cost (1 << 9) and re-add it once
            // at the end so the per-call accumulator matches the
            // 4-tile single-call accumulator (which adds eob_cost
            // four times in one call).
            sum_rate += r_i.coef_rate - (1u64 << VP9_PROB_COST_SHIFT);
            sum_dist += r_i.coef_dist;
        }
        sum_rate += 4u64 << VP9_PROB_COST_SHIFT;
        assert_eq!(
            r_all.coef_rate, sum_rate,
            "single-call rate must match summed-per-tile rate",
        );
        assert_eq!(r_all.coef_dist, sum_dist);
    }
}
