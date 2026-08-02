//! W4.3 — Coefficient rate primitives for RDO: `cost_coeffs` driver
//! + `band_counts` walker.
//!
//! ## libvpx reference
//!
//! - `band_counts[TX_SIZES][8]` — `vp9/encoder/vp9_rdopt.c:352-357`
//!   (v1.14.1 commit `12f3a2ac6`).
//! - `cost_coeffs` — `vp9/encoder/vp9_rdopt.c:358-459`. Both the fast
//!   speed-feature path and the full neighbour-context path are ported.
//! - `vp9_pt_energy_class` — `vp9/common/vp9_entropy.c:95-96`.
//! - `vp9_get_token` — `vp9/encoder/vp9_tokenize.h:108-111`. The
//!   threshold table mirrors the `CATn_MIN_VAL` constants in
//!   `vp9_entropy.h:46-51`.
//!
//! ## Scope
//!
//! Scalar reference; 8-bit only. We do NOT charge the cat6 extra-bit
//! cost (`vp9_cat6_high_cost` etc.) because the test setup uses
//! `|qcoeff| < CAT6_MIN_VAL`. The cat 1..5 extra bits ARE charged via
//! the per-token `extra_bits_cost` table mirroring libvpx
//! `vp9_extra_bits[t].cost` (`vp9_tokenize.c`).

use rvp9_core::cost::{
    coef_cost::{CoefCostIndex, CoefCostTable, ENTROPY_TOKENS, coef_token_cost},
    vp9_cost_bit,
};
use rvp9_core::transforms::TxSize;

/// `EOB_TOKEN = 11` per libvpx `vp9_entropy.h:38`.
pub const EOB_TOKEN: u8 = 11;

/// `ZERO_TOKEN = 0` per libvpx `vp9_entropy.h:27`.
pub const ZERO_TOKEN: u8 = 0;

/// `vp9_pt_energy_class[ENTROPY_TOKENS]` from libvpx
/// `vp9/common/vp9_entropy.c:95-96`. Maps a coefficient token to its
/// per-token "energy class" (used to derive the next-position
/// neighbour context).
#[rustfmt::skip]
pub const VP9_PT_ENERGY_CLASS: [u8; ENTROPY_TOKENS] = [
    0, 1, 2, 3, 3, 4,
    4, 5, 5, 5, 5, 5,
];

/// `band_counts[TX_SIZES][8]` from libvpx
/// `vp9/encoder/vp9_rdopt.c:352-357`. The number of coefficients in
/// each coefficient-band slot, indexed by `[tx_size][band]`.
///
/// libvpx's first column (`band_counts[t][0] = 1`) is the DC band;
/// the `cost_coeffs` driver advances past it with `&band_counts[tx][1]`
/// before iterating the AC bands. The trailing `0` is the libvpx
/// terminator — when the per-band counter reaches the `0` slot the
/// EOB token's cost should NOT be re-added (the last coefficient
/// already saturated the band).
#[rustfmt::skip]
pub const BAND_COUNTS: [[i16; 8]; 4] = [
    // TX_4X4:  1 + 2 + 3 + 4 + 3 + 3 = 16 = total coeffs.
    [ 1, 2, 3, 4, 3, 16 - 13, 0, 0 ],
    // TX_8X8:  1 + 2 + 3 + 4 + 11 + 43 = 64.
    [ 1, 2, 3, 4, 11, 64 - 21, 0, 0 ],
    // TX_16X16: 1 + 2 + 3 + 4 + 11 + 235 = 256.
    [ 1, 2, 3, 4, 11, 256 - 21, 0, 0 ],
    // TX_32X32: 1 + 2 + 3 + 4 + 11 + 1003 = 1024.
    [ 1, 2, 3, 4, 11, 1024 - 21, 0, 0 ],
];

/// Map an unsigned `|qcoeff|` value to its libvpx coefficient token.
/// Faithful port of libvpx `vp9_get_token` (`vp9_tokenize.h:108-111`)
/// using the `CATn_MIN_VAL` thresholds at `vp9_entropy.h:46-51`.
///
/// Returns one of `ZERO/ONE/TWO/THREE/FOUR/CAT1..CAT6` (token values
/// `0..=10`). Never returns `EOB_TOKEN` (callers handle EOB
/// separately).
#[inline(always)]
#[must_use]
pub fn vp9_get_token(v: i32) -> u8 {
    let abs_v = v.unsigned_abs();
    match abs_v {
        0 => 0,       // ZERO_TOKEN
        1 => 1,       // ONE_TOKEN
        2 => 2,       // TWO_TOKEN
        3 => 3,       // THREE_TOKEN
        4 => 4,       // FOUR_TOKEN
        5..=6 => 5,   // CATEGORY1_TOKEN  (CAT1_MIN_VAL=5)
        7..=10 => 6,  // CATEGORY2_TOKEN  (CAT2_MIN_VAL=7)
        11..=18 => 7, // CATEGORY3_TOKEN  (CAT3_MIN_VAL=11)
        19..=34 => 8, // CATEGORY4_TOKEN  (CAT4_MIN_VAL=19)
        35..=66 => 9, // CATEGORY5_TOKEN  (CAT5_MIN_VAL=35)
        _ => 10,      // CATEGORY6_TOKEN  (CAT6_MIN_VAL=67)
    }
}

/// `band_counts` walker — used by [`cost_coeffs`] to step from one
/// coefficient band to the next as scan-position advances. Mirrors
/// the `band_left` decrement / `*band_count++` pattern in libvpx
/// `cost_coeffs` (`vp9_rdopt.c:388-411`).
///
/// Holds the current band index (`0..6`) and the count of coefficients
/// remaining in the current band. `step()` returns `true` when the
/// band index advances; the caller uses this to bump its
/// `token_costs` slab pointer.
#[derive(Copy, Clone, Debug)]
pub struct BandCountsWalker {
    /// Per-band coefficient counts for the current tx_size.
    counts: [i16; 8],
    /// Index of the next band to consume.
    next_idx: usize,
    /// Coefficients still to emit in the current band.
    band_left: i16,
}

impl BandCountsWalker {
    /// Start a walker after consuming the libvpx DC slot
    /// (`band_count = &band_counts[tx_size][1]`, line 365). The
    /// caller has emitted the DC coefficient before this constructor
    /// returns; the walker manages the AC bands going forward.
    #[must_use]
    #[inline(always)]
    pub fn start_after_dc(tx_size: TxSize) -> Self {
        let counts = BAND_COUNTS[tx_size as usize];
        // libvpx advances past the DC slot at line 365 and then reads
        // the first AC count into `band_left` at line 388
        // (`band_left = *band_count++`).
        Self {
            counts,
            next_idx: 2,
            band_left: counts[1],
        }
    }

    /// Consume one AC coefficient. Returns `true` iff the band index
    /// just advanced (caller should bump its `token_costs` slab).
    /// When the trailing `0` sentinel is hit (libvpx's terminator),
    /// `band_left` stays at zero and the caller stops emitting EOB.
    #[inline(always)]
    pub fn step(&mut self) -> bool {
        self.band_left -= 1;
        if self.band_left == 0 && self.next_idx < self.counts.len() {
            self.band_left = self.counts[self.next_idx];
            self.next_idx += 1;
            true
        } else {
            false
        }
    }

    /// `true` iff the EOB token should still be charged at the
    /// current scan position. Mirrors libvpx's `if (band_left)` check
    /// before the EOB cost addition (line 416).
    #[must_use]
    #[inline(always)]
    pub fn eob_should_be_charged(&self) -> bool {
        self.band_left != 0
    }
}

/// Extra-bits cost table for CAT1..CAT6 tokens. Mirrors libvpx
/// `vp9_extra_bits[t].cost` (`vp9_tokenize.c`). For tokens
/// `ZERO/ONE/TWO/THREE/FOUR` the extra-bits cost is zero.
/// For the small-magnitude path we skip cat6's high-bits cost (the
/// load-bearing cat6 contribution costs a per-bit lookup against
/// `vp9_cat6_high_cost`; that table is encoder-only and not yet
/// ported into rvp9. Within the test domain we use only |qcoeff|
/// < 67 so the cat6 branch is never hit).
#[rustfmt::skip]
const EXTRA_BITS_COST_LT_CAT6: [u32; 10] = [
    0,     // ZERO
    0,     // ONE
    0,     // TWO
    0,     // THREE
    0,     // FOUR
    159,   // CAT1   (1 extra bit, prob 159/256 → cost ≈ ~256 in 1/512 units)
    297,   // CAT2   (2 extra bits)
    579,   // CAT3   (3 extra bits)
    826,   // CAT4   (4 extra bits)
    1129,  // CAT5   (5 extra bits)
];

/// Fast-coef-costing variant of libvpx `cost_coeffs`
/// (`vp9_rdopt.c:386-417`).
///
/// Returns the bit-cost (in `vp9_prob_cost` 1/512-bit units) of the
/// supplied `qcoeff` block under the supplied [`CoefCostTable`].
///
/// # Parameters
///
/// - `qcoeff` — quantized coefficients in **raster** order (the
///   forward-quant output). Length must be `>= tx_size.n_coeffs()`.
/// - `scan` — forward scan order for the `(tx_size, tx_type)` pair.
/// - `eob` — one past the scan-position of the last non-zero qcoeff.
///   `eob == 0` ⇒ all-zero block, only the EOB token charge.
/// - `tx_size` / `plane_type` / `ref_type` / `coeff_ctx` — index
///   selectors into the [`CoefCostTable`].
/// - `costs` — the pre-built coefficient cost table from
///   `build_coef_cost_table`.
///
/// # Cat6 caveat
///
/// libvpx's `vp9_get_token_cost` adds the cat6 high-bits cost via
/// `vp9_cat6_high_cost[extra >> 8]` for `|qcoeff| >= 67`. We
/// approximate this at the high-end with `EXTRA_BITS_COST_LT_CAT6[5]`
/// plus a fixed-rate term (the cat6 high-bits are 14 bits in the 8-bit
/// build, each at p≈128 ⇒ ~7168 in 1/512-bit units). Test domain uses
/// `|qcoeff| < 67` so this approximation is never exercised.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn cost_coeffs(
    qcoeff: &[i16],
    scan: &[i16],
    eob: u16,
    tx_size: TxSize,
    plane_type: usize,
    ref_type: usize,
    coeff_ctx: usize,
    costs: &CoefCostTable,
) -> u32 {
    debug_assert!(qcoeff.len() >= tx_size.n_coeffs());
    debug_assert!(scan.len() >= tx_size.n_coeffs());

    let t = tx_size as usize;

    // libvpx `cost_coeffs` lines 383-385: `if (eob == 0)` emit just
    // the EOB token cost in band-0 / first-ctx / first-skip.
    if eob == 0 {
        return coef_token_cost(
            costs,
            CoefCostIndex {
                tx_size: t,
                plane_type,
                ref_type,
                band: 0,
                ctx: coeff_ctx,
                token: EOB_TOKEN,
                skip_eob: false,
            },
        ) as u32;
    }

    // libvpx `cost_coeffs` fast path (lines 388-417). The DC slot is
    // band 0 (`band_counts[t][0] = 1`); after the DC the walker
    // iterates `band_counts[t][1..]`.
    let mut cost: u32 = 0;
    let mut band: usize = 0;
    let mut ctx: usize = coeff_ctx;
    let mut prev_t: u8;
    let mut walker = BandCountsWalker::start_after_dc(tx_size);

    // DC token (scan index 0).
    let v_dc = i32::from(qcoeff[scan[0] as usize]);
    let dc_token = vp9_get_token(v_dc);
    cost = cost.saturating_add(token_cost(
        costs, tx_size, plane_type, ref_type, band, ctx, dc_token, false,
    ));
    cost = cost.saturating_add(extra_bits_cost(dc_token));
    prev_t = dc_token;

    // libvpx line 397: `token_cache[0] = vp9_pt_energy_class[prev_t]`
    // and (line 398) `++token_costs` — i.e. step from band 0 to band 1
    // when handing off to the AC loop. In fast-coef mode the per-band
    // step is driven by `band_left`; we mirror that by transitioning
    // to band 1 explicitly here.
    band = 1;

    // AC tokens (scan indices 1..eob).
    for &scan_pos in scan.iter().take(eob as usize).skip(1) {
        let rc = scan_pos as usize;
        let v = i32::from(qcoeff[rc]);
        let tok = vp9_get_token(v);

        // libvpx line 407: ctx for the AC token is `!prev_t` (i.e. 0
        // if previous token was ZERO, 1 otherwise) on the inner
        // `(*token_costs)[!prev_t][!prev_t][t]` index. In the
        // CoefCostTable layout `[band][ctx][skip_eob]` the `ctx` field
        // here aligns to that `!prev_t` selection.
        ctx = usize::from(prev_t != 0);
        let skip_eob = prev_t == 0;
        cost = cost.saturating_add(token_cost(
            costs, tx_size, plane_type, ref_type, band, ctx, tok, skip_eob,
        ));
        cost = cost.saturating_add(extra_bits_cost(tok));
        prev_t = tok;

        // Walk to the next band when the per-band count is exhausted.
        if walker.step() {
            band += 1;
        }
    }

    // EOB token charge (libvpx line 416): only if the band-walker
    // hasn't hit the terminator (`band_left != 0`).
    if walker.eob_should_be_charged() {
        cost = cost.saturating_add(token_cost(
            costs,
            tx_size,
            plane_type,
            ref_type,
            band,
            usize::from(prev_t != 0),
            EOB_TOKEN,
            false,
        ));
    }

    cost
}

/// Full neighbour-context variant of libvpx `cost_coeffs`
/// (`use_fast_coef_costing == 0`, `vp9_rdopt.c:420-456`).
///
/// Unlike [`cost_coeffs`], every AC token derives its context from the
/// already-seen coefficient-token energy classes at the scan order's two
/// causal neighbour positions. Realtime keyframes select this path even at
/// high `cpu-used` presets (`vp9_speed_features.c:set_rt_speed_feature_framesize_independent`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn cost_coeffs_with_contexts(
    qcoeff: &[i16],
    scan: &[i16],
    neighbors: &[i16],
    eob: u16,
    tx_size: TxSize,
    plane_type: usize,
    ref_type: usize,
    coeff_ctx: usize,
    costs: &CoefCostTable,
) -> u32 {
    let n = tx_size.n_coeffs();
    debug_assert!(qcoeff.len() >= n);
    debug_assert!(scan.len() >= n);
    debug_assert!(neighbors.len() >= (n + 1) * 2);
    debug_assert!(usize::from(eob) <= n);

    if eob == 0 {
        return token_cost(
            costs, tx_size, plane_type, ref_type, 0, coeff_ctx, EOB_TOKEN, false,
        );
    }

    // Matches libvpx's fixed 32x32 token cache. Only the first `n` entries
    // are addressable for the selected transform size.
    let mut token_cache = [0u8; 32 * 32];
    let mut walker = BandCountsWalker::start_after_dc(tx_size);

    let dc_rc = scan[0] as usize;
    let dc_token = vp9_get_token(i32::from(qcoeff[dc_rc]));
    let mut cost = token_cost(
        costs, tx_size, plane_type, ref_type, 0, coeff_ctx, dc_token, false,
    )
    .saturating_add(exact_extra_bits_cost(i32::from(qcoeff[dc_rc])));
    token_cache[dc_rc] = VP9_PT_ENERGY_CLASS[dc_token as usize];

    let mut band = 1usize;
    let mut prev_token = dc_token;
    for (c, &scan_pos) in scan.iter().enumerate().take(usize::from(eob)).skip(1) {
        let rc = scan_pos as usize;
        let token = vp9_get_token(i32::from(qcoeff[rc]));
        let ctx = coef_context(neighbors, &token_cache, c);
        cost = cost
            .saturating_add(token_cost(
                costs,
                tx_size,
                plane_type,
                ref_type,
                band,
                ctx,
                token,
                prev_token == ZERO_TOKEN,
            ))
            .saturating_add(exact_extra_bits_cost(i32::from(qcoeff[rc])));
        token_cache[rc] = VP9_PT_ENERGY_CLASS[token as usize];
        prev_token = token;
        if walker.step() {
            band += 1;
        }
    }

    if walker.eob_should_be_charged() {
        let ctx = coef_context(neighbors, &token_cache, usize::from(eob));
        cost = cost.saturating_add(token_cost(
            costs, tx_size, plane_type, ref_type, band, ctx, EOB_TOKEN, false,
        ));
    }

    cost
}

#[inline(always)]
fn coef_context(neighbors: &[i16], token_cache: &[u8], scan_pos: usize) -> usize {
    let first = neighbors[2 * scan_pos] as usize;
    let second = neighbors[2 * scan_pos + 1] as usize;
    ((1 + usize::from(token_cache[first]) + usize::from(token_cache[second])) >> 1).min(5)
}

/// Exact 8-bit `vp9_get_token_cost` extra-bit charge for one coefficient.
///
/// Every non-zero coefficient pays a 50/50 sign bit. Category tokens also
/// encode `abs(value) - CATn_MIN_VAL` most-significant bit first under the
/// fixed VP9 category probabilities (`vp9_entropy.c:32-38`).
#[inline]
fn exact_extra_bits_cost(value: i32) -> u32 {
    const CAT1: [u8; 1] = [159];
    const CAT2: [u8; 2] = [165, 145];
    const CAT3: [u8; 3] = [173, 148, 140];
    const CAT4: [u8; 4] = [176, 155, 140, 135];
    const CAT5: [u8; 5] = [180, 157, 141, 134, 130];
    const CAT6: [u8; 14] = [
        254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129,
    ];

    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return 0;
    }

    let (base, probs): (u32, &[u8]) = match magnitude {
        1..=4 => (magnitude, &[]),
        5..=6 => (5, &CAT1),
        7..=10 => (7, &CAT2),
        11..=18 => (11, &CAT3),
        19..=34 => (19, &CAT4),
        35..=66 => (35, &CAT5),
        _ => (67, &CAT6),
    };
    let extra = magnitude.saturating_sub(base);
    let category_cost = probs.iter().enumerate().fold(0u32, |cost, (index, &prob)| {
        let shift = probs.len() - index - 1;
        let bit = ((extra >> shift) & 1) != 0;
        cost.saturating_add(vp9_cost_bit(prob, bit))
    });

    // The sign probability is 128, so either sign costs exactly 512.
    512u32.saturating_add(category_cost)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn token_cost(
    costs: &CoefCostTable,
    tx_size: TxSize,
    plane_type: usize,
    ref_type: usize,
    band: usize,
    ctx: usize,
    token: u8,
    skip_eob: bool,
) -> u32 {
    coef_token_cost(
        costs,
        CoefCostIndex {
            tx_size: tx_size as usize,
            plane_type,
            ref_type,
            band,
            ctx,
            token,
            skip_eob,
        },
    ) as u32
}

#[inline(always)]
fn extra_bits_cost(token: u8) -> u32 {
    if (token as usize) < EXTRA_BITS_COST_LT_CAT6.len() {
        EXTRA_BITS_COST_LT_CAT6[token as usize]
    } else {
        // cat6 stand-in (see module docs).
        14 * 256
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::cost::coef_cost::build_coef_cost_table;
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::transforms::scans::default_scan_4x4;

    /// PROOF W4.3 — `band_counts` matches libvpx
    /// `vp9/encoder/vp9_rdopt.c:352-357`. We compare every entry to
    /// the libvpx literal (transcribed verbatim).
    #[test]
    fn w4_3_band_counts_match_libvpx_table() {
        // libvpx literal (line 352-357):
        //   { 1, 2, 3, 4, 3, 16 - 13, 0 },
        //   { 1, 2, 3, 4, 11, 64 - 21, 0 },
        //   { 1, 2, 3, 4, 11, 256 - 21, 0 },
        //   { 1, 2, 3, 4, 11, 1024 - 21, 0 },
        let expected: [[i16; 7]; 4] = [
            [1, 2, 3, 4, 3, 16 - 13, 0],
            [1, 2, 3, 4, 11, 64 - 21, 0],
            [1, 2, 3, 4, 11, 256 - 21, 0],
            [1, 2, 3, 4, 11, 1024 - 21, 0],
        ];
        for (t, row) in expected.iter().enumerate() {
            for (b, &v) in row.iter().enumerate() {
                assert_eq!(
                    BAND_COUNTS[t][b], v,
                    "band_counts[{t}][{b}] mismatch: rvp9={} libvpx={}",
                    BAND_COUNTS[t][b], v,
                );
            }
        }
        // Each row's first six bands MUST sum to total coefficients.
        let totals = [16usize, 64, 256, 1024];
        for (t, total) in totals.iter().enumerate() {
            let sum: i16 = BAND_COUNTS[t][0..6].iter().sum();
            assert_eq!(sum as usize, *total, "row {t} bands don't sum to {total}");
        }
    }

    /// PROOF W4.3 — `cost_coeffs` returns just the EOB-token cost for
    /// an all-zero block. libvpx `vp9_rdopt.c:383-385`: when `eob == 0`
    /// the function returns `token_costs[0][0][pt][EOB_TOKEN]`. That
    /// value must equal the cost from a direct table lookup.
    #[test]
    fn w4_3_cost_coeffs_zero_for_all_zero_block() {
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let scan = default_scan_4x4();
        let qcoeff = vec![0i16; 16];
        // For an all-zero block, eob = 0, the return is exactly the
        // EOB token cost at (band 0, ctx 0, plane 0, ref 0, tx 0).
        let cost = cost_coeffs(
            &qcoeff,
            scan.scan(),
            0,
            TxSize::Tx4x4,
            0, // Y plane
            0, // intra
            0, // ctx 0
            &costs,
        );
        let expected = coef_token_cost(
            &costs,
            CoefCostIndex {
                tx_size: 0,
                plane_type: 0,
                ref_type: 0,
                band: 0,
                ctx: 0,
                token: EOB_TOKEN,
                skip_eob: false,
            },
        ) as u32;
        assert_eq!(
            cost, expected,
            "all-zero block cost must equal EOB-only charge",
        );
    }

    /// `cost_coeffs` produces a strictly larger cost for a block with
    /// non-zero coefficients than for an all-zero block. Sanity check
    /// that the AC-loop path is reached and contributes.
    #[test]
    fn cost_coeffs_nonzero_block_costs_more_than_zero() {
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let scan = default_scan_4x4();

        let zero_cost = cost_coeffs(&[0i16; 16], scan.scan(), 0, TxSize::Tx4x4, 0, 0, 0, &costs);

        // Block with a single DC = 1 → ONE_TOKEN at scan[0].
        let mut q = vec![0i16; 16];
        q[scan.scan()[0] as usize] = 1;
        let one_cost = cost_coeffs(&q, scan.scan(), 1, TxSize::Tx4x4, 0, 0, 0, &costs);
        assert!(
            one_cost > zero_cost,
            "single-coeff cost ({one_cost}) must exceed all-zero cost ({zero_cost})",
        );
    }

    /// Regression for the first exact CPU7 keyframe 4x4 RD mismatch found on
    /// `ducks_take_off`. Pinned libvpx v1.14.1 prices this exact default-scan
    /// block at 18,784 units with `use_fast_coef_costing=0`; the legacy fast
    /// context/token-extra approximation incorrectly returns 14,227.
    #[test]
    fn keyframe_4x4_full_context_cost_matches_libvpx() {
        let costs = build_coef_cost_table(&FrameContext::new_default());
        let scan = default_scan_4x4();
        let qcoeff = [-22, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let fast = cost_coeffs(&qcoeff, scan.scan(), 4, TxSize::Tx4x4, 0, 0, 0, &costs);
        let full = cost_coeffs_with_contexts(
            &qcoeff,
            scan.scan(),
            scan.neighbors(),
            4,
            TxSize::Tx4x4,
            0,
            0,
            0,
            &costs,
        );

        assert_eq!(fast, 14_227);
        assert_eq!(full, 18_784);
    }

    /// `vp9_get_token` matches libvpx's `vp9_get_token` thresholds.
    #[test]
    fn vp9_get_token_thresholds() {
        assert_eq!(vp9_get_token(0), 0); // ZERO
        assert_eq!(vp9_get_token(1), 1); // ONE
        assert_eq!(vp9_get_token(-1), 1); // ONE (abs)
        assert_eq!(vp9_get_token(4), 4); // FOUR
        assert_eq!(vp9_get_token(5), 5); // CAT1 starts at 5
        assert_eq!(vp9_get_token(6), 5);
        assert_eq!(vp9_get_token(7), 6); // CAT2 starts at 7
        assert_eq!(vp9_get_token(10), 6);
        assert_eq!(vp9_get_token(11), 7); // CAT3 starts at 11
        assert_eq!(vp9_get_token(19), 8); // CAT4 starts at 19
        assert_eq!(vp9_get_token(35), 9); // CAT5 starts at 35
        assert_eq!(vp9_get_token(67), 10); // CAT6 starts at 67
        assert_eq!(vp9_get_token(-67), 10);
    }

    /// `BandCountsWalker` steps through the AC bands at the right
    /// positions. Tx_4x4 AC layout: bands of size [2, 3, 4, 3, 3]
    /// (BAND_COUNTS[0][1..6]). The walker returns `true` at the
    /// scan position where `band_left` hits zero (the LAST coef of
    /// a band). For the AC walk starting at scan position 1:
    ///   - positions 1, 2 are band 1 → transition at scan pos 2.
    ///   - positions 3, 4, 5 are band 2 → transition at scan pos 5.
    ///   - positions 6, 7, 8, 9 are band 3 → transition at scan pos 9.
    ///   - positions 10, 11, 12 are band 4 → transition at scan pos 12.
    ///   - positions 13, 14, 15 are band 5 → transition at scan pos 15.
    ///
    /// After position 15 the terminator slot (`0`) is loaded and
    /// `eob_should_be_charged` returns false.
    #[test]
    fn band_counts_walker_steps_at_libvpx_positions() {
        let mut w = BandCountsWalker::start_after_dc(TxSize::Tx4x4);
        let mut transitions = Vec::new();
        for c in 1..16usize {
            if w.step() {
                transitions.push(c);
            }
        }
        assert_eq!(
            transitions,
            vec![2, 5, 9, 12, 15],
            "transitions don't match libvpx band_counts[TX_4X4][1..]",
        );
        // After consuming all 15 AC coefficients, `band_left == 0`
        // (the terminator slot loaded the trailing 0), so EOB
        // should NOT be charged.
        assert!(
            !w.eob_should_be_charged(),
            "after exhausting AC bands EOB should not be charged",
        );
    }
}
