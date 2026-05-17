//! Per-transform-block coefficient detokenizer — the bool-coder ➔
//! dequantised coefficient pipeline.
//!
//! Ports libvpx `decode_coefs` from
//! `vp9/decoder/vp9_detokenize.c:115-261`. One call decodes the
//! coefficients for one transform block: walks scan order, reads
//! `(EOB, ZERO, ONE, value-tree)` token branches plus per-category
//! extension bits plus sign bits from a [`BoolReader`], and writes the
//! dequantised coefficients into the caller-supplied buffer.
//!
//! ## Companion tables
//!
//! - `VP9_COEFBAND_TRANS_*` — scan-position to coef-band lookup
//!   (libvpx `vp9_entropy.c:46-93`).
//! - `VP9_CAT*_PROB` — per-category extension-bit probabilities
//!   (libvpx `vp9_entropy.c:32-38`).
//! - [`super::pareto::VP9_PARETO8_FULL`] — the 255-row model that
//!   derives the 8 non-unconstrained tree-node probs from the pivot
//!   `prob[PIVOT_NODE]`.
//!
//! ## Scope (Stage 3 Task F.B)
//!
//! This is the entropy primitive only — it has no notion of the
//! per-plane / per-tile coefficient context, scan-table selection,
//! or `(eob > 0)` writeback. The caller (Task F.C, the per-plane
//! block driver) is responsible for:
//!
//! - Deriving `initial_pt` from
//!   [`crate::entropy::coef_probs`] + the tile's
//!   `TileCoefCtx` (via `combine_entropy_contexts`).
//! - Selecting the `ScanOrder` via `scan_for(tx_size, tx_type)`.
//! - Slicing the per-frame coef-probs down to the
//!   `(tx_size, plane_type, ref_type)` shape this function expects.
//! - Writing the `(eob > 0)` flag back into `TileCoefCtx`.
//!
//! ## Bit-depth scope
//!
//! 8-bit Profile 0 only (master ledger D-01 RESOLVED). The CAT6
//! category therefore has 14 extension bits, sourced from
//! `VP9_CAT6_PROB[14]`. The 10/12-bit `VP9_CAT6_PROB_HIGH*` arrays
//! in libvpx are not ported.

use super::BoolReader;
use super::coef_probs::{COEF_BANDS, COEFF_CONTEXTS, UNCONSTRAINED_NODES};
use super::pareto::VP9_PARETO8_FULL;
use crate::transforms::scans::ScanOrder;
use crate::transforms::types::{MAX_NEIGHBORS, TxSize};

/// `vp9_coefband_trans_4x4` (`vp9_entropy.c:91-93`). Maps the 16 scan
/// positions in a 4×4 transform block to their coefficient band
/// (`0..COEF_BANDS = 0..6`).
#[rustfmt::skip]
pub static VP9_COEFBAND_TRANS_4X4: [u8; 16] = [
    0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
];

/// `vp9_coefband_trans_8x8plus` (`vp9_entropy.c:46-88`). Maps the
/// first `MAXBAND_INDEX + 1 = 22` scan positions to their coefficient
/// band; positions ≥ 22 are all band `5` (the "high" band). Sized
/// `1024` to cover every position in a 32×32 transform block; only
/// 8×8, 16×16, and 32×32 read this table.
#[rustfmt::skip]
pub static VP9_COEFBAND_TRANS_8X8PLUS: [u8; 1024] = {
    let mut t = [5u8; 1024];
    // Positions 0..22 from the explicit libvpx initialiser:
    // 0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,5
    t[0]  = 0;
    t[1]  = 1; t[2]  = 1;
    t[3]  = 2; t[4]  = 2; t[5]  = 2;
    t[6]  = 3; t[7]  = 3; t[8]  = 3; t[9]  = 3;
    t[10] = 4; t[11] = 4; t[12] = 4; t[13] = 4; t[14] = 4;
    t[15] = 4; t[16] = 4; t[17] = 4; t[18] = 4; t[19] = 4; t[20] = 4;
    t[21] = 5;
    t
};

/// `vp9_cat1_prob` (`vp9_entropy.c:32`). Single extension bit for the
/// CAT1 category (values `[5, 6]`).
pub static VP9_CAT1_PROB: [u8; 1] = [159];

/// `vp9_cat2_prob` (`vp9_entropy.c:33`). Two extension bits for CAT2
/// (values `[7, 10]`).
pub static VP9_CAT2_PROB: [u8; 2] = [165, 145];

/// `vp9_cat3_prob` (`vp9_entropy.c:34`). Three extension bits for
/// CAT3 (values `[11, 18]`).
pub static VP9_CAT3_PROB: [u8; 3] = [173, 148, 140];

/// `vp9_cat4_prob` (`vp9_entropy.c:35`). Four extension bits for CAT4
/// (values `[19, 34]`).
pub static VP9_CAT4_PROB: [u8; 4] = [176, 155, 140, 135];

/// `vp9_cat5_prob` (`vp9_entropy.c:36`). Five extension bits for
/// CAT5 (values `[35, 66]`).
pub static VP9_CAT5_PROB: [u8; 5] = [180, 157, 141, 134, 130];

/// `vp9_cat6_prob` (`vp9_entropy.c:37-38`). Fourteen extension bits
/// for CAT6 (values `[67, 67 + 2^14 - 1]`) on 8-bit Profile 0. The
/// 10/12-bit high-bd variants are not ported.
pub static VP9_CAT6_PROB: [u8; 14] = [
    254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129,
];

// CATn_MIN_VAL constants (libvpx `vp9_entropy.h:46-51`).
const CAT1_MIN_VAL: i32 = 5;
const CAT2_MIN_VAL: i32 = 7;
const CAT3_MIN_VAL: i32 = 11;
const CAT4_MIN_VAL: i32 = 19;
const CAT5_MIN_VAL: i32 = 35;
const CAT6_MIN_VAL: i32 = 67;

// Token tree node indices (`vp9_detokenize.c:23-25`).
const EOB_CONTEXT_NODE: usize = 0;
const ZERO_CONTEXT_NODE: usize = 1;
const ONE_CONTEXT_NODE: usize = 2;

/// libvpx `EOB_MODEL_TOKEN = 3` (`vp9_entropy.h:76`). The extra
/// counter slot beyond the three unconstrained tree nodes.
pub const EOB_MODEL_TOKEN: usize = 3;
/// libvpx `ZERO_TOKEN = 0` (`vp9_entropy.h:27`).
pub const ZERO_TOKEN: usize = 0;
/// libvpx `ONE_TOKEN = 1` (`vp9_entropy.h:28`).
pub const ONE_TOKEN: usize = 1;
/// libvpx `TWO_TOKEN = 2` (`vp9_entropy.h:29`).
pub const TWO_TOKEN: usize = 2;

/// `PIVOT_NODE = 2` (`vp9_entropy.h:144`). Index into the
/// per-context probability triple whose value selects the row of
/// [`VP9_PARETO8_FULL`] used for the high-magnitude tree.
const PIVOT_NODE: usize = 2;

/// A per-`(tx_size, plane_type, ref_type)` slice of the coefficient
/// probability table. Layout matches libvpx
/// `fc->coef_probs[tx][type][ref]` after one level of indexing.
///
/// This is the type Task F.C will produce by slicing
/// `FrameContext::coef_probs[tx][plane][ref]` (Y/UV plane, intra/inter
/// ref).
pub type CoefBandCtxProbs = [[[u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS];

/// Per-block coefficient counts (libvpx `vp9_coeff_count_model`
/// projected to one `(tx_size, plane_type, ref_type)`).
///
/// Indexed `[band][ctx][token]` where `token` is one of
/// `ZERO_TOKEN`, `ONE_TOKEN`, `TWO_TOKEN`, or `EOB_MODEL_TOKEN`.
pub type CoefCountsBlock = [[[u32; UNCONSTRAINED_NODES + 1]; COEFF_CONTEXTS]; COEF_BANDS];

/// Per-block EOB-branch counts (libvpx `counts->eob_branch[tx][type]
/// [ref]`). Indexed `[band][ctx]`.
pub type EobBranchCountsBlock = [[u32; COEFF_CONTEXTS]; COEF_BANDS];

/// Mutable per-block count buckets bundled into one struct so the
/// caller passes a single `Option<&mut DecodeCoefCounts>` instead of
/// juggling two count handles. Mirrors libvpx's
/// `coef_counts = counts->coef[tx][type][ref]` +
/// `eob_branch_count = counts->eob_branch[tx][type][ref]` pair from
/// `vp9_detokenize.c:155-157`.
pub struct DecodeCoefCounts<'a> {
    /// `[band][ctx][token]` tally — bumped per ZERO / ONE / TWO / EOB
    /// model-token decision the tree walk takes.
    pub coef_counts: &'a mut CoefCountsBlock,
    /// `[band][ctx]` tally — bumped once per loop iteration
    /// (libvpx increments this BEFORE the EOB read, so it counts
    /// every position at which an EOB decision was attempted).
    pub eob_branch_counts: &'a mut EobBranchCountsBlock,
}

/// Pick the coefband translate table for `tx_size`.
/// Libvpx `get_band_translate` (`vp9_entropy.h:132-135`).
#[inline]
#[must_use]
fn get_band_translate(tx_size: TxSize) -> &'static [u8] {
    if matches!(tx_size, TxSize::Tx4x4) {
        &VP9_COEFBAND_TRANS_4X4
    } else {
        &VP9_COEFBAND_TRANS_8X8PLUS
    }
}

/// `get_coef_context` (`vp9_scan.h:35-40`). Compute the next
/// scan-position's token-tree context as
/// `(1 + token_cache[nb[2c]] + token_cache[nb[2c+1]]) >> 1`.
#[inline]
#[must_use]
fn get_coef_context(neighbors: &[i16], token_cache: &[u8], c: usize) -> usize {
    let base = MAX_NEIGHBORS * c;
    let nb0 = neighbors[base] as usize;
    let nb1 = neighbors[base + 1] as usize;
    (1 + usize::from(token_cache[nb0]) + usize::from(token_cache[nb1])) >> 1
}

/// Read `n` raw bits MSB-first under the given per-bit probability
/// vector. Mirrors libvpx `read_coeff` (`vp9_detokenize.c:107-113`).
///
/// Each `probs[i]` gates one extension bit; the returned value packs
/// the bits MSB-first so they form a `0..(1 << n)` integer.
#[inline]
#[must_use]
fn read_coeff_bits(reader: &mut BoolReader<'_>, probs: &[u8]) -> i32 {
    let mut val: i32 = 0;
    for &p in probs {
        val = (val << 1) | i32::from(reader.read(p));
    }
    val
}

/// Decode one transform block's coefficients.
///
/// Mirrors libvpx `decode_coefs` (`vp9/decoder/vp9_detokenize.c:115-261`)
/// for the 8-bit (non-`CONFIG_VP9_HIGHBITDEPTH`) path.
///
/// # Parameters
/// - `reader`: the [`BoolReader`] positioned at the first token bit.
///   Advances by however many bits the block decode consumes.
/// - `coef_probs`: the per-(tx, plane, ref) coefficient probability
///   slice — caller-supplied from a `FrameContext::coef_probs[tx][plane][ref]`
///   projection. Shape is `[COEF_BANDS][COEFF_CONTEXTS][UNCONSTRAINED_NODES]`.
/// - `counts`: optional per-block count buckets for backward
///   adaptation. Pass `None` when adaptation is disabled / not yet
///   wired (Task F.C may pass `None` until Task 1f lands).
/// - `scan`: scan-order triple from
///   [`crate::transforms::scans::scan_for`].
/// - `initial_pt`: starting token-tree context (0..=2) from
///   `combine_entropy_contexts(above_coef, left_coef)`.
/// - `tx_size`: transform size — selects between the 4×4 and 8×8plus
///   coefband translate tables; for `Tx32x32` we apply the
///   `dq_shift = 1` libvpx scaling.
/// - `seg_eob`: the maximum number of coefficients allowed in this
///   block. Typically `tx_size.n_coeffs()`; the per-segment
///   `SEG_LVL_SKIP` machinery shortens it elsewhere.
/// - `dequant`: `(dc_dequant, ac_dequant)` — the per-plane dequant
///   pair. The DC dequant applies to scan position 0, AC to all
///   others.
/// - `dq_coefs_out`: caller-zeroed (we zero it ourselves) output
///   buffer of length `seg_eob` (or `tx_size.n_coeffs()` —
///   `seg_eob` ≤ buffer length must hold).
///
/// # Returns
///
/// `eob` — the number of scan positions consumed, equivalently the
/// scan-order position of the first un-decoded coefficient. The
/// caller's "had any non-zero" flag is `eob > 0`.
///
/// # Panics
///
/// - `dq_coefs_out.len() < seg_eob`.
/// - `seg_eob > tx_size.n_coeffs()` (the scan/iscan/neighbour tables
///   are sized for the full block).
/// - `initial_pt > 2` (out of the 0..=2 range produced by
///   `combine_entropy_contexts`).
#[must_use]
#[allow(clippy::too_many_arguments)] // mirrors libvpx decode_coefs's argument list 1:1
pub fn decode_coefs(
    reader: &mut BoolReader<'_>,
    coef_probs: &CoefBandCtxProbs,
    mut counts: Option<&mut DecodeCoefCounts<'_>>,
    scan: &ScanOrder,
    initial_pt: usize,
    tx_size: TxSize,
    seg_eob: usize,
    dequant: (i32, i32),
    dq_coefs_out: &mut [i32],
) -> usize {
    assert!(
        seg_eob <= tx_size.n_coeffs(),
        "decode_coefs: seg_eob ({seg_eob}) > tx_size.n_coeffs ({})",
        tx_size.n_coeffs(),
    );
    assert!(
        dq_coefs_out.len() >= seg_eob,
        "decode_coefs: dq_coefs_out.len() ({}) < seg_eob ({seg_eob})",
        dq_coefs_out.len(),
    );
    assert!(
        initial_pt <= 2,
        "decode_coefs: initial_pt ({initial_pt}) > 2",
    );

    // Zero the output region. Mirrors the caller-side `vpx_memset` of
    // `pd->dqcoeff` libvpx does at `vp9_decodeframe.c:251-253` (and
    // assumed-on-entry by the original `decode_coefs`).
    for slot in &mut dq_coefs_out[..seg_eob] {
        *slot = 0;
    }

    let band_translate = get_band_translate(tx_size);
    let dq_shift: u32 = if matches!(tx_size, TxSize::Tx32x32) { 1 } else { 0 };

    // libvpx uses a flat `uint8_t[32*32]` for token_cache; we size to
    // the actual block. The cache is indexed by raster position
    // (`scan[c]`), so it needs to cover every raster slot in the
    // block, i.e. `tx_size.n_coeffs()`.
    let mut token_cache = vec![0u8; tx_size.n_coeffs()];

    let mut ctx = initial_pt;
    let mut c = 0usize;
    let (dc_dq, ac_dq) = dequant;
    // `dqv` tracks "DC for c=0, then AC for the rest" — libvpx
    // initialises it to `dq[0]` and switches to `dq[1]` after the
    // first scan position (mirrors `vp9_detokenize.c:133` then
    // `:171,254`).
    let mut dqv = dc_dq;

    while c < seg_eob {
        let band = band_translate[c] as usize;
        let prob = &coef_probs[band][ctx];

        if let Some(ref mut cs) = counts {
            cs.eob_branch_counts[band][ctx] += 1;
        }

        // EOB bit.
        if reader.read(prob[EOB_CONTEXT_NODE]) == 0 {
            if let Some(ref mut cs) = counts {
                cs.coef_counts[band][ctx][EOB_MODEL_TOKEN] += 1;
            }
            break;
        }

        // ZERO bit — consume runs of zeros, updating context per-step.
        let mut prob_local = prob;
        let mut band_local = band;
        while reader.read(prob_local[ZERO_CONTEXT_NODE]) == 0 {
            if let Some(ref mut cs) = counts {
                cs.coef_counts[band_local][ctx][ZERO_TOKEN] += 1;
            }
            // dqv switches to AC once we leave scan position 0.
            dqv = ac_dq;
            token_cache[scan.scan[c] as usize] = 0;
            c += 1;
            if c >= seg_eob {
                // Loop exits naturally (no EOB token consumed).
                return c;
            }
            ctx = get_coef_context(scan.neighbors, &token_cache, c);
            band_local = band_translate[c] as usize;
            prob_local = &coef_probs[band_local][ctx];
        }

        // ONE-vs-higher branch.
        let v: i32;
        if reader.read(prob_local[ONE_CONTEXT_NODE]) != 0 {
            // High branch — walk the Pareto8 row.
            let p: &[u8; super::pareto::MODEL_NODES] =
                &VP9_PARETO8_FULL[(prob_local[PIVOT_NODE] - 1) as usize];
            if let Some(ref mut cs) = counts {
                cs.coef_counts[band_local][ctx][TWO_TOKEN] += 1;
            }
            if reader.read(p[0]) != 0 {
                // HIGH_LOW or CAT_THREE/FOUR/FIVE/SIX.
                if reader.read(p[3]) != 0 {
                    token_cache[scan.scan[c] as usize] = 5;
                    let val = if reader.read(p[5]) != 0 {
                        if reader.read(p[7]) != 0 {
                            // CAT6.
                            CAT6_MIN_VAL + read_coeff_bits(reader, &VP9_CAT6_PROB)
                        } else {
                            // CAT5.
                            CAT5_MIN_VAL + read_coeff_bits(reader, &VP9_CAT5_PROB)
                        }
                    } else if reader.read(p[6]) != 0 {
                        // CAT4.
                        CAT4_MIN_VAL + read_coeff_bits(reader, &VP9_CAT4_PROB)
                    } else {
                        // CAT3.
                        CAT3_MIN_VAL + read_coeff_bits(reader, &VP9_CAT3_PROB)
                    };
                    v = (val * dqv) >> dq_shift;
                } else {
                    token_cache[scan.scan[c] as usize] = 4;
                    let val = if reader.read(p[4]) != 0 {
                        // CAT2.
                        CAT2_MIN_VAL + read_coeff_bits(reader, &VP9_CAT2_PROB)
                    } else {
                        // CAT1.
                        CAT1_MIN_VAL + read_coeff_bits(reader, &VP9_CAT1_PROB)
                    };
                    v = (val * dqv) >> dq_shift;
                }
            } else {
                // No high-magnitude bits → token is TWO, THREE, or FOUR.
                if reader.read(p[1]) != 0 {
                    token_cache[scan.scan[c] as usize] = 3;
                    let three_or_four = 3 + i32::from(reader.read(p[2]));
                    v = (three_or_four * dqv) >> dq_shift;
                } else {
                    token_cache[scan.scan[c] as usize] = 2;
                    v = (2 * dqv) >> dq_shift;
                }
            }
        } else {
            // ONE_TOKEN.
            if let Some(ref mut cs) = counts {
                cs.coef_counts[band_local][ctx][ONE_TOKEN] += 1;
            }
            token_cache[scan.scan[c] as usize] = 1;
            v = dqv >> dq_shift;
        }

        // Sign bit (raw, prob 128).
        let signed_v = if reader.read_bit() != 0 { -v } else { v };
        dq_coefs_out[scan.scan[c] as usize] = signed_v;

        c += 1;
        if c < seg_eob {
            ctx = get_coef_context(scan.neighbors, &token_cache, c);
        }
        dqv = ac_dq;
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BoolWriter;
    use crate::entropy::coef_probs::{COEF_BANDS, COEFF_CONTEXTS, UNCONSTRAINED_NODES};
    use crate::transforms::scans::default_scan_4x4;
    use crate::transforms::types::TxSize;

    /// Build a trivial per-block coef-prob table where every entry is
    /// the same `[p_eob, p_zero, p_one]` triple. Lets each test fix
    /// the bit budget by picking probabilities that the encoder can
    /// emit cleanly. The PIVOT_NODE prob doubles as both the
    /// `ONE_CONTEXT_NODE` gate (libvpx's "two-or-more" decision) AND
    /// the row index into [`VP9_PARETO8_FULL`].
    fn flat_coef_probs(p_eob: u8, p_zero: u8, p_one: u8) -> CoefBandCtxProbs {
        let triple = [p_eob, p_zero, p_one];
        let ctx_row = [triple; COEFF_CONTEXTS];
        [ctx_row; COEF_BANDS]
    }

    /// Open a `BoolWriter` over a fresh 256-byte buffer, run the
    /// closure to emit some symbols, then close and return the
    /// produced bytes.
    fn encode<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolWriter<'_>),
    {
        let mut buf = vec![0u8; 256];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            f(&mut w);
            // Pad with enough literal zero-bits to make sure the
            // reader can always fill its register without overshoot.
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().unwrap()
        };
        buf.truncate(n);
        buf
    }

    /// EOB at position 0 ⇒ no coefficients decoded.
    #[test]
    fn empty_block_eob_at_zero() {
        let probs = flat_coef_probs(128, 128, 128);
        // Encoder writes a single 0-bit at the EOB node prob (= 128).
        let bytes = encode(|w| {
            w.write(0, 128); // EOB taken
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx4x4,
            16,
            (10, 20),
            &mut dqc,
        );
        assert_eq!(eob, 0);
        assert!(dqc.iter().all(|&v| v == 0), "all coeffs zero");
    }

    /// Single non-zero DC coefficient with `ONE_TOKEN`, positive sign,
    /// then EOB.
    #[test]
    fn single_one_token_dc_positive() {
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Position 0: NOT-EOB → bit 1 at EOB node.
            w.write(1, 64);
            // Position 0: NOT-ZERO → bit 1 at ZERO node.
            w.write(1, 64);
            // Position 0: NOT-TWO-OR-MORE → bit 0 at ONE node (ONE_TOKEN).
            w.write(0, 64);
            // Sign: positive → bit 0 at prob 128.
            w.write_bit(0);
            // Position 1: EOB → bit 0 at EOB node.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let dc_dq = 11;
        let ac_dq = 22;
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            1,
            TxSize::Tx4x4,
            16,
            (dc_dq, ac_dq),
            &mut dqc,
        );
        assert_eq!(eob, 1);
        // scan[0] == 0 (DC); value = +1 * dc_dq.
        assert_eq!(dqc[scan.scan[0] as usize], dc_dq);
        for (i, &v) in dqc.iter().enumerate() {
            if i == scan.scan[0] as usize {
                continue;
            }
            assert_eq!(v, 0, "non-DC slot {i}");
        }
    }

    /// Single non-zero AC coefficient at scan position 5, magnitude 2
    /// (TWO_TOKEN), negative sign; positions 0..=4 are ZERO_TOKEN.
    /// Verifies AC dequant path + the ZERO run + sign + dq_shift=0.
    #[test]
    fn single_two_token_ac_at_position_5_negative() {
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Positions 0..=4: NOT-EOB then ZERO.
            //
            // libvpx's EOB read happens first per loop iteration, so
            // the order at position 0 is:
            //   NOT-EOB → ZERO (run-of-zeros consumed) → ZERO → ...
            // After we leave the zero-run loop, we re-enter the outer
            // loop only when a non-zero value is decoded.
            //
            // Concretely position 0: NOT-EOB → ZERO at pos 0.
            //                pos 1: ZERO (within the inner zero loop).
            //                ...
            //                pos 4: ZERO.
            // At pos 5 the zero loop breaks because we emit NOT-ZERO.
            w.write(1, 64); // NOT-EOB at pos 0
            // 5 zero tokens (positions 0..=4):
            for _ in 0..5 {
                w.write(0, 64); // ZERO
            }
            // pos 5: NOT-ZERO (breaks the zero run).
            w.write(1, 64);
            // pos 5: TWO-or-more (high branch).
            w.write(1, 64);
            // Walk Pareto8 row for pivot prob = 64: low branch (p[0] = 0).
            // pareto_row = VP9_PARETO8_FULL[63]. We don't pre-compute the
            // row probs; just emit "0 at p[0]" then "0 at p[1]" (token = 2).
            let pareto = &VP9_PARETO8_FULL[63];
            w.write(0, pareto[0]); // low → not HIGH branch
            w.write(0, pareto[1]); // → token TWO (not THREE/FOUR)
            // Sign: negative → bit 1.
            w.write_bit(1);
            // pos 6: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let dc_dq = 7;
        let ac_dq = 19;
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx4x4,
            16,
            (dc_dq, ac_dq),
            &mut dqc,
        );
        assert_eq!(eob, 6, "eob is the scan position past the last non-zero");
        // The non-zero value sits at raster position `scan[5]`.
        assert_eq!(dqc[scan.scan[5] as usize], -2 * ac_dq);
        // Every other slot is zero.
        for (i, &v) in dqc.iter().enumerate() {
            if i == scan.scan[5] as usize {
                continue;
            }
            assert_eq!(v, 0, "slot {i}");
        }
    }

    /// CAT1 token at DC: HIGH branch ➔ low side ➔ CAT1 (1 extension bit).
    /// Verifies the extension-bit + sign path on a small category.
    #[test]
    fn cat1_token_with_extension_bit() {
        // Pick a pivot prob whose Pareto8 row gives identifiable
        // probabilities. We use prob_one = 100 (row 99). The encoder
        // and decoder both consult `VP9_PARETO8_FULL[99]`.
        let probs = flat_coef_probs(64, 64, 100);
        let pareto = &VP9_PARETO8_FULL[99];
        let bytes = encode(|w| {
            // Pos 0: NOT-EOB.
            w.write(1, 64);
            // Pos 0: NOT-ZERO.
            w.write(1, 64);
            // Pos 0: TWO-or-more (high branch).
            w.write(1, 100);
            // Pareto: p[0] = 1 → enter HIGH/CAT path.
            w.write(1, pareto[0]);
            // p[3] = 0 → low-cat side (CAT1 or CAT2).
            w.write(0, pareto[3]);
            // p[4] = 0 → CAT1.
            w.write(0, pareto[4]);
            // CAT1 extension bit: emit '1' under VP9_CAT1_PROB[0].
            w.write(1, VP9_CAT1_PROB[0]);
            // Sign: positive.
            w.write_bit(0);
            // Pos 1: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let dc_dq = 13;
        let ac_dq = 17;
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx4x4,
            16,
            (dc_dq, ac_dq),
            &mut dqc,
        );
        assert_eq!(eob, 1);
        // CAT1 value with extension = 1: val = CAT1_MIN_VAL + 1 = 6.
        // dqv = dc_dq for the first position; dq_shift = 0 for Tx4x4.
        assert_eq!(dqc[scan.scan[0] as usize], 6 * dc_dq);
    }

    /// CAT6 token (max category for 8-bit) — exercises the longest
    /// extension-bit read in the tree (14 bits at AC).
    /// We run on Tx8x8 so the coefband translate path uses the
    /// 8x8plus table, and pick a non-DC position to lean on the AC
    /// dequant path.
    #[test]
    fn cat6_token_full_14_extension_bits() {
        use crate::transforms::scans::default_scan_8x8;
        let probs = flat_coef_probs(64, 64, 200);
        let pareto = &VP9_PARETO8_FULL[199];
        let bytes = encode(|w| {
            // Pos 0: NOT-EOB → ZERO.
            w.write(1, 64);
            w.write(0, 64);
            // Pos 1: NOT-EOB (we re-derive the outer-loop probs after
            // a zero run; with `flat_coef_probs` they're all the same).
            //
            // Wait — the libvpx loop structure: after one ZERO at
            // pos 0, we re-read inside the inner zero loop. So we
            // just emit ZERO at pos 1 too within the same inner loop.
            // Then at pos 2 we break out with NOT-ZERO.
            w.write(0, 64); // pos 1 ZERO
            // pos 2: NOT-ZERO (within inner zero loop).
            w.write(1, 64);
            // pos 2: TWO-or-more (high branch).
            w.write(1, 200);
            // Pareto: p[0] = 1 (HIGH), p[3] = 1 (HIGH_HIGH), p[5] = 1
            // (CAT5 or CAT6), p[7] = 1 (CAT6).
            w.write(1, pareto[0]);
            w.write(1, pareto[3]);
            w.write(1, pareto[5]);
            w.write(1, pareto[7]);
            // 14 extension bits — encode the value 0xABC under
            // VP9_CAT6_PROB. Decoder reads MSB-first; we mirror.
            let ext_val: u32 = 0x0ABC; // < 2^14
            for i in (0..14).rev() {
                let bit = ((ext_val >> i) & 1) as u8;
                w.write(bit, VP9_CAT6_PROB[13 - i]);
            }
            // Sign: positive.
            w.write_bit(0);
            // pos 3: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 64];
        let scan = default_scan_8x8();
        let dc_dq = 9;
        let ac_dq = 14;
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx8x8,
            64,
            (dc_dq, ac_dq),
            &mut dqc,
        );
        assert_eq!(eob, 3);
        // Position 2 is AC (non-zero scan position). Value =
        // CAT6_MIN_VAL + 0x0ABC = 67 + 2748 = 2815. Multiplied by
        // AC dequant; dq_shift = 0 for Tx8x8.
        let expected: i32 = (CAT6_MIN_VAL + 0x0ABC) * ac_dq;
        assert_eq!(dqc[scan.scan[2] as usize], expected);
    }

    /// Block that runs to `seg_eob` without an EOB token: every scan
    /// position decodes as ZERO, the inner-zero loop exhausts the
    /// block, and we return `c == seg_eob`.
    ///
    /// Note: libvpx's actual EOB-vs-implicit-EOB invariant says that
    /// when scan-position 15 (the last 4×4 position) is reached, the
    /// EOB bit is not read (libvpx doesn't read EOB at the last
    /// position because there's nothing past it). Our port mirrors
    /// the exact structure of `decode_coefs`, which DOES read an EOB
    /// at every position including the last. The "no EOB token" exit
    /// happens via the inner zero loop running off the end of the
    /// block — exactly the path tested here.
    #[test]
    fn block_runs_to_seg_eob_via_zero_run() {
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Pos 0: NOT-EOB → ZERO. The inner zero loop then takes
            // 15 more ZEROs (positions 1..15), reaching seg_eob = 16
            // and exiting via the early-return at `c >= seg_eob`.
            w.write(1, 64); // NOT-EOB at pos 0
            for _ in 0..16 {
                w.write(0, 64); // 16 ZEROs total
            }
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx4x4,
            16,
            (10, 20),
            &mut dqc,
        );
        assert_eq!(eob, 16, "ran to seg_eob without an EOB token");
        for (i, &v) in dqc.iter().enumerate() {
            assert_eq!(v, 0, "slot {i} zero");
        }
    }

    /// Mid-loop context update: if `token_cache` weren't being updated
    /// between positions, the per-position `ctx` would be wrong and
    /// the decoder would consume the wrong bits.
    ///
    /// Construct a sequence where position 0 emits ONE_TOKEN (writing
    /// `token_cache[0] = 1`), then position 1's `ctx` is derived
    /// from `token_cache` at the neighbour positions of scan[1]. In
    /// libvpx 4×4 default scan, position 1's neighbours (per
    /// `DEFAULT_SCAN_4X4_NEIGHBORS`) include scan[0] = 0; so the
    /// token_cache write to position 0 directly changes the ctx for
    /// position 1.
    ///
    /// We use a coef-prob table where ctx 1 and ctx 2 have very
    /// different EOB probs. If the decoder reads the wrong ctx, the
    /// EOB bit will mis-decode.
    #[test]
    fn mid_loop_context_update_required() {
        // ctx-dependent EOB prob:
        // ctx=0: p_eob = 1   → EOB bit '1' decodes to bit 1 (NOT-EOB)
        //                       under prob 1 (the encoder must emit '1')
        // ctx=1: p_eob = 250 → likely EOB bit '0'
        //
        // After ONE_TOKEN at position 0, `get_coef_context` returns
        // `(1 + token_cache[nb0] + token_cache[nb1]) >> 1`. For
        // position 1 in DEFAULT_SCAN_4X4_NEIGHBORS the relevant nbs
        // are scan[0] = 0 (and the EOB-sentinel slot which is also
        // index 0). So `(1 + 1 + 1) >> 1 = 1`. ⇒ ctx becomes 1.
        let mut probs = [[[0u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS];
        for band_row in probs.iter_mut() {
            for (ctx, slot) in band_row.iter_mut().enumerate() {
                let p_eob = match ctx {
                    0 => 1,   // initial ctx
                    1 => 250, // post-ONE_TOKEN ctx
                    _ => 128,
                };
                *slot = [p_eob, 64, 64];
            }
        }
        let bytes = encode(|w| {
            // Pos 0 with ctx=0: EOB-prob=1 → NOT-EOB is bit '1'.
            w.write(1, 1);
            // ZERO-prob 64 → NOT-ZERO bit '1'.
            w.write(1, 64);
            // ONE-prob 64 → ONE-token bit '0'.
            w.write(0, 64);
            // Sign positive.
            w.write_bit(0);
            // Pos 1 with ctx=1 (mid-loop update kicks in): EOB-prob=250
            // → EOB bit '0' decodes as bit 0 with prob 250.
            w.write(0, 250);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx4x4,
            16,
            (3, 5),
            &mut dqc,
        );
        assert_eq!(eob, 1, "EOB at pos 1 after ONE_TOKEN at pos 0");
        assert_eq!(dqc[scan.scan[0] as usize], 3, "+1 * dc_dq=3");
    }

    /// Counts wiring: when `Some(counts)` is passed, the relevant
    /// model-token tallies and eob-branch tallies get bumped.
    #[test]
    fn counts_tallied_correctly() {
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Pos 0: NOT-EOB → ONE-token → positive.
            w.write(1, 64);
            w.write(1, 64);
            w.write(0, 64);
            w.write_bit(0);
            // Pos 1: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = default_scan_4x4();
        let mut coef_counts: CoefCountsBlock =
            [[[0u32; UNCONSTRAINED_NODES + 1]; COEFF_CONTEXTS]; COEF_BANDS];
        let mut eob_branch_counts: EobBranchCountsBlock =
            [[0u32; COEFF_CONTEXTS]; COEF_BANDS];
        let mut counts = DecodeCoefCounts {
            coef_counts: &mut coef_counts,
            eob_branch_counts: &mut eob_branch_counts,
        };
        let eob = decode_coefs(
            &mut r,
            &probs,
            Some(&mut counts),
            &scan,
            2,
            TxSize::Tx4x4,
            16,
            (10, 20),
            &mut dqc,
        );
        assert_eq!(eob, 1);
        // Initial ctx = 2 at band 0 (pos 0): one EOB-branch attempt,
        // one ONE_TOKEN tally.
        assert_eq!(eob_branch_counts[0][2], 1);
        assert_eq!(coef_counts[0][2][ONE_TOKEN], 1);
        // Pos 1 is at band 1 (per VP9_COEFBAND_TRANS_4X4[1] = 1).
        // After ONE_TOKEN at pos 0, ctx flips per get_coef_context;
        // we don't pin the exact value here, just verify SOMEthing in
        // band 1 got an EOB-branch attempt + an EOB_MODEL_TOKEN tally.
        let eob_band1_total: u32 = eob_branch_counts[1].iter().sum();
        assert_eq!(eob_band1_total, 1, "pos 1 eob-branch attempt");
        let eob_model_band1_total: u32 = coef_counts[1]
            .iter()
            .map(|ctx_row| ctx_row[EOB_MODEL_TOKEN])
            .sum();
        assert_eq!(eob_model_band1_total, 1, "pos 1 EOB model-token tally");
    }

    /// `get_band_translate` returns the 4×4 table for `Tx4x4` and the
    /// 8×8plus table for every other size.
    #[test]
    fn band_translate_dispatch() {
        assert_eq!(
            get_band_translate(TxSize::Tx4x4).as_ptr(),
            VP9_COEFBAND_TRANS_4X4.as_ptr()
        );
        assert_eq!(
            get_band_translate(TxSize::Tx8x8).as_ptr(),
            VP9_COEFBAND_TRANS_8X8PLUS.as_ptr()
        );
        assert_eq!(
            get_band_translate(TxSize::Tx16x16).as_ptr(),
            VP9_COEFBAND_TRANS_8X8PLUS.as_ptr()
        );
        assert_eq!(
            get_band_translate(TxSize::Tx32x32).as_ptr(),
            VP9_COEFBAND_TRANS_8X8PLUS.as_ptr()
        );
    }

    /// 4×4 coefband table values match the libvpx initializer.
    #[test]
    fn coefband_trans_4x4_values() {
        assert_eq!(
            VP9_COEFBAND_TRANS_4X4,
            [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        );
    }

    /// 8x8plus coefband table starts with the explicit 22-entry prefix
    /// and is `5` everywhere else.
    #[test]
    fn coefband_trans_8x8plus_values() {
        let expected_prefix: [u8; 22] = [
            0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5,
        ];
        assert_eq!(&VP9_COEFBAND_TRANS_8X8PLUS[..22], &expected_prefix);
        for v in &VP9_COEFBAND_TRANS_8X8PLUS[22..] {
            assert_eq!(*v, 5);
        }
    }

    /// Category-probability tables match libvpx `vp9_entropy.c:32-38`.
    #[test]
    fn cat_probs_match_libvpx() {
        assert_eq!(VP9_CAT1_PROB, [159]);
        assert_eq!(VP9_CAT2_PROB, [165, 145]);
        assert_eq!(VP9_CAT3_PROB, [173, 148, 140]);
        assert_eq!(VP9_CAT4_PROB, [176, 155, 140, 135]);
        assert_eq!(VP9_CAT5_PROB, [180, 157, 141, 134, 130]);
        assert_eq!(
            VP9_CAT6_PROB,
            [254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129]
        );
    }

    /// Tx32x32 right-shifts the magnitude by 1 (libvpx `dq_shift`).
    /// Verifies the rare-but-important `dq_shift == 1` path.
    #[test]
    fn tx32x32_applies_dq_shift_of_one() {
        use crate::transforms::scans::default_scan;
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Pos 0: NOT-EOB → NOT-ZERO → ONE_TOKEN → positive.
            w.write(1, 64);
            w.write(1, 64);
            w.write(0, 64);
            w.write_bit(0);
            // Pos 1: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 1024];
        let scan = default_scan(TxSize::Tx32x32);
        // dc_dq = 8 → with dq_shift=1, output = 8 >> 1 = 4.
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            1,
            TxSize::Tx32x32,
            1024,
            (8, 30),
            &mut dqc,
        );
        assert_eq!(eob, 1);
        assert_eq!(dqc[scan.scan[0] as usize], 4, "1 * 8 >> 1");
    }
}
