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
/// ## Runtime differential (Stage 3 stretch M.6)
///
/// When the environment variable `RVP9_DEBUG_REFCOEFS=1` is set, this
/// function snapshots the [`BoolReader`] register state on entry, runs the
/// normal production decode, then restores the snapshot and re-runs the
/// libvpx verbatim reference port ([`decode_coefs_libvpx_ref`]) on
/// identical inputs. If `(eob, dq_coefs)` disagree, the function panics
/// with a full input + output dump. Unset (the default), the env var
/// lookup is one extra branch and nothing else.
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

    // Stage 3 stretch M.6 — runtime differential against the libvpx
    // verbatim reference port. Gated by `RVP9_DEBUG_REFCOEFS=1`. When
    // active: snapshot the reader's register state before the decode loop,
    // run production as normal, then restore the snapshot and re-run the
    // reference port on the same inputs. If `(eob, dq_coefs)` diverge,
    // panic with a full input + output dump — the smoking gun for any
    // production/oracle divergence in real conformance bitstreams.
    //
    // `RVP9_DEBUG_REFCOEFS=trace` is the same differential check PLUS a
    // per-call dump of inputs, pre-reader-state, and outputs — used to
    // hand-trace a specific failing block against libvpx's actual bit
    // consumption when the differential alone agrees but the result is
    // still wrong (i.e., a transcription bug shared by prod + ref port).
    let m6_mode = std::env::var("RVP9_DEBUG_REFCOEFS").ok();
    let m6_active = matches!(m6_mode.as_deref(), Some("1") | Some("trace"));
    let m6_trace = matches!(m6_mode.as_deref(), Some("trace"));
    let m6_pre_snap = if m6_active {
        Some(reader.snapshot())
    } else {
        None
    };

    'outer: while c < seg_eob {
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
                // Loop exits naturally (no EOB token consumed). Use a
                // labelled break (not `return c`) so the M.6 differential
                // gate at the function tail still runs.
                break 'outer;
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

    // Stage 3 stretch M.6 — post-decode differential. The reference port
    // was deliberately frozen back at M.4 (verbatim transcription of
    // libvpx `decode_coefs`, N=100 fuzz-clean). If production and ref
    // disagree on the same bitstream + inputs, we have a smoking-gun
    // localization: the divergence is here (not in any upstream cursor
    // alignment), at this exact transform-block call.
    if let Some(snap) = m6_pre_snap {
        if m6_trace {
            let (r_range, r_value, r_count, r_pos) = snap.cursor_state();
            let probs_b0 = &coef_probs[0][initial_pt];
            eprintln!(
                "RVP9_TRACE decode_coefs(tx={tx_size:?}, initial_pt={initial_pt}, \
                 seg_eob={seg_eob}, dq={dequant:?}, scan[0..6]={:?}) \
                 pre=(r={r_range},v=0x{r_value:016x},c={r_count},p={r_pos}) \
                 probs[b=0][ctx={initial_pt}]={probs_b0:?} \
                 eob={c} dq={:?}",
                &scan.scan[..6.min(scan.scan.len())],
                &dq_coefs_out[..seg_eob],
            );
            // Stage 3 stretch M.8 — bitstream-bytes dump for the C oracle.
            // Triggered ONLY for the failing block (Tx4x4 + initial_pt=1 +
            // dq=(177, 231)) — that's the M.6-captured 32x32 frame-1 sub-1
            // signature. Dumping every call would flood the log; we only
            // need the one block. Mirrors the snapshot.h format the oracle
            // ingests.
            if matches!(tx_size, TxSize::Tx4x4)
                && initial_pt == 1
                && dequant == (177, 231)
            {
                let buf = reader.buffer();
                eprintln!(
                    "RVP9_TRACE_BYTES decode_coefs(tx=Tx4x4, pt=1, dq=(177,231)) \
                     buffer_len={} pre_pos={r_pos} pre_range={r_range} \
                     pre_value=0x{r_value:016x} pre_count={r_count}",
                    buf.len(),
                );
                eprint!("RVP9_TRACE_BYTES_HEX ");
                for &b in buf {
                    eprint!("{b:02x}");
                }
                eprintln!();
                // Also dump probs flattened for the oracle. The harness
                // can re-derive these from libvpx defaults, but dumping
                // them here removes one source of mismatch ambiguity.
                eprint!("RVP9_TRACE_PROBS coef_probs[Tx4x4][PLY][REF]= ");
                for band in coef_probs.iter() {
                    for ctx in band.iter() {
                        for &p in ctx.iter() {
                            eprint!("{p:02x}");
                        }
                    }
                }
                eprintln!();
            }
        }
        let prod_eob = c;
        let prod_dq: Vec<i32> = dq_coefs_out[..seg_eob].to_vec();
        let post_snap = reader.snapshot();
        // Rewind reader and re-run the libvpx reference port on the same
        // bitstream + inputs.
        reader.restore(&snap);
        let mut ref_dq = vec![0i32; seg_eob];
        let ref_eob = decode_coefs_libvpx_ref(
            reader,
            coef_probs,
            tx_size,
            dequant,
            initial_pt,
            scan.scan,
            scan.neighbors,
            &mut ref_dq,
            seg_eob,
        );
        if ref_eob != prod_eob || ref_dq[..] != prod_dq[..] {
            let (r_range, r_value, r_count, r_pos) = snap.cursor_state();
            let probs_b0 = &coef_probs[0][initial_pt];
            eprintln!(
                "RVP9_DEBUG_REFCOEFS: divergence detected\n  \
                 inputs: tx_size={tx_size:?} initial_pt={initial_pt} seg_eob={seg_eob} dequant={dequant:?}\n  \
                 reader-pre: range={r_range} value=0x{r_value:016x} count={r_count} pos={r_pos}\n  \
                 probs[band=0][ctx={initial_pt}]={probs_b0:?}\n  \
                 scan[0..6]={:?}\n  \
                 production: eob={prod_eob} dq_coefs={prod_dq:?}\n  \
                 ref-port:   eob={ref_eob} dq_coefs={ref_dq:?}",
                &scan.scan[..6.min(scan.scan.len())]
            );
            panic!(
                "decode_coefs production diverges from libvpx ref port \
                 (prod_eob={prod_eob}, ref_eob={ref_eob})"
            );
        }
        // Restore the reader to the production-final state so the caller
        // sees no observable change.
        reader.restore(&post_snap);
    }

    c
}

/// Verbatim transcription of libvpx `decode_coefs`
/// (`third_party/libvpx/vp9/decoder/vp9_detokenize.c:115-261`). Lifted out
/// of the test module (where it lived at M.4 dispatch) so the M.6 runtime
/// differential in [`decode_coefs`] can call it on real conformance
/// bitstreams. The implementation is intentionally a line-by-line mirror
/// of libvpx — not idiomatic Rust — because its job is to be *obviously*
/// correct by direct comparison with the source-of-truth C.
///
/// The differential property under M.4: for any `(coef_probs, tx_size, dq,
/// initial_pt, scan, nb, seg_eob, bitstream)` tuple, this function
/// produces the same `(eob, dq_coefs)` as [`decode_coefs`] (N=100 random
/// fuzz tested across all tx sizes and scan orders).
///
/// `dq` is the libvpx `[dc, ac]` pair. `nb` is the scan's neighbour table
/// (length `MAX_NEIGHBORS * tx_size.n_coeffs()`).
#[allow(clippy::too_many_arguments)] // matches libvpx 1:1
#[must_use]
pub fn decode_coefs_libvpx_ref(
    reader: &mut BoolReader<'_>,
    coef_probs: &CoefBandCtxProbs,
    tx_size: TxSize,
    dq: (i32, i32),
    mut ctx: usize,
    scan: &[i16],
    nb: &[i16],
    dq_coefs_out: &mut [i32],
    seg_eob: usize,
) -> usize {
    // libvpx: int max_eob = 16 << (tx_size << 1);  → 16 / 64 / 256 / 1024
    let max_eob = tx_size.n_coeffs();
    let max_eob = max_eob.min(seg_eob); // seg_eob cap for SEG_LVL_SKIP

    // Zero the output (caller-side memset in libvpx).
    for slot in &mut dq_coefs_out[..max_eob] {
        *slot = 0;
    }

    // band_translate: 4x4 or 8x8plus
    let band_translate: &[u8] = if matches!(tx_size, TxSize::Tx4x4) {
        &VP9_COEFBAND_TRANS_4X4
    } else {
        &VP9_COEFBAND_TRANS_8X8PLUS
    };
    let dq_shift: u32 = if matches!(tx_size, TxSize::Tx32x32) { 1 } else { 0 };

    let mut token_cache = vec![0u8; tx_size.n_coeffs()];

    let mut c: usize = 0;
    let mut bt_idx: usize = 0; // mirrors libvpx's post-increment
    let mut dqv: i32 = dq.0;

    while c < max_eob {
        // libvpx: band = *band_translate++;
        let mut band = band_translate[bt_idx] as usize;
        bt_idx += 1;
        // libvpx: prob = coef_probs[band][ctx];
        let mut prob = &coef_probs[band][ctx];

        // libvpx: if (!read_bool(EOB_CONTEXT_NODE)) break;
        if reader.read(prob[EOB_CONTEXT_NODE]) == 0 {
            break;
        }

        // libvpx: while (!read_bool(ZERO_CONTEXT_NODE)) { ... }
        while reader.read(prob[ZERO_CONTEXT_NODE]) == 0 {
            dqv = dq.1;
            token_cache[scan[c] as usize] = 0;
            c += 1;
            if c >= max_eob {
                return c;
            }
            let nb0 = nb[MAX_NEIGHBORS * c] as usize;
            let nb1 = nb[MAX_NEIGHBORS * c + 1] as usize;
            ctx = (1 + usize::from(token_cache[nb0]) + usize::from(token_cache[nb1])) >> 1;
            band = band_translate[bt_idx] as usize;
            bt_idx += 1;
            prob = &coef_probs[band][ctx];
        }

        // libvpx: if (read_bool(ONE_CONTEXT_NODE)) { ... }
        let v: i32;
        if reader.read(prob[ONE_CONTEXT_NODE]) != 0 {
            let p = &VP9_PARETO8_FULL[(prob[PIVOT_NODE] - 1) as usize];
            if reader.read(p[0]) != 0 {
                if reader.read(p[3]) != 0 {
                    token_cache[scan[c] as usize] = 5;
                    let val = if reader.read(p[5]) != 0 {
                        if reader.read(p[7]) != 0 {
                            CAT6_MIN_VAL + read_coeff_bits(reader, &VP9_CAT6_PROB)
                        } else {
                            CAT5_MIN_VAL + read_coeff_bits(reader, &VP9_CAT5_PROB)
                        }
                    } else if reader.read(p[6]) != 0 {
                        CAT4_MIN_VAL + read_coeff_bits(reader, &VP9_CAT4_PROB)
                    } else {
                        CAT3_MIN_VAL + read_coeff_bits(reader, &VP9_CAT3_PROB)
                    };
                    v = (val * dqv) >> dq_shift;
                } else {
                    token_cache[scan[c] as usize] = 4;
                    let val = if reader.read(p[4]) != 0 {
                        CAT2_MIN_VAL + read_coeff_bits(reader, &VP9_CAT2_PROB)
                    } else {
                        CAT1_MIN_VAL + read_coeff_bits(reader, &VP9_CAT1_PROB)
                    };
                    v = (val * dqv) >> dq_shift;
                }
            } else if reader.read(p[1]) != 0 {
                token_cache[scan[c] as usize] = 3;
                let three_or_four = 3 + i32::from(reader.read(p[2]));
                v = (three_or_four * dqv) >> dq_shift;
            } else {
                token_cache[scan[c] as usize] = 2;
                v = (2 * dqv) >> dq_shift;
            }
        } else {
            token_cache[scan[c] as usize] = 1;
            v = dqv >> dq_shift;
        }

        let signed_v = if reader.read_bit() != 0 { -v } else { v };
        dq_coefs_out[scan[c] as usize] = signed_v;

        c += 1;
        if c < max_eob {
            let nb0 = nb[MAX_NEIGHBORS * c] as usize;
            let nb1 = nb[MAX_NEIGHBORS * c + 1] as usize;
            ctx = (1 + usize::from(token_cache[nb0]) + usize::from(token_cache[nb1])) >> 1;
        }
        dqv = dq.1;
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

    /// `decode_coefs` with `col_scan_4x4` (the scan order used for
    /// `TxType::DctAdst`) must route non-zero tokens to the **raster**
    /// positions named by `scan[c]`, not to scan positions `c`. M.2
    /// bisect probe: if this fails, the bug is in `decode_coefs`'s
    /// handling of non-default scans (token_cache writeback / neighbour
    /// indexing / dqv switching). If it passes, the bug is downstream
    /// (in `iht4x4_16_add` or the dequant glue).
    ///
    /// Encoded sequence:
    /// - Scan pos 0: ONE_TOKEN positive → raster pos `scan[0] = 0` (DC).
    /// - Scan pos 1: ONE_TOKEN positive → raster pos `scan[1] = 4`
    ///   (row 1, col 0 — vertical neighbour of DC).
    /// - Scan pos 2: EOB.
    ///
    /// Expected:
    /// - `dq_coefs[0] = +1 * dc_dq` (DC dequant on first position).
    /// - `dq_coefs[4] = +1 * ac_dq` (AC dequant after the first).
    /// - All other slots zero.
    #[test]
    fn col_scan_4x4_dispatches_to_raster_positions() {
        use crate::transforms::scans::col_scan_4x4;
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Scan pos 0: NOT-EOB → NOT-ZERO → ONE-token → positive.
            w.write(1, 64);
            w.write(1, 64);
            w.write(0, 64);
            w.write_bit(0);
            // Scan pos 1: NOT-EOB → NOT-ZERO → ONE-token → positive.
            w.write(1, 64);
            w.write(1, 64);
            w.write(0, 64);
            w.write_bit(0);
            // Scan pos 2: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = col_scan_4x4();
        let dc_dq = 11;
        let ac_dq = 23;
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
        assert_eq!(eob, 2);
        // scan[0] = 0 (raster DC), gets dc_dq.
        assert_eq!(scan.scan[0], 0);
        assert_eq!(dqc[0], dc_dq, "DC at raster 0 should be dc_dq");
        // scan[1] = 4 (raster row 1 col 0), gets ac_dq.
        assert_eq!(scan.scan[1], 4);
        assert_eq!(dqc[4], ac_dq, "second coef at raster 4 should be ac_dq");
        // Everything else zero.
        for (i, &v) in dqc.iter().enumerate() {
            if i == 0 || i == 4 {
                continue;
            }
            assert_eq!(v, 0, "raster slot {i} should be zero");
        }
    }

    /// Companion to `col_scan_4x4_dispatches_to_raster_positions` —
    /// same probe but at `row_scan_4x4` (the scan order used for
    /// `TxType::AdstDct`). Verifies the row-scan path lands tokens at
    /// the correct raster positions too.
    #[test]
    fn row_scan_4x4_dispatches_to_raster_positions() {
        use crate::transforms::scans::row_scan_4x4;
        let probs = flat_coef_probs(64, 64, 64);
        let bytes = encode(|w| {
            // Scan pos 0: ONE_TOKEN positive.
            w.write(1, 64);
            w.write(1, 64);
            w.write(0, 64);
            w.write_bit(0);
            // Scan pos 1: ONE_TOKEN positive.
            w.write(1, 64);
            w.write(1, 64);
            w.write(0, 64);
            w.write_bit(0);
            // Scan pos 2: EOB.
            w.write(0, 64);
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut dqc = vec![0i32; 16];
        let scan = row_scan_4x4();
        let dc_dq = 11;
        let ac_dq = 23;
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
        assert_eq!(eob, 2);
        // row_scan_4x4: scan = [0, 1, 4, ...]. scan[0]=0 (DC), scan[1]=1
        // (row 0 col 1 — horizontal neighbour of DC).
        assert_eq!(scan.scan[0], 0);
        assert_eq!(scan.scan[1], 1);
        assert_eq!(dqc[0], dc_dq, "DC at raster 0 should be dc_dq");
        assert_eq!(dqc[1], ac_dq, "second coef at raster 1 should be ac_dq");
        for (i, &v) in dqc.iter().enumerate() {
            if i == 0 || i == 1 {
                continue;
            }
            assert_eq!(v, 0, "raster slot {i} should be zero");
        }
    }

    // ===================================================================
    // Stage 3 stretch Task M.4 — verbatim libvpx reference port.
    //
    // The reference port was lifted out of the test module at M.6 so the
    // runtime-differential hook in production `decode_coefs` (gated by
    // `RVP9_DEBUG_REFCOEFS=1`) can call it on real conformance bitstreams.
    // See [`decode_coefs_libvpx_ref`] in the parent module for the port.
    // ===================================================================

    /// Tiny deterministic PRNG (xorshift64*) — keeps the differential
    /// test reproducible without a `rand` dep.
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_range(&mut self, range: u32) -> u32 {
            (self.next_u64() as u32) % range
        }
        /// Pick a probability biased to NOT be 128 — flat 128 avoids many
        /// edge cases (Pareto pivot 127 → row 127, etc). We mix in some
        /// extreme values (low and high) along with the bulk range.
        fn next_prob(&mut self) -> u8 {
            // 1..=255 (probability 0 is not used in VP9 — `prob - 1` for
            // the Pareto pivot would underflow).
            let r = self.next_u64() as u32;
            // 10% chance of extreme high (200..=255)
            // 10% chance of extreme low   (1..=50)
            // 80% chance of mid          (51..=199)
            let bucket = r & 0xff;
            let base = (r >> 8) as u8;
            if bucket < 26 {
                1 + (base % 50)
            } else if bucket < 52 {
                200 + (base % 56)
            } else {
                51 + (base % 149)
            }
        }
    }

    /// Encode a random sequence of tokens compatible with the libvpx
    /// detokenize bit-grammar: walks the same scan-position structure
    /// the decoder will walk, emitting either an EOB at position 0, a
    /// run of ZERO tokens that exhausts the block, or some sequence
    /// of (ZERO* | ONE | TWO | THREE | FOUR | CAT1..6) ending in EOB.
    ///
    /// Drives the *encoder's* mirror of the libvpx state machine,
    /// which means we MUST track the same `ctx` / `band` / `dqv` /
    /// `token_cache` evolution the decoder will see, so the emitted
    /// bits round-trip cleanly.
    ///
    /// Returns the buffer plus the expected (eob, dq_coefs).
    #[allow(clippy::too_many_arguments)]
    fn synthesize_random_block(
        rng: &mut XorShift64,
        coef_probs: &CoefBandCtxProbs,
        tx_size: TxSize,
        dq: (i32, i32),
        initial_pt: usize,
        scan: &[i16],
        nb: &[i16],
    ) -> (Vec<u8>, usize, Vec<i32>) {
        let n = tx_size.n_coeffs();
        let band_translate: &[u8] = if matches!(tx_size, TxSize::Tx4x4) {
            &VP9_COEFBAND_TRANS_4X4
        } else {
            &VP9_COEFBAND_TRANS_8X8PLUS
        };
        let dq_shift: u32 = if matches!(tx_size, TxSize::Tx32x32) { 1 } else { 0 };

        // Determine an EOB roll: 30% empty (EOB@0), 60% truncated mid,
        // 10% full-length (run to seg_eob via zero-run).
        let eob_roll = rng.next_range(10);
        let target_eob: usize = if eob_roll < 3 {
            0
        } else if eob_roll < 9 {
            // pick a target in 1..n
            1 + (rng.next_range(n as u32) as usize % n)
        } else {
            n // attempt to run all the way (rarely reachable since the
              // last position consumes the EOB read; we cap by max_eob
              // path which exits via the inner zero-run early-return).
        };

        let mut bytes = vec![0u8; n * 8 + 256]; // generous
        let mut dq_coefs_expected = vec![0i32; n];
        let mut token_cache = vec![0u8; n];
        let mut ctx = initial_pt;
        let mut dqv = dq.0;
        let mut c: usize = 0;
        let mut bt_idx: usize = 0;
        let mut eob: usize = 0;

        let final_len = {
            let mut w = BoolWriter::start_encode(&mut bytes);
            loop {
                if c >= n {
                    // Block ran to seg_eob via zero-run inside the inner
                    // loop. No EOB token emitted. Decoder hits its
                    // early-return at `c >= max_eob`.
                    eob = c;
                    break;
                }
                let band = band_translate[bt_idx] as usize;
                bt_idx += 1;
                let prob = &coef_probs[band][ctx];

                if c >= target_eob {
                    // Emit EOB.
                    w.write(0, prob[EOB_CONTEXT_NODE]);
                    eob = c;
                    break;
                }

                // NOT-EOB.
                w.write(1, prob[EOB_CONTEXT_NODE]);

                // Decide ZERO run length here. We always emit at least
                // one not-zero per outer iteration unless we're stuffing
                // zeros to advance `c` (toward target_eob OR toward n).
                // Strategy: roll a small random count of zero-tokens
                // before the next non-zero.
                let max_zero_run = (target_eob - c).min(8);
                let zero_run = if max_zero_run > 0 {
                    rng.next_range(max_zero_run as u32 + 1) as usize
                } else {
                    0
                };

                let mut local_ctx = ctx;
                let mut local_prob = prob;
                let mut emitted_zeros = 0usize;
                while emitted_zeros < zero_run {
                    w.write(0, local_prob[ZERO_CONTEXT_NODE]);
                    dqv = dq.1;
                    token_cache[scan[c] as usize] = 0;
                    c += 1;
                    emitted_zeros += 1;
                    if c >= n {
                        // Match libvpx's early return: no further EOB
                        // bit emitted.
                        eob = c;
                        ctx = local_ctx; // unused after break
                        break;
                    }
                    let nb0 = nb[MAX_NEIGHBORS * c] as usize;
                    let nb1 = nb[MAX_NEIGHBORS * c + 1] as usize;
                    local_ctx =
                        (1 + usize::from(token_cache[nb0]) + usize::from(token_cache[nb1])) >> 1;
                    let local_band = band_translate[bt_idx] as usize;
                    bt_idx += 1;
                    local_prob = &coef_probs[local_band][local_ctx];
                }
                if c >= n {
                    break;
                }

                // NOT-ZERO.
                w.write(1, local_prob[ZERO_CONTEXT_NODE]);

                // Pick a token category. Distribute over ONE / TWO /
                // THREE / FOUR / CAT1..6.
                let kind = rng.next_range(10);
                let (val, cache_v) = match kind {
                    0..=3 => {
                        // ONE_TOKEN.
                        w.write(0, local_prob[ONE_CONTEXT_NODE]);
                        (1i32, 1u8)
                    }
                    _ => {
                        // TWO-or-more branch.
                        w.write(1, local_prob[ONE_CONTEXT_NODE]);
                        let p = &VP9_PARETO8_FULL[(local_prob[PIVOT_NODE] - 1) as usize];
                        let sub = rng.next_range(10);
                        match sub {
                            0..=2 => {
                                // TWO_TOKEN (val=2)
                                w.write(0, p[0]);
                                w.write(0, p[1]);
                                (2, 2)
                            }
                            3..=4 => {
                                // THREE or FOUR
                                w.write(0, p[0]);
                                w.write(1, p[1]);
                                let four = rng.next_range(2) as u8;
                                w.write(four, p[2]);
                                (3 + i32::from(four), 3)
                            }
                            5 => {
                                // CAT1
                                w.write(1, p[0]);
                                w.write(0, p[3]);
                                w.write(0, p[4]);
                                let extra = rng.next_range(1u32 << 1) as i32;
                                // CAT1 has 1 extension bit.
                                let bit = (extra & 1) as u8;
                                w.write(bit, VP9_CAT1_PROB[0]);
                                (CAT1_MIN_VAL + extra, 4)
                            }
                            6 => {
                                // CAT2
                                w.write(1, p[0]);
                                w.write(0, p[3]);
                                w.write(1, p[4]);
                                let extra = rng.next_range(1u32 << 2) as i32;
                                for i in (0..2).rev() {
                                    let bit = ((extra >> i) & 1) as u8;
                                    w.write(bit, VP9_CAT2_PROB[1 - i]);
                                }
                                (CAT2_MIN_VAL + extra, 4)
                            }
                            7 => {
                                // CAT3
                                w.write(1, p[0]);
                                w.write(1, p[3]);
                                w.write(0, p[5]);
                                w.write(0, p[6]);
                                let extra = rng.next_range(1u32 << 3) as i32;
                                for i in (0..3).rev() {
                                    let bit = ((extra >> i) & 1) as u8;
                                    w.write(bit, VP9_CAT3_PROB[2 - i]);
                                }
                                (CAT3_MIN_VAL + extra, 5)
                            }
                            8 => {
                                // CAT4
                                w.write(1, p[0]);
                                w.write(1, p[3]);
                                w.write(0, p[5]);
                                w.write(1, p[6]);
                                let extra = rng.next_range(1u32 << 4) as i32;
                                for i in (0..4).rev() {
                                    let bit = ((extra >> i) & 1) as u8;
                                    w.write(bit, VP9_CAT4_PROB[3 - i]);
                                }
                                (CAT4_MIN_VAL + extra, 5)
                            }
                            9 => {
                                // CAT5
                                w.write(1, p[0]);
                                w.write(1, p[3]);
                                w.write(1, p[5]);
                                w.write(0, p[7]);
                                let extra = rng.next_range(1u32 << 5) as i32;
                                for i in (0..5).rev() {
                                    let bit = ((extra >> i) & 1) as u8;
                                    w.write(bit, VP9_CAT5_PROB[4 - i]);
                                }
                                (CAT5_MIN_VAL + extra, 5)
                            }
                            _ => {
                                // CAT6 (14 ext bits)
                                w.write(1, p[0]);
                                w.write(1, p[3]);
                                w.write(1, p[5]);
                                w.write(1, p[7]);
                                let extra = rng.next_range(1u32 << 14) as i32;
                                for i in (0..14).rev() {
                                    let bit = ((extra >> i) & 1) as u8;
                                    w.write(bit, VP9_CAT6_PROB[13 - i]);
                                }
                                (CAT6_MIN_VAL + extra, 5)
                            }
                        }
                    }
                };

                // Sign + dequant + dq_shift.
                let sign_bit = (rng.next_range(2) as u8) & 1;
                w.write_bit(sign_bit);
                let signed_v_pre_shift = (val * dqv) >> dq_shift;
                let signed = if sign_bit != 0 {
                    -signed_v_pre_shift
                } else {
                    signed_v_pre_shift
                };
                dq_coefs_expected[scan[c] as usize] = signed;
                token_cache[scan[c] as usize] = cache_v;

                c += 1;
                if c < n {
                    let nb0 = nb[MAX_NEIGHBORS * c] as usize;
                    let nb1 = nb[MAX_NEIGHBORS * c + 1] as usize;
                    ctx = (1 + usize::from(token_cache[nb0]) + usize::from(token_cache[nb1])) >> 1;
                } else {
                    // Decoder will exit via outer-loop guard.
                }
                dqv = dq.1;
            }
            // Pad with zero bits so the reader can always fill its
            // register without overshoot.
            for _ in 0..32 {
                w.write_bit(0);
            }
            w.stop_encode().unwrap()
        };
        bytes.truncate(final_len);
        (bytes, eob, dq_coefs_expected)
    }

    /// Build a random `CoefBandCtxProbs` table: every (band, ctx)
    /// triple is independently rolled with the bias from
    /// `XorShift64::next_prob` (extreme-low / mid / extreme-high mix).
    fn random_coef_probs(rng: &mut XorShift64) -> CoefBandCtxProbs {
        let mut probs: CoefBandCtxProbs =
            [[[0u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS];
        for band in probs.iter_mut() {
            for ctx in band.iter_mut() {
                for slot in ctx.iter_mut() {
                    *slot = rng.next_prob();
                }
            }
        }
        probs
    }

    /// Differential fuzz harness: N random inputs across all TX sizes
    /// and scan orders. For each iteration:
    ///   1. Roll random probs / initial_pt / dequant.
    ///   2. Synthesize a random bitstream (driven by the same state
    ///      machine the decoder will walk).
    ///   3. Decode it with BOTH the libvpx-verbatim reference port
    ///      AND our production `decode_coefs`.
    ///   4. Assert (eob, dq_coefs) match between the two decoders AND
    ///      that both match the synthesizer's "expected" tally.
    ///
    /// If our impl diverges at any iteration we get the (seed,
    /// iteration, TX size, scan kind) tuple printed to stderr for
    /// localized re-investigation.
    #[test]
    fn differential_fuzz_decode_coefs_vs_libvpx_ref() {
        use crate::transforms::scans::{
            col_scan_4x4, col_scan_8x8, col_scan_16x16, default_scan_4x4, default_scan_8x8,
            default_scan_16x16, default_scan_32x32, row_scan_4x4, row_scan_8x8, row_scan_16x16,
        };

        let mut rng = XorShift64::new(0x00C0_FFEE_BABE_FACE_u64);
        #[allow(clippy::type_complexity)] // one-off test config table
        let configs: &[(TxSize, fn() -> ScanOrder, &'static str)] = &[
            (TxSize::Tx4x4, default_scan_4x4, "default_4x4"),
            (TxSize::Tx4x4, col_scan_4x4, "col_4x4"),
            (TxSize::Tx4x4, row_scan_4x4, "row_4x4"),
            (TxSize::Tx8x8, default_scan_8x8, "default_8x8"),
            (TxSize::Tx8x8, col_scan_8x8, "col_8x8"),
            (TxSize::Tx8x8, row_scan_8x8, "row_8x8"),
            (TxSize::Tx16x16, default_scan_16x16, "default_16x16"),
            (TxSize::Tx16x16, col_scan_16x16, "col_16x16"),
            (TxSize::Tx16x16, row_scan_16x16, "row_16x16"),
            (TxSize::Tx32x32, default_scan_32x32, "default_32x32"),
        ];

        const N_PER_CONFIG: usize = 10; // 100 total across 10 configs

        for (tx_size, scan_fn, label) in configs {
            for iter in 0..N_PER_CONFIG {
                let probs = random_coef_probs(&mut rng);
                let initial_pt = (rng.next_range(3) as usize) & 0x3;
                let dq_dc = 1 + (rng.next_range(48) as i32);
                let dq_ac = 1 + (rng.next_range(96) as i32);

                let scan_order = scan_fn();
                let (bytes, expected_eob, expected_dq) = synthesize_random_block(
                    &mut rng,
                    &probs,
                    *tx_size,
                    (dq_dc, dq_ac),
                    initial_pt,
                    scan_order.scan,
                    scan_order.neighbors,
                );
                let n = tx_size.n_coeffs();

                // Reference decode.
                let mut r_ref = BoolReader::new(&bytes)
                    .unwrap_or_else(|_| panic!("{label} iter {iter}: bad bitstream"));
                let mut dq_ref = vec![0i32; n];
                let eob_ref = decode_coefs_libvpx_ref(
                    &mut r_ref,
                    &probs,
                    *tx_size,
                    (dq_dc, dq_ac),
                    initial_pt,
                    scan_order.scan,
                    scan_order.neighbors,
                    &mut dq_ref,
                    n,
                );

                // Production decode.
                let mut r_prod = BoolReader::new(&bytes)
                    .unwrap_or_else(|_| panic!("{label} iter {iter}: bad bitstream"));
                let mut dq_prod = vec![0i32; n];
                let eob_prod = decode_coefs(
                    &mut r_prod,
                    &probs,
                    None,
                    &scan_order,
                    initial_pt,
                    *tx_size,
                    n,
                    (dq_dc, dq_ac),
                    &mut dq_prod,
                );

                // Three-way assertion: ref == prod == synth-expected.
                assert_eq!(
                    eob_ref, eob_prod,
                    "{label} iter {iter}: ref eob {eob_ref} != prod eob {eob_prod}",
                );
                assert_eq!(
                    dq_ref, dq_prod,
                    "{label} iter {iter}: ref dq != prod dq (first diff scan, ref={dq_ref:?} prod={dq_prod:?})",
                );
                assert_eq!(
                    eob_ref, expected_eob,
                    "{label} iter {iter}: ref eob {eob_ref} != synth-expected {expected_eob}",
                );
                assert_eq!(
                    dq_ref, expected_dq,
                    "{label} iter {iter}: ref dq != synth-expected (ref={dq_ref:?} expected={expected_dq:?})",
                );
            }
        }
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
