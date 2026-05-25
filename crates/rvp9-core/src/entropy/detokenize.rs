// Wave A G11 — workspace lint `clippy::print_stderr` is "deny" for
// library crates. This module's `eprintln!` sites are env-var-gated
// trace-debug breadcrumbs (`RVP9_TRACE`, `RVP9_DEBUG_SCAN_WRITE`,
// etc.) scheduled for triage in the HLD §G11 deliverable 3 eprintln
// audit follow-up (delete / `cfg!(rvp9_trace)`-gate / promote to
// `DecodeError`). Until that sweep lands, allow at the module level
// so the lint stays active for *new* code while ignoring the pre-G11
// inventory.
#![allow(clippy::print_stderr)]

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

use std::cell::RefCell;

use super::BoolReader;
use super::BoolWriter;
use super::adapt::CoefCountsBank;
use super::coef_probs::{COEF_BANDS, COEFF_CONTEXTS, UNCONSTRAINED_NODES};
use super::frame_context::FrameContext;
use super::pareto::VP9_PARETO8_FULL;
use crate::coef_ctx::{TileCoefCtx, combine_entropy_contexts};
use crate::transforms::scans::{ScanOrder, scan_for};
use crate::transforms::types::{MAX_NEIGHBORS, TxSize, TxType};

/// Capacity floor for the per-thread `decode_coefs` scratch buffers.
/// Matches the worst-case TX block (`TxSize::Tx32x32` → 32 × 32 = 1024
/// coefficients). Allocated once per thread on first use; reused
/// (with `Vec::clear`) across every subsequent `decode_coefs` call so
/// the hot path costs zero heap allocations.
const DECODE_COEFS_SCRATCH_CAP: usize = 1024;

/// Per-thread reusable scratch for [`decode_coefs`]. R.2 (2026-05-23):
/// hoists the 5 `Vec` allocations that previously happened on EVERY
/// `decode_coefs` call (~30k-60k calls per 1080p frame → 150k-300k
/// `malloc`/`free` pairs/frame on Windows) into a single thread-local
/// `RefCell`. Each call clears the Vecs (preserves capacity, no
/// realloc) and zeros only the prefix of `token_cache` it will read
/// (sized to `tx_size.n_coeffs()`).
///
/// Thread-locality means each tile worker has its own scratch — zero
/// contention, no Mutex needed. The `RefCell` is cheap (single byte
/// borrow flag) and decode_coefs does not recursively call itself or
/// any function that would re-enter the borrow (verified: the AVX2
/// `plane_dequant_finisher` dispatch is a quant kernel, not a
/// decoder).
struct DecodeCoefsScratch {
    /// Indexed-write buffer; reads via `get_coef_context`. Sized to
    /// `DECODE_COEFS_SCRATCH_CAP`; we zero `[..tx_size.n_coeffs()]`
    /// per call.
    token_cache: Vec<u8>,
    /// Pass-1 push targets for the Wave H H5 split. Cleared per call;
    /// capacity preserved.
    scratch_raster: Vec<u16>,
    scratch_dqv: Vec<i16>,
    scratch_val: Vec<i16>,
    scratch_sign: Vec<u8>,
}

impl DecodeCoefsScratch {
    fn new() -> Self {
        Self {
            token_cache: vec![0u8; DECODE_COEFS_SCRATCH_CAP],
            scratch_raster: Vec::with_capacity(DECODE_COEFS_SCRATCH_CAP),
            scratch_dqv: Vec::with_capacity(DECODE_COEFS_SCRATCH_CAP),
            scratch_val: Vec::with_capacity(DECODE_COEFS_SCRATCH_CAP),
            scratch_sign: Vec::with_capacity(DECODE_COEFS_SCRATCH_CAP),
        }
    }
}

thread_local! {
    static DECODE_COEFS_SCRATCH: RefCell<DecodeCoefsScratch> =
        RefCell::new(DecodeCoefsScratch::new());
}

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
///
/// # Safety contract (caller's responsibility)
///
/// - `c < n_coeffs` where `n_coeffs` is the active TX block's
///   `tx_size.n_coeffs()`. `neighbors` is sized
///   `(n_coeffs + 1) * MAX_NEIGHBORS = (n_coeffs + 1) * 2`, so
///   `MAX_NEIGHBORS * c + 1 < neighbors.len()`.
/// - `token_cache.len() >= n_coeffs`. The neighbour values themselves
///   are pre-generated table data with values in `0..n_coeffs`
///   (verified in scan-order tests; see
///   `crates/rvp9-core/src/transforms/scans.rs`).
#[inline]
#[must_use]
fn get_coef_context(neighbors: &[i16], token_cache: &[u8], c: usize) -> usize {
    let base = MAX_NEIGHBORS * c;
    debug_assert!(base + 1 < neighbors.len(), "neighbors OOB at c={c}");
    // SAFETY: Caller guarantees `c < n_coeffs` (see fn-doc safety
    // contract); `neighbors.len() = (n_coeffs + 1) * 2`, so
    // `base = 2*c` and `base + 1` are both `< neighbors.len()`.
    let (nb0, nb1) = unsafe {
        (
            *neighbors.get_unchecked(base) as usize,
            *neighbors.get_unchecked(base + 1) as usize,
        )
    };
    debug_assert!(nb0 < token_cache.len(), "token_cache OOB nb0={nb0}");
    debug_assert!(nb1 < token_cache.len(), "token_cache OOB nb1={nb1}");
    // SAFETY: neighbour values come from `scans.rs` static tables
    // whose entries are `0..n_coeffs`; `token_cache.len() >= n_coeffs`
    // by the caller's contract.
    let (t0, t1) = unsafe {
        (
            *token_cache.get_unchecked(nb0),
            *token_cache.get_unchecked(nb1),
        )
    };
    (1 + usize::from(t0) + usize::from(t1)) >> 1
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

    // R.2: hoist all 5 `decode_coefs` scratch buffers to a per-thread
    // `RefCell` (see `DecodeCoefsScratch` above). The `with_borrow_mut`
    // closure scopes the borrow for the rest of the function body;
    // we destructure into split `&mut Vec<...>` bindings so the existing
    // call-site code (`token_cache`, `scratch_raster`, etc.) compiles
    // unchanged. `Vec::clear()` preserves capacity → zero heap alloc
    // on the warm path.
    DECODE_COEFS_SCRATCH.with_borrow_mut(|scratch| {
    let DecodeCoefsScratch {
        token_cache,
        scratch_raster,
        scratch_dqv,
        scratch_val,
        scratch_sign,
    } = scratch;
    // Zero only the prefix `token_cache[..n_coeffs]` we'll actually
    // read/write. `token_cache.len() == DECODE_COEFS_SCRATCH_CAP` is
    // an invariant established at construction.
    for slot in &mut token_cache[..tx_size.n_coeffs()] {
        *slot = 0;
    }
    scratch_raster.clear();
    scratch_dqv.clear();
    scratch_val.clear();
    scratch_sign.clear();

    let mut ctx = initial_pt;
    let mut c = 0usize;
    let (dc_dq, ac_dq) = dequant;
    // `dqv` tracks "DC for c=0, then AC for the rest" — libvpx
    // initialises it to `dq[0]` and switches to `dq[1]` after the
    // first scan position (mirrors `vp9_detokenize.c:133` then
    // `:171,254`).
    let mut dqv = dc_dq;

    // Wave H H5 — full two-pass dequant split for ALL token arms.
    //
    // Architectural goal (originally filed as F12 blocker, see
    // `notes/wave-g-f12-blocker.md` and `notes/wave-h-g7-prototype.md`):
    // separate per-coefficient dequant arithmetic from the bool-coder
    // loop so that the dequant arithmetic can be vectorised. Wave G G7
    // landed the ONE_TOKEN arm as a prototype; H5 extends the same
    // pattern to every remaining token arm (TWO_TOKEN, THREE_TOKEN,
    // FOUR_TOKEN, CAT1..6).
    //
    // Pass-1 scratch layout: four parallel arrays, one entry per
    // non-zero coefficient. Each entry records:
    //   - `raster_idx` (u16): scan[c], i.e. the position in the
    //     transform block to write the dequantised coefficient into.
    //   - `dqv`        (i16): the dequant value sampled AT the time
    //     this coefficient was decoded. Captures DC vs AC: scan
    //     position 0 sees `dc_dq`, positions 1..eob see `ac_dq`. Do
    //     NOT recompute this in Pass 2 from scan position — the
    //     zero-run logic flips `dqv` to `ac_dq` mid-iteration, so the
    //     "DC vs AC by scan position" rule isn't a clean function of
    //     `raster_idx` alone here.
    //   - `val`        (i16): the pre-dequant integer multiplier. The
    //     per-arm formula was `v = (val * dqv) >> dq_shift`, with
    //     `val ∈ {1, 2, 3, 4, CAT_min + ext_bits}`. The maximum value
    //     is CAT6 = 67 + (2^14 - 1) = 16450, comfortably within i16.
    //   - `sign`       (u8): the raw sign bit, read inside the
    //     bool-coder loop (entropy-coded → cannot defer).
    //
    // Pass-2 uniform formula: `mag = (val * dqv) >> dq_shift; out =
    // if sign { -mag } else { mag }`. This collapses to the
    // pre-refactor arithmetic exactly for every arm:
    //   - ONE_TOKEN     : val=1   → mag = dqv >> dq_shift
    //   - TWO_TOKEN     : val=2   → mag = (2 * dqv) >> dq_shift
    //   - THREE/FOUR    : val=3|4 → mag = (three_or_four * dqv) >> dq_shift
    //   - CAT1..6       : val=CATn_MIN_VAL + ext bits → mag = (val * dqv) >> dq_shift
    //
    // Memory cost: ~7 bytes per entry * seg_eob = ≤ 7 KiB for a 32×32
    // block (worst case all non-zero). R.2 (2026-05-23): hoisted to a
    // per-thread `RefCell<DecodeCoefsScratch>` so the allocations
    // happen ONCE per thread, not per call. The destructured `&mut
    // Vec<...>` bindings above carry the same names as the old locals,
    // so the rest of the body is unchanged.

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
    let m6_mode = crate::trace_flags::trace_flags().debug_refcoefs.as_deref();
    let m6_active = matches!(m6_mode, Some("1") | Some("trace"));
    let m6_trace = matches!(m6_mode, Some("trace"));
    let m6_pre_snap = if m6_active {
        Some(reader.snapshot())
    } else {
        None
    };

    // Hot-loop bound invariants (proof obligations for the
    // `get_unchecked` sites below):
    //
    // 1. `seg_eob <= tx_size.n_coeffs()` — asserted at fn entry.
    // 2. `band_translate.len() >= tx_size.n_coeffs()` — for `Tx4x4`
    //    `band_translate` is `VP9_COEFBAND_TRANS_4X4` (len 16 = 4×4
    //    n_coeffs); for every other size it's `VP9_COEFBAND_TRANS_8X8PLUS`
    //    (len 1024 = 32×32 n_coeffs ≥ every other size). So
    //    `c < seg_eob ≤ n_coeffs ≤ band_translate.len()`.
    // 3. `band_translate` values are in `0..COEF_BANDS=6` (verified by
    //    inspection of the static initialisers above). `ctx <= 2`
    //    (`combine_entropy_contexts` returns `0..=2`; `get_coef_context`
    //    returns `(1 + a + b) >> 1` with `a, b ∈ {0..=5}` from the
    //    token-tree, so `ctx ∈ 0..=5 < COEFF_CONTEXTS=6`). So
    //    `coef_probs[band][ctx]` is in bounds.
    // 4. `scan.scan.len() == tx_size.n_coeffs() >= seg_eob`, and
    //    `scan.scan[c] < n_coeffs` (raster index into the same TX block).
    //    `token_cache.len() == DECODE_COEFS_SCRATCH_CAP = 1024 ≥
    //    n_coeffs`, so `token_cache[scan.scan[c]]` is in bounds.
    debug_assert!(band_translate.len() >= seg_eob);
    debug_assert!(scan.scan.len() >= seg_eob);

    'outer: while c < seg_eob {
        // SAFETY: invariant (2) above — `c < seg_eob ≤ band_translate.len()`.
        let band = unsafe { *band_translate.get_unchecked(c) } as usize;
        debug_assert!(band < COEF_BANDS);
        debug_assert!(ctx < COEFF_CONTEXTS);
        // SAFETY: invariant (3) above — `band < COEF_BANDS` and
        // `ctx < COEFF_CONTEXTS` (both from token-tree state).
        let prob = unsafe { coef_probs.get_unchecked(band).get_unchecked(ctx) };

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
            // SAFETY: invariant (4) — `c < seg_eob ≤ scan.scan.len()`
            // and `scan.scan[c]` is a raster index into the same TX
            // block (< n_coeffs ≤ token_cache.len()).
            let scan_c = unsafe { *scan.scan.get_unchecked(c) } as usize;
            debug_assert!(scan_c < token_cache.len());
            // SAFETY: see above; `scan_c < n_coeffs ≤ token_cache.len()`.
            unsafe { *token_cache.get_unchecked_mut(scan_c) = 0; }
            c += 1;
            if c >= seg_eob {
                // Loop exits naturally (no EOB token consumed). Use a
                // labelled break (not `return c`) so the M.6 differential
                // gate at the function tail still runs.
                break 'outer;
            }
            ctx = get_coef_context(scan.neighbors, &token_cache, c);
            // SAFETY: invariant (2) — `c < seg_eob ≤ band_translate.len()`.
            band_local = unsafe { *band_translate.get_unchecked(c) } as usize;
            debug_assert!(band_local < COEF_BANDS);
            debug_assert!(ctx < COEFF_CONTEXTS);
            // SAFETY: invariant (3) — `band_local < COEF_BANDS`
            // and `ctx < COEFF_CONTEXTS`.
            prob_local = unsafe { coef_probs.get_unchecked(band_local).get_unchecked(ctx) };
        }

        // ONE-vs-higher branch.
        //
        // Wave H H5: every arm now defers its dequant arithmetic into
        // pass 2. The per-arm work inside this loop is reduced to:
        //   (a) reading the entropy-coded bits that select the token,
        //   (b) updating `token_cache` (used for context selection of
        //       the NEXT coefficient — must stay in-loop, in original
        //       sequence),
        //   (c) updating `counts` for backward adaptation,
        //   (d) computing the integer multiplier `val`,
        //   (e) reading the sign bit (entropy-coded → must stay in-loop),
        //   (f) depositing `(raster_idx, dqv, val, sign)` into scratch.
        //
        // Pass 2 (after the loop) applies the uniform formula
        // `mag = (val * dqv) >> dq_shift; out = sign ? -mag : mag`
        // for every captured entry.
        // SAFETY: invariant (4) — `c < seg_eob ≤ scan.scan.len()`.
        let raster_idx = unsafe { *scan.scan.get_unchecked(c) } as usize;
        debug_assert!(raster_idx < token_cache.len());
        // SAFETY: invariant (4) — `raster_idx = scan.scan[c]` is a
        // raster position in this TX block (< n_coeffs ≤ token_cache.len()).
        // Capture an aliased mut pointer to the cell we'll update once
        // per non-zero-token branch, so we don't re-bounds-check on
        // every arm.
        let tc_slot: &mut u8 = unsafe { token_cache.get_unchecked_mut(raster_idx) };
        let val: i32;
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
                    *tc_slot = 5;
                    val = if reader.read(p[5]) != 0 {
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
                } else {
                    *tc_slot = 4;
                    val = if reader.read(p[4]) != 0 {
                        // CAT2.
                        CAT2_MIN_VAL + read_coeff_bits(reader, &VP9_CAT2_PROB)
                    } else {
                        // CAT1.
                        CAT1_MIN_VAL + read_coeff_bits(reader, &VP9_CAT1_PROB)
                    };
                }
            } else {
                // No high-magnitude bits → token is TWO, THREE, or FOUR.
                if reader.read(p[1]) != 0 {
                    *tc_slot = 3;
                    // THREE_TOKEN or FOUR_TOKEN.
                    val = 3 + i32::from(reader.read(p[2]));
                } else {
                    *tc_slot = 2;
                    // TWO_TOKEN.
                    val = 2;
                }
            }
        } else {
            // ONE_TOKEN.
            if let Some(ref mut cs) = counts {
                cs.coef_counts[band_local][ctx][ONE_TOKEN] += 1;
            }
            *tc_slot = 1;
            val = 1;
        }

        // Sign bit (raw, prob 128). MUST be read inside the
        // bool-coder loop — it is entropy-coded so deferring this
        // read would corrupt all subsequent reads.
        let sign_bit = reader.read_bit();

        // Pass-1 deposit: record (raster_idx, dqv, val, sign) for the
        // pass-2 arithmetic. `dq_coefs_out[raster_idx]` was zeroed at
        // the function top so leaving it untouched here is correct —
        // pass 2 will overwrite. CAT6's maximum `val` is
        // `67 + 2^14 - 1 = 16450`, which fits in i16; smaller arms
        // are obviously within range. `dqv` comes from
        // `plane_dequant_pair` whose return type is `(i16, i16)` so
        // the i16 narrowing is loss-free.
        debug_assert!(dqv.unsigned_abs() <= i16::MAX as u32);
        debug_assert!(raster_idx <= u16::MAX as usize);
        debug_assert!((0..=i32::from(i16::MAX)).contains(&val));
        scratch_raster.push(raster_idx as u16);
        scratch_dqv.push(dqv as i16);
        scratch_val.push(val as i16);
        scratch_sign.push(sign_bit);

        // Stage 3 stretch M.17 — print scan[c] and write index for Tx8x8.
        // The post-pass-2 dequantised value is reconstructed here so
        // the trace stays meaningful (the actual writeback happens in
        // pass 2 below).
        if matches!(tx_size, TxSize::Tx8x8)
            && crate::trace_flags::trace_flags().debug_scan_write
        {
            let mag = (val * dqv) >> dq_shift;
            let trace_v = if sign_bit != 0 { -mag } else { mag };
            eprintln!(
                "RVP9_DBG_SCAN_WRITE c={c} scan[c]={raster_idx} v={trace_v} dq_coefs_out.len={}",
                dq_coefs_out.len()
            );
        }

        c += 1;
        if c < seg_eob {
            ctx = get_coef_context(scan.neighbors, &token_cache, c);
        }
        dqv = ac_dq;
    }

    // Wave H H5 — pass 2: uniform vectorisable dequant arithmetic +
    // writeback for every coefficient deposited by pass 1.
    //
    // Each iteration is independent (no loop-carried state, no
    // bool-coder reads), so this loop is trivially auto-vectorisable
    // by LLVM and is the consumption point for the H6 AVX2
    // `plane_dequant_pair`-family kernel: gather (val, dqv) pairs,
    // perform the i32 multiply + arithmetic right shift, conditionally
    // negate by `sign`, and scatter to `dq_coefs_out[raster_idx]`.
    //
    // Bit-exactness vs the fused path: the pre-refactor arithmetic
    // for every arm reduces to `signed_v = if sign { -((val * dqv) >>
    // dq_shift) } else { (val * dqv) >> dq_shift }` with
    // arm-specific `val` constants:
    //   - ONE_TOKEN     val=1   (collapse: `v = dqv >> dq_shift`)
    //   - TWO_TOKEN     val=2
    //   - THREE/FOUR    val=3|4
    //   - CAT1..6       val=CATn_MIN_VAL + extension bits
    // The unified Pass 2 computes that formula uniformly. Because
    // `val > 0` and `dqv >= 0` for all valid VP9 streams, the product
    // is non-negative and the arithmetic right shift behaves
    // identically to the per-arm inline arithmetic the pre-refactor
    // code performed.
    debug_assert_eq!(scratch_raster.len(), scratch_dqv.len());
    debug_assert_eq!(scratch_raster.len(), scratch_val.len());
    debug_assert_eq!(scratch_raster.len(), scratch_sign.len());
    // Wave H H6 — pass-2 dequant arithmetic via the AVX2-capable
    // `plane_dequant_finisher` slot. On AVX2 hosts this processes 8
    // i32 lanes per iteration (`_mm256_mullo_epi32` +
    // `_mm256_sra_epi32` + `_mm256_sign_epi32`) with a scalar tail;
    // on non-AVX2 hosts it falls back to the bit-exact scalar
    // reference (`crate::quant::plane_dequant_finisher`). `dsp::get()`
    // is one atomic load on the process-wide OnceLock; the dispatch
    // table is constructed once per process and cached.
    // R.14: try `get_ref()` (borrowed ref) instead of `get()` (Arc clone).
    // R.13 tried this on inter_predictor and regressed; trying here in
    // decode_coefs (a different call site) — the dispatch is used
    // immediately for one indirect call and dropped, so eliminating
    // the atomic increment may help on coefficient-dense fixtures.
    let dispatch = crate::dsp::get_ref();
    (dispatch.plane_dequant_finisher())(
        &scratch_raster,
        &scratch_dqv,
        &scratch_val,
        &scratch_sign,
        dq_shift,
        dq_coefs_out,
    );

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
    }) // close DECODE_COEFS_SCRATCH.with_borrow_mut(|scratch| { ... })
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


/// Write `n` raw bits MSB-first under the given per-bit probability
/// vector. Mirror of [`read_coeff_bits`] (libvpx `vp9_detokenize.c:107-113`)
/// for the encoder side: emits `probs.len()` bits, with bit `i` of `val`
/// (MSB first) gated by `probs[i]`.
#[inline]
fn write_coeff_bits(bw: &mut BoolWriter<'_>, val: i32, probs: &[u8]) {
    let n = probs.len();
    for (i, &p) in probs.iter().enumerate() {
        // MSB-first: bit position is `n - 1 - i` from the LSB.
        let shift = (n - 1 - i) as u32;
        let bit = ((val >> shift) & 1) as u8;
        bw.write(bit, p);
    }
}

/// Encode one transform block's coefficients — the encoder-side mirror
/// of [`decode_coefs`].
///
/// Mirrors libvpx `vp9_encode_coefs` / `tokenize_b`
/// (`vp9/encoder/vp9_tokenize.c`) at the bit-grammar level by inverting
/// the `decode_coefs` state machine in
/// `vp9/decoder/vp9_detokenize.c:115-261`. Walks the scan in order,
/// emitting `(EOB, ZERO, ONE, value-tree)` + extension bits + sign for
/// each raw quantized coefficient indexed by `scan.scan[c]`.
///
/// ## Round-trip contract
///
/// For any `(coef_probs, scan, initial_pt, tx_size, seg_eob,
/// raw_coefs, eob)`, the bits this function emits, when fed back into
/// [`decode_coefs`] with the identity dequant pair (the multiplier
/// after `dq_shift` evaluates to 1 — `(1, 1)` for Tx{4,8,16}, `(2, 2)`
/// for Tx32x32), will recover `eob` and `raw_coefs[..seg_eob]`
/// verbatim.
///
/// ## EOB termination
///
/// - If `eob < seg_eob`, this function emits one terminal `0` bit at
///   the `EOB_CONTEXT_NODE` after the `eob`-th coefficient.
/// - If `eob == seg_eob`, NO terminal EOB bit is emitted — the decoder
///   exits via its `c >= seg_eob` outer-loop guard.
///
/// # Parameters
/// - `bw`: the [`BoolWriter`] positioned where this block's tokens
///   should land. Advances by however many bits the block encode emits.
/// - `coef_probs`: per-`(tx, plane, ref)` coefficient probability slice
///   — the SAME table the matching `decode_coefs` call will consult.
/// - `scan`: scan-order triple (must match what the decoder will use).
/// - `initial_pt`: starting token-tree context (0..=2) from
///   `combine_entropy_contexts(above_coef, left_coef)`.
/// - `tx_size`: transform size — selects 4×4 vs 8×8plus coefband
///   translate table.
/// - `seg_eob`: per-segment EOB cap; typically `tx_size.n_coeffs()`.
/// - `raw_coefs`: raster-order quantized coefficient indices. Length
///   must be `>= tx_size.n_coeffs()`.
/// - `eob`: number of scan positions to encode (0..=seg_eob). Slots
///   beyond `eob` in the scan must read as 0 from `raw_coefs` for the
///   round-trip to make sense, but this function does NOT inspect them.
///
/// # Panics
///
/// - `seg_eob > tx_size.n_coeffs()`.
/// - `eob > seg_eob`.
/// - `raw_coefs.len() < tx_size.n_coeffs()`.
/// - `initial_pt > 2`.
#[allow(clippy::too_many_arguments)] // mirrors decode_coefs 1:1
pub fn encode_coefs(
    bw: &mut BoolWriter<'_>,
    coef_probs: &CoefBandCtxProbs,
    scan: &ScanOrder,
    initial_pt: usize,
    tx_size: TxSize,
    seg_eob: usize,
    raw_coefs: &[i32],
    eob: usize,
    mut counts: Option<&mut DecodeCoefCounts<'_>>,
) {
    assert!(
        seg_eob <= tx_size.n_coeffs(),
        "encode_coefs: seg_eob ({seg_eob}) > tx_size.n_coeffs ({})",
        tx_size.n_coeffs(),
    );
    assert!(
        eob <= seg_eob,
        "encode_coefs: eob ({eob}) > seg_eob ({seg_eob})",
    );
    assert!(
        raw_coefs.len() >= tx_size.n_coeffs(),
        "encode_coefs: raw_coefs.len() ({}) < tx_size.n_coeffs ({})",
        raw_coefs.len(),
        tx_size.n_coeffs(),
    );
    assert!(
        initial_pt <= 2,
        "encode_coefs: initial_pt ({initial_pt}) > 2",
    );

    let band_translate = get_band_translate(tx_size);
    let mut token_cache = vec![0u8; tx_size.n_coeffs()];
    let mut ctx = initial_pt;

    // Outer loop — one iteration per (NOT-EOB, zero-run, non-zero token,
    // sign) tuple OR the final EOB token. Mirrors the structure of
    // `decode_coefs`'s outer loop at `vp9_detokenize.c:130-244`.
    let mut c = 0usize;
    'outer: while c < eob {
        let band = band_translate[c] as usize;
        let prob = &coef_probs[band][ctx];

        // (A) G3 — eob_branch: this position attempts an EOB decision.
        if let Some(ref mut ct) = counts {
            ct.eob_branch_counts[band][ctx] += 1;
        }

        // Outer-loop EOB decision — we are NOT at EOB, emit '1'.
        bw.write(1, prob[EOB_CONTEXT_NODE]);

        // Inner zero-run loop. Each iteration here emits ONE ZERO bit
        // ('0' at the ZERO node) and DOES NOT re-read EOB. Mirrors the
        // decoder's `while reader.read(prob[ZERO_CONTEXT_NODE]) == 0`
        // inner loop.
        let mut local_prob = prob;
        loop {
            let raster_idx = scan.scan[c] as usize;
            let coef = raw_coefs[raster_idx];
            if coef != 0 {
                // Break out — emit NOT-ZERO and proceed to the value
                // tree below.
                bw.write(1, local_prob[ZERO_CONTEXT_NODE]);
                break;
            }
            // ZERO_TOKEN at this position.
            bw.write(0, local_prob[ZERO_CONTEXT_NODE]);
            // (B) G3 — count ZERO_TOKEN.
            if let Some(ref mut ct) = counts {
                let b = band_translate[c] as usize;
                ct.coef_counts[b][ctx][ZERO_TOKEN] += 1;
            }
            token_cache[raster_idx] = 0;
            c += 1;
            if c >= eob {
                // Decoder will exit via its `c >= seg_eob` guard if eob
                // == seg_eob, OR will consume one more EOB bit at the
                // next outer iteration. Either way, we're done with the
                // value tree path here — fall through to the terminal
                // EOB emission below.
                //
                // Update `ctx` so the terminal EOB emission (below)
                // sees the post-zero-run context. Match the decoder's
                // behaviour: when the inner zero loop ends inside the
                // outer loop, the next outer iteration recomputes
                // (band, ctx) from the updated token_cache.
                if c < seg_eob {
                    ctx = get_coef_context(scan.neighbors, &token_cache, c);
                }
                break 'outer;
            }
            ctx = get_coef_context(scan.neighbors, &token_cache, c);
            let band_local = band_translate[c] as usize;
            local_prob = &coef_probs[band_local][ctx];
        }

        // Non-zero coefficient — emit the value tree.
        let raster_idx = scan.scan[c] as usize;
        let coef = raw_coefs[raster_idx];
        let abs_val = coef.unsigned_abs() as i32;
        let sign_neg = coef < 0;

        if abs_val == 1 {
            // ONE_TOKEN — emit '0' at the ONE node.
            bw.write(0, local_prob[ONE_CONTEXT_NODE]);
            // (C) G3 — count ONE_TOKEN.
            if let Some(ref mut ct) = counts {
                let b = band_translate[c] as usize;
                ct.coef_counts[b][ctx][ONE_TOKEN] += 1;
            }
            token_cache[raster_idx] = 1;
        } else {
            // TWO-or-higher branch.
            bw.write(1, local_prob[ONE_CONTEXT_NODE]);
            // (D) G3 — count TWO_TOKEN (absorbs TWO through CAT6).
            if let Some(ref mut ct) = counts {
                let b = band_translate[c] as usize;
                ct.coef_counts[b][ctx][TWO_TOKEN] += 1;
            }
            let p = &VP9_PARETO8_FULL[(local_prob[PIVOT_NODE] - 1) as usize];
            if abs_val <= 4 {
                // Low branch — TWO / THREE / FOUR.
                bw.write(0, p[0]);
                match abs_val {
                    2 => {
                        bw.write(0, p[1]);
                        token_cache[raster_idx] = 2;
                    }
                    3 => {
                        bw.write(1, p[1]);
                        bw.write(0, p[2]);
                        token_cache[raster_idx] = 3;
                    }
                    4 => {
                        bw.write(1, p[1]);
                        bw.write(1, p[2]);
                        token_cache[raster_idx] = 3;
                    }
                    _ => unreachable!(),
                }
            } else {
                // High branch — CAT1..CAT6.
                bw.write(1, p[0]);
                if abs_val <= 10 {
                    // p[3] = 0 → low-cat side.
                    bw.write(0, p[3]);
                    token_cache[raster_idx] = 4;
                    if abs_val <= 6 {
                        // CAT1 (1 ext bit; values 5..=6).
                        bw.write(0, p[4]);
                        let extra = abs_val - CAT1_MIN_VAL;
                        write_coeff_bits(bw, extra, &VP9_CAT1_PROB);
                    } else {
                        // CAT2 (2 ext bits; values 7..=10).
                        bw.write(1, p[4]);
                        let extra = abs_val - CAT2_MIN_VAL;
                        write_coeff_bits(bw, extra, &VP9_CAT2_PROB);
                    }
                } else {
                    // p[3] = 1 → high-cat side.
                    bw.write(1, p[3]);
                    token_cache[raster_idx] = 5;
                    if abs_val <= 34 {
                        // p[5] = 0 → CAT3 / CAT4.
                        bw.write(0, p[5]);
                        if abs_val <= 18 {
                            // CAT3 (3 ext bits; values 11..=18).
                            bw.write(0, p[6]);
                            let extra = abs_val - CAT3_MIN_VAL;
                            write_coeff_bits(bw, extra, &VP9_CAT3_PROB);
                        } else {
                            // CAT4 (4 ext bits; values 19..=34).
                            bw.write(1, p[6]);
                            let extra = abs_val - CAT4_MIN_VAL;
                            write_coeff_bits(bw, extra, &VP9_CAT4_PROB);
                        }
                    } else {
                        // p[5] = 1 → CAT5 / CAT6.
                        bw.write(1, p[5]);
                        if abs_val <= 66 {
                            // CAT5 (5 ext bits; values 35..=66).
                            bw.write(0, p[7]);
                            let extra = abs_val - CAT5_MIN_VAL;
                            write_coeff_bits(bw, extra, &VP9_CAT5_PROB);
                        } else {
                            // CAT6 (14 ext bits on 8-bit Profile 0;
                            // values 67..=67 + 2^14 - 1 = 16450).
                            bw.write(1, p[7]);
                            let extra = abs_val - CAT6_MIN_VAL;
                            write_coeff_bits(bw, extra, &VP9_CAT6_PROB);
                        }
                    }
                }
            }
        }

        // Sign bit — raw, prob 128 (`write_bit`).
        bw.write_bit(u8::from(sign_neg));

        c += 1;
        // Update ctx for the next iteration. We update whenever there's
        // another scan position to consider — either another value-tree
        // emit (c < eob) OR the terminal EOB emission below
        // (eob < seg_eob, in which case we need ctx for position `eob`).
        if c < seg_eob {
            ctx = get_coef_context(scan.neighbors, &token_cache, c);
        }
    }

    // Terminal EOB bit — only if we stopped short of seg_eob. If we
    // ran to seg_eob (either via the value-tree path or the inner
    // zero-run), the decoder exits via its `c >= seg_eob` guard
    // without consuming any further bit. Mirrors the libvpx behaviour
    // at `vp9_detokenize.c:158-160` + the inner zero-run early-return.
    if eob < seg_eob {
        // Recompute the EOB context for the terminal position. When
        // `eob` is 0 nothing has been emitted yet, so the context is
        // still `initial_pt`. Otherwise the loop above already updated
        // `ctx` to the next position's context.
        let band = band_translate[eob] as usize;
        // (E) G3 — eob_branch + EOB_MODEL_TOKEN at the terminal position.
        if let Some(ref mut ct) = counts {
            ct.eob_branch_counts[band][ctx] += 1;
            ct.coef_counts[band][ctx][EOB_MODEL_TOKEN] += 1;
        }
        let prob = &coef_probs[band][ctx];
        bw.write(0, prob[EOB_CONTEXT_NODE]);
    }
}

/// Encode one transform block's coefficients with full per-TX-block
/// neighbour-context setup — the encoder-side mirror of
/// `vp9_decode_block_tokens` (`vp9/decoder/vp9_detokenize.c:275-333`).
///
/// This is the wrapper layer on top of [`encode_coefs`]: it slices
/// `fc.coef_probs[tx_size][plane_type][ref_type]` to the per-`(tx,
/// plane, ref)` table, looks up the scan order, OR-folds the spanned
/// above-row + left-column [`TileCoefCtx`] cells to derive
/// `initial_pt`, calls [`encode_coefs`], and writes the resulting
/// `(eob > 0)` flag back into the covered sub-block region. After
/// this call, the decoder-side `vp9_decode_block_tokens` will recover
/// the same coefficients verbatim (modulo the identity-dequant trick
/// the round-trip tests use to invert the dequant multiply).
///
/// # Parameters
///
/// - `bw`: bool-coder positioned where this block's tokens should
///   land; advances by however many bits the block encode emits.
/// - `fc`: per-frame probability bank — we slice
///   `fc.coef_probs[tx_size][plane_type][ref_type]` for the per-`(tx,
///   plane_type, ref)` table.
/// - `coef_ctx`: per-tile coefficient-context state — read for the
///   initial `pt`, written for the post-encode `(eob > 0)` flag.
/// - `plane`: `0` / `1` / `2` for Y / U / V. `plane > 0` ⇒ chroma
///   plane-type (libvpx `get_plane_type` at `vp9_blockd.h:210-212`).
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
/// - `seg_eob`: per-segment EOB cap; typically `tx_size.n_coeffs()`.
/// - `raw_coefs`: raster-order quantized coefficient indices. Length
///   must be `>= tx_size.n_coeffs()`.
/// - `eob`: number of scan positions to encode (`0..=seg_eob`).
///
/// # Panics
///
/// Inherits [`encode_coefs`]'s assertions: `seg_eob > tx_size.n_coeffs()`,
/// `eob > seg_eob`, or `raw_coefs.len() < tx_size.n_coeffs()` trip.
#[allow(clippy::too_many_arguments)] // mirrors decode_block_tokens 1:1
pub fn write_block_tokens(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    coef_ctx: &mut TileCoefCtx,
    plane: usize,
    x4: usize,
    y4_in_sb: usize,
    tx_size: TxSize,
    tx_type: TxType,
    ref_type: usize,
    seg_eob: usize,
    raw_coefs: &[i32],
    eob: usize,
    counts: Option<&mut CoefCountsBank>,
) {
    // libvpx `get_plane_type`: Y → 0, UV → 1
    // (`vp9_blockd.h:210-212`).
    let plane_type = usize::from(plane > 0);

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
    // cells. Mirrors the decoder-side `fold_any_nonzero` helper in
    // `rvp9_decoder::detokenize`.
    let n = 1usize << (tx_size as usize); // 1 / 2 / 4 / 8
    let mut a_any: u8 = 0;
    for i in 0..n {
        if coef_ctx.above_coef(plane, x4 + i) != 0 {
            a_any = 1;
            break;
        }
    }
    let mut l_any: u8 = 0;
    for i in 0..n {
        if coef_ctx.left_coef(plane, y4_in_sb + i) != 0 {
            l_any = 1;
            break;
        }
    }
    let initial_pt = combine_entropy_contexts(a_any, l_any);

    // G3 — slice the per-frame counts to per-block DecodeCoefCounts.
    let mut block_counts = counts.map(|c| {
        let plane_type = usize::from(plane > 0);
        DecodeCoefCounts {
            coef_counts: &mut c.coef[tx_size as usize][plane_type][ref_type],
            eob_branch_counts: &mut c.eob_branch[tx_size as usize][plane_type][ref_type],
        }
    });
    encode_coefs(bw, coef_probs, &scan, initial_pt, tx_size, seg_eob, raw_coefs, eob,
                 block_counts.as_mut());

    // Writeback: splat `(eob > 0)` across the (n × n) sub-block
    // region. libvpx `vp9_detokenize.c:294/302/311/323`.
    coef_ctx.set_coef_flags(plane, x4, y4_in_sb, n, n, eob > 0);
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

    // ===================================================================
    // Stage 4 task S4.15 — encoder-side `encode_coefs` round-trip tests.
    //
    // Each test builds a raw quantized-coefficient block, runs
    // `encode_coefs` into a `BoolWriter`, then decodes the same bytes
    // with `decode_coefs` using the identity dequant pair (`(1, 1)` for
    // Tx{4,8,16}, `(2, 2)` for Tx32x32 — the `(val * 2) >> 1` cancels
    // the `dq_shift = 1` so the recovered value equals the raw input).
    // ===================================================================

    /// Helper: encode `raw_coefs[..eob]` into a fresh buffer and return
    /// the byte slice. Adds 32 trailing zero bits of padding so the
    /// `BoolReader` register can fill without overshoot.
    fn enc_block(
        probs: &CoefBandCtxProbs,
        scan: &ScanOrder,
        initial_pt: usize,
        tx_size: TxSize,
        seg_eob: usize,
        raw_coefs: &[i32],
        eob: usize,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; tx_size.n_coeffs() * 8 + 256];
        let n = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            encode_coefs(&mut bw, probs, scan, initial_pt, tx_size, seg_eob, raw_coefs, eob, None);
            for _ in 0..32 {
                bw.write_bit(0);
            }
            bw.stop_encode().unwrap()
        };
        buf.truncate(n);
        buf
    }

    /// Decode the bytes produced by `enc_block` with the identity-dq
    /// pair appropriate for `tx_size` and assert that the recovered
    /// `(eob, dq_coefs)` equal `(expected_eob, expected_raw)`. Returns
    /// the recovered dq buffer for additional caller-side assertions.
    #[allow(clippy::too_many_arguments)] // mirrors enc_block 1:1
    fn assert_round_trip(
        bytes: &[u8],
        probs: &CoefBandCtxProbs,
        scan: &ScanOrder,
        initial_pt: usize,
        tx_size: TxSize,
        seg_eob: usize,
        expected_raw: &[i32],
        expected_eob: usize,
    ) -> Vec<i32> {
        // Identity dequant: `(val * dq) >> dq_shift` must equal `val`.
        // For Tx32x32 dq_shift=1, so use (2, 2); otherwise (1, 1).
        let dq = if matches!(tx_size, TxSize::Tx32x32) {
            (2i32, 2i32)
        } else {
            (1i32, 1i32)
        };
        let mut dq_out = vec![0i32; tx_size.n_coeffs()];
        let mut r = BoolReader::new(bytes).unwrap();
        let eob = decode_coefs(
            &mut r,
            probs,
            None,
            scan,
            initial_pt,
            tx_size,
            seg_eob,
            dq,
            &mut dq_out,
        );
        assert_eq!(eob, expected_eob, "round-trip eob mismatch");
        assert_eq!(
            &dq_out[..],
            &expected_raw[..tx_size.n_coeffs()],
            "round-trip dq_coefs mismatch"
        );
        dq_out
    }

    #[test]
    fn encode_coefs_round_trips_empty_block() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let raw = vec![0i32; 16];
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 0);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 0);
    }

    #[test]
    fn encode_coefs_round_trips_single_one_dc_positive() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 1;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
    }

    #[test]
    fn encode_coefs_round_trips_single_one_negative() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = -1;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
    }

    #[test]
    fn encode_coefs_round_trips_two_token() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 2;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
    }

    #[test]
    fn encode_coefs_round_trips_three_token() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 3;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
    }

    #[test]
    fn encode_coefs_round_trips_four_token() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 4;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
    }

    #[test]
    fn encode_coefs_round_trips_cat1_full_range() {
        // CAT1 covers absolute values 5..=6 (one extension bit).
        for &v in &[5i32, 6] {
            let probs = flat_coef_probs(64, 64, 100);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_cat2_full_range() {
        // CAT2 covers absolute values 7..=10 (two extension bits).
        for &v in &[7i32, 8, 9, 10] {
            let probs = flat_coef_probs(64, 64, 100);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_cat3() {
        // CAT3 covers absolute values 11..=18 (three extension bits).
        for &v in &[11i32, 18] {
            let probs = flat_coef_probs(64, 64, 100);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_cat4() {
        // CAT4 covers absolute values 19..=34 (four extension bits).
        for &v in &[19i32, 34] {
            let probs = flat_coef_probs(64, 64, 100);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_cat5() {
        // CAT5 covers absolute values 35..=66 (five extension bits).
        for &v in &[35i32, 66] {
            let probs = flat_coef_probs(64, 64, 100);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_cat6() {
        // CAT6 covers absolute values 67..=67 + 2^14 - 1 = 16450
        // (fourteen extension bits) on 8-bit Profile 0.
        for &v in &[67i32, 128, 1000, 8194, 16450] {
            let probs = flat_coef_probs(64, 64, 200);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_zero_run_then_one() {
        // Mirrors `single_two_token_ac_at_position_5_negative`: scan
        // positions 0..=4 are zero, position 5 is -2, eob = 6.
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[5] as usize] = -2;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 6);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 6);
    }

    #[test]
    fn encode_coefs_round_trips_block_full_to_seg_eob() {
        // Every scan position non-zero, eob = seg_eob = 16. NO terminal
        // EOB bit is emitted — decoder exits via `c >= seg_eob` guard.
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let raw = vec![1i32; 16];
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 16);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 16);
    }

    #[test]
    fn encode_coefs_round_trips_full_block_random_fuzz() {
        // 100 iterations of random 4x4 blocks with abs-cap 200 (hits
        // CAT5 cleanly, occasionally CAT6).
        let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_F00D_u64);
        let scan = default_scan_4x4();
        for iter in 0..100 {
            let probs = random_coef_probs(&mut rng);
            let initial_pt = (rng.next_range(3) as usize) & 0x3;
            let target_eob = rng.next_range(17) as usize; // 0..=16
            let mut raw = vec![0i32; 16];
            // Populate scan positions 0..target_eob with random non-
            // zero magnitudes capped at 200 (with random sign). Scan
            // positions >= target_eob remain zero.
            //
            // To exercise the inner zero-run path we ALSO inject a few
            // zeros within the run: each slot has a 25% chance of being
            // zero (skipped). The last filled position must be non-zero
            // so eob is well-defined; we enforce this by computing the
            // last-non-zero index and using it as eob.
            for c in 0..target_eob {
                let r = rng.next_range(100);
                if r < 25 {
                    // leave zero
                } else {
                    let mag = 1 + (rng.next_range(200) as i32);
                    let sign = if rng.next_range(2) == 0 { 1 } else { -1 };
                    raw[scan.scan[c] as usize] = mag * sign;
                }
            }
            // Compute effective eob = last-non-zero scan position + 1.
            let mut effective_eob = 0usize;
            for c in 0..target_eob {
                if raw[scan.scan[c] as usize] != 0 {
                    effective_eob = c + 1;
                }
            }
            let bytes = enc_block(
                &probs,
                &scan,
                initial_pt,
                TxSize::Tx4x4,
                16,
                &raw,
                effective_eob,
            );
            let mut dq_out = vec![0i32; 16];
            let mut r = BoolReader::new(&bytes).unwrap();
            let eob_got = decode_coefs(
                &mut r,
                &probs,
                None,
                &scan,
                initial_pt,
                TxSize::Tx4x4,
                16,
                (1, 1),
                &mut dq_out,
            );
            assert_eq!(
                eob_got, effective_eob,
                "iter {iter}: eob mismatch (raw={raw:?}, initial_pt={initial_pt})",
            );
            assert_eq!(
                dq_out, raw,
                "iter {iter}: dq round-trip mismatch (initial_pt={initial_pt})",
            );
        }
    }

    // ------------------------------------------------------------------
    // Stage 9 — additional `decode_coefs` branch coverage
    // ------------------------------------------------------------------

    /// `decode_coefs` accepts `seg_eob < n_coeffs` (per-segment cap) and
    /// stops cleanly when the EOB token arrives before `seg_eob`. The
    /// existing `block_runs_to_seg_eob_via_zero_run` test exercises the
    /// "zero-run advances c past seg_eob" path; this test exercises a
    /// short seg_eob with normal NOT-EOB + EOB termination.
    #[test]
    fn decode_coefs_short_seg_eob_stops_early() {
        let probs = flat_coef_probs(64, 64, 64);
        // Encode: pos 0 ONE_TOKEN, then EOB at pos 1. seg_eob = 5
        // (smaller than n_coeffs=16) — both `seg_eob` and the EOB token
        // bound the loop.
        let bytes = encode(|w| {
            w.write(1, 64); // NOT-EOB at pos 0
            w.write(1, 64); // NOT-ZERO
            w.write(0, 64); // ONE_TOKEN
            w.write_bit(0); // positive
            w.write(0, 64); // EOB at pos 1
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let scan = default_scan_4x4();
        let mut dqc = vec![0i32; 16];
        let eob = decode_coefs(
            &mut r,
            &probs,
            None,
            &scan,
            0,
            TxSize::Tx4x4,
            5, // shortened seg_eob
            (3, 5),
            &mut dqc,
        );
        assert_eq!(eob, 1);
        assert_eq!(dqc[scan.scan[0] as usize], 3);
    }

    /// `decode_coefs` with `counts = Some(&mut ...)` AND a zero-run
    /// AND a non-zero-token must bump every applicable counter on every
    /// loop iteration. Hits the `if let Some(ref mut cs) = counts`
    /// branch inside the inner zero-run loop (previously only exercised
    /// without a zero-run).
    #[test]
    fn decode_coefs_counts_with_zero_run_and_value_tree() {
        let probs = flat_coef_probs(64, 64, 64);
        // Pos 0: NOT-EOB, ZERO (run of 1).
        // Pos 1: NOT-ZERO, ONE_TOKEN, sign positive.
        // Pos 2: EOB.
        let bytes = encode(|w| {
            w.write(1, 64); // NOT-EOB pos 0
            w.write(0, 64); // ZERO at pos 0
            w.write(1, 64); // NOT-ZERO at pos 1
            w.write(0, 64); // ONE_TOKEN
            w.write_bit(0); // positive
            w.write(0, 64); // EOB pos 2
        });
        let mut r = BoolReader::new(&bytes).unwrap();
        let scan = default_scan_4x4();
        let mut dqc = vec![0i32; 16];
        let mut coef_counts: CoefCountsBlock = [[[0; UNCONSTRAINED_NODES + 1]; COEFF_CONTEXTS]; COEF_BANDS];
        let mut eob_branch_counts: EobBranchCountsBlock = [[0; COEFF_CONTEXTS]; COEF_BANDS];
        let mut counts = DecodeCoefCounts {
            coef_counts: &mut coef_counts,
            eob_branch_counts: &mut eob_branch_counts,
        };
        let eob = decode_coefs(
            &mut r,
            &probs,
            Some(&mut counts),
            &scan,
            0,
            TxSize::Tx4x4,
            16,
            (7, 11),
            &mut dqc,
        );
        assert_eq!(eob, 2);
        // Pin a ZERO_TOKEN bump (inner zero-run path's counts update)
        // and an EOB tally for at least one band.
        let zero_total: u32 = coef_counts.iter().flatten().map(|t| t[ZERO_TOKEN]).sum();
        assert!(zero_total >= 1, "expected at least one ZERO_TOKEN tally");
        let one_total: u32 = coef_counts.iter().flatten().map(|t| t[ONE_TOKEN]).sum();
        assert_eq!(one_total, 1, "exactly one ONE_TOKEN");
        let eob_total: u32 = coef_counts.iter().flatten().map(|t| t[EOB_MODEL_TOKEN]).sum();
        assert_eq!(eob_total, 1, "exactly one EOB_MODEL_TOKEN");
        let eob_branch_total: u32 = eob_branch_counts.iter().flatten().sum();
        // Two outer iterations: pos 0 (NOT-EOB → ZERO → NOT-ZERO →
        // ONE_TOKEN at pos 1, inner zero-run advances c) and pos 2
        // (EOB). The eob_branch_counts bumps once at the top of each
        // outer iteration before the EOB read.
        assert_eq!(eob_branch_total, 2, "two eob branch tallies");
    }

    /// `encode_coefs` with `eob == seg_eob` AND a leading zero-run of
    /// length seg_eob-1 followed by one non-zero token at scan[seg_eob-1]:
    /// the inner zero-loop runs for 15 positions (emits ZERO ×15), then
    /// hits coef != 0 at c=15, emits the value tree, advances c to 16.
    /// The outer loop's `while c < eob` exits naturally (c=16=eob),
    /// and the `if eob < seg_eob` terminal-EOB guard is FALSE (no
    /// terminal EOB emitted). Decoder exits via its `c >= max_eob`
    /// guard. Pins the encoder's terminal-EOB `eob < seg_eob` false arm.
    #[test]
    fn encode_coefs_full_block_with_leading_zero_run_no_terminal_eob() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        // Last scan position non-zero — forces eob = 16 == seg_eob.
        raw[scan.scan[15] as usize] = 1;
        let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 16);
        assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 16);
    }

    /// `encode_coefs` THREE vs FOUR_TOKEN split via Pareto p[2]:
    /// 3 emits `(0 @ p[2])` and 4 emits `(1 @ p[2])`. Round-trip both
    /// to pin both arms of the THREE-or-FOUR branch.
    #[test]
    fn encode_coefs_three_and_four_tokens_round_trip() {
        for &v in &[3i32, -3, 4, -4] {
            let probs = flat_coef_probs(64, 64, 100);
            let scan = default_scan_4x4();
            let mut raw = vec![0i32; 16];
            raw[scan.scan[0] as usize] = v;
            let bytes = enc_block(&probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1);
        }
    }

    #[test]
    fn encode_coefs_round_trips_all_tx_sizes() {
        use crate::transforms::scans::{
            default_scan_16x16, default_scan_32x32, default_scan_8x8,
        };
        // Per-tx_size table of (TxSize, ScanOrder ctor). Tx32x32 will
        // use dq=(2,2) via `assert_round_trip`'s built-in switch.
        #[allow(clippy::type_complexity)] // one-off test config table
        let configs: &[(TxSize, fn() -> ScanOrder)] = &[
            (TxSize::Tx4x4, default_scan_4x4),
            (TxSize::Tx8x8, default_scan_8x8),
            (TxSize::Tx16x16, default_scan_16x16),
            (TxSize::Tx32x32, default_scan_32x32),
        ];
        for &(tx_size, scan_fn) in configs {
            let n = tx_size.n_coeffs();
            let scan = scan_fn();
            let probs = flat_coef_probs(64, 64, 64);

            // Case 1: empty block.
            let raw = vec![0i32; n];
            let bytes = enc_block(&probs, &scan, 0, tx_size, n, &raw, 0);
            assert_round_trip(&bytes, &probs, &scan, 0, tx_size, n, &raw, 0);

            // Case 2: single ONE_TOKEN at scan[0], positive.
            let mut raw = vec![0i32; n];
            raw[scan.scan[0] as usize] = 1;
            let bytes = enc_block(&probs, &scan, 0, tx_size, n, &raw, 1);
            assert_round_trip(&bytes, &probs, &scan, 0, tx_size, n, &raw, 1);

            // Case 3: zero run of 5 then a TWO_TOKEN at scan[5].
            let mut raw = vec![0i32; n];
            raw[scan.scan[5] as usize] = 2;
            let bytes = enc_block(&probs, &scan, 0, tx_size, n, &raw, 6);
            assert_round_trip(&bytes, &probs, &scan, 0, tx_size, n, &raw, 6);
        }
    }

    // ===================================================================
    // `write_block_tokens` round-trip tests (S4.16g phase 2).
    //
    // The decoder-side mirror (`vp9_decode_block_tokens`) lives in the
    // `rvp9-decoder` crate, which we cannot depend on from here without
    // creating a cycle. Each test therefore (a) drives `write_block_tokens`
    // through a fresh `TileCoefCtx`, then (b) replicates the same per-TX
    // setup (`fc.coef_probs[..]` slice, `scan_for`, OR-fold) over a
    // SEPARATE decoder-side `TileCoefCtx` and feeds the result into
    // `decode_coefs`. This mirrors what `vp9_decode_block_tokens` does
    // verbatim and pins the encode/decode contract on the OR-fold and
    // writeback steps.
    //
    // Dequant: identity-dequant trick — `(1, 1)` for Tx{4,8,16}, `(2, 2)`
    // for Tx32x32 (the `>> dq_shift=1` cancels the `* 2`), so the
    // recovered `dq_coefs_out` slot equals the raw input value.
    // ===================================================================

    use crate::coef_ctx::TileCoefCtx;
    use crate::transforms::types::TxType;

    /// Identity-dequant pair for `tx_size`.
    fn identity_dq(tx_size: TxSize) -> (i32, i32) {
        if matches!(tx_size, TxSize::Tx32x32) {
            (2, 2)
        } else {
            (1, 1)
        }
    }

    /// OR-fold `n` cells of an above-row or left-column slice — the
    /// decoder-side mirror of the inlined OR-fold inside
    /// `write_block_tokens`. Cell-by-cell because `TileCoefCtx` stores
    /// one `u8` per 4×4 sub-block (rather than the packed `uintN_t`
    /// libvpx uses).
    fn fold_any_nonzero(
        coef_ctx: &TileCoefCtx,
        plane: usize,
        start: usize,
        n: usize,
        above: bool,
    ) -> u8 {
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

    /// End-to-end round-trip helper:
    /// 1. Run `write_block_tokens` over a fresh `TileCoefCtx`.
    /// 2. Re-derive `initial_pt` from a FRESH decoder-side `TileCoefCtx`
    ///    that has been seeded with the SAME pre-encode state.
    /// 3. Feed the bytes into `decode_coefs` and assert `(eob, dq_coefs)`
    ///    match the input.
    /// 4. Return the (post-encode) encoder-side `TileCoefCtx` for any
    ///    caller-side writeback assertions.
    ///
    /// `seed_enc_ctx` runs before encode on the encoder-side coef-ctx
    /// (typically `|_| {}` for fresh-context tests). The decoder-side
    /// coef-ctx receives the same seeding so the OR-fold matches.
    #[allow(clippy::too_many_arguments)] // mirror of write_block_tokens
    fn round_trip_block_tokens<F: Fn(&mut TileCoefCtx)>(
        plane: usize,
        x4: usize,
        y4_in_sb: usize,
        tx_size: TxSize,
        tx_type: TxType,
        ref_type: usize,
        seg_eob: usize,
        raw_coefs: &[i32],
        eob: usize,
        seed_enc_ctx: F,
    ) -> TileCoefCtx {
        let fc = FrameContext::new_default();
        let n = tx_size.n_coeffs();
        assert!(raw_coefs.len() >= n);

        // tile_mi_cols=8 gives Y above_coef width=16, UV width=8.
        // Sufficient for every case below (max Tx32x32 at origin needs
        // 8 cells, which fits even on UV).
        let mut enc_ctx = TileCoefCtx::new(8);
        seed_enc_ctx(&mut enc_ctx);
        let mut dec_ctx = TileCoefCtx::new(8);
        seed_enc_ctx(&mut dec_ctx);

        // Encode side: 8 bytes-per-coef + slack handles the worst case
        // (every coef CAT6 + 14 ext bits ≈ 32 bits ⇒ 4 bytes; 8B per
        // coef leaves room).
        let mut buf = vec![0u8; n * 8 + 256];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            write_block_tokens(
                &mut bw,
                &fc,
                &mut enc_ctx,
                plane,
                x4,
                y4_in_sb,
                tx_size,
                tx_type,
                ref_type,
                seg_eob,
                raw_coefs,
                eob,
                None,
            );
            // Pad with trailing zero bits so the reader can fill its
            // register without overshoot (matches `enc_block` helper).
            for _ in 0..32 {
                bw.write_bit(0);
            }
            bw.stop_encode().unwrap()
        };
        buf.truncate(written);

        // Decode side: replicate the per-TX-block setup
        // `decode_block_tokens` performs before its `decode_coefs` call.
        let plane_type = usize::from(plane > 0);
        let coef_probs: &CoefBandCtxProbs =
            &fc.coef_probs[tx_size as usize][plane_type][ref_type];
        let scan = scan_for(tx_size, tx_type);
        let n_span = 1usize << (tx_size as usize);
        let a_any = fold_any_nonzero(&dec_ctx, plane, x4, n_span, true);
        let l_any = fold_any_nonzero(&dec_ctx, plane, y4_in_sb, n_span, false);
        let initial_pt = combine_entropy_contexts(a_any, l_any);

        let mut r = BoolReader::new(&buf).unwrap();
        let mut dq_out = vec![0i32; n];
        let dq = identity_dq(tx_size);
        let recovered_eob = decode_coefs(
            &mut r, coef_probs, None, &scan, initial_pt, tx_size, seg_eob, dq,
            &mut dq_out,
        );

        assert_eq!(recovered_eob, eob, "eob mismatch (plane={plane}, x4={x4}, y4_in_sb={y4_in_sb})");
        assert_eq!(&dq_out[..], &raw_coefs[..n], "coef mismatch");

        // Mirror the writeback into dec_ctx so the caller can compare
        // the post-encode + post-decode states (they must agree).
        dec_ctx.set_coef_flags(plane, x4, y4_in_sb, n_span, n_span, recovered_eob > 0);
        for plane_idx in 0..3 {
            for x in 0..enc_ctx.above_coef_width(plane_idx) {
                assert_eq!(
                    enc_ctx.above_coef(plane_idx, x),
                    dec_ctx.above_coef(plane_idx, x),
                    "above[{plane_idx}][{x}] enc/dec mismatch"
                );
            }
            for y in 0..enc_ctx.left_coef_height(plane_idx) {
                assert_eq!(
                    enc_ctx.left_coef(plane_idx, y),
                    dec_ctx.left_coef(plane_idx, y),
                    "left[{plane_idx}][{y}] enc/dec mismatch"
                );
            }
        }

        enc_ctx
    }

    /// Y intra Tx4x4 DctDct, all-zero coefs (eob=0). Trivial proof-of-
    /// life — emits a single EOB-at-pos-0 bit, decodes to eob=0, leaves
    /// the writeback region cleared.
    #[test]
    fn write_block_tokens_y_intra_tx4x4_empty() {
        let raw = vec![0i32; 16];
        let ctx = round_trip_block_tokens(
            0, 0, 0,
            TxSize::Tx4x4,
            TxType::DctDct,
            0, // intra
            16,
            &raw,
            0,
            |_| {},
        );
        // eob == 0 ⇒ writeback splatted zero. All cells remain 0.
        assert_eq!(ctx.above_coef(0, 0), 0);
        assert_eq!(ctx.left_coef(0, 0), 0);
    }

    /// Y intra Tx4x4 DctDct, 3 non-zero coefs of mixed magnitude /
    /// sign. Round-trip oracle covers the value-tree + sign-bit paths
    /// at three different scan positions.
    #[test]
    fn write_block_tokens_y_intra_tx4x4_three_nonzero() {
        let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 1;   // ONE_TOKEN, positive
        raw[scan.scan[1] as usize] = -2;  // TWO_TOKEN, negative
        raw[scan.scan[2] as usize] = 7;   // CAT2 (7..=10), positive
        let ctx = round_trip_block_tokens(
            0, 0, 0,
            TxSize::Tx4x4,
            TxType::DctDct,
            0,
            16,
            &raw,
            3,
            |_| {},
        );
        // Writeback: above[0] and left[0] flipped to 1 (Tx4x4 spans 1×1).
        assert_eq!(ctx.above_coef(0, 0), 1);
        assert_eq!(ctx.left_coef(0, 0), 1);
    }

    /// Y inter Tx8x8 DctDct, eob=10. Tx8x8 spans 2×2 sub-blocks ⇒
    /// writeback splats above[0..2] = left[0..2] = 1. ref_type=1
    /// selects the inter coef-prob bank.
    #[test]
    fn write_block_tokens_y_inter_tx8x8_eob10() {
        let scan = scan_for(TxSize::Tx8x8, TxType::DctDct);
        let mut raw = vec![0i32; 64];
        // Fill scan[0..10] with small non-zero values.
        for c in 0..10 {
            let v = if c % 2 == 0 { 1 + (c as i32) } else { -(1 + c as i32) };
            raw[scan.scan[c] as usize] = v;
        }
        let ctx = round_trip_block_tokens(
            0, 0, 0,
            TxSize::Tx8x8,
            TxType::DctDct,
            1, // inter
            64,
            &raw,
            10,
            |_| {},
        );
        for i in 0..2 {
            assert_eq!(ctx.above_coef(0, i), 1, "Y above[{i}] splat");
            assert_eq!(ctx.left_coef(0, i), 1, "Y left[{i}] splat");
        }
    }

    /// UV inter Tx16x16, eob=20. Plane=1 (U) selects plane_type=1 in
    /// the coef_probs index. Tx16x16 spans 4×4 ⇒ writeback splats
    /// above[0..4] = left[0..4] = 1 on the U plane only.
    #[test]
    fn write_block_tokens_uv_inter_tx16x16_eob20() {
        let scan = scan_for(TxSize::Tx16x16, TxType::DctDct);
        let mut raw = vec![0i32; 256];
        for c in 0..20 {
            let v: i32 = match c % 5 {
                0 => 1,
                1 => -1,
                2 => 5,    // CAT1 (5..=6)
                3 => -11,  // CAT3 (11..=18)
                _ => 3,
            };
            raw[scan.scan[c] as usize] = v;
        }
        let ctx = round_trip_block_tokens(
            1, 0, 0,
            TxSize::Tx16x16,
            TxType::DctDct,
            1,
            256,
            &raw,
            20,
            |_| {},
        );
        // U-plane writeback: 4×4 region (above[0..4], left[0..4]).
        for i in 0..4 {
            assert_eq!(ctx.above_coef(1, i), 1, "U above[{i}] splat");
            assert_eq!(ctx.left_coef(1, i), 1, "U left[{i}] splat");
        }
        // Y and V untouched.
        for plane in [0usize, 2] {
            for x in 0..ctx.above_coef_width(plane) {
                assert_eq!(ctx.above_coef(plane, x), 0, "plane {plane} above[{x}]");
            }
            for y in 0..ctx.left_coef_height(plane) {
                assert_eq!(ctx.left_coef(plane, y), 0, "plane {plane} left[{y}]");
            }
        }
    }

    /// Y intra Tx32x32, eob=64 (well under n_coeffs=1024). Largest TX
    /// size — exercises the `dq_shift=1` identity-dequant path
    /// (dq=(2,2) ⇒ `(val * 2) >> 1 == val`).
    #[test]
    fn write_block_tokens_y_intra_tx32x32_eob64() {
        let scan = scan_for(TxSize::Tx32x32, TxType::DctDct);
        let mut raw = vec![0i32; 1024];
        for c in 0..64 {
            // Mix: ONE, TWO, CAT1, occasional zero (to hit zero-run
            // path inside the value-tree loop).
            let v: i32 = match c % 7 {
                0 | 4 => 0, // zero within run
                1 => 1,
                2 => -2,
                3 => 5,    // CAT1
                5 => 3,
                _ => -1,
            };
            raw[scan.scan[c] as usize] = v;
        }
        // Effective eob = last-non-zero scan position + 1; if scan[63]
        // landed on a zero slot, scan back. For our mod-7 pattern
        // c=63 → 63 % 7 = 0 ⇒ zero. Force a non-zero at scan[63] so
        // eob=64 is well-defined.
        raw[scan.scan[63] as usize] = 1;

        let ctx = round_trip_block_tokens(
            0, 0, 0,
            TxSize::Tx32x32,
            TxType::DctDct,
            0,
            1024,
            &raw,
            64,
            |_| {},
        );
        // 8×8 writeback splat across above and left.
        for i in 0..8 {
            assert_eq!(ctx.above_coef(0, i), 1, "above[{i}] splat");
            assert_eq!(ctx.left_coef(0, i), 1, "left[{i}] splat");
        }
    }

    /// Above/left context propagation: write block A at (x4=0,y4=0)
    /// with eob>0, then write block B at (x4=1,y4=0). B's encode must
    /// observe above[0]=1 / left[0]=1 (well, left[0]=1 since the
    /// previous block sits to the left) and pick the corresponding
    /// `initial_pt`. The round-trip helper re-derives the same
    /// `initial_pt` on the decode side from the matching seed
    /// state. If propagation works, the bytes round-trip cleanly.
    #[test]
    fn write_block_tokens_above_left_propagation() {
        let fc = FrameContext::new_default();
        let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
        let mut raw_a = vec![0i32; 16];
        raw_a[scan.scan[0] as usize] = 1; // ONE_TOKEN, positive
        let eob_a = 1;

        let mut raw_b = vec![0i32; 16];
        raw_b[scan.scan[0] as usize] = -3; // THREE_TOKEN, negative
        let eob_b = 1;

        // Encode both blocks into a SHARED enc_ctx + buffer.
        let mut enc_ctx = TileCoefCtx::new(8);
        let mut buf = vec![0u8; 1024];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            write_block_tokens(
                &mut bw, &fc, &mut enc_ctx, 0, 0, 0,
                TxSize::Tx4x4, TxType::DctDct, 0, 16, &raw_a, eob_a, None,
            );
            write_block_tokens(
                &mut bw, &fc, &mut enc_ctx, 0, 1, 0,
                TxSize::Tx4x4, TxType::DctDct, 0, 16, &raw_b, eob_b, None,
            );
            for _ in 0..32 {
                bw.write_bit(0);
            }
            bw.stop_encode().unwrap()
        };
        buf.truncate(written);

        // After A: above[0] = left[0] = 1. After B (eob>0): above[1] =
        // left[0] = 1.
        assert_eq!(enc_ctx.above_coef(0, 0), 1, "post-AB above[0]");
        assert_eq!(enc_ctx.above_coef(0, 1), 1, "post-AB above[1]");
        assert_eq!(enc_ctx.left_coef(0, 0), 1, "post-AB left[0]");

        // Decode side: replicate the per-block setup. Mirror the
        // `write_block_tokens` ctx evolution as we decode.
        let mut r = BoolReader::new(&buf).unwrap();
        let mut dec_ctx = TileCoefCtx::new(8);

        // Block A at (0, 0).
        let coef_probs = &fc.coef_probs[TxSize::Tx4x4 as usize][0][0];
        let a_any = fold_any_nonzero(&dec_ctx, 0, 0, 1, true);
        let l_any = fold_any_nonzero(&dec_ctx, 0, 0, 1, false);
        let initial_pt_a = combine_entropy_contexts(a_any, l_any);
        assert_eq!(initial_pt_a, 0, "A at origin: pt=0");
        let mut dq_out_a = vec![0i32; 16];
        let recovered_eob_a = decode_coefs(
            &mut r, coef_probs, None, &scan, initial_pt_a,
            TxSize::Tx4x4, 16, (1, 1), &mut dq_out_a,
        );
        assert_eq!(recovered_eob_a, eob_a);
        assert_eq!(dq_out_a, raw_a);
        dec_ctx.set_coef_flags(0, 0, 0, 1, 1, recovered_eob_a > 0);

        // Block B at (1, 0). At this point dec_ctx.above[0]=1, left[0]=1.
        // B reads above[1] (=0) and left[0] (=1) ⇒ initial_pt = 0 + 1 = 1.
        let a_any = fold_any_nonzero(&dec_ctx, 0, 1, 1, true);
        let l_any = fold_any_nonzero(&dec_ctx, 0, 0, 1, false);
        let initial_pt_b = combine_entropy_contexts(a_any, l_any);
        assert_eq!(initial_pt_b, 1, "B right of A: pt = 0 + 1 = 1");
        let mut dq_out_b = vec![0i32; 16];
        let recovered_eob_b = decode_coefs(
            &mut r, coef_probs, None, &scan, initial_pt_b,
            TxSize::Tx4x4, 16, (1, 1), &mut dq_out_b,
        );
        assert_eq!(recovered_eob_b, eob_b);
        assert_eq!(dq_out_b, raw_b);
        dec_ctx.set_coef_flags(0, 1, 0, 1, 1, recovered_eob_b > 0);

        // Encoder / decoder ctxs end in the same state.
        for plane in 0..3 {
            for x in 0..enc_ctx.above_coef_width(plane) {
                assert_eq!(
                    enc_ctx.above_coef(plane, x),
                    dec_ctx.above_coef(plane, x),
                );
            }
            for y in 0..enc_ctx.left_coef_height(plane) {
                assert_eq!(
                    enc_ctx.left_coef(plane, y),
                    dec_ctx.left_coef(plane, y),
                );
            }
        }
    }

    /// `seg_eob < n_coeffs`: shorten the per-segment EOB cap to 5 for
    /// a Tx4x4 block (whose n_coeffs is 16). Encode 3 non-zero coefs.
    /// Decoder side must use the same seg_eob to bound its scan walk.
    #[test]
    fn write_block_tokens_seg_eob_shortened() {
        let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 1;
        raw[scan.scan[1] as usize] = -2;
        raw[scan.scan[2] as usize] = 3;
        // seg_eob=5 (< n_coeffs=16), eob=3.
        round_trip_block_tokens(
            0, 0, 0,
            TxSize::Tx4x4,
            TxType::DctDct,
            0,
            5,
            &raw,
            3,
            |_| {},
        );
    }

    /// Pre-seeded above/left context: seed both neighbours non-zero
    /// before encoding so initial_pt = 2. Round-trips via the helper,
    /// which re-derives the same initial_pt on decode side from the
    /// same seed.
    #[test]
    fn write_block_tokens_initial_pt_two_via_seeded_context() {
        let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 1;
        round_trip_block_tokens(
            0,
            2, 2, // place the block away from origin so the seed cells aren't overwritten
            TxSize::Tx4x4,
            TxType::DctDct,
            0,
            16,
            &raw,
            1,
            |ctx| {
                // Seed Y above[2] = 1 and Y left[2] = 1 — the two cells
                // the block at (x4=2, y4_in_sb=2) reads for its OR-fold.
                ctx.set_coef_flags(0, 2, 2, 1, 1, true);
                // The set above wrote both above[2] and left[2]; perfect.
            },
        );
    }

    /// G3 proof test — verifies that `encode_coefs` actually accumulates
    /// counts when given a non-`None` counts handle. Without this test
    /// the count accumulation could silently be a no-op (which is
    /// exactly the bug the reviewer caught in the original G3 attempt).
    ///
    /// Encodes a single ONE_TOKEN at scan[0] (positive), eob=1, with a
    /// 4x4 scan. The expected counts are:
    /// - One eob_branch attempt before the EOB decision (band 0, ctx 0)
    /// - One ONE_TOKEN tally (band 0, ctx 0)
    /// - One eob_branch attempt at terminal EOB (band depends on scan)
    /// - One EOB_MODEL_TOKEN tally at terminal EOB
    #[test]
    fn g3_encode_coefs_accumulates_counts_when_handle_present() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 1; // single ONE_TOKEN at DC

        let mut coef_counts: CoefCountsBlock =
            [[[0u32; UNCONSTRAINED_NODES + 1]; COEFF_CONTEXTS]; COEF_BANDS];
        let mut eob_branch_counts: EobBranchCountsBlock =
            [[0u32; COEFF_CONTEXTS]; COEF_BANDS];
        let mut buf = vec![0u8; 256];
        let _ = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let mut counts = DecodeCoefCounts {
                coef_counts: &mut coef_counts,
                eob_branch_counts: &mut eob_branch_counts,
            };
            encode_coefs(&mut bw, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1,
                         Some(&mut counts));
            bw.stop_encode().unwrap()
        };

        // Outer-loop iteration at c=0: eob_branch[0][0] += 1.
        assert_eq!(eob_branch_counts[0][0], 1,
                   "outer-loop eob_branch attempt at c=0");
        // ONE_TOKEN at c=0: coef_counts[0][0][ONE_TOKEN] += 1.
        assert_eq!(coef_counts[0][0][ONE_TOKEN], 1,
                   "ONE_TOKEN tally at c=0");

        // Terminal EOB at eob=1 (eob<seg_eob=16 so terminal bit fires).
        // band = band_translate[1] (4x4: band 1), ctx = post-ONE_TOKEN ctx.
        // The exact (band, ctx) depends on `get_coef_context`, but
        // exactly one EOB_MODEL_TOKEN + one eob_branch must be tallied
        // SOMEWHERE in the table.
        let total_eob_model: u32 = coef_counts.iter()
            .flat_map(|b| b.iter())
            .map(|ctx| ctx[EOB_MODEL_TOKEN])
            .sum();
        assert_eq!(total_eob_model, 1, "exactly one terminal EOB_MODEL tally");
        let total_eob_branch: u32 = eob_branch_counts.iter()
            .flat_map(|b| b.iter())
            .sum();
        assert_eq!(total_eob_branch, 2,
                   "two eob_branch attempts: one outer-loop, one terminal");

        // Nothing else should have been touched.
        let total_zero: u32 = coef_counts.iter()
            .flat_map(|b| b.iter())
            .map(|ctx| ctx[ZERO_TOKEN])
            .sum();
        assert_eq!(total_zero, 0, "no ZERO_TOKEN tallies (no zeros in input)");
        let total_two: u32 = coef_counts.iter()
            .flat_map(|b| b.iter())
            .map(|ctx| ctx[TWO_TOKEN])
            .sum();
        assert_eq!(total_two, 0, "no TWO_TOKEN tallies (no twos in input)");
    }

    /// G3 proof test — `encode_coefs` with `None` counts is a no-op
    /// (no panic, no behavioural change from the pre-G3 version).
    #[test]
    fn g3_encode_coefs_with_none_counts_does_not_panic() {
        let probs = flat_coef_probs(64, 64, 64);
        let scan = default_scan_4x4();
        let mut raw = vec![0i32; 16];
        raw[scan.scan[0] as usize] = 2; // TWO_TOKEN

        let mut buf = vec![0u8; 256];
        let _ = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            encode_coefs(&mut bw, &probs, &scan, 0, TxSize::Tx4x4, 16, &raw, 1,
                         None);
            bw.stop_encode().unwrap()
        };
        // If we got here, the None path didn't panic.
    }
}
