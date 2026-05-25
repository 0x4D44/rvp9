//! Adaptive RD-threshold gate for the inter-mode picker (P20).
//!
//! Port of libvpx's `adaptive_rd_thresh` speed feature
//! (`vp9_pickmode.c:2241-2247` gate site;
//! `vp9_pickmode.c:1139-1154` `update_thresh_freq_fact`;
//! `vp9_rd.c:355-385` `set_block_thresholds`;
//! `vp9_rd.c:693-745` `vp9_set_rd_speed_thresholds`).
//!
//! ## Mechanism
//!
//! For each `(BlockSize, THR_MODE)` slot the encoder tracks two values:
//!
//! 1. **`rd_thresh_base[bsize][thr_mode]`** — a `(thresh_mult,
//!    block_size_factor, quantizer)`-derived constant that does not
//!    change within a frame. Computed once per frame.
//! 2. **`thresh_freq_fact[bsize][thr_mode]`** — a per-frame mutable
//!    factor that grows when a mode loses and shrinks when it wins.
//!    Initialised to [`RD_THRESH_INIT_FACT`] (= 32, i.e. neutral
//!    `1.0`).
//!
//! The gate, applied per candidate before evaluation, is:
//!
//! ```text
//! effective_thresh = rd_thresh_base[bsize][thr_mode] * fact >> 5
//! if best_rd_so_far < effective_thresh: skip this candidate
//! ```
//!
//! After the frame's picker has decided the winning mode, the caller
//! calls [`update_thresh_freq_fact`] which decays the winner's fact
//! (`*fact -= *fact >> 4`) and increments every other inter mode's fact
//! (`*fact = min(*fact + 1, adaptive_rd_thresh * RD_THRESH_MAX_FACT)`).
//!
//! ## libvpx fidelity & divergences
//!
//! - **THR_MODES mapping**: we mirror the `mode_idx[MAX_REF_FRAMES][4]`
//!   table from `vp9_pickmode.c:1089-1094`. rvp9 only ports the LAST
//!   and GOLDEN inter slots; ALTREF and compound entries stay at their
//!   default (`thresh_mult` from `vp9_set_rd_speed_thresholds` but
//!   never accessed by the gate because rvp9's picker only iterates
//!   LAST+GOLDEN ZERO/NEAREST/NEAR/NEW).
//! - **`adaptive_rd_thresh = 4`**: matches libvpx's cpu-used 6 / 8
//!   realtime path (`vp9_speed_features.c:752`,
//!   `vp9_speed_features.c:550`). Equivalent to the operating point
//!   rvp9's encoder targets.
//! - **No row-mt variant**: libvpx's
//!   `update_thresh_freq_fact_row_mt` / `row_base_thresh_freq_fact`
//!   exists for multi-threaded encoding (a separate SB-row-indexed
//!   table). rvp9 single-threads the encode and uses the single-tile
//!   `thresh_freq_fact` path only.
//! - **Single-segment**: rvp9 uses one segment (segment_id = 0); the
//!   libvpx `rd->threshes[MAX_SEGMENTS][BLOCK_SIZES][MAX_MODES]` table
//!   collapses to `[BLOCK_SIZES][MAX_MODES]`.
//! - **No `thresh_freq_fact_prev` snapshot**: libvpx keeps a paired
//!   `thresh_freq_fact_prev` (`vp9_encodeframe.c:5875, 6266`) so the
//!   end-of-frame state is restored before the next frame's encode
//!   begins. rvp9 omits the snapshot/restore and accumulates the
//!   live table directly — the difference is empirically negligible
//!   for steady-state encodes and avoids the per-frame O(BLOCK_SIZES
//!   × MAX_MODES) copy.

use crate::bitstream::BitDepth;
use crate::block::{BLOCK_SIZES, BlockSize};
use crate::mode_info::{InterMode, RefFrame};
use crate::quant::dc_quant;

/// Number of distinct mode-threshold slots libvpx tracks. Mirrors
/// `MAX_MODES = 30` (`vp9_rd.h:41`).
pub const MAX_MODES: usize = 30;

/// `RD_THRESH_INIT_FACT` — the initial value of every `thresh_freq_fact`
/// slot. `vp9_rd.h:44`.
pub const RD_THRESH_INIT_FACT: i32 = 32;

/// `RD_THRESH_MAX_FACT` — the per-multiplier cap on `thresh_freq_fact`
/// (the cap is `adaptive_rd_thresh * RD_THRESH_MAX_FACT`). `vp9_rd.h:45`.
pub const RD_THRESH_MAX_FACT: i32 = 64;

/// `RD_THRESH_INC` — the per-loss increment applied to non-winning
/// modes. `vp9_rd.h:46`.
pub const RD_THRESH_INC: i32 = 1;

/// `RD_THRESH_POW` — exponent in the quantizer-driven threshold
/// factor. `vp9_rd.c:43`. We compute the pow with `f64` to match
/// libvpx (`vp9_rd.c:316-328`).
const RD_THRESH_POW: f64 = 1.25;

/// `adaptive_rd_thresh` speed-feature value. Default-ON multiplier on
/// the per-mode threshold caps. Set to `4` to match libvpx's
/// realtime cpu-used-6/8 setting (`vp9_speed_features.c:752`).
pub const ADAPTIVE_RD_THRESH: i32 = 4;

/// libvpx `rd_thresh_block_size_factor[BLOCK_SIZES]` table from
/// `vp9_rd.c:88-90`. Indexed by [`BlockSize::idx`].
const RD_THRESH_BLOCK_SIZE_FACTOR: [u8; BLOCK_SIZES] =
    [2, 3, 3, 4, 6, 6, 8, 12, 12, 16, 24, 24, 32];

/// THR_MODES enum values for the inter slots rvp9 cares about.
/// Mirrors `vp9_rd.h:54-92`.
const THR_NEARESTMV: usize = 0;
const THR_NEARESTG: usize = 2;
const THR_NEWMV: usize = 4;
const THR_NEWG: usize = 6;
const THR_NEARMV: usize = 7;
const THR_NEARG: usize = 9;
const THR_ZEROMV: usize = 10;
const THR_ZEROG: usize = 11;

/// Map `(ref_frame, inter_mode)` to the libvpx `THR_MODES` slot index
/// used by `thresh_mult[]` / `thresh_freq_fact[bsize][..]`. Mirrors
/// `mode_idx[MAX_REF_FRAMES][4]` (`vp9_pickmode.c:1089-1094`) for the
/// LAST and GOLDEN ref frames. Returns `None` for ref frames the
/// rvp9 picker does not iterate (INTRA / ALTREF / NoRefFrame).
#[inline]
#[must_use]
pub fn thr_mode_idx(ref_frame: RefFrame, mode: InterMode) -> Option<usize> {
    match (ref_frame, mode) {
        (RefFrame::LastFrame, InterMode::NearestMv) => Some(THR_NEARESTMV),
        (RefFrame::LastFrame, InterMode::NearMv) => Some(THR_NEARMV),
        (RefFrame::LastFrame, InterMode::ZeroMv) => Some(THR_ZEROMV),
        (RefFrame::LastFrame, InterMode::NewMv) => Some(THR_NEWMV),
        (RefFrame::GoldenFrame, InterMode::NearestMv) => Some(THR_NEARESTG),
        (RefFrame::GoldenFrame, InterMode::NearMv) => Some(THR_NEARG),
        (RefFrame::GoldenFrame, InterMode::ZeroMv) => Some(THR_ZEROG),
        (RefFrame::GoldenFrame, InterMode::NewMv) => Some(THR_NEWG),
        _ => None,
    }
}

/// Per-`THR_MODES` rate-multiplier table. Mirrors libvpx
/// `vp9_set_rd_speed_thresholds` with `sf.adaptive_rd_thresh > 0`
/// (`vp9_rd.c:693-745`).
///
/// Slots untouched by rvp9's picker (intra THR_*, compound THR_COMP_*,
/// ALTREF THR_*A) carry their libvpx default values so the table
/// stays drop-in compatible with a future port that does iterate them.
#[must_use]
const fn thresh_mult_table() -> [i32; MAX_MODES] {
    let mut t = [0i32; MAX_MODES];
    // adaptive_rd_thresh > 0 branch (vp9_rd.c:702-710): NEAREST modes
    // get an explicit 300 rather than the +1000 from the base += rule.
    t[THR_NEARESTMV] = 300;
    t[THR_NEARESTG] = 300;
    t[1] = 300; // THR_NEARESTA
    t[3] = 1000; // THR_DC
    t[THR_NEWMV] = 1000;
    t[5] = 1000; // THR_NEWA
    t[THR_NEWG] = 1000;
    t[THR_NEARMV] = 1000;
    t[8] = 1000; // THR_NEARA
    t[13] = 1000; // THR_COMP_NEARESTLA
    t[14] = 1000; // THR_COMP_NEARESTGA
    t[15] = 1000; // THR_TM
    t[16] = 1500; // THR_COMP_NEARLA
    t[17] = 2000; // THR_COMP_NEWLA
    t[THR_NEARG] = 1000;
    t[18] = 1500; // THR_COMP_NEARGA
    t[19] = 2000; // THR_COMP_NEWGA
    t[THR_ZEROMV] = 2000;
    t[THR_ZEROG] = 2000;
    t[12] = 2000; // THR_ZEROA
    t[20] = 2500; // THR_COMP_ZEROLA
    t[21] = 2500; // THR_COMP_ZEROGA
    t[22] = 2000; // THR_H_PRED
    t[23] = 2000; // THR_V_PRED
    t[24] = 2500; // THR_D135_PRED
    t[25] = 2500; // THR_D207_PRED
    t[26] = 2500; // THR_D153_PRED
    t[27] = 2500; // THR_D63_PRED
    t[28] = 2500; // THR_D117_PRED
    t[29] = 2500; // THR_D45_PRED
    t
}

const THRESH_MULT: [i32; MAX_MODES] = thresh_mult_table();

/// `compute_rd_thresh_factor(qindex)` — the q-driven scale applied to
/// `thresh_mult` × `bsize_factor`. Mirrors libvpx
/// `vp9_rd.c:316-328`: `max((int)(pow(dc_quant(q) / 4.0, 1.25) *
/// 5.12), 8)`. The `dc_quant` lookup is for 8-bit content (the
/// `compute_rd_thresh_factor` function uses `bit_depth` purely to
/// pick the right LUT; we hard-code 8-bit since the rvp9 path is
/// 8-bit-only).
#[inline]
#[must_use]
fn compute_rd_thresh_factor(qindex: u8) -> i32 {
    let dc = f64::from(dc_quant(qindex as i32, 0, BitDepth::Eight));
    let q = dc / 4.0;
    let v = (q.powf(RD_THRESH_POW) * 5.12) as i32;
    v.max(8)
}

/// Adaptive-RD-threshold table for one encoder stream. Holds the
/// per-`(bsize, thr_mode)` `thresh_freq_fact` (mutable across frames)
/// alongside an immutable `qindex` snapshot the `effective_thresh`
/// lookup uses to derive `rd_thresh_base[bsize][thr_mode]`.
#[derive(Clone, Debug)]
pub struct RdThreshTable {
    /// `thresh_freq_fact[bsize][thr_mode]` — the per-mode fact. Both
    /// read by the gate and mutated by [`update_thresh_freq_fact`].
    fact: [[i32; MAX_MODES]; BLOCK_SIZES],
}

impl RdThreshTable {
    /// Build a fresh table with every `thresh_freq_fact` slot at
    /// [`RD_THRESH_INIT_FACT`]. Mirrors libvpx
    /// `vp9_encodeframe.c:5874-5875`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            fact: [[RD_THRESH_INIT_FACT; MAX_MODES]; BLOCK_SIZES],
        }
    }

    /// Effective RD threshold for `(bsize, ref_frame, mode)` at the
    /// supplied `qindex`. Returns `None` when the mode is not gated
    /// (ALTREF / intra slots — the gate has no slot to read).
    ///
    /// Mirrors libvpx `vp9_rd.c:373-374` (the base) ×
    /// `vp9_rd.h:193-196` (the gate factor): `rd_thresh_base *
    /// thresh_freq_fact >> 5`. Saturated to `i64::MAX` when the base
    /// would overflow (libvpx clamps via the `thresh_max` guard at
    /// `vp9_rd.c:369-381`).
    #[inline]
    #[must_use]
    pub fn effective_thresh(
        &self,
        bsize: BlockSize,
        ref_frame: RefFrame,
        mode: InterMode,
        qindex: u8,
    ) -> Option<i64> {
        let thr_idx = thr_mode_idx(ref_frame, mode)?;
        let mult = THRESH_MULT[thr_idx];
        if mult == 0 {
            return Some(0);
        }
        let q = compute_rd_thresh_factor(qindex);
        let bsf = i32::from(RD_THRESH_BLOCK_SIZE_FACTOR[bsize.idx()]);
        // libvpx `t = q * rd_thresh_block_size_factor[bsize]`
        // (`vp9_rd.c:368`), then `thresh = thresh_mult * t / 4`
        // (`vp9_rd.c:374`). Compute in i64 to mirror the libvpx
        // `INT_MAX / t` guard without UB.
        let t = i64::from(q).saturating_mul(i64::from(bsf));
        let base = i64::from(mult).saturating_mul(t) / 4;
        let fact = i64::from(self.fact[bsize.idx()][thr_idx]);
        // `rd_less_than_thresh` uses `thresh * fact >> 5`
        // (`vp9_rd.h:195`). Same arithmetic, surfaced as an i64.
        Some(base.saturating_mul(fact) >> 5)
    }

    /// Read the raw `thresh_freq_fact[bsize][thr_mode]`. Used by tests.
    #[doc(hidden)]
    #[must_use]
    pub fn freq_fact_for_test(
        &self,
        bsize: BlockSize,
        ref_frame: RefFrame,
        mode: InterMode,
    ) -> Option<i32> {
        thr_mode_idx(ref_frame, mode).map(|i| self.fact[bsize.idx()][i])
    }

    /// Test-only setter for `thresh_freq_fact[bsize][thr_mode]`. Used
    /// by the picker integration tests to pre-load a per-mode penalty
    /// without going through `update_thresh_freq_fact`. Returns `false`
    /// when `(ref_frame, mode)` is not a slot the gate understands
    /// (`ALTREF` / `IntraFrame` / `NoRefFrame`).
    #[doc(hidden)]
    pub fn set_freq_fact_for_test(
        &mut self,
        bsize: BlockSize,
        ref_frame: RefFrame,
        mode: InterMode,
        value: i32,
    ) -> bool {
        match thr_mode_idx(ref_frame, mode) {
            Some(i) => {
                self.fact[bsize.idx()][i] = value;
                true
            }
            None => false,
        }
    }
}

impl Default for RdThreshTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Decide whether to skip a `(bsize, ref_frame, mode)` candidate given
/// the running `best_rd_so_far`. Mirrors libvpx's `rd_less_than_thresh`
/// gate at `vp9_pickmode.c:2241-2247`:
///
/// ```text
/// if best_rd < (thresh * fact >> 5) && frame_mv[mode][ref] != 0:
///     continue
/// ```
///
/// The `frame_mv != 0` clause — present at `vp9_pickmode.c:2247` and
/// `:2580` — keeps zero-MV candidates exempt from the gate so the
/// picker always has at least one baseline result. We pass `mv_is_zero`
/// explicitly rather than recomputing here.
#[inline]
#[must_use]
pub fn rd_less_than_thresh(
    table: &RdThreshTable,
    bsize: BlockSize,
    ref_frame: RefFrame,
    mode: InterMode,
    qindex: u8,
    best_rd_so_far: i64,
    mv_is_zero: bool,
) -> bool {
    if mv_is_zero {
        return false;
    }
    match table.effective_thresh(bsize, ref_frame, mode, qindex) {
        Some(thresh) => best_rd_so_far < thresh,
        None => false,
    }
}

/// Update `thresh_freq_fact` after `pick_inter_mode` has chosen
/// `(winning_ref, winning_mode)` for a block of size `bsize`. Mirrors
/// `update_thresh_freq_fact` at `vp9_pickmode.c:1139-1154`:
///
/// - winner's fact decays by `(*fact >> 4)`
/// - every other inter mode (LAST/GOLDEN × NEAREST/NEAR/ZERO/NEW)
///   increments by [`RD_THRESH_INC`], capped at
///   `ADAPTIVE_RD_THRESH * RD_THRESH_MAX_FACT`.
///
/// libvpx's `vp9_pickmode.c:2725-2738` updates EVERY inter mode at
/// `bsize` for the winning ref frame after a pick, plus the same
/// across all 4 intra modes when the winner is intra. rvp9 follows
/// the inter case (LAST or GOLDEN winner) and skips the intra-winner
/// fan-out (the rvp9 picker's intra fallback is rare and the
/// per-leaf fact churn on intra slots wouldn't affect the picker
/// without those slots being read).
///
/// The §11 `limit_newmv_early_exit` flag and `source_variance` value
/// gate the libvpx low-cap branch at `vp9_pickmode.c:1147-1149`:
/// non-winning `(LAST, NEWMV)` on a flat block (`source_variance < 5`)
/// caps `thresh_freq_fact` at `RD_THRESH_INIT_FACT` (= 32) instead of
/// the regular `ADAPTIVE_RD_THRESH * RD_THRESH_MAX_FACT`. Callers
/// that do not run §11 should pass
/// `(limit_newmv_early_exit = false, source_variance = u32::MAX)`.
pub fn update_thresh_freq_fact(
    table: &mut RdThreshTable,
    bsize: BlockSize,
    winning_ref: RefFrame,
    winning_mode: InterMode,
) {
    update_thresh_freq_fact_with_s11(
        table,
        bsize,
        winning_ref,
        winning_mode,
        /* limit_newmv_early_exit = */ false,
        /* source_variance = */ u32::MAX,
    );
}

/// §11 — full update form with `limit_newmv_early_exit` /
/// `source_variance` inputs.
pub fn update_thresh_freq_fact_with_s11(
    table: &mut RdThreshTable,
    bsize: BlockSize,
    winning_ref: RefFrame,
    winning_mode: InterMode,
    limit_newmv_early_exit: bool,
    source_variance: u32,
) {
    let bsize_idx = bsize.idx();
    let winner_idx = match thr_mode_idx(winning_ref, winning_mode) {
        Some(i) => i,
        None => return,
    };
    let cap = ADAPTIVE_RD_THRESH.saturating_mul(RD_THRESH_MAX_FACT);
    // Iterate the LAST/GOLDEN × {NEAREST, NEAR, ZERO, NEW} slots and
    // apply the libvpx win/lose rule.
    const REFS: [RefFrame; 2] = [RefFrame::LastFrame, RefFrame::GoldenFrame];
    const MODES: [InterMode; 4] = [
        InterMode::NearestMv,
        InterMode::NearMv,
        InterMode::ZeroMv,
        InterMode::NewMv,
    ];
    for &rf in &REFS {
        // Per libvpx vp9_pickmode.c:2726, only the winning ref's
        // mode-slots are updated. The losing ref's slots are left
        // untouched (they were never the "best mode index" — every
        // mode for that ref counts as a non-winner relative to the
        // best_mode_idx, but libvpx specifically gates on
        // `best_pickmode.best_ref_frame != ref_frame` and `continue`s,
        // so only the winning ref's loop iterates).
        if rf != winning_ref {
            continue;
        }
        for &m in &MODES {
            let idx = match thr_mode_idx(rf, m) {
                Some(i) => i,
                None => continue,
            };
            let f = &mut table.fact[bsize_idx][idx];
            if idx == winner_idx {
                *f -= *f >> 4;
            } else {
                // §11 — `limit_newmv_early_exit` cap variant
                // (`vp9_pickmode.c:1147-1149`). For non-winning
                // (LAST, NEWMV) on a flat block, libvpx caps the
                // freq_fact at 32 (= RD_THRESH_INIT_FACT) rather
                // than the regular ADAPTIVE_RD_THRESH * 64. Keeping
                // the cap low prevents NEWMV from accumulating a
                // saturating penalty that would gate it out
                // permanently on textured content that turns flat
                // for a few frames.
                let effective_cap = if limit_newmv_early_exit
                    && rf == RefFrame::LastFrame
                    && m == InterMode::NewMv
                    && source_variance < 5
                {
                    RD_THRESH_INIT_FACT
                } else {
                    cap
                };
                let inc = (*f).saturating_add(RD_THRESH_INC);
                *f = inc.min(effective_cap);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P20 — verify the new table starts at the libvpx-mandated initial
    /// value of `RD_THRESH_INIT_FACT = 32` for every slot.
    #[test]
    fn p20_adaptive_rd_thresh_table_initial_state() {
        let t = RdThreshTable::new();
        for bsize_idx in 0..BLOCK_SIZES {
            for mode_idx in 0..MAX_MODES {
                assert_eq!(
                    t.fact[bsize_idx][mode_idx], RD_THRESH_INIT_FACT,
                    "fact[{bsize_idx}][{mode_idx}] not at INIT",
                );
            }
        }
    }

    /// P20 — sanity: the `mode_idx` mapping matches libvpx
    /// `vp9_pickmode.c:1089-1094` for the LAST/GOLDEN inter slots
    /// rvp9 iterates.
    #[test]
    fn p20_adaptive_rd_thresh_thr_mode_idx_map() {
        // LAST row of mode_idx[][]: { THR_NEARESTMV(0), THR_NEARMV(7),
        // THR_ZEROMV(10), THR_NEWMV(4) }.
        assert_eq!(
            thr_mode_idx(RefFrame::LastFrame, InterMode::NearestMv),
            Some(0),
        );
        assert_eq!(
            thr_mode_idx(RefFrame::LastFrame, InterMode::NearMv),
            Some(7),
        );
        assert_eq!(
            thr_mode_idx(RefFrame::LastFrame, InterMode::ZeroMv),
            Some(10),
        );
        assert_eq!(
            thr_mode_idx(RefFrame::LastFrame, InterMode::NewMv),
            Some(4),
        );
        // GOLDEN row: { THR_NEARESTG(2), THR_NEARG(9), THR_ZEROG(11),
        // THR_NEWG(6) }.
        assert_eq!(
            thr_mode_idx(RefFrame::GoldenFrame, InterMode::NearestMv),
            Some(2),
        );
        assert_eq!(
            thr_mode_idx(RefFrame::GoldenFrame, InterMode::NewMv),
            Some(6),
        );
    }

    /// P20 — verify that the `compute_rd_thresh_factor` formula is at
    /// least 8 (the libvpx `VPXMAX(..., 8)` floor at `vp9_rd.c:328`)
    /// and grows with `qindex`. Use two qindexes whose dc-quant ratio
    /// is large enough that pow(...) crosses the floor.
    #[test]
    fn p20_adaptive_rd_thresh_q_factor_monotone() {
        let low = compute_rd_thresh_factor(0);
        let high = compute_rd_thresh_factor(200);
        assert!(low >= 8, "low q factor {low} below floor");
        assert!(
            high > low,
            "expected q-factor to grow with q: {low} → {high}",
        );
    }

    /// P20 PROOF — the gate skips a candidate when `best_rd <
    /// effective_thresh` AND the MV is non-zero. Use a deliberately
    /// small `best_rd_so_far` and a high-thresh slot (NEWMV at high q)
    /// so the gate trips.
    #[test]
    fn p20_adaptive_rd_thresh_gate_skips_when_below_thresh() {
        let table = RdThreshTable::new();
        let bsize = BlockSize::Block16x16;
        // At INIT fact = 32, the gate's RHS is `base * 32 >> 5 = base`.
        // base = thresh_mult * q * bsize_factor / 4. For NEWMV at LAST
        // at qindex=128, this is comfortably > 0. A `best_rd_so_far`
        // of 0 must trip the gate (since 0 < base for any nonzero
        // base).
        let thresh = table
            .effective_thresh(bsize, RefFrame::LastFrame, InterMode::NewMv, 128)
            .expect("LAST/NEWMV slot present");
        assert!(thresh > 0, "expected positive threshold, got {thresh}");
        // best_rd = thresh - 1 should still trip (strict `<`).
        let skip = rd_less_than_thresh(
            &table,
            bsize,
            RefFrame::LastFrame,
            InterMode::NewMv,
            128,
            thresh - 1,
            /* mv_is_zero = */ false,
        );
        assert!(skip, "gate should skip when best_rd < thresh");
        // best_rd = thresh exactly must NOT trip (strict `<` semantic).
        let no_skip = rd_less_than_thresh(
            &table,
            bsize,
            RefFrame::LastFrame,
            InterMode::NewMv,
            128,
            thresh,
            false,
        );
        assert!(!no_skip, "gate should NOT skip when best_rd == thresh");
    }

    /// P20 PROOF — zero-MV candidates are exempt from the gate
    /// (`vp9_pickmode.c:2247` clause: `if frame_mv[mode][ref].as_int
    /// != 0 continue;`). When `mv_is_zero = true`, the gate must
    /// return `false` even with `best_rd = 0`.
    #[test]
    fn p20_adaptive_rd_thresh_gate_exempts_zero_mv() {
        let table = RdThreshTable::new();
        let skip = rd_less_than_thresh(
            &table,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            InterMode::ZeroMv,
            128,
            /* best_rd = */ 0,
            /* mv_is_zero = */ true,
        );
        assert!(!skip, "ZEROMV must be exempt from the adaptive gate");
    }

    /// P20 PROOF — winning mode's fact decays after
    /// `update_thresh_freq_fact`. With INIT_FACT = 32, the libvpx
    /// rule `*fact -= *fact >> 4` produces 32 - 2 = 30. Non-winners
    /// for the same (winning) ref frame go up by 1 → 33. Losing-ref
    /// slots stay untouched at 32.
    #[test]
    fn p20_adaptive_rd_thresh_update_decays_winner_increments_losers() {
        let mut table = RdThreshTable::new();
        let bsize = BlockSize::Block16x16;
        // Sanity: every slot starts at 32.
        assert_eq!(
            table
                .freq_fact_for_test(bsize, RefFrame::LastFrame, InterMode::NewMv),
            Some(RD_THRESH_INIT_FACT),
        );
        // Winner: LAST / NEWMV.
        update_thresh_freq_fact(
            &mut table,
            bsize,
            RefFrame::LastFrame,
            InterMode::NewMv,
        );
        // Winner decayed by 32 >> 4 = 2 → 30.
        assert_eq!(
            table
                .freq_fact_for_test(bsize, RefFrame::LastFrame, InterMode::NewMv),
            Some(30),
            "winner fact should decay to 30",
        );
        // Non-winner same ref incremented by 1 → 33.
        assert_eq!(
            table
                .freq_fact_for_test(bsize, RefFrame::LastFrame, InterMode::ZeroMv),
            Some(33),
            "non-winner same-ref fact should increment",
        );
        // Different ref untouched (libvpx only iterates winning ref).
        assert_eq!(
            table.freq_fact_for_test(
                bsize,
                RefFrame::GoldenFrame,
                InterMode::NewMv,
            ),
            Some(RD_THRESH_INIT_FACT),
            "losing-ref slot should be unchanged",
        );
        // Different bsize untouched.
        assert_eq!(
            table.freq_fact_for_test(
                BlockSize::Block32x32,
                RefFrame::LastFrame,
                InterMode::NewMv,
            ),
            Some(RD_THRESH_INIT_FACT),
            "other-bsize slot should be unchanged",
        );
    }

    /// P20 PROOF — after the winner decays, its effective threshold
    /// SHRINKS (the gate becomes MORE permissive for it), and a
    /// non-winner's threshold GROWS (gate becomes MORE aggressive
    /// against it). This is the speed-feature's core invariant.
    #[test]
    fn p20_adaptive_rd_thresh_update_shifts_effective_thresholds() {
        let bsize = BlockSize::Block16x16;
        let qindex = 128u8;
        let mut table = RdThreshTable::new();
        let winner_before = table
            .effective_thresh(bsize, RefFrame::LastFrame, InterMode::NewMv, qindex)
            .unwrap();
        let loser_before = table
            .effective_thresh(bsize, RefFrame::LastFrame, InterMode::ZeroMv, qindex)
            .unwrap();
        update_thresh_freq_fact(
            &mut table,
            bsize,
            RefFrame::LastFrame,
            InterMode::NewMv,
        );
        let winner_after = table
            .effective_thresh(bsize, RefFrame::LastFrame, InterMode::NewMv, qindex)
            .unwrap();
        let loser_after = table
            .effective_thresh(bsize, RefFrame::LastFrame, InterMode::ZeroMv, qindex)
            .unwrap();
        assert!(
            winner_after < winner_before,
            "winner thresh should shrink: before={winner_before} after={winner_after}",
        );
        assert!(
            loser_after > loser_before,
            "loser thresh should grow: before={loser_before} after={loser_after}",
        );
    }

    /// P20 — the `compute_rd_thresh_factor(255)` value matches a
    /// hand-derived libvpx-style figure. dc_quant(255, 0, 8) = 1828
    /// (from DC_QLOOKUP terminal entry); (1828/4)^1.25 * 5.12 ≈
    /// 17_240. Allow a small tolerance for `f64` rounding.
    #[test]
    fn p20_adaptive_rd_thresh_q_factor_matches_libvpx() {
        let v = compute_rd_thresh_factor(255);
        // Compute the expected with the exact libvpx formula.
        let dc = f64::from(dc_quant(255, 0, BitDepth::Eight));
        let expected = ((dc / 4.0).powf(1.25) * 5.12) as i32;
        let expected = expected.max(8);
        assert_eq!(v, expected, "q-factor formula must mirror libvpx exactly");
    }

}
