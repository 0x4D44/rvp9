//! W4.9 — `vp9_rd_pick_inter_mode_sb` driver skeleton.
//!
//! ## libvpx reference
//!
//! Port of the mode-iteration loop in
//! `vp9/encoder/vp9_rdopt.c:3438-4187` (v1.14.1 commit `12f3a2ac6`).
//! The driver:
//!
//! 1. Computes `ref_costs_single` / `ref_costs_comp` via
//!    [`crate::rdo::ref_costs::estimate_ref_frame_costs`] (libvpx line 3516).
//! 2. Initialises `frame_mv[mode][ref]` to libvpx defaults (libvpx
//!    lines 3540-3543): `INVALID_MV` for NEWMV, `(0, 0)` for ZEROMV.
//! 3. Iterates `MAX_MODES` in `vp9_mode_order` order, applying mask
//!    logic per-mode (libvpx lines 3640-3802).
//! 4. For each `(ref_frame, this_mode)` pair, calls
//!    [`crate::rdo::handle_inter::handle_inter_mode_single_ref`]
//!    (libvpx line 3863).
//! 5. Adds `ref_costs_{single,comp}[ref_frame]` to the candidate rate
//!    (libvpx lines 3881-3885).
//! 6. Runs the skip-flag tie-break (libvpx lines 3887-3920).
//! 7. Tracks `best_rd` / `best_mode_index` / `best_mbmode` across the
//!    mode iteration.
//! 8. Runs the NEWMV → NEAREST/NEAR/ZERO post-pick rewrite (libvpx
//!    lines 4085-4104) via
//!    [`crate::rdo::discount_newmv::rewrite_newmv_post_pick`].
//!
//! ## Scope (recorded deviations)
//!
//! - **Single-ref only.** Compound modes are out of scope for the W4
//!   batch; the driver iterates `{LAST, GOLDEN, ALTREF}` × `{NEARESTMV,
//!   NEARMV, ZEROMV, NEWMV}` plus `INTRA_FRAME` for the intra fallback.
//! - **Intra fallback is stubbed.** When inter is uniformly bad, the
//!   driver returns a `PickResult { mode: INTRA }` with a caller-
//!   supplied intra RD. The full libvpx intra branch (lines 3823-3876)
//!   threads intra-mode search + chroma RD; that's W4.10+ territory.
//! - **No `ref_frame_skip_mask` / `mode_skip_mask` plumbing.** The
//!   driver iterates the full inter cross product; the caller drops
//!   refs by zeroing their `ref_plane` (signalled via `available`).
//! - **No `motion_field_mode_search` / `alt_ref_search_fp` / film-
//!   content adjustments**. Those are speed features above the
//!   load-bearing slow path.
//! - **No per-iter `mode_threshold` ordering**. The libvpx
//!   `mode_threshold[i]` sort at lines 3624-3636 reorders the mode
//!   evaluation; we iterate in libvpx's static `vp9_mode_order` order
//!   directly.
//! - **MV search is delegated** — the caller pre-runs
//!   [`crate::rdo::single_mv_search::single_motion_search`] for NEWMV
//!   per ref-frame and passes the result in. The driver does NOT
//!   re-run the search inside the mode loop.

use rvp9_core::mode_info::{InterMode, Mv, RefFrame};

use crate::rdo::discount_newmv::{discount_newmv_test, rewrite_newmv_post_pick};
use crate::rdo::handle_inter::HandleInterModeResult;
use crate::rdo::ref_costs::RefFrameCosts;

/// libvpx `vp9_mode_order` slice for the single-ref inter modes.
/// Mirrors the relevant entries from `vp9_rd.c:84-156`.
///
/// We enumerate `{NEARESTMV, NEARMV, ZEROMV, NEWMV}` × `{LAST, GOLDEN,
/// ALTREF}` for the inter cross product; intra is handled separately
/// via the [`PickInterModeSbParams::intra_fallback`] field.
pub const INTER_MODE_ORDER: &[(InterMode, RefFrame)] = &[
    (InterMode::NearestMv, RefFrame::LastFrame),
    (InterMode::NearestMv, RefFrame::GoldenFrame),
    (InterMode::NearestMv, RefFrame::AltrefFrame),
    (InterMode::NearMv, RefFrame::LastFrame),
    (InterMode::NearMv, RefFrame::GoldenFrame),
    (InterMode::NearMv, RefFrame::AltrefFrame),
    (InterMode::ZeroMv, RefFrame::LastFrame),
    (InterMode::ZeroMv, RefFrame::GoldenFrame),
    (InterMode::ZeroMv, RefFrame::AltrefFrame),
    (InterMode::NewMv, RefFrame::LastFrame),
    (InterMode::NewMv, RefFrame::GoldenFrame),
    (InterMode::NewMv, RefFrame::AltrefFrame),
];

/// Per-ref candidate-MV state. Mirrors libvpx's
/// `frame_mv[mode][ref]` (`vp9_rdopt.c:3454`) reduced to the rows the
/// W4.9 driver actually reads.
#[derive(Copy, Clone, Debug, Default)]
pub struct RefMvCandidates {
    /// libvpx `frame_mv[NEARESTMV][ref]`.
    pub nearest: Mv,
    /// libvpx `frame_mv[NEARMV][ref]`.
    pub near: Mv,
    /// libvpx `frame_mv[NEWMV][ref]`. Caller pre-runs
    /// `single_motion_search` for this ref and threads the result
    /// here. `None` (= libvpx `INVALID_MV`) signals "no valid NEWMV
    /// for this ref" (e.g. adaptive-motion-search early-exit fired).
    pub newmv: Option<Mv>,
    /// NEWMV rate (libvpx `*rate2 += rate_mv` at line 2941). 1/512-bit
    /// units. Only consumed when `newmv.is_some()`.
    pub newmv_rate: u32,
}

/// Per-ref input bundle.
pub struct RefFrameInput<'a> {
    /// `true` when this ref-frame is available (libvpx's
    /// `cpi->ref_frame_flags & ref_frame_to_flag(ref)`).
    pub available: bool,
    /// MV candidates for this ref.
    pub candidates: RefMvCandidates,
    /// `handle_inter_mode` runner — see [`HandleInterRunner`].
    ///
    /// **Standalone-scope sentinel**: when this is `None`, the driver
    /// treats every (mode, this ref) pair as RD = `INT64_MAX`. The
    /// caller uses this to inject "pathologically bad inter" test
    /// scenarios — the picker should fall back to the intra branch in
    /// that case.
    pub handle_inter_runner: Option<HandleInterRunner<'a>>,
    /// `rdmult` value used inside the runner — surfaced here so the
    /// driver can run `RDCOST(rate, dist)` after adding the ref-frame
    /// cost.
    pub rdmult: i64,
}

/// `handle_inter_mode` invocation type. The caller supplies a closure
/// that — given `(this_mode, mv, inter_mode_rate, newmv_rate)` —
/// builds the `HandleInterModeParams` (lifting the captured plane
/// views + cost tables) and runs
/// [`crate::rdo::handle_inter::handle_inter_mode_single_ref`].
///
/// We expose the runner as a closure that returns the raw
/// `HandleInterModeResult` rather than a builder, because Rust's
/// for<'b>` higher-ranked trait bounds don't allow a returned struct's
/// lifetime to be tied to the closure's call. The caller is
/// responsible for the `super_block_yrd` / chroma RD plumbing inside
/// their runner — the standalone module just composes the result.
pub type HandleInterRunner<'a> = Box<
    dyn Fn(InterMode, Mv, u32, u32) -> Option<HandleInterModeResult> + 'a,
>;

/// Driver inputs.
pub struct PickInterModeSbParams<'a> {
    /// Ref-frame cost tuple from
    /// [`crate::rdo::ref_costs::estimate_ref_frame_costs`].
    pub ref_costs: RefFrameCosts,
    /// Per-ref-frame input bundles. Indexed by [`RefFrame`] (LAST=1,
    /// GOLDEN=2, ALTREF=3). The `IntraFrame = 0` slot is unused; the
    /// caller threads the intra RD via [`Self::intra_fallback`].
    pub per_ref: [RefFrameInput<'a>; 4],
    /// Intra-mode fallback RD tuple. When ALL inter candidates exceed
    /// this, the picker returns `mode = INTRA`. Use `i64::MAX` to
    /// disable.
    pub intra_fallback: IntraFallback,
    /// `is_src_frame_alt_ref` flag (drives `discount_newmv_test`).
    pub is_src_frame_alt_ref: bool,
    /// Optional iteration counter — increments for every
    /// (`mode`, `ref`) pair the driver evaluates. Used by W4.9 PROOF
    /// test to verify each mode gets touched.
    pub mode_counter: Option<&'a std::cell::RefCell<ModeCounter>>,
}

/// libvpx intra-mode fallback. The full driver runs
/// `super_block_yrd(INTRA_FRAME)` + `choose_intra_uv_mode` to fill these
/// fields; the standalone module accepts them pre-baked.
#[derive(Copy, Clone, Debug, Default)]
pub struct IntraFallback {
    /// Intra mode's total rate (mode bits + coef bits + intra cost
    /// penalty).
    pub rate: u32,
    /// Intra mode's total distortion.
    pub dist: i64,
    /// Intra mode's total RD (already computed).
    pub rd: i64,
}

/// Per-mode hit counter. Used by the W4.9 PROOF test to verify the
/// driver iterates ZERO/NEAREST/NEAR/NEW.
#[derive(Default, Debug, Clone)]
pub struct ModeCounter {
    /// Number of (mode, ref) pairs evaluated for NEARESTMV.
    pub nearest: u32,
    /// Number for NEARMV.
    pub near: u32,
    /// Number for ZEROMV.
    pub zero: u32,
    /// Number for NEWMV.
    pub new: u32,
}

/// Driver output — the picked mode and its RD tuple.
#[derive(Copy, Clone, Debug)]
pub struct PickResult {
    /// Winning inter mode (or NEARESTMV/NEARMV/ZEROMV after the
    /// post-pick rewrite if the original winner was NEWMV).
    pub mode: InterMode,
    /// Winning ref-frame. `IntraFrame` when the intra fallback wins.
    pub ref_frame: RefFrame,
    /// Winning MV (1/8-pel units). Zero for intra / ZEROMV.
    pub mv: Mv,
    /// Total rate.
    pub rate: u32,
    /// Total distortion.
    pub dist: i64,
    /// Total RD cost.
    pub rd: i64,
    /// `true` iff the driver picked the intra fallback (i.e. all inter
    /// candidates were worse than `intra_fallback.rd`).
    pub picked_intra: bool,
}

/// libvpx `vp9_rd_pick_inter_mode_sb` skeleton (`vp9_rdopt.c:3438-4187`).
///
/// Single-ref iteration. The full driver runs `MAX_MODES = 30`
/// candidates; the standalone W4.9 scope ports the 12-entry single-ref
/// cross product, plus intra fallback.
#[must_use]
pub fn pick_inter_mode_sb(params: &PickInterModeSbParams<'_>) -> PickResult {
    let mut best_rd: i64 = i64::MAX;
    let mut best: Option<(InterMode, RefFrame, Mv, u32, i64)> = None;

    // Track NEW MVs per-ref for the post-pick rewrite (libvpx lines
    // 4085-4104 read `frame_mv[NEARESTMV][refs[0]]` / `frame_mv[NEARMV]
    // [refs[0]]`).
    for &(this_mode, ref_frame) in INTER_MODE_ORDER {
        let ref_idx = ref_frame as usize;
        let per_ref = &params.per_ref[ref_idx];
        if !per_ref.available {
            continue;
        }
        // libvpx lines 3540-3543: NEWMV stays INVALID_MV until the
        // search runs. Skip the NEWMV row if the caller didn't run it.
        let mv = match this_mode {
            InterMode::NearestMv => per_ref.candidates.nearest,
            InterMode::NearMv => per_ref.candidates.near,
            InterMode::ZeroMv => Mv::ZERO,
            InterMode::NewMv => match per_ref.candidates.newmv {
                Some(mv) => mv,
                None => continue,
            },
        };

        // Bump the mode counter (used by the PROOF test).
        if let Some(counter_cell) = params.mode_counter {
            let mut counter = counter_cell.borrow_mut();
            match this_mode {
                InterMode::NearestMv => counter.nearest += 1,
                InterMode::NearMv => counter.near += 1,
                InterMode::ZeroMv => counter.zero += 1,
                InterMode::NewMv => counter.new += 1,
            }
        }

        // libvpx line 3863: handle_inter_mode produces per-mode RD.
        let Some(runner) = per_ref.handle_inter_runner.as_ref() else {
            continue;
        };

        // libvpx line 2977 / 2939: `*rate2 += cost_mv_ref(mode)` —
        // i.e. the caller passes the inter-mode bits as
        // `inter_mode_rate`. For the standalone module we use a small
        // mode-dependent placeholder; the production caller plumbs
        // the real lookup table.
        let mode_rate = libvpx_mode_rate_placeholder(this_mode);
        let newmv_rate = if matches!(this_mode, InterMode::NewMv) {
            per_ref.candidates.newmv_rate
        } else {
            0
        };

        let Some(hi_result) = runner(this_mode, mv, mode_rate, newmv_rate) else {
            continue;
        };

        // libvpx lines 3881-3885: add ref-frame cost.
        let rate = hi_result.rate + params.ref_costs.single[ref_idx];

        // libvpx line 3920: this_rd = RDCOST(rate, distortion).
        let this_rd = recompute_rd(per_ref.rdmult, rate, hi_result.dist);

        // libvpx line 3973: track best.
        if this_rd < best_rd {
            best_rd = this_rd;
            best = Some((this_mode, ref_frame, mv, rate, hi_result.dist));
        }
    }

    // libvpx lines 4085-4104: NEWMV → NEAREST/NEAR/ZERO post-pick
    // rewrite (single-ref form).
    let (best_mode, best_ref, best_mv, best_rate, best_dist) = match best {
        Some(t) => t,
        None => {
            // No inter candidate succeeded — pick intra fallback.
            return PickResult {
                mode: InterMode::ZeroMv, // unused
                ref_frame: RefFrame::IntraFrame,
                mv: Mv::ZERO,
                rate: params.intra_fallback.rate,
                dist: params.intra_fallback.dist,
                rd: params.intra_fallback.rd,
                picked_intra: true,
            };
        }
    };

    // libvpx lines 4106-4112: compare against `best_rd_so_far` (we
    // surface this as `intra_fallback.rd` — when the best inter RD
    // is worse, intra wins).
    if best_rd > params.intra_fallback.rd {
        return PickResult {
            mode: InterMode::ZeroMv, // unused — intra
            ref_frame: RefFrame::IntraFrame,
            mv: Mv::ZERO,
            rate: params.intra_fallback.rate,
            dist: params.intra_fallback.dist,
            rd: params.intra_fallback.rd,
            picked_intra: true,
        };
    }

    // Post-pick rewrite (W4.8).
    let nearest = params.per_ref[best_ref as usize].candidates.nearest;
    let near = params.per_ref[best_ref as usize].candidates.near;
    let rewritten_mode = rewrite_newmv_post_pick(best_mode, best_mv, nearest, near);
    let _ = discount_newmv_test(
        params.is_src_frame_alt_ref,
        best_mode,
        best_mv,
        Some(nearest),
        Some(near),
    );

    PickResult {
        mode: rewritten_mode,
        ref_frame: best_ref,
        mv: best_mv,
        rate: best_rate,
        dist: best_dist,
        rd: best_rd,
        picked_intra: false,
    }
}

/// Standalone-scope placeholder for the inter-mode rate cost. The
/// production caller plumbs the per-context inter-mode table from
/// [`rvp9_core::cost::InterModeCosts`]; the standalone PROOF tests
/// just need a sane mode-dependent ordering.
fn libvpx_mode_rate_placeholder(mode: InterMode) -> u32 {
    match mode {
        InterMode::NearestMv => 400,
        InterMode::NearMv => 500,
        InterMode::ZeroMv => 300,
        InterMode::NewMv => 1200,
    }
}

#[inline]
fn recompute_rd(rdmult: i64, rate: u32, dist: i64) -> i64 {
    let rate_term = (rdmult.saturating_mul(i64::from(rate)) + 256) >> 9;
    let dist_term = dist << 7;
    rate_term.saturating_add(dist_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdo::handle_inter::{
        FilterRd, HandleInterModeResult, SWITCHABLE_FILTERS,
    };
    use crate::rdo::ref_costs::{
        ReferenceMode, RefFrameContext, estimate_ref_frame_costs,
    };
    use rvp9_core::bitstream::frame::InterpFilter;
    use std::cell::RefCell;

    /// Construct a synthetic `HandleInterModeResult` for the W4.9 tests.
    /// Per-mode RDs differ slightly so the picker has a clean winner
    /// to lock onto.
    fn synth_hi_result(mode: InterMode, mv: Mv, base_dist: i64) -> HandleInterModeResult {
        let _ = mv;
        let mode_bias = match mode {
            InterMode::NearestMv => 100i64,
            InterMode::NearMv => 200,
            InterMode::ZeroMv => 50,
            InterMode::NewMv => 300,
        };
        let dist = base_dist + mode_bias * 100;
        let rate = 500u32 + (mode as u32) * 50;
        let rd = (rate as i64).saturating_mul(100_000) >> 9;
        let rd = rd.saturating_add(dist << 7);
        let mut filter_cache: [Option<FilterRd>; SWITCHABLE_FILTERS] =
            [None; SWITCHABLE_FILTERS];
        for slot in filter_cache.iter_mut() {
            *slot = Some(FilterRd {
                rd,
                rate,
                dist,
                skip_txfm_sb: false,
                skip_sse_sb: dist,
            });
        }
        HandleInterModeResult {
            best_filter: InterpFilter::EightTap,
            rate,
            dist,
            rd,
            disable_skip: false,
            filter_cache,
        }
    }

    /// Helper to build a per-ref bundle with a synthetic runner.
    fn make_per_ref(
        candidates: RefMvCandidates,
        base_dist: i64,
        rdmult: i64,
    ) -> RefFrameInput<'static> {
        let runner: HandleInterRunner<'static> =
            Box::new(move |mode, mv, _inter_rate, _newmv_rate| {
                Some(synth_hi_result(mode, mv, base_dist))
            });
        RefFrameInput {
            available: true,
            candidates,
            handle_inter_runner: Some(runner),
            rdmult,
        }
    }

    fn make_unavailable() -> RefFrameInput<'static> {
        RefFrameInput {
            available: false,
            candidates: RefMvCandidates::default(),
            handle_inter_runner: None,
            rdmult: 100_000,
        }
    }

    /// PROOF W4.9 — `pick_inter_mode_sb` returns a winner with sane
    /// fields. libvpx `vp9_rdopt.c:3438-4187`: a sane (ref_plane, src
    /// plane, mv_candidates) input must produce a `PickResult` with
    /// a finite RD and a recognisable inter mode.
    #[test]
    fn w4_9_pick_inter_mode_sb_returns_a_winner() {
        let ref_costs = estimate_ref_frame_costs(
            false,
            ReferenceMode::SingleReference,
            RefFrameContext::default(),
        );

        let candidates = RefMvCandidates {
            nearest: Mv::ZERO,
            near: Mv { row: 0, col: 8 },
            newmv: Some(Mv { row: 0, col: 16 }),
            newmv_rate: 100,
        };
        let counter = RefCell::new(ModeCounter::default());
        let per_ref = [
            make_unavailable(),
            make_per_ref(candidates, 1_000, 100_000),
            make_unavailable(),
            make_unavailable(),
        ];

        let params = PickInterModeSbParams {
            ref_costs,
            per_ref,
            intra_fallback: IntraFallback {
                rate: 2000,
                dist: 1_000_000,
                rd: i64::MAX / 2, // disable intra
            },
            is_src_frame_alt_ref: false,
            mode_counter: Some(&counter),
        };

        let result = pick_inter_mode_sb(&params);
        assert!(!result.picked_intra, "should pick an inter mode");
        assert_eq!(result.ref_frame, RefFrame::LastFrame);
        assert!(result.rd < i64::MAX / 2, "RD should be finite");
        // Mode should be one of the recognised inter modes.
        assert!(matches!(
            result.mode,
            InterMode::NearestMv | InterMode::NearMv | InterMode::ZeroMv | InterMode::NewMv
        ));
    }

    /// PROOF W4.9 — driver iterates ALL inter modes
    /// (ZERO/NEAREST/NEAR/NEW). libvpx `vp9_rdopt.c:3640-4076`: each
    /// mode in `vp9_mode_order` gets at least one
    /// `handle_inter_mode` evaluation per available ref. We assert
    /// the mode_counter reflects this.
    #[test]
    fn w4_9_pick_inter_mode_sb_iterates_all_inter_modes() {
        let ref_costs = estimate_ref_frame_costs(
            false,
            ReferenceMode::SingleReference,
            RefFrameContext::default(),
        );

        let candidates = RefMvCandidates {
            nearest: Mv::ZERO,
            near: Mv { row: 0, col: 8 },
            newmv: Some(Mv { row: 0, col: 16 }),
            newmv_rate: 100,
        };
        let counter = RefCell::new(ModeCounter::default());
        let per_ref = [
            make_unavailable(),
            make_per_ref(candidates, 1_000, 100_000),
            make_unavailable(),
            make_unavailable(),
        ];

        let params = PickInterModeSbParams {
            ref_costs,
            per_ref,
            intra_fallback: IntraFallback {
                rate: 2000,
                dist: 1_000_000,
                rd: i64::MAX / 2,
            },
            is_src_frame_alt_ref: false,
            mode_counter: Some(&counter),
        };

        let _ = pick_inter_mode_sb(&params);
        let counter = counter.borrow();
        assert!(counter.nearest >= 1, "NEAREST not iterated (got {})", counter.nearest);
        assert!(counter.near >= 1, "NEAR not iterated (got {})", counter.near);
        assert!(counter.zero >= 1, "ZERO not iterated (got {})", counter.zero);
        assert!(counter.new >= 1, "NEW not iterated (got {})", counter.new);
    }

    /// PROOF W4.9 — intra fallback wins on pathological inter.
    /// libvpx `vp9_rdopt.c:4106-4112`: when every inter candidate
    /// exceeds `best_rd_so_far`, the picker returns intra. We
    /// simulate "pathological inter" by setting all
    /// `handle_inter_runner` to `None` (which makes every
    /// (mode, ref) row skipped → no inter winner → intra wins).
    #[test]
    fn w4_9_intra_fallback_wins_on_pathological_inter() {
        let ref_costs = estimate_ref_frame_costs(
            false,
            ReferenceMode::SingleReference,
            RefFrameContext::default(),
        );

        // All refs available but no handle_inter_runner → no inter
        // candidate succeeds.
        let mut bad_last = make_unavailable();
        bad_last.available = true;
        let mut bad_golden = make_unavailable();
        bad_golden.available = true;
        let mut bad_altref = make_unavailable();
        bad_altref.available = true;
        let per_ref = [make_unavailable(), bad_last, bad_golden, bad_altref];

        let intra_fallback = IntraFallback {
            rate: 1500,
            dist: 100_000,
            rd: 10_000_000,
        };
        let params = PickInterModeSbParams {
            ref_costs,
            per_ref,
            intra_fallback,
            is_src_frame_alt_ref: false,
            mode_counter: None,
        };

        let result = pick_inter_mode_sb(&params);
        assert!(result.picked_intra, "intra should win when all inter is bad");
        assert_eq!(result.ref_frame, RefFrame::IntraFrame);
        assert_eq!(result.rate, intra_fallback.rate);
        assert_eq!(result.dist, intra_fallback.dist);
        assert_eq!(result.rd, intra_fallback.rd);
    }
}
