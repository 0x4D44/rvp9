//! W4.8 — `discount_newmv_test` + NEWMV → NEAREST/NEAR/ZERO post-pick
//! rewrite.
//!
//! ## libvpx references
//!
//! - `discount_newmv_test` — `vp9/encoder/vp9_rdopt.c:2764-2810`
//!   (v1.14.1 commit `12f3a2ac6`). The `!CONFIG_NON_GREEDY_MV` branch
//!   (lines 2799-2809) is the load-bearing one for the default build.
//! - Post-pick rewrite — `vp9/encoder/vp9_rdopt.c:4085-4104`. After
//!   `vp9_rd_pick_inter_mode_sb` finishes its mode loop, the chosen
//!   `best_mbmode.mode = NEWMV` is rewritten to `NEARESTMV` /
//!   `NEARMV` / `ZEROMV` when the picked MV matches the corresponding
//!   candidate's MV.
//!
//! ## Scope
//!
//! This is the "biggest cheap RD win" item in the wave. Wiring it
//! reduces NEW-MV rate accounting bias and makes the post-pick mode
//! reflect cheaper-to-signal modes when the MV happens to match a
//! candidate.

use rvp9_core::mode_info::{InterMode, Mv};

/// libvpx `NEW_MV_DISCOUNT_FACTOR` (`vp9_rdopt.c:50`).
pub const NEW_MV_DISCOUNT_FACTOR: u32 = 8;

/// libvpx `INVALID_MV = 0x80008000` (`vp9_mv.h:23`).
pub const INVALID_MV_INT: u32 = 0x8000_8000;

/// libvpx `discount_newmv_test` `!CONFIG_NON_GREEDY_MV` branch
/// (`vp9_rdopt.c:2799-2809`).
///
/// Returns `true` when the picker should discount the NEWMV rate cost
/// by `NEW_MV_DISCOUNT_FACTOR` (to encourage initiation of a weak
/// motion field). All of:
///
/// 1. The current source frame is NOT an altref reuse
///    (`!cpi->rc.is_src_frame_alt_ref`).
/// 2. `this_mode == NEWMV`.
/// 3. The candidate MV is non-zero (`this_mv != 0`).
/// 4. Both `NEARESTMV` and `NEARMV` for this ref-frame are either
///    `(0,0)` OR `INVALID_MV` (i.e. there's no informative ref MV).
///
/// All four conditions are required; the libvpx C is a single
/// short-circuiting `&&` expression.
#[must_use]
pub fn discount_newmv_test(
    is_src_frame_alt_ref: bool,
    this_mode: InterMode,
    this_mv: Mv,
    nearest_mv: Option<Mv>,
    near_mv: Option<Mv>,
) -> bool {
    if is_src_frame_alt_ref {
        return false;
    }
    if !matches!(this_mode, InterMode::NewMv) {
        return false;
    }
    if mv_as_int(this_mv) == 0 {
        return false;
    }
    // libvpx encodes "INVALID_MV" as the `INVALID_MV_INT` sentinel.
    // We surface that as `None` in the rvp9 API — `Some(Mv::ZERO)`
    // covers the "real zero MV" half of the `||`.
    let nearest_ok = match nearest_mv {
        None => true,                   // INVALID_MV
        Some(mv) => mv_as_int(mv) == 0, // (0, 0)
    };
    let near_ok = match near_mv {
        None => true,
        Some(mv) => mv_as_int(mv) == 0,
    };
    nearest_ok && near_ok
}

/// libvpx post-pick rewrite (`vp9_rdopt.c:4085-4104`).
///
/// Single-ref form: after the mode loop has picked
/// `best_mode == NEWMV` with motion vector `best_mv`, rewrite the mode
/// to a cheaper-to-signal alternative when the picked MV happens to
/// match a candidate's MV:
///
/// - If `best_mv == frame_mv[NEARESTMV]` → `NEARESTMV`.
/// - Else if `best_mv == frame_mv[NEARMV]` → `NEARMV`.
/// - Else if `best_mv == (0, 0)` → `ZEROMV`.
///
/// The libvpx pseudocode uses `int_mv.as_int` for equality (i.e.
/// bit-exact MV equality); we mirror that via [`mv_as_int`].
///
/// `comp_pred_mode = false` (single-ref) is the load-bearing case.
/// The compound path also requires `best_mv[1]` to match the second
/// ref's candidate; that's a follow-up wave item.
#[must_use]
pub fn rewrite_newmv_post_pick(
    best_mode: InterMode,
    best_mv: Mv,
    nearest_frame_mv: Mv,
    near_frame_mv: Mv,
) -> InterMode {
    if !matches!(best_mode, InterMode::NewMv) {
        return best_mode;
    }
    let bmv = mv_as_int(best_mv);
    if mv_as_int(nearest_frame_mv) == bmv {
        InterMode::NearestMv
    } else if mv_as_int(near_frame_mv) == bmv {
        InterMode::NearMv
    } else if bmv == 0 {
        InterMode::ZeroMv
    } else {
        InterMode::NewMv
    }
}

/// libvpx `int_mv.as_int` — packs `(row, col)` into a `u32`. Used for
/// the `==` checks at lines 4090, 4095, 4100.
#[inline]
#[must_use]
pub fn mv_as_int(mv: Mv) -> u32 {
    ((mv.row as u16 as u32) << 16) | (mv.col as u16 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PROOF W4.8 — `rewrite_newmv_post_pick` rewrites NEWMV to NEAREST
    /// when the picked MV matches the NEAREST candidate. libvpx
    /// `vp9_rdopt.c:4090-4094`: if `frame_mv[NEARESTMV][refs[0]].as_int
    /// == best_mbmode.mv[0].as_int` then `best_mbmode.mode = NEARESTMV`.
    #[test]
    fn w4_8_discount_newmv_rewrites_to_nearest_when_mv_matches() {
        let nearest = Mv { row: 16, col: -32 };
        let near = Mv { row: 8, col: 8 };
        // Picker chose NEWMV with mv == nearest → should be rewritten.
        let out = rewrite_newmv_post_pick(InterMode::NewMv, nearest, nearest, near);
        assert!(
            matches!(out, InterMode::NearestMv),
            "expected NEARESTMV, got {out:?}",
        );
    }

    /// PROOF W4.8 — NEWMV(0,0) rewritten to ZEROMV. libvpx
    /// `vp9_rdopt.c:4100-4103`: `best_mbmode.mv[0].as_int == 0` ⇒
    /// `best_mbmode.mode = ZEROMV` (when the NEAREST/NEAR-match
    /// branches above don't fire).
    #[test]
    fn w4_8_post_pick_rewrite_preserves_rate_after_zero_match() {
        // NEAREST / NEAR are both non-zero, so the NEAREST/NEAR branches
        // don't fire; the `mv == 0` branch must.
        let nearest = Mv { row: 16, col: -32 };
        let near = Mv { row: 8, col: 8 };
        let out = rewrite_newmv_post_pick(InterMode::NewMv, Mv::ZERO, nearest, near);
        assert!(
            matches!(out, InterMode::ZeroMv),
            "expected ZEROMV, got {out:?}",
        );
    }

    /// NEAR match takes precedence over the ZEROMV branch (libvpx
    /// lines 4095-4099 are `else if` of the NEAREST branch).
    #[test]
    fn rewrites_to_near_when_mv_matches_and_not_nearest() {
        let nearest = Mv { row: 100, col: 50 };
        let near = Mv { row: 16, col: -32 };
        let out = rewrite_newmv_post_pick(InterMode::NewMv, near, nearest, near);
        assert!(matches!(out, InterMode::NearMv));
    }

    /// When NEAREST is the same MV as the picked one, libvpx's chain
    /// stops at the NEARESTMV rewrite (even if NEAR also matches).
    #[test]
    fn nearest_precedes_near_when_both_match() {
        let mv = Mv { row: 16, col: -32 };
        let out = rewrite_newmv_post_pick(InterMode::NewMv, mv, mv, mv);
        assert!(matches!(out, InterMode::NearestMv));
    }

    /// `discount_newmv_test` returns `false` for non-NEWMV.
    #[test]
    fn discount_false_for_non_newmv() {
        let mv = Mv { row: 16, col: -32 };
        let ok = discount_newmv_test(
            false,
            InterMode::NearestMv,
            mv,
            Some(Mv::ZERO),
            Some(Mv::ZERO),
        );
        assert!(!ok, "discount must be false for non-NEWMV");
    }

    /// `discount_newmv_test` requires all of the conditions in the
    /// libvpx `!CONFIG_NON_GREEDY_MV` branch. With NEWMV + non-zero MV
    /// + zero NEAREST/NEAR, returns true.
    #[test]
    fn discount_true_when_all_conditions_met() {
        let mv = Mv { row: 16, col: -32 };
        let ok = discount_newmv_test(false, InterMode::NewMv, mv, Some(Mv::ZERO), Some(Mv::ZERO));
        assert!(ok, "discount must be true for NEWMV with zero ref-MVs");
    }

    /// When the NEAREST MV is non-zero, the discount must NOT fire.
    /// (We DO have an informative ref MV, so the rate term shouldn't
    /// be discounted.)
    #[test]
    fn discount_false_when_nearest_is_nonzero() {
        let mv = Mv { row: 16, col: -32 };
        let nearest = Mv { row: 8, col: 0 };
        let ok = discount_newmv_test(false, InterMode::NewMv, mv, Some(nearest), Some(Mv::ZERO));
        assert!(!ok);
    }

    /// `is_src_frame_alt_ref = true` disables discount.
    #[test]
    fn discount_false_for_alt_ref_source() {
        let mv = Mv { row: 16, col: -32 };
        let ok = discount_newmv_test(true, InterMode::NewMv, mv, Some(Mv::ZERO), Some(Mv::ZERO));
        assert!(!ok);
    }
}
