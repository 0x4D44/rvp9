//! Encoder mode-decision drivers.
//!
//! This module hosts the RDO-style "try several candidates, pick the
//! one minimising `D + lambda * R`" search loops that the encoder runs
//! per block. Each sub-module corresponds to one of libvpx's
//! `vp9_pick_*` realtime drivers.
//!
//! ## Sub-modules
//!
//! - [`intra`] — luma intra-mode picker. Mirrors libvpx
//!   `vp9_pick_intra_mode` (`vp9_pickmode.c:1155-1213`). F8.0.
//! - [`inter`] — luma inter-mode picker. Mirrors libvpx
//!   `vp9_pick_inter_mode` (`vp9_pickmode.c:1300-1700`) for the
//!   realtime variant. F8.1 — `{ZEROMV, NEARESTMV, NEWMV}` ×
//!   `{LASTFRAME}`, single-ref only.
//! - [`partition`] — variance-based partition picker. Mirrors libvpx
//!   `choose_partitioning` (`vp9_encodeframe.c:1261-...`). F8.2 —
//!   variance descent over a 64x64 SB, NONE / SPLIT only.
//! - [`nonrd_partition`] — motion-aware partition descent (F.3
//!   infrastructure). Mirrors libvpx `nonrd_pick_partition`
//!   (`vp9_encodeframe.c:4922-5210`). Default OFF; gated behind
//!   `RVP9_F3_NONRD_PARTITION_ENABLE` (see F.3 HLD §11).
//!
//! UV-mode search lands as a further sub-module here later.
//!
//! ## RD cost convention
//!
//! All sub-modules use libvpx's `RDCOST` macro
//! (`vp9_rd.h:29-30`):
//!
//! ```text
//! RDCOST(RM, DM, R, D) = ROUND_POWER_OF_TWO(R * RM, 9) + (D << DM)
//! ```
//!
//! where `RM` is the RD multiplier (lambda) from
//! [`crate::cost::compute_rd_mult`], `DM = RDDIV_BITS = 7`, `R` is the
//! rate in 1/512-bit `vp9_prob_cost` units, and `D` is the distortion
//! in SSE units. The shared helper [`rd_cost`] implements this exactly.
//!
//! Returns an `i64` so the term `D << 7` can grow without overflow for
//! large blocks (max `D` is `32 * 32 * 255^2 = 66_585_600 << 7 ≈ 8.5e9`,
//! comfortably below `i64::MAX`).

pub mod block_yrd;
pub mod hadamard;
pub mod inter;
pub mod intra;
pub mod model_rd;
pub mod nonrd_partition;
pub mod partition;
pub mod rd_thresh;
pub mod speed_features;

/// libvpx `RDDIV_BITS` (`vp9_rd.h:26`).
pub const RDDIV_BITS: u32 = 7;

/// libvpx `RDCOST(RM, DM, R, D) = ROUND_POWER_OF_TWO(R * RM, 9) +
/// (D << DM)` with `DM = RDDIV_BITS = 7`.
///
/// `rd_mult` is the qindex-derived lambda from
/// [`crate::cost::compute_rd_mult`]. `rate` is in `vp9_prob_cost` units
/// (1/512-bit). `dist` is sum-of-squared-error.
#[inline]
#[must_use]
pub fn rd_cost(rd_mult: u32, rate: u32, dist: u32) -> i64 {
    use crate::cost::VP9_PROB_COST_SHIFT;
    // R * RM as i64 (saturating to avoid pathological overflow if a
    // caller plumbs in a bogus rate × mult product).
    let r_rm = (rate as i64).saturating_mul(rd_mult as i64);
    // ROUND_POWER_OF_TWO(r_rm, 9).
    let rate_term = (r_rm + (1i64 << (VP9_PROB_COST_SHIFT - 1))) >> VP9_PROB_COST_SHIFT;
    // D << 7 (dist is u32 → i64 cast is exact).
    let dist_term = (dist as i64) << RDDIV_BITS;
    rate_term + dist_term
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: `rd_cost(0, _, _) = dist << 7` — when lambda is zero the
    /// rate term collapses and we get pure distortion in 1/128 units.
    #[test]
    fn rd_cost_zero_lambda_is_pure_distortion() {
        assert_eq!(rd_cost(0, 1000, 42), 42i64 << 7);
        assert_eq!(rd_cost(0, 0, 0), 0);
        assert_eq!(rd_cost(0, u32::MAX, 100), 100i64 << 7);
    }

    /// Sanity: `rd_cost(_, 0, _) = dist << 7` — zero rate means the
    /// rate term is zero (after rounding) regardless of multiplier.
    #[test]
    fn rd_cost_zero_rate_is_pure_distortion() {
        assert_eq!(rd_cost(1_000_000, 0, 42), 42i64 << 7);
    }

    /// `RDCOST(512, 7, 1024, 5)` — the libvpx-style hand-derivation
    /// helper. `(1024 * 512 + 256) >> 9 = 1024`, plus `5 << 7 = 640`,
    /// total = 1664.
    #[test]
    fn rd_cost_hand_derivation() {
        assert_eq!(rd_cost(512, 1024, 5), 1024 + 640);
    }

    /// Monotone: increasing rate at fixed dist gives non-decreasing
    /// rd-cost; increasing dist at fixed rate gives strictly greater
    /// rd-cost (since dist << 7 changes by at least 128 per unit).
    #[test]
    fn rd_cost_monotone() {
        let base = rd_cost(100_000, 500, 1000);
        assert!(rd_cost(100_000, 600, 1000) >= base);
        assert!(rd_cost(100_000, 500, 1001) > base);
    }
}
