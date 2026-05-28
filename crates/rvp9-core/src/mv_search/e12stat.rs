//! E13 — env-gated diamond-search-quality oracle (diagnostic only).
//!
//! Reproduces the E12 finding inside the encoder: for every NEWMV leaf
//! processed by [`crate::mode_decision::inter`], record the raw SAD at
//! three points and accumulate them so the harness can convert the
//! aggregate SAD into an equivalent prediction PSNR:
//!
//! - `sad_zero`   — SAD at the (0,0) prediction (no motion).
//! - `sad_newmv`  — SAD at the diamond search's full-pel output
//!   (`NEWMVraw`, before sub-pel).
//! - `sad_oracle` — SAD at the exhaustive full-pel minimum over the
//!   same `mv_limits` window (the best a full-pel search could do).
//!
//! These are SAD (sum of abs diff), not SSE, so they are NOT directly
//! a PSNR; the harness reports the SAD ratios and the
//! diamond-recovers-fraction `(zero-newmv)/(zero-oracle)`. A fraction
//! near 1.0 means the diamond reaches the oracle; near 0.0 means it is
//! stuck at its seed (the E12 symptom).
//!
//! No coding logic changes: the gate is a single relaxed atomic load
//! when `RVP9_E13_DIAMOND` is unset.

use core::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;

static ON: AtomicBool = AtomicBool::new(false);

/// Per-frame diamond-search-quality aggregate.
#[derive(Copy, Clone, Debug, Default)]
pub struct DiamondStats {
    /// Number of NEWMV leaves observed.
    pub leaves: u64,
    /// Sum of raw SAD at (0,0) over all leaves.
    pub sad_zero: u64,
    /// Sum of raw SAD at the diamond full-pel output over all leaves.
    pub sad_newmv: u64,
    /// Sum of raw SAD at the exhaustive full-pel minimum over all leaves.
    pub sad_oracle: u64,
    /// Count of leaves where the diamond reached the oracle SAD exactly.
    pub reached_oracle: u64,
    /// Count of leaves where the diamond improved on (0,0) at all.
    pub improved_on_zero: u64,
    /// Count of leaves that took the int_pro-seed short-circuit path
    /// (diamond skipped).
    pub seeded_leaves: u64,
    /// Sum of |newmv_row|+|newmv_col| (full-pel) — how far the chosen
    /// MV travels from origin on average.
    pub mv_mag_sum: u64,
    /// Count of leaves where NEWMV won the RD argmin (the shipped mode).
    pub newmv_won: u64,
    /// Count of leaves where the argmin was observed at all.
    pub argmin_leaves: u64,
}

thread_local! {
    static STATS: RefCell<DiamondStats> = const { RefCell::new(DiamondStats {
        leaves: 0, sad_zero: 0, sad_newmv: 0, sad_oracle: 0,
        reached_oracle: 0, improved_on_zero: 0, seeded_leaves: 0, mv_mag_sum: 0,
        newmv_won: 0, argmin_leaves: 0,
    }) };
}

/// Enable the oracle if `RVP9_E13_DIAMOND` is set and non-zero.
pub fn init_from_env() {
    let on = std::env::var_os("RVP9_E13_DIAMOND").is_some_and(|v| v != "0");
    ON.store(on, Ordering::Relaxed);
}

/// Whether the oracle is active.
#[inline]
#[must_use]
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Record one NEWMV leaf's three SAD measurements.
#[inline]
pub fn record(sad_zero: u32, sad_newmv: u32, sad_oracle: u32, seeded: bool, mv_mag: u32) {
    if !enabled() {
        return;
    }
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.leaves += 1;
        s.sad_zero += u64::from(sad_zero);
        s.sad_newmv += u64::from(sad_newmv);
        s.sad_oracle += u64::from(sad_oracle);
        if sad_newmv <= sad_oracle {
            s.reached_oracle += 1;
        }
        if sad_newmv < sad_zero {
            s.improved_on_zero += 1;
        }
        if seeded {
            s.seeded_leaves += 1;
        }
        s.mv_mag_sum += u64::from(mv_mag);
    });
}

/// Record the RD-argmin outcome for one inter leaf: whether NEWMV won.
#[inline]
pub fn record_argmin(newmv_won: bool) {
    if !enabled() {
        return;
    }
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.argmin_leaves += 1;
        if newmv_won {
            s.newmv_won += 1;
        }
    });
}

/// Take and reset the per-frame stats for the current thread.
pub fn take_snapshot() -> DiamondStats {
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        let snap = *s;
        *s = DiamondStats::default();
        snap
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-test sweep covering every branch:
    ///   * `enabled()` gate in `record`
    ///   * `enabled()` gate in `record_argmin`
    ///   * `sad_newmv <= sad_oracle` arm in `record`
    ///   * `sad_newmv < sad_zero` arm in `record`
    ///   * `seeded` arm in `record`
    ///   * `newmv_won` arm in `record_argmin`
    #[test]
    fn full_coverage_sweep() {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                ON.store(self.0, Ordering::Relaxed);
            }
        }
        let _restore = Restore(ON.load(Ordering::Relaxed));

        // Drain thread-local state.
        let _ = take_snapshot();

        // ---------------------------------------------------------
        // Branch: enabled() == false → both record* are no-ops.
        // ---------------------------------------------------------
        ON.store(false, Ordering::Relaxed);
        assert!(!enabled());
        record(1000, 500, 300, true, 7);
        record_argmin(true);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 0, "record disabled no-op");
        assert_eq!(snap.argmin_leaves, 0, "record_argmin disabled no-op");
        assert_eq!(snap.seeded_leaves, 0);
        assert_eq!(snap.newmv_won, 0);

        // ---------------------------------------------------------
        // Enable and hit every arm of `record`:
        // (a) newmv == oracle AND improved on zero AND seeded.
        //     sad_zero=100, sad_newmv=50, sad_oracle=50 → reached_oracle++,
        //     improved_on_zero++, seeded_leaves++.
        // ---------------------------------------------------------
        ON.store(true, Ordering::Relaxed);
        assert!(enabled());

        record(100, 50, 50, true, 7);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 1);
        assert_eq!(snap.sad_zero, 100);
        assert_eq!(snap.sad_newmv, 50);
        assert_eq!(snap.sad_oracle, 50);
        assert_eq!(snap.reached_oracle, 1, "newmv == oracle");
        assert_eq!(snap.improved_on_zero, 1, "newmv < zero");
        assert_eq!(snap.seeded_leaves, 1, "seeded=true");
        assert_eq!(snap.mv_mag_sum, 7);

        // (b) newmv < oracle (also <= oracle) AND improved on zero AND NOT seeded.
        record(200, 30, 40, false, 5);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 1);
        assert_eq!(snap.reached_oracle, 1, "30 <= 40 still counts");
        assert_eq!(snap.improved_on_zero, 1);
        assert_eq!(snap.seeded_leaves, 0, "seeded=false must NOT bump");
        assert_eq!(snap.mv_mag_sum, 5);

        // (c) newmv > oracle (failed to reach) AND did NOT improve on zero AND not seeded.
        //     sad_zero=50, sad_newmv=100, sad_oracle=30 → none of the optional bumps.
        record(50, 100, 30, false, 0);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 1);
        assert_eq!(snap.sad_zero, 50);
        assert_eq!(snap.sad_newmv, 100);
        assert_eq!(snap.sad_oracle, 30);
        assert_eq!(snap.reached_oracle, 0, "100 > 30, did NOT reach oracle");
        assert_eq!(
            snap.improved_on_zero, 0,
            "100 >= 50, did NOT improve on zero"
        );
        assert_eq!(snap.seeded_leaves, 0);
        assert_eq!(snap.mv_mag_sum, 0);

        // (d) Boundary: newmv == zero is NOT strictly less than zero
        //     (improved_on_zero uses `<`, not `<=`).
        record(100, 100, 50, false, 3);
        let snap = take_snapshot();
        assert_eq!(snap.improved_on_zero, 0, "newmv == zero is not 'improved'");

        // ---------------------------------------------------------
        // record_argmin: newmv_won==true bumps both; ==false bumps only argmin_leaves.
        // ---------------------------------------------------------
        record_argmin(true);
        let snap = take_snapshot();
        assert_eq!(snap.argmin_leaves, 1);
        assert_eq!(snap.newmv_won, 1);

        record_argmin(false);
        let snap = take_snapshot();
        assert_eq!(snap.argmin_leaves, 1);
        assert_eq!(snap.newmv_won, 0, "newmv_won=false must NOT bump");

        // ---------------------------------------------------------
        // Accumulation across many calls.
        // ---------------------------------------------------------
        record(100, 50, 60, true, 2);
        record(80, 40, 40, false, 1);
        record_argmin(true);
        record_argmin(true);
        record_argmin(false);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 2);
        assert_eq!(snap.sad_zero, 180);
        assert_eq!(snap.sad_newmv, 90);
        assert_eq!(snap.sad_oracle, 100);
        assert_eq!(snap.reached_oracle, 2, "both 50<=60 and 40<=40");
        assert_eq!(snap.improved_on_zero, 2);
        assert_eq!(snap.seeded_leaves, 1);
        assert_eq!(snap.mv_mag_sum, 3);
        assert_eq!(snap.argmin_leaves, 3);
        assert_eq!(snap.newmv_won, 2);

        // take_snapshot resets state.
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 0);
        assert_eq!(snap.sad_zero, 0);
        assert_eq!(snap.sad_newmv, 0);
        assert_eq!(snap.sad_oracle, 0);
        assert_eq!(snap.reached_oracle, 0);
        assert_eq!(snap.improved_on_zero, 0);
        assert_eq!(snap.seeded_leaves, 0);
        assert_eq!(snap.mv_mag_sum, 0);
        assert_eq!(snap.argmin_leaves, 0);
        assert_eq!(snap.newmv_won, 0);
    }
}
